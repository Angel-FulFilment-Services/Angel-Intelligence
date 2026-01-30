"""
Angel Intelligence - Call Analysis Service

AI-powered call analysis using Qwen2.5-Omni-7B.
Supports two modes:
- Audio mode: Analyses audio directly for tone, emotion, sentiment
- Transcript mode: Analyses transcript text only

All outputs use British English (en-GB).
"""

import gc
import json
import logging
import os
import time
import uuid
from typing import Dict, Any, List, Optional

import torch

from src.config import get_settings
from src.database.models import calculate_quality_score, get_quality_zone
from src.services.enquiry_context import get_enquiry_context_service
from src.services.order_context import get_order_context_service

logger = logging.getLogger(__name__)

# Check for model availability
try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor, BitsAndBytesConfig
    QWEN_OMNI_AVAILABLE = True
except ImportError:
    QWEN_OMNI_AVAILABLE = False
    logger.warning("Qwen2.5-Omni not available - install with: pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview")

# Text-only model support (for transcript mode)
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TEXT_MODEL_AVAILABLE = True
except ImportError:
    TEXT_MODEL_AVAILABLE = False
    logger.warning("transformers AutoModelForCausalLM not available")

# LoRA adapter support (for fine-tuned models)
try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.debug("peft not available - fine-tuned adapters will not be loaded")

try:
    from qwen_omni_utils import process_mm_info
    QWEN_OMNI_UTILS_AVAILABLE = True
except ImportError:
    QWEN_OMNI_UTILS_AVAILABLE = False
    logger.warning("qwen_omni_utils not available - install with: pip install qwen-omni-utils[decord]")

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# JSON repair for LLM outputs
try:
    from json_repair import repair_json
    JSON_REPAIR_AVAILABLE = True
except ImportError:
    JSON_REPAIR_AVAILABLE = False
    logger.warning("json-repair not available - install with: pip install json-repair")


# British English system prompt for all analysis
BRITISH_ENGLISH_INSTRUCTION = """
IMPORTANT: All output text must use British English (en-GB) spelling and conventions:
- Use -ise not -ize (organise, recognise, analyse)
- Use -our not -or (colour, behaviour, favour)
- Use -re not -er (centre, metre)
- Use -ogue not -og (catalogue, dialogue)
- Use -ence not -ense (licence, defence)
- Use DD/MM/YYYY date format
- Use £ (GBP) for currency
"""


class AnalysisService:
    """
    Call analysis service using AI models.
    
    Supports:
    - Audio analysis: Direct audio processing with Qwen2.5-Omni
    - Transcript analysis: Text-based analysis
    - Mock mode: Returns deterministic test data for development
    """
    
    def __init__(self):
        """Initialise the analysis service."""
        self.settings = get_settings()
        settings = self.settings  # Local alias for convenience
        
        # Configuration
        self.analysis_mode = settings.analysis_mode  # 'audio' or 'transcript'
        self.use_mock = settings.use_mock_models
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu"
        
        # Quantization setting
        self.quantization = settings.analysis_model_quantization  # 'int4', 'int8', or ''
        
        # Model paths
        self.analysis_model_path = settings.get_analysis_model_path()
        self.chat_model_path = settings.get_chat_model_path()
        
        # Transcript length limit - smaller models need shorter transcripts
        # Set to 0 for unlimited (recommended for 14B+ models)
        self.max_transcript_length = getattr(settings, 'max_transcript_length', 0)
        
        # External LLM API (vLLM/TGI) - when set, uses API instead of local models
        self.llm_api_url = getattr(settings, 'llm_api_url', '') or ''
        self.llm_api_key = getattr(settings, 'llm_api_key', '') or ''
        
        # Audio analysis API (vLLM serving Qwen2.5-Omni) - when set, uses API for audio mode
        self.audio_analysis_api_url = getattr(settings, 'audio_analysis_api_url', '') or ''
        
        # LoRA adapter name for vLLM (e.g., "call-analysis", "email-analysis")
        self.analysis_adapter_name = getattr(settings, 'analysis_adapter_name', 'call-analysis') or 'call-analysis'
        
        # Models (lazy loaded) - only used when llm_api_url is not set
        self._analysis_model = None  # Qwen2.5-Omni for audio mode
        self._analysis_processor = None
        self._text_model = None  # Text-only model for transcript mode
        self._text_tokenizer = None
        self._chat_model = None
        self._chat_processor = None
        
        # Config service for three-tier configuration
        from src.services.config import get_config_service
        self._config_service = get_config_service()
        
        # Legacy: Load config for backwards compatibility
        self._config = self._load_config()
        
        # Log initialisation with backend details
        if self.analysis_mode == "audio":
            backend_desc = "vLLM API" if self.audio_analysis_api_url else f"local ({self.device})"
        else:
            backend_desc = "vLLM API" if self.llm_api_url else f"local ({self.device})"
        logger.info(f"AnalysisService initialised: mode={self.analysis_mode}, backend={backend_desc}")
    
    def _load_config(self, config_path: str = "call_analysis_config.json") -> Dict[str, Any]:
        """Load analysis configuration."""
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"topics": [], "agent_actions": [], "performance_rubric": []}

    def _ensure_analysis_model_loaded(self) -> None:
        """Ensure the analysis model is loaded."""
        if self.use_mock:
            logger.debug("Using mock mode - skipping model load")
            return
        
        if not QWEN_OMNI_AVAILABLE or not QWEN_OMNI_UTILS_AVAILABLE:
            raise RuntimeError("Qwen2.5-Omni or qwen_omni_utils not installed")
        
        if self._analysis_model is None:
            logger.info(f"Loading analysis model: {self.analysis_model_path}")
            
            # Build load kwargs
            load_kwargs = {
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True,
            }
            
            # Apply quantization if configured (GPU only - no CPU offload for int4)
            if self.quantization == "int4" and self.device == "cuda":
                logger.info("Using int4 quantization for analysis model")
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif self.quantization == "int8" and self.device == "cuda":
                logger.info("Using int8 quantization for analysis model")
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
            else:
                load_kwargs["torch_dtype"] = torch.bfloat16 if self.device == "cuda" else torch.float32
            
            self._analysis_model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self.analysis_model_path,
                **load_kwargs
            )
            
            # Disable audio output to save VRAM
            self._analysis_model.disable_talker()
            
            self._analysis_processor = Qwen2_5OmniProcessor.from_pretrained(
                self.analysis_model_path,
                trust_remote_code=True
            )
            
            logger.info(f"Analysis model loaded on {self.device}")
    
    def _ensure_text_model_loaded(self) -> None:
        """Ensure the text-only model is loaded (for transcript mode)."""
        if self.use_mock:
            logger.debug("Using mock mode - skipping model load")
            return
        
        if not TEXT_MODEL_AVAILABLE:
            raise RuntimeError("transformers AutoModelForCausalLM not installed")
        
        if self._text_model is None:
            logger.info(f"Loading text model for transcript analysis: {self.analysis_model_path}")
            
            # Load tokenizer
            self._text_tokenizer = AutoTokenizer.from_pretrained(
                self.analysis_model_path,
                trust_remote_code=True
            )
            
            # Build load kwargs
            load_kwargs = {
                "device_map": "auto" if self.device == "cuda" else None,
                "trust_remote_code": True,
            }
            
            # Apply quantization if configured (GPU only - no CPU offload for int4)
            if self.quantization == "int4" and self.device == "cuda":
                logger.info("Using int4 quantization for text model")
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            elif self.quantization == "int8" and self.device == "cuda":
                logger.info("Using int8 quantization for text model")
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True,
                )
            else:
                load_kwargs["torch_dtype"] = torch.bfloat16 if self.device == "cuda" else torch.float32
            
            try:
                self._text_model = AutoModelForCausalLM.from_pretrained(
                    self.analysis_model_path,
                    **load_kwargs
                )
                
                # Load LoRA adapter if available
                self._text_model = self._load_adapter_if_available(self._text_model)
                
            except Exception as e:
                # Fallback to CPU with float32 if quantized loading fails
                logger.warning(f"Quantized model loading failed: {e}")
                logger.info("Falling back to CPU with float32 (slower but compatible)")
                load_kwargs = {
                    "device_map": None,
                    "trust_remote_code": True,
                    "torch_dtype": torch.float32,
                }
                self._text_model = AutoModelForCausalLM.from_pretrained(
                    self.analysis_model_path,
                    **load_kwargs
                )
                self._text_model = self._text_model.to("cpu")
                self.device = "cpu"  # Update device for this session
                
                # Try to load adapter even on CPU
                self._text_model = self._load_adapter_if_available(self._text_model)
            
            logger.info(f"Text model loaded on {self.device}")
    
    def _load_adapter_if_available(self, model):
        """
        Load LoRA adapter if one has been trained.
        
        Uses TrainingService to find the current versioned adapter.
        
        Args:
            model: Base model to apply adapter to
            
        Returns:
            Model with adapter applied, or original model if no adapter
        """
        if not PEFT_AVAILABLE:
            return model
        
        from pathlib import Path
        
        # Use TrainingService to get the current adapter path (supports versioning)
        try:
            from src.services.trainer import TrainingService
            adapter_name = getattr(self.settings, 'analysis_adapter_name', 'call-analysis')
            trainer = TrainingService(adapter_name=adapter_name)
            adapter_path = trainer.get_current_adapter_path()
            
            if adapter_path is None:
                logger.debug("No trained adapter found, using base model")
                return model
                
        except ImportError:
            # Fallback to legacy path if TrainingService not available
            adapter_path = Path(self.settings.models_base_path) / "adapters" / "call-analysis"
            if not (adapter_path / "adapter_config.json").exists():
                logger.debug("No trained adapter found, using base model")
                return model
        
        adapter_config = adapter_path / "adapter_config.json"
        if not adapter_config.exists():
            logger.debug("No trained adapter found, using base model")
            return model
        
        try:
            logger.info(f"Loading LoRA adapter from {adapter_path}")
            model = PeftModel.from_pretrained(model, str(adapter_path))
            
            # Log adapter info
            metadata_path = adapter_path / "training_metadata.json"
            if metadata_path.exists():
                import json
                with open(metadata_path) as f:
                    metadata = json.load(f)
                version = metadata.get('version', 'unknown')
                logger.info(f"Adapter version '{version}' trained at {metadata.get('trained_at')} with {metadata.get('samples_used')} samples")
            
            return model
            
        except Exception as e:
            logger.warning(f"Failed to load adapter: {e}. Using base model.")
            return model
    
    def analyse(
        self, 
        audio_path: Optional[str],
        transcript: Dict[str, Any],
        recording_id: int,
        client_ref: Optional[str] = None,
        campaign_type: Optional[str] = None,
        direction: Optional[str] = None,
        enqref: Optional[str] = None,
        orderref: Optional[str] = None,
        ddi: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyse a call recording.
        
        Args:
            audio_path: Path to audio file (required for audio mode)
            transcript: Transcript dictionary from transcription service
            recording_id: Database recording ID
            client_ref: Client reference for client-specific config (optional)
            campaign_type: Campaign type for campaign-specific config (optional)
            direction: Call direction for direction-specific config (optional)
            enqref: Enquiry reference for calltype validation (optional)
            orderref: Order reference for data capture validation (optional)
            ddi: DDI phone number for calltype group lookup (optional)
            
        Returns:
            Analysis results dictionary
        """
        start_time = time.time()
        
        # Calculate speaker metrics (always needed)
        speaker_metrics = self._calculate_speaker_metrics(transcript)
        
        # Build combined context from enquiry and order data
        context_parts = []
        
        # Get enquiry context for calltype validation (if enqref provided)
        if enqref and client_ref:
            try:
                enquiry_service = get_enquiry_context_service()
                enquiry_context = enquiry_service.get_enquiry_context(
                    enqref=enqref,
                    client_ref=client_ref,
                    ddi=ddi
                )
                if enquiry_context.has_context:
                    context_parts.append(enquiry_context.to_prompt_context())
                    logger.info(f"Enquiry context loaded for {enqref}: {len(enquiry_context.available_calltypes)} calltypes available")
                elif enquiry_context.error:
                    logger.debug(f"No enquiry context for {enqref}: {enquiry_context.error}")
            except Exception as e:
                logger.warning(f"Failed to get enquiry context: {e}")
        
        # Get order context for data capture validation (if orderref provided)
        if orderref and client_ref:
            try:
                order_service = get_order_context_service()
                order_context = order_service.get_order_context(
                    orderref=orderref,
                    client_ref=client_ref,
                    ddi=ddi
                )
                if order_context.has_context:
                    context_parts.append(order_context.to_prompt_context())
                    logger.info(f"Order context loaded for {orderref}")
                elif order_context.error:
                    logger.debug(f"No order context for {orderref}: {order_context.error}")
            except Exception as e:
                logger.warning(f"Failed to get order context: {e}")
        
        # Combine context parts
        combined_context_str = "\n\n".join(context_parts) if context_parts else ""
        
        if self.use_mock:
            logger.info("Using mock analysis response")
            return self._mock_analysis(speaker_metrics, start_time)
        
        if self.analysis_mode == "audio" and audio_path:
            return self._analyse_audio(audio_path, transcript, speaker_metrics, start_time, client_ref, campaign_type, direction, combined_context_str)
        else:
            return self._analyse_transcript(transcript, speaker_metrics, start_time, client_ref, campaign_type, direction, combined_context_str)
    
    def _calculate_speaker_metrics(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for each speaker matching specification format."""
        raw_metrics = {}
        total_time = 0
        
        for segment in transcript.get("segments", []):
            speaker = segment.get("speaker", "unknown")
            duration = segment.get("end", 0) - segment.get("start", 0)
            total_time += duration
            
            # Normalize speaker label to agent/supporter
            if speaker in ["agent", "SPEAKER_00"] or speaker.startswith("agent"):
                speaker_label = "agent"
            elif speaker in ["supporter", "SPEAKER_01"] or speaker.startswith("supporter"):
                speaker_label = "supporter"
            else:
                # Unknown speaker - try to infer from speaker ID
                speaker_label = "supporter" if "01" in speaker else "agent"
            
            if speaker_label not in raw_metrics:
                raw_metrics[speaker_label] = {
                    "talk_time_seconds": 0,
                    "word_count": 0,
                    "segment_count": 0,
                    "interruptions": 0,
                }
            
            text = segment.get("text", "")
            raw_metrics[speaker_label]["talk_time_seconds"] += duration
            raw_metrics[speaker_label]["word_count"] += len(text.split())
            raw_metrics[speaker_label]["segment_count"] += 1
        
        # Calculate percentages and WPM
        formatted_metrics = {}
        for speaker_label, data in raw_metrics.items():
            talk_time = data["talk_time_seconds"]
            word_count = data["word_count"]
            
            formatted_metrics[speaker_label] = {
                "talk_time_seconds": round(talk_time, 1),
                "talk_time_percentage": round((talk_time / total_time * 100) if total_time > 0 else 0, 1),
                "word_count": word_count,
                "average_pace_wpm": round((word_count / (talk_time / 60)) if talk_time > 0 else 0),
                "interruptions": data["interruptions"],
                "silence_percentage": 0,  # Would need more analysis
            }
        
        return formatted_metrics
    
    def _format_transcript_with_timestamps(self, transcript: Dict[str, Any]) -> tuple[str, Dict[str, Dict]]:
        """
        Format transcript segments with timestamps and IDs for LLM analysis.
        
        Uses existing segment_id from transcript (added by transcriber service).
        segment_id is UUID-based (e.g., seg_a1b2c3d4) for lifecycle traceability.
        
        Produces format like:
        [seg_a1b2c3d4] [0.0s - 15.2s] Agent: Good morning, thank you for calling...
        [seg_e5f6g7h8] [15.2s - 22.5s] Supporter: Hello, I wanted to ask about...
        
        Returns:
            tuple: (formatted_transcript_string, segment_map)
                - segment_map: {segment_id: {"start": float, "end": float, "speaker": str}}
        """
        segments = transcript.get("segments", [])
        
        if not segments:
            # Fall back to plain text if no segments
            return transcript.get("full_transcript", ""), {}
        
        formatted_lines = []
        segment_map = {}
        
        for i, seg in enumerate(segments):
            start = seg.get("start", 0)
            end = seg.get("end", 0)
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "").strip()
            
            if not text:
                continue
            
            # Use existing segment_id from transcript, or generate UUID-based fallback
            segment_id = seg.get("segment_id") or f"seg_{uuid.uuid4().hex[:8]}"
            
            # Normalise speaker label for readability
            if speaker in ["SPEAKER_00", "agent"] or speaker.startswith("agent"):
                speaker_label = "Agent"
            elif speaker in ["SPEAKER_01", "supporter"] or speaker.startswith("supporter"):
                speaker_label = "Supporter"
            else:
                speaker_label = speaker.title()
            
            # Build segment map for reference
            segment_map[segment_id] = {
                "start": start,
                "end": end,
                "speaker": speaker_label.lower()
            }
            
            # Format with segment ID and timestamp
            formatted_lines.append(f"[{segment_id}] [{start:.1f}s - {end:.1f}s] {speaker_label}: {text}")
        
        return "\n".join(formatted_lines), segment_map
    
    def _analyse_audio(
        self, 
        audio_path: str, 
        transcript: Dict[str, Any],
        speaker_metrics: Dict[str, Any],
        start_time: float,
        client_ref: Optional[str] = None,
        campaign_type: Optional[str] = None,
        direction: Optional[str] = None,
        additional_context_str: str = ""
    ) -> Dict[str, Any]:
        """Analyse call using audio directly with transcript context."""
        try:
            # Format transcript with timestamps and segment IDs (same as transcript analysis)
            formatted_transcript, segment_map = self._format_transcript_with_timestamps(transcript)
            
            # Calculate max timestamp from segments
            segments = transcript.get("segments", [])
            max_timestamp = max((seg.get("end", 0) for seg in segments), default=0)
            
            # Get merged config for this context (all four tiers) - same as transcript analysis
            merged_config = self._config_service.get_merged_config(
                campaign_type=campaign_type,
                direction=direction,
                client_ref=client_ref
            )
            
            # Merge additional context (enquiry/order data) into prompt_context
            if additional_context_str:
                existing_context = merged_config.get("prompt_context", "")
                merged_config["prompt_context"] = f"{existing_context}\n\n{additional_context_str}".strip()
            
            # Use vLLM API if configured, otherwise use local model
            if self.audio_analysis_api_url:
                return self._analyse_audio_via_api(
                    audio_path=audio_path,
                    transcript=transcript,
                    formatted_transcript=formatted_transcript,
                    segment_map=segment_map,
                    speaker_metrics=speaker_metrics,
                    max_timestamp=max_timestamp,
                    merged_config=merged_config,
                    start_time=start_time
                )
            
            # Local model path - load Qwen2.5-Omni
            self._ensure_analysis_model_loaded()
            
            # Preprocess audio
            processed_audio = self._preprocess_audio(audio_path)
            
            # Chunk audio for processing
            chunks = self._chunk_audio(processed_audio, max_duration=30)
            
            # Sample chunks for efficiency (beginning, middle, end)
            if len(chunks) > 3:
                sample_chunks = [chunks[0], chunks[len(chunks)//2], chunks[-1]]
            else:
                sample_chunks = chunks
            
            # Get transcript segments for each chunk
            def get_segments_for_chunk(chunk_start: float, chunk_end: float) -> str:
                """Get formatted transcript lines that fall within the chunk time range."""
                chunk_lines = []
                for seg in segments:
                    seg_start = seg.get("start", 0)
                    seg_end = seg.get("end", 0)
                    # Include segment if it overlaps with chunk
                    if seg_start < chunk_end and seg_end > chunk_start:
                        segment_id = seg.get("segment_id") or f"seg_{uuid.uuid4().hex[:8]}"
                        speaker = seg.get("speaker", "Unknown")
                        if speaker in ["SPEAKER_00", "agent"] or speaker.startswith("agent"):
                            speaker_label = "Agent"
                        elif speaker in ["SPEAKER_01", "supporter"] or speaker.startswith("supporter"):
                            speaker_label = "Supporter"
                        else:
                            speaker_label = speaker.title()
                        text = seg.get("text", "").strip()
                        if text:
                            chunk_lines.append(f"[{segment_id}] [{seg_start:.1f}s - {seg_end:.1f}s] {speaker_label}: {text}")
                return "\n".join(chunk_lines)
            
            # Analyse each chunk with transcript context
            chunk_analyses = []
            for chunk_start, chunk_end, chunk_path in sample_chunks:
                chunk_transcript = get_segments_for_chunk(chunk_start, chunk_end)
                analysis = self._analyse_audio_chunk(
                    chunk_path,
                    chunk_start=chunk_start,
                    chunk_end=chunk_end,
                    chunk_transcript=chunk_transcript,
                    merged_config=merged_config
                )
                chunk_analyses.append(analysis)
                
                # Clean up temp chunk
                if chunk_path != processed_audio and os.path.exists(chunk_path):
                    os.unlink(chunk_path)
            
            # Clean up preprocessed audio
            if processed_audio != audio_path and os.path.exists(processed_audio):
                os.unlink(processed_audio)
            
            # Aggregate results into same structure as transcript analysis
            result = self._aggregate_audio_analyses(chunk_analyses, speaker_metrics, segment_map, max_timestamp)
            result["processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}", exc_info=True)
            return self._empty_analysis(speaker_metrics, start_time)
        finally:
            # Free GPU memory for next recording's transcription (only if using local model)
            if not self.audio_analysis_api_url:
                self._unload_analysis_model()
    
    def _unload_analysis_model(self) -> None:
        """Unload audio analysis model to free GPU memory."""
        if self._analysis_model is not None:
            logger.debug("Unloading analysis model to free GPU memory")
            del self._analysis_model
            del self._analysis_processor
            self._analysis_model = None
            self._analysis_processor = None
            
            # Force garbage collection and clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _analyse_audio_via_api(
        self,
        audio_path: str,
        transcript: Dict[str, Any],
        formatted_transcript: str,
        segment_map: Dict[str, Dict],
        speaker_metrics: Dict[str, Any],
        max_timestamp: float,
        merged_config: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """
        Analyse call audio using vLLM API serving Qwen2.5-Omni.
        
        Uses OpenAI-compatible multimodal endpoint with base64-encoded audio.
        """
        import base64
        import httpx
        
        try:
            # Build the analysis prompt (same structure as chunk analysis)
            prompt = self._build_audio_analysis_prompt(
                formatted_transcript=formatted_transcript,
                max_timestamp=max_timestamp,
                merged_config=merged_config
            )
            
            # Debug: Log full prompt for testing
            logger.debug("=" * 80)
            logger.debug("FULL LLM PROMPT (audio API mode):")
            logger.debug("=" * 80)
            logger.debug(prompt)
            logger.debug("=" * 80)
            
            # Read and encode audio as base64
            with open(audio_path, 'rb') as f:
                audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
            
            # Determine audio MIME type
            if audio_path.endswith('.mp3'):
                mime_type = 'audio/mpeg'
            elif audio_path.endswith('.wav'):
                mime_type = 'audio/wav'
            elif audio_path.endswith('.ogg'):
                mime_type = 'audio/ogg'
            else:
                mime_type = 'audio/wav'  # Default assumption
            
            # Build multimodal message with audio
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "audio_url",
                            "audio_url": {
                                "url": f"data:{mime_type};base64,{audio_b64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
            
            headers = {"Content-Type": "application/json"}
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"
            
            payload = {
                "model": self.settings.analysis_model,  # Qwen2.5-Omni-7B
                "messages": messages,
                "max_tokens": 2000,
                "temperature": 0,  # Deterministic for structured output
            }
            
            logger.info(f"Sending audio analysis request to vLLM API: {self.audio_analysis_api_url}")
            
            with httpx.Client(timeout=600.0) as client:  # Long timeout for audio processing
                response = client.post(
                    f"{self.audio_analysis_api_url}/chat/completions",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
            
            result_json = response.json()
            content = result_json["choices"][0]["message"]["content"]
            
            logger.debug(f"Audio API response (first 500 chars): {content[:500]}")
            
            # Parse response using same parser as local analysis
            result = self._parse_analysis_response(content, speaker_metrics)
            
            # Enrich segment_ids with timestamps
            result = self._enrich_segment_timestamps(result, segment_map)
            
            result["processing_time"] = time.time() - start_time
            result["analysis_type"] = "audio_api"
            
            return result
            
        except httpx.HTTPStatusError as e:
            logger.error(f"Audio analysis API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Audio analysis via API failed: {e}", exc_info=True)
            raise
    
    def _build_audio_analysis_prompt(
        self,
        formatted_transcript: str,
        max_timestamp: float,
        merged_config: Dict[str, Any]
    ) -> str:
        """Build prompt for audio-based analysis via API."""
        # Extract config values
        topics = merged_config.get("topics", [])
        actions = merged_config.get("agent_actions", [])
        rubric = merged_config.get("performance_rubric", [])
        prompt_context = merged_config.get("prompt_context", "")
        
        # Build lists
        topic_list = ", ".join(topics) if topics else "donation, gift aid, regular giving, complaints, account updates"
        action_list = ", ".join(actions) if actions else "greeting, verification, explanation, objection handling, closing"
        
        # Convert rubric items to valid JSON keys
        def to_key(name: str) -> str:
            if isinstance(name, dict):
                name = name.get("name", "")
            return name.replace(" ", "_").replace("&", "and").replace("/", "_")
        
        perf_keys = [to_key(r) for r in rubric] if rubric else [
            "Empathy", "Clarity", "Listening", "Script_adherence",
            "Product_knowledge", "Rapport_building", "Objection_handling", "Closing_effectiveness"
        ]
        perf_keys_list = ", ".join(perf_keys)
        perf_scores_json = ",\n        ".join([f'"{k}": 1-10' for k in perf_keys])
        
        context_section = f"\n{prompt_context}\n" if prompt_context else ""
        
        return f"""You are an expert call quality analyst for charity fundraising and engagement.
Analyse this audio recording alongside its transcript and provide detailed coaching insights.
{context_section}
CALL DURATION: {max_timestamp:.1f} seconds

TRANSCRIPT (each line has [segment_id] [start - end] Speaker: text):
{formatted_transcript}

TOPICS TO IDENTIFY: {topic_list}
AGENT ACTIONS TO DETECT: {action_list}
PERFORMANCE CRITERIA: {perf_keys_list}

Return your analysis as valid JSON matching this exact structure:

{{
    "summary": "2-3 sentence British English summary describing the call purpose, outcome, and notable moments",
    "sentiment_score": 0 to 10 (0=hostile, 5=neutral, 10=very friendly),
    "key_topics": [
        {{"name": "Topic Name In Title Case", "confidence": 0.0-1.0}}
    ],
    "agent_actions": [
        {{"action": "specific action the agent took", "segment_ids": ["seg_a1b2c3d4", "seg_e5f6g7h8"]}}
    ],
    "score_impacts": [
        {{"segment_ids": ["seg_a1b2c3d4"], "impact": -5 to +5, "category": "one of: {perf_keys_list}", "reason": "why this affected the score", "quote": "exact quote from transcript"}}
    ],
    "compliance_flags": [
        {{"type": "GDPR|payment_security|misleading_info|rudeness|data_protection", "segment_ids": ["seg_a1b2c3d4", "seg_e5f6g7h8"], "severity": "low/medium/high/critical", "issue": "detailed description", "quote": "exact quote from transcript"}}
    ],
    "performance_scores": {{
        {perf_scores_json}
    }},
    "audio_observations": {{
        "agent_tone": "warm/neutral/cold/hostile",
        "agent_emotion": "calm/frustrated/happy/anxious/other",
        "supporter_emotion": "calm/frustrated/happy/anxious/other",
        "speech_clarity": 1-10,
        "background_noise": "low/medium/high",
        "pacing": "rushed/steady/slow"
    }},
    "action_items": [
        {{"description": "specific follow-up action needed", "priority": "high/medium/low"}}
    ]
}}

AUDIO ANALYSIS GUIDANCE:
- Listen for TONE OF VOICE - warmth, empathy, frustration, impatience
- Listen for PACING - rushing through information, giving supporter time to respond
- Listen for INTERRUPTIONS - agent cutting off supporter
- Listen for BACKGROUND NOISE affecting call quality
- Use segment_ids ARRAY to reference specific moments - actions/impacts can span multiple segments
- Include specific quotes from the transcript as evidence
- ANALYSE THE ENTIRE CALL from start to finish

SEGMENT IDS - CRITICAL:
- segment_ids is an ARRAY - use ["seg_a1b2c3d4"] for single segment or ["seg_a1b2c3d4", "seg_e5f6g7h8"] for spans
- Copy segment_id values EXACTLY from the transcript (they are UUID-based like seg_a1b2c3d4)
- Do NOT invent segment IDs - only use ones that appear in the transcript

SCORE IMPACT SCALE:
- +5: Exceptional moment (exemplary)
- +3 to +4: Strong positive
- +1 to +2: Minor positive
- -1 to -2: Minor negative
- -3 to -4: Significant negative
- -5: Severe issue

COMPLIANCE FLAGS: Only include if actual issues detected. Use empty array [] if none.

Return ONLY valid JSON - no text before or after."""

    def _analyse_transcript(
        self, 
        transcript: Dict[str, Any],
        speaker_metrics: Dict[str, Any],
        start_time: float,
        client_ref: Optional[str] = None,
        campaign_type: Optional[str] = None,
        direction: Optional[str] = None,
        additional_context_str: str = ""
    ) -> Dict[str, Any]:
        """Analyse call using transcript text only (with text-only model or API)."""
        try:
            # Only load local model if not using external API
            if not self.llm_api_url:
                self._ensure_text_model_loaded()
            
            # Format transcript with timestamps and segment IDs for the LLM
            formatted_transcript, segment_map = self._format_transcript_with_timestamps(transcript)
            
            # Calculate max timestamp from segments for validation
            segments = transcript.get("segments", [])
            max_timestamp = max((seg.get("end", 0) for seg in segments), default=0)
            
            # Get merged config for this context (all four tiers)
            merged_config = self._config_service.get_merged_config(
                campaign_type=campaign_type,
                direction=direction,
                client_ref=client_ref
            )
            
            # Merge additional context (enquiry/order data) into prompt_context
            if additional_context_str:
                existing_context = merged_config.get("prompt_context", "")
                merged_config["prompt_context"] = f"{existing_context}\n\n{additional_context_str}".strip()
            
            # Build prompt with context and merged config (includes overrides)
            prompt = self._build_transcript_analysis_prompt(
                formatted_transcript, 
                max_timestamp,
                prompt_context=merged_config.get("prompt_context", ""),
                merged_config=merged_config
            )
            
            # Log transcript info for debugging
            segment_count = len(segments)
            logger.info(f"Transcript: {segment_count} segments, {len(segment_map)} mapped, max_ts={max_timestamp:.1f}s, {len(prompt)} chars prompt")
            
            # Debug: Log full prompt for testing
            logger.debug("=" * 80)
            logger.debug("FULL LLM PROMPT (transcript mode):")
            logger.debug("=" * 80)
            logger.debug(prompt)
            logger.debug("=" * 80)
            
            # Generate analysis using text-only model or API
            response = self._generate_text_only_response(prompt)
            
            # Parse response
            result = self._parse_analysis_response(response, speaker_metrics)
            
            # Enrich segment_ids with timestamps from the segment_map
            result = self._enrich_segment_timestamps(result, segment_map)
            
            result["processing_time"] = time.time() - start_time
            result["analysis_type"] = "transcript_api" if self.llm_api_url else "transcript"
            
            return result
            
        except Exception as e:
            logger.error(f"Transcript analysis failed: {e}", exc_info=True)
            return self._empty_analysis(speaker_metrics, start_time)
        finally:
            # Free GPU memory for next recording's transcription (only if using local model)
            if not self.llm_api_url:
                self._unload_text_model()
    
    def _unload_text_model(self) -> None:
        """Unload text model to free GPU memory."""
        if self._text_model is not None:
            logger.debug("Unloading text model to free GPU memory")
            del self._text_model
            del self._text_tokenizer
            self._text_model = None
            self._text_tokenizer = None
            
            # Force garbage collection and clear CUDA cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _generate_text_only_response(self, prompt: str) -> str:
        """Generate text response using the text-only model or external API."""
        # Use external vLLM API if configured
        if self.llm_api_url:
            return self._generate_via_api(prompt, adapter_name=self.analysis_adapter_name)
        
        # Otherwise use local model
        return self._generate_via_local_model(prompt)
    
    def _generate_via_api(self, prompt: str, adapter_name: Optional[str] = "call-analysis") -> str:
        """
        Generate text response using external LLM API (vLLM/TGI).
        
        Args:
            prompt: The prompt to send
            adapter_name: LoRA adapter name to use (default: "call-analysis")
                         Set to None to use base model without adapter
        """
        import httpx
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        
        # Build model identifier - base model or with adapter
        model_id = self.settings.analysis_model
        
        payload = {
            "model": model_id,
            "messages": messages,
            "max_tokens": 2000,
            "temperature": 0,  # Deterministic for structured output
        }
        
        # Only add repetition_penalty for vLLM (not supported by OpenAI/Groq)
        if "groq.com" not in self.llm_api_url and "openai.com" not in self.llm_api_url:
            payload["repetition_penalty"] = 1.1
        
        # Add LoRA adapter if specified and available
        if adapter_name:
            # Get the versioned adapter path from TrainingService
            adapter_path = f"/models/adapters/{adapter_name}"
            try:
                from src.services.trainer import TrainingService
                trainer = TrainingService(adapter_name=adapter_name)
                current_version = trainer.get_current_version_name()
                if current_version:
                    adapter_path = f"/models/adapters/{adapter_name}/{current_version}"
            except Exception as e:
                logger.debug(f"Could not get versioned adapter path: {e}")
            
            # vLLM LoRA format: specify adapter in extra_body
            payload["extra_body"] = {
                "lora_request": {
                    "lora_name": adapter_name,
                    "lora_path": adapter_path,
                }
            }
        
        try:
            # Use longer timeout for large model inference
            with httpx.Client(timeout=300.0) as client:
                response = client.post(
                    f"{self.llm_api_url}/chat/completions",
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            logger.debug(f"API response (first 500 chars): {content[:500]}")
            return content
            
        except httpx.HTTPStatusError as e:
            # If adapter not found, retry without adapter
            if e.response.status_code == 400 and adapter_name:
                logger.warning(f"LoRA adapter '{adapter_name}' not found, falling back to base model")
                return self._generate_via_api(prompt, adapter_name=None)
            logger.error(f"LLM API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"LLM API request failed: {e}")
            raise
    
    def _generate_via_local_model(self, prompt: str) -> str:
        """Generate text response using locally loaded model."""
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        text = self._text_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self._text_tokenizer(text, return_tensors="pt")
        inputs = inputs.to(self._text_model.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            output_ids = self._text_model.generate(
                **inputs,
                max_new_tokens=2000,  # Reduced to prevent runaway generation
                do_sample=False,
                repetition_penalty=1.1,  # Penalise repetition to prevent loops
                no_repeat_ngram_size=4,  # Prevent 4-gram repetition
            )
        
        generated_ids = output_ids[:, input_len:]
        response = self._text_tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        # Detect and truncate repetition loops
        response = self._truncate_repetition(response)
        
        logger.debug(f"Model response (first 500 chars): {response[:500]}")
        
        return response
    
    def _truncate_repetition(self, text: str) -> str:
        """Detect and truncate text that has entered a repetition loop."""
        # Look for the last valid JSON closing structure
        # Find patterns like }] or }} that would close the main object
        
        # First, try to find where improvement_areas array closes properly
        import re
        
        # Look for the pattern: improvement_areas array followed by proper closing
        match = re.search(r'"improvement_areas"\s*:\s*\[.*?\}\s*\]\s*\}', text, re.DOTALL)
        if match:
            # Found a complete structure, truncate there
            return text[:match.end()]
        
        # If no clean end found, look for last complete-looking object
        # Find all positions where we have }] or }}
        last_good_end = -1
        
        # Find last }] pattern (end of array in object)
        bracket_matches = list(re.finditer(r'\}\s*\]', text))
        if bracket_matches:
            last_good_end = bracket_matches[-1].end()
        
        # Check for repetition: if we see the same phrase 3+ times, truncate before the 3rd
        lines = text.split('\n')
        seen_lines = {}
        truncate_at = len(text)
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if len(stripped) > 20:  # Only check substantial lines
                if stripped in seen_lines:
                    seen_lines[stripped] += 1
                    if seen_lines[stripped] >= 3:
                        # Found repetition - truncate at this point
                        truncate_at = sum(len(l) + 1 for l in lines[:i])
                        logger.warning(f"Detected repetition loop at line {i}, truncating")
                        break
                else:
                    seen_lines[stripped] = 1
        
        if truncate_at < len(text):
            text = text[:truncate_at]
        
        return text
    
    def _preprocess_audio(self, audio_path: str) -> str:
        """Preprocess audio to 16kHz mono WAV."""
        if not SOUNDFILE_AVAILABLE:
            return audio_path
        
        try:
            audio_data, sample_rate = sf.read(audio_path)
            
            # Already good format
            if sample_rate == 16000 and len(audio_data.shape) == 1:
                return audio_path
            
            # Need to convert
            import tempfile
            import subprocess
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()
            
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-ar', '16000',
                '-ac', '1',
                '-f', 'wav',
                temp_file.name
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            return temp_file.name
            
        except Exception as e:
            logger.warning(f"Audio preprocessing failed: {e}")
            return audio_path
    
    def _chunk_audio(self, audio_path: str, max_duration: int = 30) -> List[tuple]:
        """Split audio into chunks."""
        if not SOUNDFILE_AVAILABLE:
            return [(0, 0, audio_path)]
        
        try:
            audio_data, sample_rate = sf.read(audio_path)
            total_duration = len(audio_data) / sample_rate
            
            if total_duration <= max_duration:
                return [(0, total_duration, audio_path)]
            
            import tempfile
            
            chunks = []
            chunk_samples = max_duration * sample_rate
            
            for i, start_sample in enumerate(range(0, len(audio_data), chunk_samples)):
                end_sample = min(start_sample + chunk_samples, len(audio_data))
                chunk_data = audio_data[start_sample:end_sample]
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                sf.write(temp_file.name, chunk_data, sample_rate)
                
                start_time = start_sample / sample_rate
                end_time = end_sample / sample_rate
                chunks.append((start_time, end_time, temp_file.name))
            
            return chunks
            
        except Exception as e:
            logger.warning(f"Audio chunking failed: {e}")
            return [(0, 0, audio_path)]
    
    def _analyse_audio_chunk(
        self, 
        audio_path: str, 
        chunk_start: float = 0,
        chunk_end: float = 0,
        chunk_transcript: str = "",
        merged_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyse a single audio chunk with transcript context.
        
        Uses the same output schema as transcript analysis for consistency.
        """
        # Use merged config if provided (includes overrides from client/campaign_type)
        if merged_config:
            topics = merged_config.get("topics", [])
            actions = merged_config.get("agent_actions", [])
            rubric = merged_config.get("performance_rubric", [])
            prompt_context = merged_config.get("prompt_context", "")
        else:
            # Fallback to global config
            global_cfg = self._config_service.get_global_config()
            topics = global_cfg.topics or self._config.get("topics", [])
            actions = global_cfg.agent_actions or self._config.get("agent_actions", [])
            rubric = global_cfg.performance_rubric or self._config.get("performance_rubric", [])
            prompt_context = ""
        
        # Build lists from config
        topic_list = ", ".join(topics) if topics else "donation, gift aid, regular giving, complaints, account updates"
        action_list = ", ".join(actions) if actions else "greeting, verification, explanation, objection handling, closing"
        
        # Convert rubric items to valid JSON keys
        def to_key(name: str) -> str:
            if isinstance(name, dict):
                name = name.get("name", "")
            return name.replace(" ", "_").replace("&", "and").replace("/", "_")
        
        perf_keys = [to_key(r) for r in rubric] if rubric else [
            "Empathy", "Clarity", "Listening", "Script_adherence",
            "Product_knowledge", "Rapport_building", "Objection_handling", "Closing_effectiveness"
        ]
        perf_keys_list = ", ".join(perf_keys)
        perf_scores_json = ",\n        ".join([f'"{k}": 1-10' for k in perf_keys])
        
        # Build context section (includes quality signals from merged config)
        context_section = f"\n{prompt_context}\n" if prompt_context else ""
        
        # Build transcript section for this chunk
        transcript_section = ""
        if chunk_transcript:
            transcript_section = f"""
TRANSCRIPT FOR THIS SEGMENT (use segment_ids for references):
{chunk_transcript}
"""
        
        system_prompt = f"""You are an expert call quality analyst for charity fundraising and engagement.
Analyse this audio recording segment alongside its transcript and provide detailed coaching insights.
{context_section}
AUDIO SEGMENT: {chunk_start:.1f}s to {chunk_end:.1f}s
{transcript_section}
TOPICS TO IDENTIFY: {topic_list}
AGENT ACTIONS TO DETECT: {action_list}
PERFORMANCE CRITERIA: {perf_keys_list}

Return your analysis as valid JSON matching this exact structure:

{{
    "summary": "1-2 sentence observation about this segment",
    "sentiment_score": 0 to 10 (0=hostile, 5=neutral, 10=very friendly),
    "key_topics": [
        {{"name": "Topic Name", "confidence": 0.0-1.0}}
    ],
    "agent_actions": [
        {{"action": "specific action", "segment_ids": ["<seg_id_from_transcript>"]}}
    ],
    "score_impacts": [
        {{"segment_ids": ["<seg_id_from_transcript>"], "impact": -5 to +5, "category": "one of: {perf_keys_list}", "reason": "why this affected the score", "quote": "exact quote"}}
    ],
    "compliance_flags": [
        {{"type": "GDPR|payment_security|misleading_info|rudeness|data_protection", "segment_ids": ["<seg_id_from_transcript>"], "severity": "low/medium/high/critical", "issue": "description", "quote": "exact quote"}}
    ],
    "performance_scores": {{
        {perf_scores_json}
    }},
    "audio_observations": {{
        "agent_tone": "warm/neutral/cold/hostile",
        "agent_emotion": "calm/frustrated/happy/anxious/other",
        "supporter_emotion": "calm/frustrated/happy/anxious/other",
        "speech_clarity": 1-10,
        "background_noise": "low/medium/high",
        "pacing": "rushed/steady/slow"
    }}
}}

AUDIO ANALYSIS GUIDANCE:
- Listen for TONE OF VOICE - warmth, empathy, frustration, impatience
- Listen for PACING - rushing through information, giving supporter time to respond
- Listen for INTERRUPTIONS - agent cutting off supporter
- Listen for BACKGROUND NOISE affecting call quality
- Use segment_ids ARRAY to reference specific moments - can span multiple segments
- Include quotes from the transcript as evidence

SEGMENT IDS: Copy segment_ids EXACTLY from the transcript (UUID-based like seg_a1b2c3d4). Use array format.

SCORE IMPACT SCALE:
- +5: Exceptional moment (exemplary)
- +3 to +4: Strong positive
- +1 to +2: Minor positive
- -1 to -2: Minor negative
- -3 to -4: Significant negative
- -5: Severe issue

COMPLIANCE FLAGS: Only include if actual issues detected. Use empty array [] if none.

Return ONLY valid JSON - no text before or after."""

        try:
            # Build conversation with audio
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": system_prompt}
                    ]
                }
            ]
            
            # Process with Qwen audio model
            text = self._analysis_processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )
            
            audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
            
            inputs = self._analysis_processor(
                text=text,
                audio=audios,
                images=images,
                videos=videos,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=False
            )
            
            inputs = inputs.to(self._analysis_model.device).to(self._analysis_model.dtype)
            input_len = inputs.input_ids.shape[1]
            
            with torch.no_grad():
                text_ids = self._analysis_model.generate(
                    **inputs,
                    max_new_tokens=1500,  # Increased for fuller response
                    do_sample=False,
                    return_audio=False
                )
            
            generated_ids = text_ids[:, input_len:]
            response = self._analysis_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return self._parse_audio_chunk_response(response, chunk_start, chunk_end)
            
        except Exception as e:
            logger.error(f"Chunk analysis failed: {e}")
            return self._default_audio_chunk_analysis(chunk_start, chunk_end)
    
    def _build_transcript_analysis_prompt(self, transcript: str, max_timestamp: float = 0, prompt_context: str = "", merged_config: Optional[Dict[str, Any]] = None) -> str:
        """Build prompt for transcript-based analysis."""
        # Use merged config if provided (includes overrides from client/campaign_type)
        if merged_config:
            topics = merged_config.get("topics", [])
            actions = merged_config.get("agent_actions", [])
            rubric = merged_config.get("performance_rubric", [])
        else:
            # Fallback to global config directly
            global_cfg = self._config_service.get_global_config()
            topics = global_cfg.topics or self._config.get("topics", [])
            actions = global_cfg.agent_actions or self._config.get("agent_actions", [])
            rubric = global_cfg.performance_rubric or self._config.get("performance_rubric", [])
        
        # Build topic and action lists for the prompt - use full lists from config
        topic_list = ", ".join(topics) if topics else "donation, gift aid, regular giving, complaints, updates, account changes, direct debit, legacy giving, event registration, volunteer enquiry"
        action_list = ", ".join(actions) if actions else "greeting, verification, explanation, objection handling, closing, upselling, complaint resolution, payment processing"
        rubric_list = ", ".join([r.get("name", r) if isinstance(r, dict) else r for r in rubric]) if rubric else "Empathy, Clarity, Listening, Script adherence, Product knowledge, Rapport building, Objection handling, Closing effectiveness"
        
        # Convert rubric items to valid JSON keys for performance_scores
        def to_key(name: str) -> str:
            """Convert rubric name to valid JSON key."""
            if isinstance(name, dict):
                name = name.get("name", "")
            return name.replace(" ", "_").replace("&", "and").replace("/", "_")
        
        perf_keys = [to_key(r) for r in rubric] if rubric else [
            "Empathy", "Clarity", "Listening", "Script_adherence",
            "Product_knowledge", "Rapport_building", "Objection_handling", "Closing_effectiveness"
        ]
        perf_keys_list = ", ".join(perf_keys)
        perf_scores_json = ",\n        ".join([f'"{k}": 1-10' for k in perf_keys])
        
        # Optionally truncate transcript for smaller models
        if self.max_transcript_length > 0:
            transcript = transcript[:self.max_transcript_length]
            # Use simplified prompt for constrained models (no quality_context to save tokens)
            return self._build_simple_analysis_prompt(transcript, topic_list, perf_keys, perf_keys_list, max_timestamp, prompt_context)
        
        # Build context section (prompt_context already includes all four tiers' context including quality signals)
        context_section = f"\n\n{prompt_context}\n" if prompt_context else ""
        
        return f"""You are an expert call quality analyst for charity fundraising and engagement.
Analyse this transcript thoroughly and provide detailed coaching insights.
{context_section}
CALL DURATION: {max_timestamp:.1f} seconds

TRANSCRIPT (each line has [segment_id] [start - end] Speaker: text):
{transcript}

TOPICS TO IDENTIFY: {topic_list}
AGENT ACTIONS TO DETECT: {action_list}
PERFORMANCE CRITERIA: {rubric_list}

Return your analysis as valid JSON matching this exact structure:

{{
    "summary": "2-3 sentence British English summary describing the call purpose, outcome, and notable moments",
    "sentiment_score": 0 to 10 (0=hostile, 5=neutral, 10=very friendly),
    "key_topics": [
        {{"name": "Topic Name In Title Case", "confidence": 0.0-1.0}}
    ],
    "agent_actions": [
        {{"action": "specific action the agent took", "segment_ids": ["seg_a1b2c3d4", "seg_e5f6g7h8"]}}
    ],
    "score_impacts": [
        {{"segment_ids": ["seg_a1b2c3d4"], "impact": -5 to +5, "category": "one of: {perf_keys_list}", "reason": "why this affected the score", "quote": "exact quote from transcript"}}
    ],
    "compliance_flags": [
        {{"type": "GDPR|payment_security|misleading_info|rudeness|data_protection", "segment_ids": ["seg_a1b2c3d4", "seg_e5f6g7h8"], "severity": "low/medium/high/critical", "issue": "detailed description", "quote": "exact quote from transcript"}}
    ],
    "performance_scores": {{
        {perf_scores_json}
    }},
    "action_items": [
        {{"description": "specific follow-up action needed", "priority": "high/medium/low"}}
    ]
}}

ANALYSIS GUIDANCE:
- ANALYSE THE ENTIRE TRANSCRIPT from start to finish - do not focus only on the beginning
- Capture actions from all parts of the call: opening, middle, AND closing sections
- Use segment_ids ARRAY to reference specific moments - actions/impacts can span multiple segments
- Include specific quotes from the transcript as evidence
- Flag any compliance issues: GDPR, payment security, misleading information, rudeness
- Identify: rushing, interrupting, poor listening, weak objection handling, unclear explanations, lack of empathy, missed upsell opportunities

AGENT ACTIONS (neutral identification):
- Simply identify WHAT the agent did and WHEN - no judgement
- Use segment_ids array to reference where in the call this happened (can span multiple segments)
- Examples: "Greeted supporter", "Verified identity", "Explained Gift Aid", "Handled objection", "Closed call"

SCORE IMPACTS (quality judgements):
- Identify specific moments that POSITIVELY or NEGATIVELY affected the call quality
- Use the "impact" field from -5 to +5:
  - +5: Exceptional - exemplary moment, could be used as training example
  - +3 to +4: Strong positive - notably good handling
  - +1 to +2: Minor positive - good practice observed
  - -1 to -2: Minor negative - could improve
  - -3 to -4: Significant negative - clear problem
  - -5: Severe negative - major issue affecting call quality
- "category" MUST be one of: {perf_keys_list}
- Include the exact "quote" from the transcript showing this moment
- Compliance issues should ALSO appear in score_impacts with negative impact (-3 to -5)

COMPLIANCE FLAGS (separate for alerting):
- ONLY include if there is an ACTUAL compliance issue detected
- If NO compliance issues are found, use an empty array: "compliance_flags": []
- Types: GDPR, payment_security, misleading_info, rudeness, data_protection
- Severity: low (minor procedural issue), medium (needs attention), high (serious breach), critical (immediate action required)
- These issues should ALSO appear in score_impacts with appropriate negative impact

ARRAY LIMITS (do not exceed):
- key_topics: maximum 5 items, use Title Case (e.g., "Regular Giving", "Gift Aid", "Direct Debit")
- agent_actions: maximum 15 items (capture key actions throughout the entire call)
- score_impacts: maximum 15 items (focus on most significant positive and negative moments)
- action_items: maximum 5 items
- compliance_flags: only actual issues found - use empty array [] if none

SEGMENT IDs - CRITICAL:
- segment_ids is an ARRAY - use ["seg_a1b2c3d4"] for single or ["seg_a1b2c3d4", "seg_e5f6g7h8"] for spans
- Copy segment_id values EXACTLY from the transcript (UUID-based like seg_a1b2c3d4)
- Each segment_id references a specific line in the transcript above
- Do NOT invent segment IDs - only use ones that appear in the transcript
- Use multiple segment_ids when an action/impact spans multiple transcript segments

CRITICAL RULES:
- Each array item must be unique - DO NOT repeat yourself
- Return ONLY valid JSON - no text before or after the JSON object
- Stop generating immediately after the final closing brace"""
    
    def _build_simple_analysis_prompt(self, transcript: str, topic_list: str, perf_keys: list, perf_keys_list: str, max_timestamp: float = 0, prompt_context: str = "") -> str:
        """Simplified prompt for smaller models (3B-7B) - outputs full structure."""
        # Build performance_scores example with dynamic keys
        perf_scores_example = ",\n        ".join([f'"{k}": 7' for k in perf_keys[:8]])  # Limit to 8 for readability
        
        # Build context section
        context_section = f"\n{prompt_context}\n" if prompt_context else ""
        
        return f"""You are a call quality analyst. Analyse this charity supporter call transcript.
{context_section}
CALL DURATION: {max_timestamp:.1f} seconds

TRANSCRIPT (each line: [segment_id] [start - end] Speaker: text):
{transcript}

Return your analysis as valid JSON matching this EXACT structure:

{{
    "summary": "2-3 sentence summary describing what happened in the call, its purpose and outcome",
    "sentiment_score": 7,
    "key_topics": [
        {{"name": "Regular Giving", "confidence": 0.9}},
        {{"name": "Gift Aid", "confidence": 0.8}}
    ],
    "agent_actions": [
        {{"action": "Greeted supporter", "segment_ids": ["seg_a1b2c3d4"]}},
        {{"action": "Verified identity", "segment_ids": ["seg_e5f6g7h8", "seg_i9j0k1l2"]}},
        {{"action": "Explained donation options", "segment_ids": ["seg_m3n4o5p6", "seg_q7r8s9t0"]}},
        {{"action": "Closed call", "segment_ids": ["seg_u1v2w3x4"]}}  
    ],
    "score_impacts": [
        {{"segment_ids": ["seg_a1b2c3d4"], "impact": 3, "category": "{perf_keys[0] if perf_keys else 'Empathy'}", "reason": "Warm, friendly greeting", "quote": "Good morning, lovely to speak with you!"}},
        {{"segment_ids": ["seg_m3n4o5p6"], "impact": -2, "category": "{perf_keys[2] if len(perf_keys) > 2 else 'Listening'}", "reason": "Interrupted supporter", "quote": "Yes but what I meant was--"}}
    ],
    "compliance_flags": [],
    "performance_scores": {{
        {perf_scores_example}
    }},
    "action_items": [
        {{"description": "Send confirmation email", "priority": "high"}}
    ]
}}

SCORING GUIDE:
- sentiment_score: 0 (hostile) to 10 (very friendly), 5 is neutral
- confidence: 0.0 to 1.0
- performance scores: 1-10 for each criterion
- impact (for score_impacts): -5 to +5
  - +5: Exceptional moment
  - +3 to +4: Strong positive
  - +1 to +2: Minor positive
  - -1 to -2: Minor negative
  - -3 to -4: Significant negative
  - -5: Severe issue

SCORE IMPACT CATEGORIES (must use exactly):
{perf_keys_list}

TOPICS TO IDENTIFY: {topic_list}

CRITICAL RULES:
1. Use lowercase field names with underscores exactly as shown
2. Include empty arrays [] if no items found (don't omit fields)
3. MAXIMUM array sizes: key_topics (3), agent_actions (5), score_impacts (8), action_items (3)
4. DO NOT repeat yourself - each item in an array must be unique
5. Keep descriptions concise - one sentence maximum
6. Return ONLY the JSON object, no text before or after
7. Stop generating after the final closing brace
8. For compliance_flags: ONLY include ACTUAL issues - use empty array [] if no issues found
9. Use Title Case for topic names (e.g., "Regular Giving", "Gift Aid", "Direct Debit")
10. Compliance issues should ALSO appear in score_impacts with negative impact
11. SEGMENT IDs - copy segment_ids EXACTLY from transcript (UUID-based like seg_a1b2c3d4) - use array format"""
    
    def _generate_text_response(self, prompt: str) -> str:
        """Generate text response from model."""
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
        
        text = self._analysis_processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False
        )
        
        inputs = self._analysis_processor(
            text=text,
            return_tensors="pt",
            padding=True
        )
        
        inputs = inputs.to(self._analysis_model.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            output_ids = self._analysis_model.generate(
                **inputs,
                max_new_tokens=800,
                do_sample=False
            )
        
        generated_ids = output_ids[:, input_len:]
        response = self._analysis_processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return response
    
    def _parse_chunk_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from chunk analysis response."""
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                # Use json_repair if available
                if JSON_REPAIR_AVAILABLE:
                    try:
                        json_str = repair_json(json_str)
                    except Exception:
                        pass  # Fall through to regular parsing
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}")
        
        return self._default_chunk_analysis()
    
    def _parse_analysis_response(
        self, 
        response: str, 
        speaker_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Parse analysis response into structured format."""
        # Log truncated response for debugging (full response at DEBUG level)
        response_preview = response[:500] + "..." if len(response) > 500 else response
        logger.info(f"LLM response received ({len(response)} chars): {response_preview}")
        logger.debug(f"Full LLM response: {response}")
        
        try:
            start = response.find('{')
            end = response.rfind('}')
            logger.debug(f"Parsing response: found JSON from {start} to {end}")
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                
                # Use json_repair if available (handles most LLM JSON issues)
                if JSON_REPAIR_AVAILABLE:
                    try:
                        json_str = repair_json(json_str)
                        logger.debug("JSON repaired successfully with json_repair")
                    except Exception as e:
                        logger.warning(f"json_repair failed: {e}, falling back to manual cleaning")
                        json_str = self._clean_json_response(json_str)
                else:
                    # Fallback to manual cleaning if json_repair not installed
                    json_str = self._clean_json_response(json_str)
                
                parsed = json.loads(json_str)
                
                # Handle case where model output an array instead of object
                if isinstance(parsed, list):
                    logger.warning(f"Model returned array with {len(parsed)} items instead of object, taking first item")
                    if parsed and isinstance(parsed[0], dict):
                        parsed = parsed[0]
                    else:
                        logger.error("Array does not contain valid object, falling back")
                        return self._fallback_parse_response(response, speaker_metrics)
                
                # Fix performance_scores if model used wrong structure
                perf_scores = parsed.get("performance_scores", {})
                if isinstance(perf_scores, dict) and "criterion_name" in perf_scores and "score" in perf_scores:
                    # Model output: {"criterion_name": "Empathy", "score": 8}
                    # Convert to: {"Empathy": 8}
                    perf_scores = {perf_scores["criterion_name"]: perf_scores["score"]}
                
                # Extract score_impacts and calculate quality_score
                score_impacts = parsed.get("score_impacts", [])
                quality_score = self._calculate_quality_score(score_impacts)
                
                # Ensure all required fields with correct naming
                result = {
                    "summary": parsed.get("summary", parsed.get("summarized", "Analysis completed")),
                    "sentiment_score": self._extract_number(parsed.get("sentiment_score", 5)),
                    "quality_score": quality_score,
                    "key_topics": self._add_keys_to_array(parsed.get("key_topics", parsed.get("key_topic", []))),
                    "agent_actions": parsed.get("agent_actions", []),
                    "score_impacts": self._add_keys_to_array(score_impacts),
                    "performance_scores": perf_scores,
                    "action_items": parsed.get("action_items", []),
                    "compliance_flags": self._add_keys_to_array(parsed.get("compliance_flags", [])),
                    "speaker_metrics": speaker_metrics,
                    "model_used": self.analysis_model_path,
                }
                logger.info(f"Parsed analysis: summary='{result['summary'][:50]}...', quality={result['quality_score']:.1f}% (from {len(score_impacts)} impacts)")
                return result
            else:
                logger.warning("No JSON object found in model response")
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}. Attempting fallback extraction.")
            # Try to extract what we can from partial/malformed JSON
            return self._fallback_parse_response(response, speaker_metrics)
        
        return self._empty_analysis(speaker_metrics, 0)
    
    def _enrich_segment_timestamps(self, result: Dict[str, Any], segment_map: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Enrich segment_ids references with actual timestamps from the segment_map.
        
        The LLM outputs segment_ids arrays (e.g., ["seg_a1b2c3d4", "seg_e5f6g7h8"]), 
        and we look up the actual start/end timestamps from the transcript segments.
        
        For arrays, timestamp_start is the start of the first segment and timestamp_end
        is the end of the last segment in the array.
        
        Also handles legacy single segment_id format for backwards compatibility.
        
        This ensures timestamps are always accurate since they come from the 
        original transcript, not from LLM hallucination.
        """
        if not segment_map:
            return result
        
        enriched_count = 0
        invalid_count = 0
        
        def enrich(item: Dict) -> Dict:
            nonlocal enriched_count, invalid_count
            
            # Handle new array format (segment_ids)
            segment_ids = item.get("segment_ids")
            
            # Backwards compatibility: convert legacy segment_id string to array
            if not segment_ids and item.get("segment_id"):
                segment_ids = [item.get("segment_id")]
                item["segment_ids"] = segment_ids  # Normalise to array format
                del item["segment_id"]  # Remove legacy field
            
            if segment_ids and isinstance(segment_ids, list) and len(segment_ids) > 0:
                # Filter to valid segment_ids only
                valid_ids = [sid for sid in segment_ids if sid in segment_map]
                invalid_ids = [sid for sid in segment_ids if sid not in segment_map]
                
                if invalid_ids:
                    logger.debug(f"Invalid segment_ids {invalid_ids} not in transcript")
                    invalid_count += len(invalid_ids)
                
                if valid_ids:
                    # Get timestamps spanning from first to last valid segment
                    first_seg = segment_map[valid_ids[0]]
                    last_seg = segment_map[valid_ids[-1]]
                    item["timestamp_start"] = first_seg["start"]
                    item["timestamp_end"] = last_seg["end"]
                    enriched_count += 1
                else:
                    # No valid segment_ids - set to 0
                    item["timestamp_start"] = 0.0
                    item["timestamp_end"] = 0.0
            
            return item
        
        # Enrich agent_actions
        for action in result.get("agent_actions", []):
            if isinstance(action, dict):
                enrich(action)
        
        # Enrich score_impacts
        for impact in result.get("score_impacts", []):
            if isinstance(impact, dict):
                enrich(impact)
        
        # Enrich compliance_flags
        for flag in result.get("compliance_flags", []):
            if isinstance(flag, dict):
                enrich(flag)
        
        if enriched_count > 0 or invalid_count > 0:
            logger.info(f"Enriched {enriched_count} segment references with timestamps ({invalid_count} invalid segment_ids)")
        
        return result
    
    def _calculate_quality_score(self, score_impacts: List[Dict[str, Any]]) -> float:
        """
        Calculate quality score from score impacts.
        Delegates to shared function in models.py.
        """
        return calculate_quality_score(score_impacts)
    
    def _add_keys_to_array(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add a unique 'key' field (UUID v4) to each item in an array.
        
        This is needed by the frontend for React list rendering.
        """
        if not items or not isinstance(items, list):
            return items
        
        for item in items:
            if isinstance(item, dict) and "key" not in item:
                item["key"] = str(uuid.uuid4())
        
        return items
    
    def _fallback_parse_response(
        self,
        response: str,
        speaker_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract what we can from malformed JSON response."""
        import re
        
        result = self._empty_analysis(speaker_metrics, 0)
        
        # Try to extract summary (handle variations like "summarized", "Summary")
        summary_match = re.search(r'"(?:summary|summarized|Summary)"\s*:\s*"([^"]+)"', response)
        if summary_match:
            result["summary"] = summary_match.group(1)
        
        # Extract sentiment_score (handle variations)
        sentiment_match = re.search(r'"(?:sentiment_score|sentivity_score|Sentiment_Score)"\s*:\s*(-?\d+(?:\.\d+)?)', response)
        if sentiment_match:
            result["sentiment_score"] = float(sentiment_match.group(1))
        
        # Try to extract arrays by finding valid JSON subarrays (handle variations)
        result["key_topics"] = self._add_keys_to_array(
            self._extract_json_array(response, "key_topics") or self._extract_json_array(response, "Key_Topics") or []
        )
        result["agent_actions"] = self._extract_json_array(response, "agent_actions") or self._extract_json_array(response, "Agent_Actions")
        result["score_impacts"] = self._add_keys_to_array(
            self._extract_json_array(response, "score_impacts") or self._extract_json_array(response, "Score_Impacts") or []
        )
        result["action_items"] = self._extract_json_array(response, "action_items") or self._extract_json_array(response, "Action_Items")
        result["compliance_flags"] = self._add_keys_to_array(
            self._extract_json_array(response, "compliance_flags") or self._extract_json_array(response, "compliance_issues") or self._extract_json_array(response, "Compliance_Issues") or []
        )
        
        # Calculate quality_score from score_impacts
        result["quality_score"] = self._calculate_quality_score(result["score_impacts"])
        
        # Try to extract performance_scores object (handle variations)
        perf_match = re.search(r'"(?:performance_scores|Performance_Scores)"\s*:\s*(\{[^}]+\})', response)
        if perf_match:
            try:
                perf_obj = json.loads(perf_match.group(1))
                # Fix wrong structure if needed
                if "criterion_name" in perf_obj and "score" in perf_obj:
                    perf_obj = {perf_obj["criterion_name"]: perf_obj["score"]}
                result["performance_scores"] = perf_obj
            except json.JSONDecodeError:
                pass
        
        result["model_used"] = self.analysis_model_path
        result["analysis_type"] = "transcript_partial"
        
        extracted_count = sum(1 for k in ["summary", "key_topics", "agent_actions", "score_impacts", "action_items"] 
                              if result.get(k) and result[k] != "Analysis unavailable" and result[k] != [])
        logger.info(f"Fallback extraction: recovered {extracted_count} fields from malformed JSON")
        
        if result["summary"] != "Analysis unavailable":
            logger.info(f"Fallback extraction succeeded: summary='{result['summary'][:50]}...'")
        
        return result
    
    def _extract_json_array(self, response: str, field_name: str) -> list:
        """Extract a JSON array field from potentially malformed JSON."""
        import re
        
        # Find the start of the array for this field
        pattern = rf'"{field_name}"\s*:\s*\['
        match = re.search(pattern, response)
        if not match:
            return []
        
        start_pos = match.end() - 1  # Position of opening [
        
        # Find matching closing bracket by counting nesting
        depth = 0
        in_string = False
        escape_next = False
        end_pos = start_pos
        
        for i, char in enumerate(response[start_pos:], start=start_pos):
            if escape_next:
                escape_next = False
                continue
            if char == '\\':
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == '[':
                depth += 1
            elif char == ']':
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break
        
        if depth != 0:
            # Unbalanced - try to find last valid object and close there
            array_content = response[start_pos:end_pos+1] if end_pos > start_pos else response[start_pos:]
            # Find last complete object (ending with })
            last_obj_end = array_content.rfind('}')
            if last_obj_end > 0:
                array_content = array_content[:last_obj_end+1] + ']'
            else:
                return []
        else:
            array_content = response[start_pos:end_pos+1]
        
        # Try to parse the array
        try:
            # Clean up the array content
            array_content = self._clean_json_response(array_content)
            parsed = json.loads(array_content)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            # Try parsing individual objects
            objects = []
            obj_pattern = r'\{[^{}]*\}'
            for obj_match in re.finditer(obj_pattern, array_content):
                try:
                    obj = json.loads(obj_match.group())
                    objects.append(obj)
                except json.JSONDecodeError:
                    continue
            return objects
        
        return []
    
    def _clean_json_response(self, json_str: str) -> str:
        """Clean up common LLM JSON formatting issues."""
        import re
        
        # Fix JavaScript-style unquoted keys: action: "value" -> "action": "value"
        # Handle keys at start of line or after { or ,
        json_str = re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', json_str)
        
        # Fix broken lines like: }  "action: : " "action: -> remove entire garbage line
        json_str = re.sub(r'\}\s*"[^"]*:\s*:\s*"[^"]*"[^}\]]*', '},', json_str)
        
        # Fix missing opening quote on keys at start of line: action_items": -> "action_items":
        json_str = re.sub(r'^(\s*)(\w+)":', r'\1"\2":', json_str, flags=re.MULTILINE)
        
        # Fix keys without any quotes that have colon at end of line
        json_str = re.sub(r'^(\s*)(\w+):\s*$', r'\1"\2":', json_str, flags=re.MULTILINE)
        
        # Fix patterns like: 7 (neutral) -> 7
        json_str = re.sub(r':\s*(-?\d+(?:\.\d+)?)\s*\([^)]+\)', r': \1', json_str)
        
        # Fix patterns like: "6 (out of 10)" -> 6
        json_str = re.sub(r':\s*"(-?\d+(?:\.\d+)?)\s*\([^)]+\)"', r': \1', json_str)
        
        # Remove trailing commas before } or ]
        json_str = re.sub(r',\s*([\]}])', r'\1', json_str)
        
        # Fix }}} or }} followed by ] - extra closing braces
        json_str = re.sub(r'\}\s*\}\}(\s*\])', r'}\1', json_str)
        json_str = re.sub(r'\}\}\}', '}}', json_str)
        
        # Fix },} or ],] patterns (double closing with comma)
        json_str = re.sub(r'\},\s*\}', '}}', json_str)
        json_str = re.sub(r'\],\s*\]', ']]', json_str)
        
        # Remove garbage after the main JSON object closes
        # Find where improvement_areas array should end and truncate anything after
        # Pattern: ] followed by } (end of improvement_areas and main object)
        # But there may be garbage between them
        match = re.search(r'("improvement_areas"\s*:\s*\[.*?\]\s*)\}', json_str, re.DOTALL)
        if match:
            # Find the position after the last valid closing
            # Truncate everything after the final }
            last_brace = json_str.rfind('}')
            if last_brace > 0:
                # Check if there's garbage after the last }
                after_brace = json_str[last_brace+1:].strip()
                if after_brace:
                    logger.debug(f"Removing trailing garbage: {after_brace[:50]}")
                    json_str = json_str[:last_brace+1]
        
        # Remove random quoted strings floating between } and ]
        # Pattern: } followed by quoted string followed by , or ]
        json_str = re.sub(r'\}\s*"[^"]*"\s*,?\s*(\])', r'}\1', json_str)
        
        # Remove lines that are clearly garbage (unquoted text not part of JSON structure)
        lines = json_str.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip empty lines
            if not stripped:
                cleaned_lines.append(line)
                continue
            # Keep lines that start with valid JSON characters
            if stripped.startswith(('"', '{', '}', '[', ']', ',')):
                cleaned_lines.append(line)
            # Keep lines with colon (likely key-value pairs)
            elif ':' in stripped and '"' in stripped:
                cleaned_lines.append(line)
            # Skip garbage lines (unquoted text without structure)
            else:
                logger.debug(f"Removing garbage line: {stripped[:50]}")
        json_str = '\n'.join(cleaned_lines)
        
        # Balance braces/brackets
        open_braces = json_str.count('{') - json_str.count('}')
        open_brackets = json_str.count('[') - json_str.count(']')
        
        # Remove excess closing braces/brackets
        if open_braces < 0:
            for _ in range(-open_braces):
                # Remove last }
                idx = json_str.rfind('}')
                if idx > 0:
                    json_str = json_str[:idx] + json_str[idx+1:]
        if open_brackets < 0:
            for _ in range(-open_brackets):
                idx = json_str.rfind(']')
                if idx > 0:
                    json_str = json_str[:idx] + json_str[idx+1:]
        
        # Add missing closing braces/brackets
        if open_braces > 0:
            json_str += '}' * open_braces
        if open_brackets > 0:
            json_str += ']' * open_brackets
        
        return json_str
    
    def _extract_number(self, value) -> float:
        """Extract numeric value from potentially mixed content."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            import re
            match = re.search(r'-?\d+(?:\.\d+)?', value)
            if match:
                return float(match.group())
        return 0.0
    
    def _aggregate_analyses(
        self, 
        chunk_analyses: List[Dict[str, Any]],
        speaker_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Aggregate chunk analyses into final result."""
        if not chunk_analyses:
            return self._empty_analysis(speaker_metrics, 0)
        
        # Average numeric scores
        sentiment_scores = [c.get("sentiment_score", 5) for c in chunk_analyses]
        clarity_scores = [c.get("speech_clarity", 7) for c in chunk_analyses]
        professionalism_scores = [c.get("professionalism", 7) for c in chunk_analyses]
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        avg_clarity = sum(clarity_scores) / len(clarity_scores)
        avg_professionalism = sum(professionalism_scores) / len(professionalism_scores)
        
        # Collect all items
        all_topics = []
        all_actions = []
        all_score_impacts = []
        all_issues = []
        all_positives = []
        observations = []
        all_performance_scores = {}
        
        for c in chunk_analyses:
            all_topics.extend(c.get("detected_topics", []))
            all_actions.extend(c.get("detected_actions", []))
            all_score_impacts.extend(c.get("score_impacts", []))
            all_issues.extend(c.get("detected_issues", []))
            all_positives.extend(c.get("positive_indicators", []))
            if c.get("brief_observation"):
                observations.append(c["brief_observation"])
            
            # Merge performance scores
            for key, value in c.get("performance_scores", {}).items():
                if key not in all_performance_scores:
                    all_performance_scores[key] = []
                all_performance_scores[key].append(value)
        
        # Average performance scores
        final_performance_scores = {
            k: round(sum(v) / len(v)) for k, v in all_performance_scores.items()
        }
        
        # Determine dominant tones
        tones = [c.get("agent_tone", "neutral") for c in chunk_analyses]
        agent_emotions = [c.get("agent_emotion", "calm") for c in chunk_analyses]
        supporter_emotions = [c.get("supporter_emotion", c.get("customer_emotion", "calm")) for c in chunk_analyses]
        
        dominant_tone = max(set(tones), key=tones.count) if tones else "neutral"
        dominant_agent_emotion = max(set(agent_emotions), key=agent_emotions.count) if agent_emotions else "calm"
        dominant_supporter_emotion = max(set(supporter_emotions), key=supporter_emotions.count) if supporter_emotions else "calm"
        
        # Calculate quality score from score_impacts (or fallback to sentiment-based)
        if all_score_impacts:
            quality_score = self._calculate_quality_score(all_score_impacts)
        else:
            quality_score = self._sentiment_to_quality(avg_sentiment)
        
        # Build summary
        summary = " ".join(observations[:3]) if observations else "Audio analysis completed."
        
        # Format topics for database
        key_topics = [
            {"name": t, "confidence": 0.8} 
            for t in list(set(all_topics))[:10]
        ]
        
        # Format actions for database (neutral identification)
        agent_actions = [
            {"action": a, "timestamp_start": 0.0, "timestamp_end": 0.0}
            for a in list(set(all_actions))[:15]
        ]
        
        return {
            "summary": summary[:500],
            "sentiment_score": round(avg_sentiment, 1),
            "quality_score": round(quality_score, 1),
            "key_topics": key_topics,
            "agent_actions": agent_actions,
            "score_impacts": all_score_impacts[:15],  # Limit to 15 most significant
            "performance_scores": final_performance_scores,
            "action_items": [],
            "compliance_flags": [{"type": "issue", "issue": i, "severity": "medium", "timestamp_start": 0.0, "timestamp_end": 0.0} for i in list(set(all_issues))[:5]],
            "speaker_metrics": speaker_metrics,
            "audio_observations": {
                "call_quality": "good" if avg_clarity >= 7 else "fair" if avg_clarity >= 5 else "poor",
                "background_noise": "low",
                "agent_tone": dominant_tone,
                "supporter_tone": dominant_supporter_emotion,
            },
            "model_used": self.analysis_model_path,
            "analysis_type": "audio",
        }
    
    def _sentiment_to_quality(self, sentiment: float) -> float:
        """Convert sentiment score (0 to 10) to quality score (0 to 100)."""
        # Scale: 0=hostile, 5=neutral, 10=very friendly
        if sentiment >= 9:
            return 95 + (sentiment - 9) * 5
        elif sentiment >= 7:
            return 80 + (sentiment - 7) * 7.5
        elif sentiment >= 5:
            return 60 + (sentiment - 5) * 10
        elif sentiment >= 3:
            return 40 + (sentiment - 3) * 10
        elif sentiment >= 1:
            return 20 + (sentiment - 1) * 10
        else:
            return max(0, sentiment * 20)
    
    def _default_chunk_analysis(self) -> Dict[str, Any]:
        """Default chunk analysis when processing fails (legacy format)."""
        return {
            "agent_tone": "neutral",
            "agent_emotion": "calm",
            "supporter_emotion": "calm",
            "speech_clarity": 7,
            "professionalism": 7,
            "detected_topics": [],
            "detected_actions": [],
            "score_impacts": [],
            "performance_scores": {},
            "detected_issues": [],
            "positive_indicators": [],
            "sentiment_score": 5,
            "brief_observation": "Analysis unavailable for this segment",
        }
    
    def _default_audio_chunk_analysis(self, chunk_start: float = 0, chunk_end: float = 0) -> Dict[str, Any]:
        """Default audio chunk analysis matching the new aligned format."""
        return {
            "summary": f"Analysis unavailable for segment {chunk_start:.1f}s - {chunk_end:.1f}s",
            "sentiment_score": 5,
            "key_topics": [],
            "agent_actions": [],
            "score_impacts": [],
            "compliance_flags": [],
            "performance_scores": {},
            "audio_observations": {
                "agent_tone": "neutral",
                "agent_emotion": "calm",
                "supporter_emotion": "calm",
                "speech_clarity": 7,
                "background_noise": "low",
                "pacing": "steady"
            },
            "chunk_start": chunk_start,
            "chunk_end": chunk_end,
        }
    
    def _parse_audio_chunk_response(self, response: str, chunk_start: float, chunk_end: float) -> Dict[str, Any]:
        """Parse JSON from audio chunk analysis response (new aligned format)."""
        try:
            start = response.find('{')
            end = response.rfind('}')
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                # Use json_repair if available
                if JSON_REPAIR_AVAILABLE:
                    try:
                        json_str = repair_json(json_str)
                    except Exception:
                        pass
                parsed = json.loads(json_str)
                parsed["chunk_start"] = chunk_start
                parsed["chunk_end"] = chunk_end
                return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"Audio chunk JSON parsing failed: {e}")
        
        return self._default_audio_chunk_analysis(chunk_start, chunk_end)
    
    def _aggregate_audio_analyses(
        self, 
        chunk_analyses: List[Dict[str, Any]],
        speaker_metrics: Dict[str, Any],
        segment_map: Dict[str, Dict],
        max_timestamp: float
    ) -> Dict[str, Any]:
        """
        Aggregate audio chunk analyses into final result.
        
        Produces the same output structure as transcript analysis.
        """
        if not chunk_analyses:
            return self._empty_analysis(speaker_metrics, 0)
        
        # Collect all items from chunks
        all_summaries = []
        all_topics = []
        all_actions = []
        all_score_impacts = []
        all_compliance_flags = []
        all_performance_scores = {}
        sentiment_scores = []
        
        # Audio-specific observations
        tones = []
        agent_emotions = []
        supporter_emotions = []
        clarity_scores = []
        
        for chunk in chunk_analyses:
            # Summary
            if chunk.get("summary"):
                all_summaries.append(chunk["summary"])
            
            # Topics (with confidence)
            for topic in chunk.get("key_topics", []):
                if isinstance(topic, dict):
                    all_topics.append(topic)
                else:
                    all_topics.append({"name": topic, "confidence": 0.8})
            
            # Agent actions (with segment_id)
            for action in chunk.get("agent_actions", []):
                if isinstance(action, dict):
                    all_actions.append(action)
                else:
                    all_actions.append({"action": action, "segment_id": None})
            
            # Score impacts
            all_score_impacts.extend(chunk.get("score_impacts", []))
            
            # Compliance flags
            all_compliance_flags.extend(chunk.get("compliance_flags", []))
            
            # Performance scores (average across chunks)
            for key, value in chunk.get("performance_scores", {}).items():
                if key not in all_performance_scores:
                    all_performance_scores[key] = []
                if isinstance(value, (int, float)):
                    all_performance_scores[key].append(value)
            
            # Sentiment
            if "sentiment_score" in chunk:
                sentiment_scores.append(chunk["sentiment_score"])
            
            # Audio observations
            audio_obs = chunk.get("audio_observations", {})
            if audio_obs:
                if audio_obs.get("agent_tone"):
                    tones.append(audio_obs["agent_tone"])
                if audio_obs.get("agent_emotion"):
                    agent_emotions.append(audio_obs["agent_emotion"])
                if audio_obs.get("supporter_emotion"):
                    supporter_emotions.append(audio_obs["supporter_emotion"])
                if audio_obs.get("speech_clarity"):
                    clarity_scores.append(audio_obs["speech_clarity"])
        
        # Calculate averages
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 5.0
        avg_clarity = sum(clarity_scores) / len(clarity_scores) if clarity_scores else 7.0
        
        # Average performance scores
        final_performance_scores = {
            k: round(sum(v) / len(v)) for k, v in all_performance_scores.items() if v
        }
        
        # Determine dominant audio qualities
        dominant_tone = max(set(tones), key=tones.count) if tones else "neutral"
        dominant_agent_emotion = max(set(agent_emotions), key=agent_emotions.count) if agent_emotions else "calm"
        dominant_supporter_emotion = max(set(supporter_emotions), key=supporter_emotions.count) if supporter_emotions else "calm"
        
        # Calculate quality score from score_impacts
        if all_score_impacts:
            quality_score = self._calculate_quality_score(all_score_impacts)
        else:
            quality_score = self._sentiment_to_quality(avg_sentiment)
        
        # Build summary from chunk observations
        summary = " ".join(all_summaries[:3]) if all_summaries else "Audio analysis completed."
        if len(summary) > 500:
            summary = summary[:497] + "..."
        
        # Deduplicate topics by name
        seen_topics = set()
        unique_topics = []
        for topic in all_topics:
            name = topic.get("name", "")
            if name and name not in seen_topics:
                seen_topics.add(name)
                unique_topics.append(topic)
        
        # Deduplicate actions by action text
        seen_actions = set()
        unique_actions = []
        for action in all_actions:
            action_text = action.get("action", "")
            if action_text and action_text not in seen_actions:
                seen_actions.add(action_text)
                unique_actions.append(action)
        
        # Enrich with timestamps from segment_map
        result = {
            "summary": summary,
            "sentiment_score": round(avg_sentiment, 1),
            "quality_score": round(quality_score, 1),
            "key_topics": unique_topics[:10],
            "agent_actions": unique_actions[:15],
            "score_impacts": all_score_impacts[:15],
            "performance_scores": final_performance_scores,
            "action_items": [],  # Could be derived from compliance_flags
            "compliance_flags": all_compliance_flags[:5],
            "speaker_metrics": speaker_metrics,
            "audio_observations": {
                "call_quality": "good" if avg_clarity >= 7 else "fair" if avg_clarity >= 5 else "poor",
                "background_noise": "low",
                "agent_tone": dominant_tone,
                "agent_emotion": dominant_agent_emotion,
                "supporter_emotion": dominant_supporter_emotion,
                "speech_clarity": round(avg_clarity, 1),
            },
            "model_used": self.analysis_model_path,
            "analysis_type": "audio",
        }
        
        # Enrich segment_ids with timestamps
        result = self._enrich_segment_timestamps(result, segment_map)
        
        return result
    
    def _empty_analysis(
        self, 
        speaker_metrics: Dict[str, Any], 
        start_time: float
    ) -> Dict[str, Any]:
        """Empty analysis when processing fails."""
        return {
            "summary": "Analysis unavailable",
            "sentiment_score": 5,
            "sentiment_label": "neutral",
            "quality_score": 65.0,  # Baseline score
            "key_topics": [],
            "agent_actions": [],
            "score_impacts": [],
            "performance_scores": {},
            "action_items": [],
            "compliance_flags": [],
            "speaker_metrics": speaker_metrics,
            "audio_observations": None,
            "model_used": "none",
            "model_version": None,
            "analysis_type": "failed",
            "processing_time": time.time() - start_time,
        }
    
    def _mock_analysis(
        self, 
        speaker_metrics: Dict[str, Any], 
        start_time: float
    ) -> Dict[str, Any]:
        """
        Mock analysis for development/testing.
        
        Returns deterministic test data matching the full specification.
        All text uses British English.
        """
        # Mock score impacts that would produce ~82% quality score
        mock_score_impacts = [
            {"timestamp_start": 0.0, "timestamp_end": 12.0, "impact": 4, "category": "Rapport_building", "reason": "Warm, personalised greeting", "quote": "Good morning, lovely to speak with you today!"},
            {"timestamp_start": 15.0, "timestamp_end": 30.0, "impact": 2, "category": "Script_adherence", "reason": "Proper identity verification", "quote": "Could I just confirm your postcode please?"},
            {"timestamp_start": 45.0, "timestamp_end": 90.0, "impact": 3, "category": "Product_knowledge", "reason": "Clear explanation of giving options", "quote": "We have several ways you can support us..."},
            {"timestamp_start": 120.0, "timestamp_end": 180.0, "impact": 4, "category": "Clarity", "reason": "Excellent Gift Aid explanation", "quote": "Gift Aid means we can claim an extra 25p for every pound you donate..."},
            {"timestamp_start": 200.0, "timestamp_end": 240.0, "impact": 2, "category": "Empathy", "reason": "Acknowledged supporter's generosity", "quote": "That's really wonderful, thank you so much for your support."},
            {"timestamp_start": 280.0, "timestamp_end": 300.0, "impact": 3, "category": "Closing_effectiveness", "reason": "Strong professional close", "quote": "Thank you again for your time today. Take care!"},
        ]
        
        return {
            "summary": "The supporter enquired about regular giving options. The agent explained the monthly donation programme and successfully enrolled the supporter at £10 per month with Gift Aid.",
            "sentiment_score": 7.5,
            "sentiment_label": "positive",
            "quality_score": self._calculate_quality_score(mock_score_impacts),
            "key_topics": self._add_keys_to_array([
                {"name": "Regular Giving", "confidence": 0.95},
                {"name": "Gift Aid", "confidence": 0.88},
                {"name": "Payment Processing", "confidence": 0.82},
            ]),
            "agent_actions": [
                {"action": "Greeted supporter", "timestamp_start": 0.0, "timestamp_end": 12.0},
                {"action": "Verified supporter identity", "timestamp_start": 15.0, "timestamp_end": 30.0},
                {"action": "Explained charity mission", "timestamp_start": 45.0, "timestamp_end": 90.0},
                {"action": "Explained Gift Aid", "timestamp_start": 120.0, "timestamp_end": 180.0},
                {"action": "Processed donation payment", "timestamp_start": 200.0, "timestamp_end": 240.0},
                {"action": "Closed call professionally", "timestamp_start": 280.0, "timestamp_end": 300.0},
            ],
            "score_impacts": self._add_keys_to_array(mock_score_impacts),
            "performance_scores": {
                "Empathy": 8,
                "Clarity": 9,
                "Listening": 7,
                "Script_adherence": 8,
                "Product_knowledge": 9,
                "Rapport_building": 8,
                "Objection_handling": 6,
                "Closing_effectiveness": 8,
            },
            "action_items": [
                {"description": "Send confirmation email with Gift Aid declaration", "priority": "high"},
            ],
            "compliance_flags": [],  # Empty - no keys needed
            "speaker_metrics": speaker_metrics if speaker_metrics else {
                "agent": {
                    "talk_time_seconds": 245,
                    "talk_time_percentage": 58,
                    "interruptions": 2,
                    "average_pace_wpm": 145,
                    "silence_percentage": 12
                },
                "supporter": {
                    "talk_time_seconds": 180,
                    "talk_time_percentage": 42,
                    "interruptions": 1,
                    "average_pace_wpm": 130,
                    "sentiment_trend": "positive"
                }
            },
            "audio_observations": {
                "call_quality": "good",
                "background_noise": "low",
                "agent_tone": "friendly",
                "supporter_tone": "happy",
            },
            "model_used": "mock",
            "model_version": "mock-v1.0.0",
            "analysis_type": "mock",
            "processing_time": time.time() - start_time,
        }
    
    def is_available(self) -> bool:
        """Check if analysis service is available."""
        if self.use_mock:
            return True
        return QWEN_OMNI_AVAILABLE and QWEN_OMNI_UTILS_AVAILABLE
