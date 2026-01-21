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
from typing import Dict, Any, List, Optional

import torch

from src.config import get_settings

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
        settings = get_settings()
        
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
        
        # Models (lazy loaded)
        self._analysis_model = None  # Qwen2.5-Omni for audio mode
        self._analysis_processor = None
        self._text_model = None  # Text-only model for transcript mode
        self._text_tokenizer = None
        self._chat_model = None
        self._chat_processor = None
        
        # Load configuration
        self._config = self._load_config()
        
        logger.info(f"AnalysisService initialised: mode={self.analysis_mode}, device={self.device}")
    
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
            
            logger.info(f"Text model loaded on {self.device}")
    
    def analyse(
        self, 
        audio_path: Optional[str],
        transcript: Dict[str, Any],
        recording_id: int
    ) -> Dict[str, Any]:
        """
        Analyse a call recording.
        
        Args:
            audio_path: Path to audio file (required for audio mode)
            transcript: Transcript dictionary from transcription service
            recording_id: Database recording ID
            
        Returns:
            Analysis results dictionary
        """
        start_time = time.time()
        
        # Calculate speaker metrics (always needed)
        speaker_metrics = self._calculate_speaker_metrics(transcript)
        
        if self.use_mock:
            logger.info("Using mock analysis response")
            return self._mock_analysis(speaker_metrics, start_time)
        
        if self.analysis_mode == "audio" and audio_path:
            return self._analyse_audio(audio_path, transcript, speaker_metrics, start_time)
        else:
            return self._analyse_transcript(transcript, speaker_metrics, start_time)
    
    def _calculate_speaker_metrics(self, transcript: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for each speaker matching specification format."""
        raw_metrics = {}
        total_time = 0
        
        for segment in transcript.get("segments", []):
            speaker = segment.get("speaker", "SPEAKER_00")
            duration = segment.get("end", 0) - segment.get("start", 0)
            total_time += duration
            
            if speaker not in raw_metrics:
                raw_metrics[speaker] = {
                    "talk_time_seconds": 0,
                    "word_count": 0,
                    "segment_count": 0,
                    "interruptions": 0,
                }
            
            text = segment.get("text", "")
            raw_metrics[speaker]["talk_time_seconds"] += duration
            raw_metrics[speaker]["word_count"] += len(text.split())
            raw_metrics[speaker]["segment_count"] += 1
        
        # Calculate percentages and WPM
        formatted_metrics = {}
        for speaker, data in raw_metrics.items():
            talk_time = data["talk_time_seconds"]
            word_count = data["word_count"]
            
            # Determine if agent or supporter (first speaker is usually agent)
            speaker_label = "agent" if speaker == "SPEAKER_00" else "supporter"
            
            formatted_metrics[speaker_label] = {
                "talk_time_seconds": round(talk_time, 1),
                "talk_time_percentage": round((talk_time / total_time * 100) if total_time > 0 else 0, 1),
                "word_count": word_count,
                "average_pace_wpm": round((word_count / (talk_time / 60)) if talk_time > 0 else 0),
                "interruptions": data["interruptions"],
                "silence_percentage": 0,  # Would need more analysis
            }
        
        return formatted_metrics
    
    def _analyse_audio(
        self, 
        audio_path: str, 
        transcript: Dict[str, Any],
        speaker_metrics: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Analyse call using audio directly."""
        try:
            self._ensure_analysis_model_loaded()
            
            # Get full transcript text for context
            full_text = transcript.get("full_transcript", "")
            
            # Preprocess audio
            processed_audio = self._preprocess_audio(audio_path)
            
            # Chunk audio for processing
            chunks = self._chunk_audio(processed_audio, max_duration=30)
            
            # Sample chunks for efficiency
            if len(chunks) > 3:
                sample_chunks = [chunks[0], chunks[len(chunks)//2], chunks[-1]]
            else:
                sample_chunks = chunks
            
            # Analyse each chunk
            chunk_analyses = []
            for chunk_start, chunk_end, chunk_path in sample_chunks:
                analysis = self._analyse_audio_chunk(
                    chunk_path,
                    context=f"Segment from {chunk_start:.0f}s to {chunk_end:.0f}s"
                )
                chunk_analyses.append(analysis)
                
                # Clean up temp chunk
                if chunk_path != processed_audio and os.path.exists(chunk_path):
                    os.unlink(chunk_path)
            
            # Clean up preprocessed audio
            if processed_audio != audio_path and os.path.exists(processed_audio):
                os.unlink(processed_audio)
            
            # Aggregate results
            result = self._aggregate_analyses(chunk_analyses, speaker_metrics)
            result["processing_time"] = time.time() - start_time
            
            return result
            
        except Exception as e:
            logger.error(f"Audio analysis failed: {e}", exc_info=True)
            return self._empty_analysis(speaker_metrics, start_time)
        finally:
            # Free GPU memory for next recording's transcription
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
    
    def _analyse_transcript(
        self, 
        transcript: Dict[str, Any],
        speaker_metrics: Dict[str, Any],
        start_time: float
    ) -> Dict[str, Any]:
        """Analyse call using transcript text only (with text-only model)."""
        try:
            self._ensure_text_model_loaded()
            
            full_text = transcript.get("full_transcript", "")
            
            # Build prompt
            prompt = self._build_transcript_analysis_prompt(full_text)
            
            # Generate analysis using text-only model
            response = self._generate_text_only_response(prompt)
            
            # Parse response
            result = self._parse_analysis_response(response, speaker_metrics)
            result["processing_time"] = time.time() - start_time
            result["analysis_type"] = "transcript"
            
            return result
            
        except Exception as e:
            logger.error(f"Transcript analysis failed: {e}", exc_info=True)
            return self._empty_analysis(speaker_metrics, start_time)
        finally:
            # Free GPU memory for next recording's transcription
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
        """Generate text response using the text-only model."""
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
    
    def _analyse_audio_chunk(self, audio_path: str, context: str = "") -> Dict[str, Any]:
        """Analyse a single audio chunk."""
        topics = self._config.get("topics", [])
        actions = self._config.get("agent_actions", [])
        rubric = self._config.get("performance_rubric", [])
        
        # Build lists from config
        topic_list = ", ".join(topics) if topics else "donation, gift aid, regular giving, complaints, account updates"
        action_list = ", ".join(actions) if actions else "greeting, verification, explanation, objection handling, closing"
        rubric_list = ", ".join([r.get("name", r) if isinstance(r, dict) else r for r in rubric]) if rubric else "Empathy, Clarity, Listening, Rapport building"
        
        system_prompt = f"""You are an expert call quality analyst for charity supporter engagement calls.
{BRITISH_ENGLISH_INSTRUCTION}

Analyse this audio recording segment for:
1. TONE: Agent's voice (warm/neutral/cold/hostile)
2. EMOTION: Detected emotions (calm/frustrated/happy/anxious)
3. PACING: Speech speed (rushed/steady/slow)
4. PROFESSIONALISM: Overall professional conduct
5. SUPPORTER EXPERIENCE: How the supporter seems to feel
6. TOPICS: What subjects are being discussed
7. ACTIONS: What the agent is doing"""

        user_prompt = f"""Analyse this call recording audio.

{context}

TOPICS TO IDENTIFY: {topic_list}
AGENT ACTIONS TO DETECT: {action_list}
PERFORMANCE CRITERIA: {rubric_list}

Return ONLY valid JSON matching this structure:
{{
    "agent_tone": "warm/neutral/cold/hostile",
    "agent_emotion": "calm/frustrated/happy/anxious/other",
    "supporter_emotion": "calm/frustrated/happy/anxious/other",
    "speech_clarity": 1-10,
    "professionalism": 1-10,
    "detected_topics": ["topics heard in this segment"],
    "detected_actions": ["actions performed by agent"],
    "performance_scores": {{
        "Empathy": 1-10,
        "Clarity": 1-10,
        "Listening": 1-10,
        "Rapport_building": 1-10
    }},
    "detected_issues": ["any compliance or quality issues"],
    "positive_indicators": ["things the agent did well"],
    "sentiment_score": -10 to +10,
    "brief_observation": "One sentence observation about this segment"
}}

Return ONLY the JSON object, no other text."""

        try:
            # Build conversation with audio
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": audio_path},
                        {"type": "text", "text": system_prompt + "\n\n" + user_prompt}
                    ]
                }
            ]
            
            # Process with Qwen
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
                    max_new_tokens=600,
                    do_sample=False,
                    return_audio=False
                )
            
            generated_ids = text_ids[:, input_len:]
            response = self._analysis_processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
            return self._parse_chunk_response(response)
            
        except Exception as e:
            logger.error(f"Chunk analysis failed: {e}")
            return self._default_chunk_analysis()
    
    def _build_transcript_analysis_prompt(self, transcript: str) -> str:
        """Build prompt for transcript-based analysis."""
        topics = self._config.get("topics", [])
        actions = self._config.get("agent_actions", [])
        rubric = self._config.get("performance_rubric", [])
        
        # Build topic and action lists for the prompt - use full lists from config
        topic_list = ", ".join(topics) if topics else "donation, gift aid, regular giving, complaints, updates, account changes, direct debit, legacy giving, event registration, volunteer enquiry"
        action_list = ", ".join(actions) if actions else "greeting, verification, explanation, objection handling, closing, upselling, complaint resolution, payment processing"
        rubric_list = ", ".join([r.get("name", r) if isinstance(r, dict) else r for r in rubric]) if rubric else "Empathy, Clarity, Listening, Script adherence, Product knowledge, Rapport building, Objection handling, Closing effectiveness"
        
        # Optionally truncate transcript for smaller models
        if self.max_transcript_length > 0:
            transcript = transcript[:self.max_transcript_length]
            # Use simplified prompt for constrained models
            return self._build_simple_analysis_prompt(transcript, topic_list, rubric_list)
        
        return f"""You are an expert call quality analyst for charity supporter engagement calls.
Analyse this transcript thoroughly and provide detailed coaching insights.

TRANSCRIPT:
{transcript}

TOPICS TO IDENTIFY: {topic_list}
AGENT ACTIONS TO DETECT: {action_list}
PERFORMANCE CRITERIA: {rubric_list}

Return your analysis as valid JSON matching this exact structure:

{{
    "summary": "2-3 sentence British English summary describing the call purpose, outcome, and notable moments",
    "sentiment_score": -10 to +10 (negative=hostile, 0=neutral, positive=friendly),
    "quality_score": 0 to 100 (overall call quality rating),
    "key_topics": [
        {{"name": "topic from the list above", "confidence": 0.0-1.0}}
    ],
    "agent_actions_performed": [
        {{"action": "specific action the agent took", "timestamp_start": 0.0, "quality": 1-5}}
    ],
    "performance_scores": {{
        "Empathy": 1-10,
        "Clarity": 1-10,
        "Listening": 1-10,
        "Script_adherence": 1-10,
        "Product_knowledge": 1-10,
        "Rapport_building": 1-10,
        "Objection_handling": 1-10,
        "Closing_effectiveness": 1-10
    }},
    "action_items": [
        {{"description": "specific follow-up action needed", "priority": "high/medium/low"}}
    ],
    "compliance_flags": [
        {{"type": "category", "issue": "detailed description of compliance concern", "severity": "low/medium/high/critical"}}
    ],
    "improvement_areas": [
        {{
            "area": "specific skill or behaviour needing improvement",
            "description": "detailed explanation of what went wrong and exactly how to improve",
            "priority": "high/medium/low",
            "examples": ["direct quote from transcript showing the issue"]
        }}
    ]
}}

ANALYSIS GUIDANCE:
- Be thorough in identifying improvement areas - this is used for agent coaching
- Include specific quotes from the transcript as evidence
- Score performance honestly - most calls should score 60-80, not 90+
- Flag any compliance issues: GDPR, payment security, misleading information, rudeness
- Identify: rushing, interrupting, poor listening, weak objection handling, unclear explanations, lack of empathy, missed upsell opportunities

Return ONLY valid JSON. No text before or after the JSON object."""
    
    def _build_simple_analysis_prompt(self, transcript: str, topic_list: str, rubric_list: str) -> str:
        """Simplified prompt for smaller models (3B-7B)."""
        return f"""Analyse this charity call transcript.

TRANSCRIPT:
{transcript}

Return JSON with these EXACT field names (copy exactly):

{{
    "summary": "Brief 2 sentence description of what happened in the call",
    "sentiment_score": 5,
    "quality_score": 75,
    "key_topics": ["Gift Aid", "Donation"],
    "performance_scores": {{"Empathy": 7, "Clarity": 7, "Listening": 7}},
    "improvement_areas": ["Could explain Gift Aid more clearly"],
    "compliance_issues": []
}}

IMPORTANT - Use these EXACT field names:
- "summary" (not "summarized" or "Summary")
- "sentiment_score" (not "sentivity_score")
- "quality_score" (not "Quality_Score")
- "key_topics" (not "Key_Topics")
- "performance_scores" (not "Performance_Scores")
- "improvement_areas" (not "Improvement_Areas")
- "compliance_issues" (not "Compliance_Issues")

All field names must be lowercase with underscores.
Return ONLY the JSON object, nothing else."""
    
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
        try:
            start = response.find('{')
            end = response.rfind('}')
            logger.debug(f"Parsing response: found JSON from {start} to {end}")
            if start != -1 and end != -1:
                json_str = response[start:end+1]
                
                # Clean up common LLM JSON formatting issues
                json_str = self._clean_json_response(json_str)
                
                # Dump full JSON for debugging (after cleaning)
                logger.info(f"=== CLEANED JSON RESPONSE ===\n{json_str}\n=== END JSON ===")
                
                parsed = json.loads(json_str)
                
                # Fix performance_scores if model used wrong structure
                perf_scores = parsed.get("performance_scores", {})
                if isinstance(perf_scores, dict) and "criterion_name" in perf_scores and "score" in perf_scores:
                    # Model output: {"criterion_name": "Empathy", "score": 8}
                    # Convert to: {"Empathy": 8}
                    perf_scores = {perf_scores["criterion_name"]: perf_scores["score"]}
                
                # Ensure all required fields with correct naming
                result = {
                    "summary": parsed.get("summary", "Analysis completed"),
                    "sentiment_score": self._extract_number(parsed.get("sentiment_score", 0)),
                    "quality_score": self._extract_number(parsed.get("quality_score", 50)),
                    "key_topics": parsed.get("key_topics", []),
                    "agent_actions_performed": parsed.get("agent_actions_performed", parsed.get("agent_actions", [])),
                    "performance_scores": perf_scores,
                    "action_items": parsed.get("action_items", []),
                    "compliance_flags": parsed.get("compliance_flags", []),
                    "improvement_areas": parsed.get("improvement_areas", []),
                    "speaker_metrics": speaker_metrics,
                    "model_used": self.analysis_model_path,
                }
                logger.info(f"Parsed analysis: summary='{result['summary'][:50]}...', quality={result['quality_score']}")
                return result
            else:
                logger.warning(f"No JSON found in response. Full response:\n{response}")
                
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed: {e}. Attempting fallback extraction.")
            logger.info(f"=== CLEANED JSON THAT FAILED ===\n{json_str}\n=== END ===")
            # Try to extract what we can from partial/malformed JSON
            return self._fallback_parse_response(response, speaker_metrics)
        
        return self._empty_analysis(speaker_metrics, 0)
    
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
        
        # Extract quality_score (handle variations)
        quality_match = re.search(r'"(?:quality_score|Quality_Score|quality)"\s*:\s*(\d+(?:\.\d+)?)', response)
        if quality_match:
            result["quality_score"] = float(quality_match.group(1))
        
        # Try to extract arrays by finding valid JSON subarrays (handle variations)
        result["key_topics"] = self._extract_json_array(response, "key_topics") or self._extract_json_array(response, "Key_Topics")
        result["agent_actions_performed"] = self._extract_json_array(response, "agent_actions_performed") or self._extract_json_array(response, "Agent_Actions")
        result["action_items"] = self._extract_json_array(response, "action_items") or self._extract_json_array(response, "Action_Items")
        result["compliance_flags"] = self._extract_json_array(response, "compliance_flags") or self._extract_json_array(response, "compliance_issues") or self._extract_json_array(response, "Compliance_Issues")
        result["improvement_areas"] = self._extract_json_array(response, "improvement_areas") or self._extract_json_array(response, "Improvement_Areas")
        
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
        
        extracted_count = sum(1 for k in ["summary", "key_topics", "agent_actions_performed", "action_items", "improvement_areas"] 
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
        sentiment_scores = [c.get("sentiment_score", 0) for c in chunk_analyses]
        clarity_scores = [c.get("speech_clarity", 7) for c in chunk_analyses]
        professionalism_scores = [c.get("professionalism", 7) for c in chunk_analyses]
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        avg_clarity = sum(clarity_scores) / len(clarity_scores)
        avg_professionalism = sum(professionalism_scores) / len(professionalism_scores)
        
        # Collect all items
        all_topics = []
        all_actions = []
        all_issues = []
        all_positives = []
        observations = []
        all_performance_scores = {}
        
        for c in chunk_analyses:
            all_topics.extend(c.get("detected_topics", []))
            all_actions.extend(c.get("detected_actions", []))
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
        
        # Calculate quality score
        quality_score = self._sentiment_to_quality(avg_sentiment)
        
        # Build summary
        summary = " ".join(observations[:3]) if observations else "Audio analysis completed."
        
        # Format topics for database
        key_topics = [
            {"name": t, "confidence": 0.8} 
            for t in list(set(all_topics))[:10]
        ]
        
        # Format actions for database (using spec format)
        agent_actions_performed = [
            {"action": a, "timestamp_start": 0.0, "quality": 4}
            for a in list(set(all_actions))[:10]
        ]
        
        return {
            "summary": summary[:500],
            "sentiment_score": round(avg_sentiment, 1),
            "quality_score": round(quality_score, 1),
            "key_topics": key_topics,
            "agent_actions_performed": agent_actions_performed,
            "performance_scores": final_performance_scores,
            "action_items": [],
            "compliance_flags": [{"type": "issue", "issue": i, "severity": "medium"} for i in list(set(all_issues))[:5]],
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
        """Convert sentiment score (-10 to +10) to quality score (0 to 100)."""
        if sentiment >= 8:
            return 95 + (sentiment - 8) * 2.5
        elif sentiment >= 5:
            return 85 + (sentiment - 5) * 3
        elif sentiment >= 2:
            return 70 + (sentiment - 2) * 4.67
        elif sentiment >= -1:
            return 50 + (sentiment + 1) * 6.33
        elif sentiment >= -4:
            return 30 + (sentiment + 4) * 6.33
        elif sentiment >= -7:
            return 10 + (sentiment + 7) * 6.33
        else:
            return max(0, min(9, (sentiment + 10) * 3))
    
    def _default_chunk_analysis(self) -> Dict[str, Any]:
        """Default chunk analysis when processing fails."""
        return {
            "agent_tone": "neutral",
            "agent_emotion": "calm",
            "supporter_emotion": "calm",
            "speech_clarity": 7,
            "professionalism": 7,
            "detected_topics": [],
            "detected_actions": [],
            "performance_scores": {},
            "detected_issues": [],
            "positive_indicators": [],
            "sentiment_score": 0,
            "brief_observation": "Analysis unavailable for this segment",
        }
    
    def _empty_analysis(
        self, 
        speaker_metrics: Dict[str, Any], 
        start_time: float
    ) -> Dict[str, Any]:
        """Empty analysis when processing fails."""
        return {
            "summary": "Analysis unavailable",
            "sentiment_score": 0,
            "sentiment_label": "neutral",
            "quality_score": 50.0,
            "key_topics": [],
            "agent_actions_performed": [],
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
        return {
            "summary": "The supporter enquired about regular giving options. The agent explained the monthly donation programme and successfully enrolled the supporter at £10 per month with Gift Aid.",
            "sentiment_score": 7.5,
            "sentiment_label": "positive",
            "quality_score": 85.0,
            "key_topics": [
                {"name": "Regular giving signup", "confidence": 0.95, "timestamp_start": 45.5, "timestamp_end": 78.2},
                {"name": "Gift Aid explanation/enrolment", "confidence": 0.88, "timestamp_start": 120.0, "timestamp_end": 180.5},
                {"name": "Donation completion/payment processing", "confidence": 0.82, "timestamp_start": 200.0, "timestamp_end": 245.0},
            ],
            "agent_actions_performed": [
                {"action": "Greeted supporter", "timestamp_start": 0.0, "quality": 5},
                {"action": "Verified supporter identity", "timestamp_start": 15.2, "quality": 4},
                {"action": "Explained charity mission/programs", "timestamp_start": 45.5, "quality": 5},
                {"action": "Requested regular giving signup", "timestamp_start": 120.5, "quality": 5},
                {"action": "Processed donation payment", "timestamp_start": 200.0, "quality": 4},
                {"action": "Thanked supporter for their time", "timestamp_start": 280.0, "quality": 5},
            ],
            "performance_scores": {
                "Clarity of speech": 8,
                "Tone control": 9,
                "Active listening": 7,
                "Empathy & rapport": 8,
                "Confidence & authority": 7,
                "Accurate information delivery": 9,
                "Script/protocol adherence": 8,
                "Payment and data protection compliance": 10,
                "Recording of mandatory information": 9,
                "Call structure/flow control": 7,
                "Quality of donation ask or conversion attempt": 8,
                "Objection handling skill": 6,
                "Engagement effectiveness": 8,
                "Problem solving": 7,
                "Effective closing": 8,
            },
            "action_items": [
                {"description": "Send confirmation email with Gift Aid declaration", "priority": "high", "due_date": "2026-01-20"},
            ],
            "compliance_flags": [],
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
