"""
Angel Intelligence - Transcription Service

Audio transcription using WhisperX with word-level timestamps and speaker diarisation.
Outputs British English transcripts.

Supports two modes:
1. Shared service mode: When TRANSCRIPTION_SERVICE_URL is set, uses HTTP proxy to call
   a shared WhisperX service (allows multiple workers to share one GPU model)
2. In-memory mode: When no service URL, loads model locally (fallback/standalone)
"""

import asyncio
import gc
import logging
import os
import time
import uuid
from typing import Dict, Any, List, Optional

import torch

# PyTorch 2.6+ compatibility fix for pyannote/HuggingFace model loading
# These models were saved with older PyTorch and contain omegaconf objects
# that aren't in the new safe globals list. Since we trust HuggingFace models,
# we patch torch.load to force weights_only=False for model loading.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    # Force weights_only=False for trusted model checkpoints
    # Lightning/pyannote explicitly pass weights_only=True which we need to override
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from src.config import get_settings

logger = logging.getLogger(__name__)

# Import WhisperX (may not be available in all environments)
try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    logger.warning("WhisperX not available - transcription disabled")


class TranscriptionService:
    """
    Audio transcription service using WhisperX.
    
    Features:
    - Word-level timestamp alignment
    - Speaker diarisation (requires HuggingFace token for pyannote)
    - Configurable model size (tiny, base, small, medium, large)
    - Shared service mode for multi-worker GPU sharing
    """
    
    def __init__(self):
        """Initialise the transcription service."""
        settings = get_settings()
        
        # Check for shared transcription service URL
        self.service_url = settings.transcription_service_url or ""
        self._proxy = None
        
        # Device configuration (for local mode)
        self.device = "cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.batch_size = 16 if self.device == "cuda" else 4
        
        # Model configuration
        self.model_size = settings.whisper_model
        self.segmentation = settings.transcript_segmentation
        
        # HuggingFace token for pyannote diarization
        self.hf_token = settings.huggingface_token or None
        
        # Models (lazy loaded for local mode)
        self._whisper_model = None
        self._alignment_model = None
        self._alignment_metadata = None
        self._diarize_model = None
        
        if self.service_url:
            logger.info(f"TranscriptionService initialised in PROXY mode: {self.service_url}")
        else:
            logger.info(f"TranscriptionService initialised in LOCAL mode: device={self.device}, model={self.model_size}, diarization={'pyannote' if self.hf_token else 'fallback'}")
    
    def _get_proxy(self):
        """Get or create the transcription proxy client."""
        if self._proxy is None and self.service_url:
            from src.services.transcriber_proxy import TranscriptionProxy
            self._proxy = TranscriptionProxy(self.service_url)
        return self._proxy
    
    async def _check_proxy_available(self) -> bool:
        """Check if the transcription service is available."""
        proxy = self._get_proxy()
        if proxy:
            try:
                return await proxy.health_check()
            except Exception as e:
                logger.debug(f"Transcription service unavailable: {e}")
                return False
        return False
    
    def _ensure_model_loaded(self) -> None:
        """Ensure the Whisper model is loaded."""
        if not WHISPERX_AVAILABLE:
            raise RuntimeError("WhisperX is not installed. Please install with: pip install git+https://github.com/m-bain/whisperx.git")
        
        if self._whisper_model is None:
            logger.info(f"Loading WhisperX model: {self.model_size}")
            self._whisper_model = whisperx.load_model(
                self.model_size,
                self.device,
                compute_type=self.compute_type
            )
            logger.info(f"WhisperX model loaded on {self.device}")
    
    def transcribe(
        self, 
        audio_path: str, 
        language: str = "en",
        diarize: bool = True
    ) -> Dict[str, Any]:
        """
        Transcribe an audio file.
        
        Tries the shared transcription service first (if configured),
        then falls back to local in-memory processing.
        
        Args:
            audio_path: Path to audio file (WAV format preferred)
            language: Language code (default 'en' for English)
            diarize: Whether to perform speaker diarisation
            
        Returns:
            Dictionary containing:
            - full_transcript: Complete transcript text
            - segments: List of word/sentence segments with timestamps
            - language_detected: Detected language code
            - confidence: Overall confidence score
        """
        # Try proxy mode first if configured
        if self.service_url:
            try:
                return self._transcribe_via_proxy(audio_path, language, diarize)
            except Exception as e:
                logger.warning(f"Proxy transcription failed, falling back to local: {e}")
        
        # Fall back to local transcription
        return self._transcribe_local(audio_path, language, diarize)
    
    def _transcribe_via_proxy(
        self,
        audio_path: str,
        language: str = "en",
        diarize: bool = True
    ) -> Dict[str, Any]:
        """Transcribe via the shared transcription service."""
        import asyncio
        
        proxy = self._get_proxy()
        if not proxy:
            raise RuntimeError("Transcription proxy not available")
        
        logger.info(f"Transcribing via proxy: {audio_path}")
        
        # Run async proxy call in sync context
        loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else asyncio.new_event_loop()
        
        async def _do_transcribe():
            result = await proxy.transcribe(
                audio_path=audio_path,
                diarize=diarize,
                language=language if language != "auto" else None,
            )
            return result
        
        # Handle both sync and async contexts
        try:
            if asyncio.get_event_loop().is_running():
                # We're in an async context, create a new thread to run the async call
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, _do_transcribe())
                    result = future.result(timeout=300)
            else:
                result = asyncio.run(_do_transcribe())
        except RuntimeError:
            # No event loop, create one
            result = asyncio.run(_do_transcribe())
        
        logger.info(f"Proxy transcription complete: {len(result.segments)} segments, {result.processing_time:.1f}s")
        
        # Convert to standard format
        return {
            "full_transcript": result.text,
            "text": result.text,
            "segments": result.segments,
            "language_detected": result.language,
            "language": result.language,
            "confidence": 0.95,
            "model_used": f"whisperx-{self.model_size}",
            "processing_time": result.processing_time,
            "duration": result.duration,
        }
    
    def _transcribe_local(
        self, 
        audio_path: str, 
        language: str = "en",
        diarize: bool = True
    ) -> Dict[str, Any]:
        """Transcribe locally using in-memory WhisperX model."""
        self._ensure_model_loaded()
        
        start_time = time.time()
        logger.info(f"Starting local transcription: {audio_path}")
        
        try:
            # Load audio
            audio = whisperx.load_audio(audio_path)
            
            # Transcribe
            transcribe_options = {
                "batch_size": self.batch_size,
                "language": language if language != "auto" else None,
            }
            
            result = self._whisper_model.transcribe(audio, **transcribe_options)
            
            detected_language = result.get("language", language)
            logger.info(f"Transcription complete, language: {detected_language}")
            
            # Align for better timestamps
            result = self._align_transcript(result, audio, detected_language)
            
            # Apply segmentation preference
            result = self._apply_segmentation(result)
            
            # Add speaker labels (diarization)
            if diarize:
                result = self._add_speaker_labels(result, audio)
            
            # Format response
            segments = self._format_segments(result["segments"])
            full_transcript = self._build_full_transcript(segments)
            
            # Clean up memory
            del audio
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.1f}s with {len(segments)} segments")
            
            return {
                "full_transcript": full_transcript,
                "text": full_transcript,
                "segments": segments,
                "language_detected": detected_language,
                "language": detected_language,
                "confidence": 0.95,  # WhisperX doesn't provide overall confidence
                "model_used": f"whisperx-{self.model_size}",
                "processing_time": processing_time,
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)
            raise
    
    def _align_transcript(
        self, 
        result: Dict[str, Any], 
        audio, 
        language: str
    ) -> Dict[str, Any]:
        """Align transcript for better word-level timestamps."""
        try:
            # Load alignment model if needed
            if self._alignment_model is None:
                self._alignment_model, self._alignment_metadata = whisperx.load_align_model(
                    language_code=language,
                    device=self.device
                )
            
            # Perform alignment
            result = whisperx.align(
                result["segments"],
                self._alignment_model,
                self._alignment_metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            
            logger.debug("Transcript alignment complete")
            return result
            
        except Exception as e:
            logger.warning(f"Alignment failed, using unaligned timestamps: {e}")
            return result
    
    def _apply_segmentation(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply configured segmentation preference.
        
        word mode: Sentence segments with nested word-level timestamps (for karaoke)
        sentence mode: Sentence segments only (no word breakdown)
        """
        segments = result.get("segments", [])
        
        if self.segmentation == "word":
            # Keep sentence-level segments but ensure words are included
            # This gives us: sentence -> words hierarchy for karaoke
            for segment in segments:
                # Words should already be present from alignment
                # Just ensure they exist
                if "words" not in segment:
                    # No word-level data available, create a single "word" from the segment
                    segment["words"] = [{
                        "word": segment.get("text", ""),
                        "start": segment.get("start", 0),
                        "end": segment.get("end", 0),
                        "score": segment.get("score", 0.95)
                    }]
            logger.debug(f"Using sentence segments with word timestamps: {len(segments)} segments")
        else:
            # Sentence mode - remove word-level data to reduce payload size
            for segment in segments:
                segment.pop("words", None)
            logger.debug(f"Using sentence-level segments only: {len(segments)} segments")
        
        return result
    
    def _add_speaker_labels(self, result: Dict[str, Any], audio) -> Dict[str, Any]:
        """
        Add speaker labels to segments using pyannote diarization.
        
        Falls back to gap-based heuristic if HuggingFace token not configured.
        """
        # Try pyannote diarization if token available
        if self.hf_token:
            try:
                return self._diarize_with_pyannote(result, audio)
            except Exception as e:
                logger.warning(f"Pyannote diarization failed, using fallback: {e}")
        
        # Fallback: Simple alternating speakers based on gaps
        logger.debug("Using gap-based speaker diarization fallback")
        current_speaker = 0
        last_end = 0
        speaker_gap_threshold = 2.0  # seconds
        
        for segment in result.get("segments", []):
            if segment.get("start", 0) - last_end > speaker_gap_threshold:
                current_speaker = (current_speaker + 1) % 2
            
            segment["speaker"] = f"SPEAKER_{current_speaker:02d}"
            last_end = segment.get("end", 0)
        
        return result
    
    def _diarize_with_pyannote(self, result: Dict[str, Any], audio) -> Dict[str, Any]:
        """
        Perform speaker diarization using pyannote via WhisperX.
        
        Requires HuggingFace token with access to pyannote models.
        """
        from whisperx.diarize import DiarizationPipeline
        
        # Load diarization model if needed
        if self._diarize_model is None:
            logger.info("Loading pyannote diarization model...")
            self._diarize_model = DiarizationPipeline(
                use_auth_token=self.hf_token,
                device=self.device
            )
            logger.info("Pyannote diarization model loaded")
        
        # Run diarization (expects 2 speakers for call recordings)
        diarize_segments = self._diarize_model(
            audio,
            min_speakers=2,
            max_speakers=2
        )
        
        # Assign speaker labels to transcript segments
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        logger.info("Pyannote diarization complete")
        return result
    
    def _format_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format segments for output.
        
        Format matches specification for karaoke feature:
        {
            "segment_id": "seg_001",
            "text": "Hello, thank you for calling.",
            "start": 0.0,
            "end": 2.5,
            "speaker": "agent",
            "speaker_id": "agent_001",
            "confidence": 0.95,
            "words": [
                {"word": "Hello", "start": 0.0, "end": 0.4, "confidence": 0.98},
                {"word": "thank", "start": 0.5, "end": 0.7, "confidence": 0.96}
            ]
        }
        """
        formatted = []
        for i, segment in enumerate(segments):
            formatted_segment = {
                "segment_id": f"seg_{i+1:03d}",  # Unique ID for this segment
                "text": segment.get("text", segment.get("word", "")).strip(),
                "start": round(segment.get("start", 0), 3),
                "end": round(segment.get("end", 0), 3),
                "speaker": segment.get("speaker", "SPEAKER_00"),
                "speaker_id": segment.get("speaker", "SPEAKER_00"),
                "confidence": round(segment.get("score", segment.get("confidence", 0.95)), 4),
            }
            
            # Include word-level timestamps if available (for karaoke)
            if "words" in segment:
                formatted_segment["words"] = [
                    {
                        "word": w.get("word", w.get("text", "")),
                        "start": round(w.get("start", 0), 3),
                        "end": round(w.get("end", 0), 3),
                        "confidence": round(w.get("score", 0.95), 4),
                    }
                    for w in segment["words"]
                ]
            
            formatted.append(formatted_segment)
        return formatted
    
    def _build_full_transcript(self, segments: List[Dict[str, Any]]) -> str:
        """Build full transcript from segments."""
        return " ".join([s["text"] for s in segments if s["text"]])
    
    def unload(self) -> None:
        """
        Unload all models to free GPU memory.
        
        Call this after transcription is complete and before loading
        other models (e.g., analysis model) to avoid CUDA OOM errors.
        
        In proxy mode, this closes the HTTP client.
        """
        logger.debug("Unloading transcription resources")
        
        # Close proxy client if in proxy mode
        if self._proxy is not None:
            import asyncio
            try:
                asyncio.run(self._proxy.close())
            except Exception:
                pass
            self._proxy = None
        
        # Unload local models
        if self._whisper_model is not None:
            del self._whisper_model
            self._whisper_model = None
        
        if self._alignment_model is not None:
            del self._alignment_model
            del self._alignment_metadata
            self._alignment_model = None
            self._alignment_metadata = None
        
        if self._diarize_model is not None:
            del self._diarize_model
            self._diarize_model = None
        
        # Force garbage collection and clear CUDA cache
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Transcription resources unloaded")
    
    def is_available(self) -> bool:
        """
        Check if transcription service is available.
        
        Returns True if either:
        - A transcription service URL is configured, or
        - WhisperX is installed locally
        """
        if self.service_url:
            return True
        return WHISPERX_AVAILABLE
    
    def get_mode(self) -> str:
        """Return the current transcription mode."""
        if self.service_url:
            return "proxy"
        return "local"
