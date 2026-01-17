"""
Angel Intelligence - Transcription Service

Audio transcription using WhisperX with word-level timestamps and speaker diarisation.
Outputs British English transcripts.
"""

import gc
import logging
import os
import time
from typing import Dict, Any, List, Optional

import torch

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
    """
    
    def __init__(self):
        """Initialise the transcription service."""
        settings = get_settings()
        
        # Device configuration
        self.device = "cuda" if torch.cuda.is_available() and settings.use_gpu else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.batch_size = 16 if self.device == "cuda" else 4
        
        # Model configuration
        self.model_size = settings.whisper_model
        self.segmentation = settings.transcript_segmentation
        
        # Models (lazy loaded)
        self._whisper_model = None
        self._alignment_model = None
        self._alignment_metadata = None
        
        logger.info(f"TranscriptionService initialised: device={self.device}, model={self.model_size}")
    
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
        self._ensure_model_loaded()
        
        start_time = time.time()
        logger.info(f"Starting transcription: {audio_path}")
        
        try:
            # Load audio
            audio = whisperx.load_audio(audio_path)
            
            # Transcribe
            result = self._whisper_model.transcribe(
                audio,
                batch_size=self.batch_size,
                language=language if language != "auto" else None
            )
            
            detected_language = result.get("language", language)
            logger.info(f"Transcription complete, language: {detected_language}")
            
            # Align for better timestamps
            result = self._align_transcript(result, audio, detected_language)
            
            # Apply segmentation preference
            result = self._apply_segmentation(result)
            
            # Add speaker labels
            if diarize:
                result = self._add_speaker_labels(result)
            
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
                "segments": segments,
                "language_detected": detected_language,
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
        """Apply configured segmentation preference (word or sentence level)."""
        if self.segmentation == "word":
            # Try to get word-level segments
            if "word_segments" in result:
                result["segments"] = result["word_segments"]
                logger.debug(f"Using word-level segments: {len(result['segments'])} words")
            elif "segments" in result and len(result["segments"]) > 0:
                # Extract words from sentence segments
                if "words" in result["segments"][0]:
                    word_segments = []
                    for segment in result["segments"]:
                        if "words" in segment:
                            for word in segment["words"]:
                                word_segments.append({
                                    "start": word.get("start", segment["start"]),
                                    "end": word.get("end", segment["end"]),
                                    "text": word.get("word", word.get("text", "")),
                                    "score": word.get("score", segment.get("score", 0.95))
                                })
                    if word_segments:
                        result["segments"] = word_segments
                        logger.debug(f"Extracted {len(word_segments)} word-level segments")
        else:
            logger.debug(f"Using sentence-level segments: {len(result.get('segments', []))} segments")
        
        return result
    
    def _add_speaker_labels(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add speaker labels to segments.
        
        Uses simple gap-based diarisation as fallback.
        For proper diarisation, pyannote.audio with HuggingFace token is needed.
        """
        # Simple alternating speakers based on gaps
        # TODO: Integrate pyannote.audio for proper diarisation
        current_speaker = 0
        last_end = 0
        speaker_gap_threshold = 2.0  # seconds
        
        for segment in result.get("segments", []):
            if segment.get("start", 0) - last_end > speaker_gap_threshold:
                current_speaker = (current_speaker + 1) % 2
            
            segment["speaker"] = f"SPEAKER_{current_speaker:02d}"
            last_end = segment.get("end", 0)
        
        return result
    
    def _format_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format segments for output.
        
        Format matches specification for karaoke feature:
        {
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
        for segment in segments:
            formatted_segment = {
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
    
    def is_available(self) -> bool:
        """Check if transcription service is available."""
        return WHISPERX_AVAILABLE
