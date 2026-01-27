"""
Angel Intelligence - Call Processor

Processes individual call recordings through the full pipeline:
1. Download audio from PBX/R2
2. Convert GSM to WAV
3. Transcribe with Whisper
4. Identify speakers via voice fingerprinting
5. Detect and redact PII
6. Load client-specific configuration
7. Analyse with LLM
8. Save results to database
9. Handle audio retention/deletion
"""

import logging
import os
import time
from typing import Optional, Tuple, Dict, Any, List

from src.config import get_settings
from src.database import CallRecording, CallTranscription, CallAnalysis, ClientConfig
from src.services import (
    AudioDownloader, 
    TranscriptionService, 
    PIIDetector, 
    get_analysis_service,
    VoiceFingerprintService,
)

logger = logging.getLogger(__name__)


# Error codes for structured error messages
class ErrorCodes:
    RECORDING_NOT_FOUND = "RECORDING_NOT_FOUND"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    CONVERSION_FAILED = "CONVERSION_FAILED"
    TRANSCRIPTION_FAILED = "TRANSCRIPTION_FAILED"
    ANALYSIS_FAILED = "ANALYSIS_FAILED"
    ANALYSIS_PARSE_ERROR = "ANALYSIS_PARSE_ERROR"
    ANALYSIS_VALIDATION_ERROR = "ANALYSIS_VALIDATION_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    MODEL_LOAD_ERROR = "MODEL_LOAD_ERROR"


class CallProcessor:
    """
    Processes call recordings through the complete AI pipeline.
    
    Handles:
    - Audio download from PBX or R2
    - GSM to WAV conversion
    - Transcription with word-level timestamps
    - Voice fingerprint matching and speaker identification
    - PII detection and redaction
    - Client-specific configuration loading
    - AI analysis (audio or transcript mode)
    - Database persistence
    - Audio retention policy
    """
    
    def __init__(self):
        """Initialise the call processor."""
        settings = get_settings()
        
        # Log GPU info for debugging
        self._log_gpu_info()
        
        # Initialise services
        self.downloader = AudioDownloader()
        self.transcriber = TranscriptionService()
        self.pii_detector = PIIDetector()
        AnalysisService = get_analysis_service()
        self.analyser = AnalysisService()
        self.voice_fingerprint = VoiceFingerprintService()
        
        # Configuration
        self.retain_audio_default = False
        
        logger.info("CallProcessor initialised")
    
    def _log_gpu_info(self):
        """Log GPU information for debugging."""
        try:
            import torch
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
            logger.info(f"CUDA_VISIBLE_DEVICES: {cuda_visible}")
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info(f"CUDA available: {device_count} device(s)")
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    logger.info(f"  GPU {i}: {props.name} ({props.total_memory // (1024**2)} MB)")
            else:
                logger.warning("CUDA not available - running on CPU")
        except Exception as e:
            logger.warning(f"Could not get GPU info: {e}")
    
    def process(self, recording: CallRecording) -> bool:
        """
        Process a single call recording.
        
        Args:
            recording: CallRecording instance to process
            
        Returns:
            True if processing succeeded, False otherwise
        """
        audio_path = None
        is_local_file = False
        
        try:
            logger.info(f"Processing recording {recording.id}: {recording.apex_id}")
            
            # Mark as processing
            recording.mark_processing()
            
            # Load client-specific configuration
            client_config = self._load_client_config(recording.client_ref)
            
            # Check if we have an existing transcription for this apex_id (from Dojo)
            existing_transcription = CallTranscription.get_by_apex_id(recording.apex_id)
            
            if existing_transcription:
                # Reuse existing transcription - link it to this recording
                logger.info(f"Found existing transcription for apex_id {recording.apex_id}, reusing")
                existing_transcription.link_to_recording(recording.id)
                
                # Use existing transcript data
                transcript_result = {
                    "full_transcript": existing_transcription.full_transcript,
                    "segments": existing_transcription.segments,
                    "language_detected": existing_transcription.language_detected,
                    "confidence": existing_transcription.confidence_score,
                    "model_used": existing_transcription.model_used,
                }
                pii_result = None
                if existing_transcription.pii_detected:
                    pii_result = {
                        "pii_detected": existing_transcription.pii_detected,
                        "pii_count": len(existing_transcription.pii_detected),
                        "redacted_text": existing_transcription.redacted_transcript,
                    }
                transcription_id = existing_transcription.id
                
                # Still need to download audio for analysis
                audio_path, is_local_file = self._download_audio(recording)
            else:
                # No existing transcription - full pipeline
                # Step 1: Download and convert audio
                audio_path, is_local_file = self._download_audio(recording)
                
                # Step 2: Transcribe
                transcript_result = self._transcribe(audio_path)
                
                # Step 3: Identify speakers using voice fingerprinting
                transcript_result = self._identify_speakers(
                    audio_path,
                    transcript_result,
                    recording.halo_id
                )
                
                # Step 4: Detect and redact PII
                pii_result = self._detect_pii(transcript_result)
                
                # Step 5: Save transcription (with apex_id for future reuse)
                transcription_id = self._save_transcription(
                    recording, 
                    transcript_result, 
                    pii_result
                )
            
            # Step 5.5: Unload transcription models to free GPU for analysis
            self.transcriber.unload()
            
            # Step 6: Analyse (with client config and context)
            analysis_result = self._analyse(
                audio_path, 
                transcript_result, 
                recording.id,
                client_ref=recording.client_ref,
                campaign_type=recording.campaign_type,
                client_config=client_config
            )
            
            # Step 7: Save analysis
            if analysis_result:
                self._save_analysis(recording, analysis_result)
            
            # Step 8: Update voice fingerprint if we have a known agent
            if recording.halo_id and recording.agent_name:
                self._update_voice_fingerprint(
                    recording.halo_id,
                    recording.agent_name,
                    audio_path,
                    transcript_result.get("segments", [])
                )
            
            # Step 9: Handle audio retention
            r2_path = self._handle_audio_retention(
                recording,
                audio_path,
                pii_result,
                is_local_file
            )
            
            # Mark as completed
            recording.mark_completed(r2_path)
            
            logger.info(f"Successfully processed recording {recording.id}")
            return True
            
        except FileNotFoundError as e:
            logger.error(f"Recording not found: {e}")
            recording.mark_failed(f"{ErrorCodes.RECORDING_NOT_FOUND}: {e}")
            return False
            
        except RuntimeError as e:
            error_msg = str(e)
            if "download" in error_msg.lower():
                recording.mark_failed(f"{ErrorCodes.DOWNLOAD_FAILED}: {e}")
            elif "conversion" in error_msg.lower():
                recording.mark_failed(f"{ErrorCodes.CONVERSION_FAILED}: {e}")
            elif "model" in error_msg.lower():
                recording.mark_failed(f"{ErrorCodes.MODEL_LOAD_ERROR}: {e}")
            else:
                recording.mark_failed(str(e))
            logger.error(f"Processing error: {e}", exc_info=True)
            return False
            
        except Exception as e:
            logger.error(f"Failed to process recording {recording.id}: {e}", exc_info=True)
            recording.mark_failed(str(e))
            return False
            
        finally:
            # Clean up audio file
            self._cleanup_audio(audio_path, is_local_file)
    
    def _load_client_config(self, client_ref: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration for a client.
        
        Checks:
        1. Client-specific database config
        2. Global database config
        3. File-based defaults
        """
        config = {}
        
        for config_type in ["topics", "agent_actions", "performance_rubric", "prompt", "analysis_mode"]:
            db_config = ClientConfig.get_config(client_ref, config_type)
            if db_config:
                config[config_type] = db_config
        
        # If no database config, load from file
        if not config:
            import json
            config_path = "call_analysis_config.json"
            if os.path.exists(config_path):
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
        
        return config
    
    def _download_audio(self, recording: CallRecording) -> Tuple[str, bool]:
        """Download and convert audio file."""
        logger.info(f"Downloading audio for {recording.apex_id}")
        
        return self.downloader.download_recording(
            apex_id=recording.apex_id,
            call_date=recording.call_date,
            r2_path=recording.r2_path,
            r2_bucket=recording.r2_bucket
        )
    
    def _transcribe(self, audio_path: str) -> dict:
        """Transcribe audio file."""
        logger.info(f"Transcribing audio: {audio_path}")
        
        result = self.transcriber.transcribe(
            audio_path=audio_path,
            language="en",
            diarize=True
        )
        
        if not result or not result.get("full_transcript"):
            raise RuntimeError(f"{ErrorCodes.TRANSCRIPTION_FAILED}: Empty transcription result")
        
        return result
    
    def _identify_speakers(
        self,
        audio_path: str,
        transcript: dict,
        known_halo_id: Optional[int] = None
    ) -> dict:
        """Identify speakers using voice fingerprinting."""
        segments = transcript.get("segments", [])
        
        if self.voice_fingerprint.is_available():
            logger.info("Identifying speakers via voice fingerprint")
            
            identified_segments = self.voice_fingerprint.identify_speakers(
                audio_path,
                segments,
                known_halo_id
            )
            
            # Check for call transfers
            transfer_info = self.voice_fingerprint.detect_call_transfer(identified_segments)
            if transfer_info:
                logger.info(f"Call transfer detected: {transfer_info['agent_count']} agents")
                transcript["call_transfer"] = transfer_info
            
            transcript["segments"] = identified_segments
        else:
            # Fallback: Map SPEAKER_00 -> agent, others -> supporter
            logger.info("Voice fingerprinting not available - using fallback speaker labels")
            transcript["segments"] = self._normalize_speaker_labels(segments)
        
        return transcript
    
    def _normalize_speaker_labels(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalize speaker labels to agent/supporter.
        
        Uses heuristics to determine which speaker is the agent:
        1. Agent typically speaks FIRST (outbound calls)
        2. Agent typically speaks MORE (guiding the conversation)
        
        We calculate talk time per speaker and use first-speaker as tiebreaker.
        """
        if not segments:
            return segments
        
        # Calculate talk time per speaker
        speaker_talk_time = {}
        first_speaker = None
        
        for seg in segments:
            speaker = seg.get("speaker", "SPEAKER_00")
            if speaker.startswith("agent") or speaker == "agent":
                # Already labelled, skip calculation
                return self._simple_normalize(segments)
            
            duration = seg.get("end", 0) - seg.get("start", 0)
            speaker_talk_time[speaker] = speaker_talk_time.get(speaker, 0) + duration
            
            if first_speaker is None:
                first_speaker = speaker
        
        if not speaker_talk_time:
            return self._simple_normalize(segments)
        
        # Determine agent: speaker with most talk time (agent guides call)
        # For outbound charity calls, agent typically talks 55-70% of the time
        sorted_speakers = sorted(speaker_talk_time.items(), key=lambda x: x[1], reverse=True)
        most_talkative = sorted_speakers[0][0]
        
        # Use first speaker as tiebreaker if talk times are close (within 20%)
        if len(sorted_speakers) > 1:
            ratio = sorted_speakers[1][1] / sorted_speakers[0][1] if sorted_speakers[0][1] > 0 else 0
            if ratio > 0.8:  # Talk times are close, use first speaker heuristic
                agent_speaker = first_speaker
                logger.debug(f"Talk times close ({ratio:.2f}), using first speaker as agent: {first_speaker}")
            else:
                agent_speaker = most_talkative
                logger.debug(f"Using most talkative speaker as agent: {most_talkative} ({sorted_speakers[0][1]:.1f}s)")
        else:
            agent_speaker = most_talkative
        
        # Normalize labels
        normalized = []
        for seg in segments:
            new_seg = seg.copy()
            speaker = seg.get("speaker", "SPEAKER_00")
            
            if speaker == agent_speaker:
                new_seg["speaker"] = "agent"
                new_seg["speaker_id"] = "agent"
            else:
                new_seg["speaker"] = "supporter"
                new_seg["speaker_id"] = "supporter"
            
            normalized.append(new_seg)
        
        return normalized
    
    def _simple_normalize(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Simple fallback normalization using SPEAKER_00 = agent."""
        normalized = []
        for seg in segments:
            new_seg = seg.copy()
            speaker = seg.get("speaker", "SPEAKER_00")
            
            if speaker in ["SPEAKER_00", "agent"] or speaker.startswith("agent_"):
                new_seg["speaker"] = "agent"
                new_seg["speaker_id"] = "agent"
            else:
                new_seg["speaker"] = "supporter"
                new_seg["speaker_id"] = "supporter"
            
            normalized.append(new_seg)
        
        return normalized
    
    def _update_voice_fingerprint(
        self,
        halo_id: int,
        agent_name: str,
        audio_path: str,
        segments: List[Dict[str, Any]]
    ) -> None:
        """Update voice fingerprint for known agent (agent segments only, never supporter)."""
        if not self.voice_fingerprint.is_available():
            return
        
        try:
            self.voice_fingerprint.update_fingerprint(halo_id, agent_name, audio_path, segments)
        except Exception as e:
            # Non-fatal - log but don't fail processing
            logger.warning(f"Failed to update voice fingerprint: {e}")
    
    def _detect_pii(self, transcript: dict) -> Optional[dict]:
        """Detect and prepare PII redaction data."""
        if not self.pii_detector.is_available():
            logger.info("PII detection not available")
            return None
        
        logger.info("Detecting PII in transcript")
        
        full_text = transcript.get("full_transcript", "")
        segments = transcript.get("segments", [])
        
        pii_result = self.pii_detector.detect(full_text, segments)
        
        if pii_result.get("pii_count", 0) > 0:
            logger.info(f"Detected {pii_result['pii_count']} PII items: {pii_result['pii_types']}")
            
            # Redact segments
            pii_result["redacted_segments"] = self.pii_detector.redact_segments(
                segments,
                pii_result["pii_detected"],
                full_text
            )
        
        return pii_result
    
    def _save_transcription(
        self, 
        recording: CallRecording, 
        transcript: dict,
        pii_result: Optional[dict]
    ) -> int:
        """Save transcription to database."""
        logger.info(f"Saving transcription for recording {recording.id}")
        
        # Use redacted versions if PII was detected
        if pii_result and pii_result.get("pii_count", 0) > 0:
            full_transcript = pii_result["redacted_text"]
            segments = pii_result["redacted_segments"]
            redacted_transcript = pii_result["redacted_text"]
            pii_detected = pii_result["pii_detected"]
        else:
            full_transcript = transcript.get("full_transcript", "")
            segments = transcript.get("segments", [])
            redacted_transcript = None
            pii_detected = None
        
        transcription = CallTranscription(
            ai_call_recording_id=recording.id,
            apex_id=recording.apex_id,  # Store apex_id for future reuse
            full_transcript=full_transcript,
            segments=segments,
            redacted_transcript=redacted_transcript,
            pii_detected=pii_detected,
            language_detected=transcript.get("language_detected", "en"),
            confidence_score=transcript.get("confidence", 0.95),
            model_used=transcript.get("model_used", "whisperx-medium"),
            processing_time_seconds=int(transcript.get("processing_time", 0)),
        )
        
        return transcription.save()
    
    def _analyse(
        self, 
        audio_path: str, 
        transcript: dict,
        recording_id: int,
        client_ref: Optional[str] = None,
        campaign_type: Optional[str] = None,
        client_config: Optional[Dict[str, Any]] = None
    ) -> Optional[dict]:
        """Analyse the call recording with optional client-specific config."""
        if not self.analyser.is_available():
            logger.info("Analysis service not available - skipping")
            return None
        
        logger.info(f"Analysing recording {recording_id} (client={client_ref}, campaign={campaign_type})")
        
        # Pass client config to analyser if available
        if client_config:
            # Override analysis mode if client specifies
            if "analysis_mode" in client_config:
                mode = client_config["analysis_mode"].get("mode")
                if mode in ("audio", "transcript"):
                    self.analyser.analysis_mode = mode
                    logger.info(f"Using client-specific analysis mode: {mode}")
        
        return self.analyser.analyse(
            audio_path=audio_path,
            transcript=transcript,
            recording_id=recording_id,
            client_ref=client_ref,
            campaign_type=campaign_type
        )
    
    def _save_analysis(self, recording: CallRecording, analysis: dict) -> int:
        """Save analysis to database."""
        logger.info(f"Saving analysis for recording {recording.id}")
        
        # Import zone helper
        from src.database.models import get_quality_zone
        
        # Calculate quality zone from score
        quality_score = analysis.get("quality_score", 65.0)
        quality_zone = get_quality_zone(quality_score)
        
        analysis_record = CallAnalysis(
            ai_call_recording_id=recording.id,
            summary=analysis.get("summary", ""),
            sentiment_score=analysis.get("sentiment_score", 0),
            quality_score=quality_score,
            quality_zone=quality_zone,
            key_topics=analysis.get("key_topics", []),
            agent_actions=analysis.get("agent_actions", []),
            score_impacts=analysis.get("score_impacts", []),
            performance_scores=analysis.get("performance_scores", {}),
            action_items=analysis.get("action_items", []),
            compliance_flags=analysis.get("compliance_flags", []),
            speaker_metrics=analysis.get("speaker_metrics", {}),
            audio_analysis=analysis.get("audio_analysis", analysis.get("audio_observations")),
            model_used=analysis.get("model_used", ""),
            model_version=analysis.get("model_version"),
            processing_time_seconds=int(analysis.get("processing_time", 0)),
        )
        
        return analysis_record.save()
    
    def _handle_audio_retention(
        self,
        recording: CallRecording,
        audio_path: str,
        pii_result: Optional[dict],
        is_local_file: bool
    ) -> Optional[str]:
        """
        Handle audio file retention based on recording settings.
        
        If retain_audio is True:
        - Create PII-redacted version of audio
        - Upload to R2
        - Return R2 path
        
        If retain_audio is False:
        - Audio will be deleted
        - Return None
        """
        if not recording.retain_audio:
            logger.info(f"Audio retention disabled for recording {recording.id}")
            return None
        
        logger.info(f"Retaining audio for recording {recording.id}")
        
        # Get PII timestamps for audio redaction
        pii_timestamps = []
        if pii_result and pii_result.get("pii_detected"):
            pii_timestamps = pii_result["pii_detected"]
        
        # Create redacted audio
        redacted_audio_path = self.pii_detector.redact_audio(
            audio_path,
            pii_timestamps
        )
        
        # Upload to R2
        if redacted_audio_path:
            r2_path = f"processed/{recording.apex_id}_redacted.wav"
            
            try:
                self.downloader.upload_to_r2(
                    local_path=redacted_audio_path,
                    r2_path=r2_path
                )
                
                # Clean up redacted file if it's a temp file
                if redacted_audio_path != audio_path and os.path.exists(redacted_audio_path):
                    os.unlink(redacted_audio_path)
                
                logger.info(f"Uploaded redacted audio to R2: {r2_path}")
                return r2_path
                
            except Exception as e:
                logger.error(f"Failed to upload audio to R2: {e}")
                return None
        else:
            # No PII to redact - upload original
            r2_path = f"processed/{recording.apex_id}.wav"
            
            try:
                self.downloader.upload_to_r2(
                    local_path=audio_path,
                    r2_path=r2_path
                )
                logger.info(f"Uploaded audio to R2: {r2_path}")
                return r2_path
                
            except Exception as e:
                logger.error(f"Failed to upload audio to R2: {e}")
                return None
    
    def _cleanup_audio(self, audio_path: Optional[str], is_local_file: bool) -> None:
        """Clean up temporary audio files."""
        if audio_path:
            self.downloader.cleanup_temp_file(audio_path, is_local_file)
