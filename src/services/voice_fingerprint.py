"""
Angel Intelligence - Voice Fingerprinting Service

Speaker identification using voice embeddings.
Identifies agents based on voice characteristics for:
- Automatic agent identification
- Speaker labelling in transcripts
- Call transfer detection
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from src.config import get_settings
from src.database import VoiceFingerprint

logger = logging.getLogger(__name__)

# Check for speaker embedding libraries
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    RESEMBLYZER_AVAILABLE = True
except ImportError:
    RESEMBLYZER_AVAILABLE = False
    logger.warning("resemblyzer not available - voice fingerprinting disabled")

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False


class VoiceFingerprintService:
    """
    Voice fingerprinting for agent identification.
    
    Features:
    - Build fingerprints from known agent calls
    - Match speakers against known fingerprints
    - Detect call transfers (multiple agents)
    - Update fingerprints with new samples
    """
    
    def __init__(self):
        """Initialise the voice fingerprint service."""
        settings = get_settings()
        self.use_mock = settings.use_mock_models
        
        # Voice encoder (lazy loaded)
        self._encoder = None
        
        # Cache of known fingerprints
        self._fingerprint_cache: Dict[int, np.ndarray] = {}
        
        logger.info(f"VoiceFingerprintService initialised (mock={self.use_mock})")
    
    def _ensure_encoder_loaded(self) -> None:
        """Ensure the voice encoder is loaded."""
        if self.use_mock:
            return
        
        if not RESEMBLYZER_AVAILABLE:
            raise RuntimeError("resemblyzer not installed - run: pip install resemblyzer")
        
        if self._encoder is None:
            logger.info("Loading voice encoder...")
            self._encoder = VoiceEncoder()
            logger.info("Voice encoder loaded")
    
    def _load_fingerprint_cache(self) -> None:
        """Load all fingerprints from database into cache."""
        fingerprints = VoiceFingerprint.get_all()
        
        for fp in fingerprints:
            if fp.fingerprint_data:
                try:
                    embedding = np.frombuffer(fp.fingerprint_data, dtype=np.float32)
                    self._fingerprint_cache[fp.halo_id] = embedding
                except Exception as e:
                    logger.warning(f"Failed to load fingerprint for halo_id {fp.halo_id}: {e}")
        
        logger.info(f"Loaded {len(self._fingerprint_cache)} voice fingerprints")
    
    def extract_embedding(self, audio_path: str) -> Optional[np.ndarray]:
        """
        Extract voice embedding from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            256-dimensional embedding vector, or None if failed
        """
        if self.use_mock:
            return np.random.randn(256).astype(np.float32)
        
        try:
            self._ensure_encoder_loaded()
            
            wav = preprocess_wav(audio_path)
            embedding = self._encoder.embed_utterance(wav)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to extract embedding: {e}")
            return None
    
    def extract_segment_embeddings(
        self, 
        audio_path: str,
        segments: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """
        Extract embeddings for each speaker segment.
        
        Args:
            audio_path: Path to audio file
            segments: List of transcript segments with speaker labels
            
        Returns:
            Dict mapping speaker label to average embedding
        """
        if self.use_mock:
            return {
                "SPEAKER_00": np.random.randn(256).astype(np.float32),
                "SPEAKER_01": np.random.randn(256).astype(np.float32),
            }
        
        try:
            self._ensure_encoder_loaded()
            
            # Load full audio
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_path)
            
            speaker_samples = {}
            
            for segment in segments:
                speaker = segment.get("speaker", "SPEAKER_00")
                start_sample = int(segment.get("start", 0) * sample_rate)
                end_sample = int(segment.get("end", 0) * sample_rate)
                
                if end_sample > start_sample and end_sample <= len(audio_data):
                    segment_audio = audio_data[start_sample:end_sample]
                    
                    if speaker not in speaker_samples:
                        speaker_samples[speaker] = []
                    speaker_samples[speaker].append(segment_audio)
            
            # Calculate average embedding per speaker
            speaker_embeddings = {}
            
            for speaker, samples in speaker_samples.items():
                if samples:
                    # Concatenate all samples for this speaker
                    combined = np.concatenate(samples)
                    
                    # Get embedding
                    if len(combined) > sample_rate * 0.5:  # At least 0.5 seconds
                        embedding = self._encoder.embed_utterance(combined)
                        speaker_embeddings[speaker] = embedding.astype(np.float32)
            
            return speaker_embeddings
            
        except Exception as e:
            logger.error(f"Failed to extract segment embeddings: {e}")
            return {}
    
    def match_speaker(
        self, 
        embedding: np.ndarray,
        threshold: float = 0.85
    ) -> Optional[Tuple[int, str, float]]:
        """
        Match embedding against known agent fingerprints.
        
        Args:
            embedding: 256-dimensional voice embedding
            threshold: Minimum similarity score (0-1)
            
        Returns:
            Tuple of (halo_id, agent_name, confidence) or None
        """
        if not self._fingerprint_cache:
            self._load_fingerprint_cache()
        
        best_match = None
        best_score = 0
        
        for halo_id, known_embedding in self._fingerprint_cache.items():
            # Cosine similarity
            score = np.dot(embedding, known_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(known_embedding)
            )
            
            if score > best_score and score >= threshold:
                best_score = score
                best_match = halo_id
        
        if best_match:
            # Get agent name from database
            fp = VoiceFingerprint.get_by_halo_id(best_match)
            if fp:
                return (best_match, fp.agent_name, float(best_score))
        
        return None
    
    def identify_speakers(
        self,
        audio_path: str,
        segments: List[Dict[str, Any]],
        known_halo_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify speakers in transcript segments.
        
        Args:
            audio_path: Path to audio file
            segments: List of transcript segments
            known_halo_id: Known agent ID (from database record)
            
        Returns:
            Updated segments with speaker identification
        """
        if self.use_mock:
            # In mock mode, just label first speaker as agent
            updated = []
            for seg in segments:
                new_seg = seg.copy()
                if seg.get("speaker") == "SPEAKER_00":
                    new_seg["speaker"] = "agent"
                    new_seg["speaker_id"] = f"agent_{known_halo_id or 'unknown'}"
                    new_seg["speaker_confidence"] = 0.95
                else:
                    new_seg["speaker"] = "supporter"
                    new_seg["speaker_id"] = "supporter"
                    new_seg["speaker_confidence"] = 0.90
                updated.append(new_seg)
            return updated
        
        try:
            # Extract embeddings for each speaker
            speaker_embeddings = self.extract_segment_embeddings(audio_path, segments)
            
            # Try to identify each speaker
            speaker_identities = {}
            
            for speaker_label, embedding in speaker_embeddings.items():
                match = self.match_speaker(embedding)
                
                if match:
                    halo_id, agent_name, confidence = match
                    speaker_identities[speaker_label] = {
                        "type": "agent",
                        "halo_id": halo_id,
                        "name": agent_name,
                        "confidence": confidence,
                    }
                elif known_halo_id and speaker_label == "SPEAKER_00":
                    # Assume first speaker is the known agent
                    speaker_identities[speaker_label] = {
                        "type": "agent",
                        "halo_id": known_halo_id,
                        "name": None,
                        "confidence": 0.70,  # Lower confidence since not voice-matched
                    }
                else:
                    speaker_identities[speaker_label] = {
                        "type": "supporter",
                        "halo_id": None,
                        "name": None,
                        "confidence": 0.60,
                    }
            
            # Update segments with identities
            updated = []
            for seg in segments:
                new_seg = seg.copy()
                speaker_label = seg.get("speaker", "SPEAKER_00")
                identity = speaker_identities.get(speaker_label, {})
                
                new_seg["speaker"] = identity.get("type", "unknown")
                if identity.get("halo_id"):
                    new_seg["speaker_id"] = f"agent_{identity['halo_id']}"
                else:
                    new_seg["speaker_id"] = identity.get("type", "unknown")
                new_seg["speaker_confidence"] = identity.get("confidence", 0.5)
                
                updated.append(new_seg)
            
            return updated
            
        except Exception as e:
            logger.error(f"Speaker identification failed: {e}")
            return segments
    
    def _extract_agent_only_embedding(
        self,
        audio_path: str,
        segments: List[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """
        Extract embedding from agent segments only.
        
        Filters segments to only include those labelled as 'agent' or 'SPEAKER_00',
        concatenates the audio, and extracts a single embedding.
        
        Args:
            audio_path: Path to audio file
            segments: Transcript segments with speaker labels
            
        Returns:
            256-dimensional embedding for agent voice only, or None if failed
        """
        if self.use_mock:
            return np.random.randn(256).astype(np.float32)
        
        try:
            self._ensure_encoder_loaded()
            
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_path)
            
            # Collect only agent audio segments
            agent_samples = []
            
            for segment in segments:
                speaker = segment.get("speaker", "")
                # Include agent-labelled segments and SPEAKER_00 (typically agent)
                if speaker in ["agent", "SPEAKER_00"] or speaker.startswith("agent_"):
                    start_sample = int(segment.get("start", 0) * sample_rate)
                    end_sample = int(segment.get("end", 0) * sample_rate)
                    
                    if end_sample > start_sample and end_sample <= len(audio_data):
                        agent_samples.append(audio_data[start_sample:end_sample])
            
            if not agent_samples:
                logger.warning("No agent segments found in transcript")
                return None
            
            # Concatenate all agent audio
            combined_audio = np.concatenate(agent_samples)
            
            # Need at least 1 second of audio for reliable embedding
            min_samples = sample_rate * 1
            if len(combined_audio) < min_samples:
                logger.warning(f"Not enough agent audio for fingerprint ({len(combined_audio)/sample_rate:.1f}s)")
                return None
            
            # Extract embedding from agent-only audio
            embedding = self._encoder.embed_utterance(combined_audio)
            
            logger.debug(f"Extracted agent embedding from {len(combined_audio)/sample_rate:.1f}s of audio")
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Failed to extract agent-only embedding: {e}")
            return None

    def update_fingerprint(
        self,
        halo_id: int,
        agent_name: str,
        audio_path: str,
        segments: Optional[List[Dict[str, Any]]] = None
    ) -> bool:
        """
        Update or create fingerprint for an agent.
        
        IMPORTANT: Only extracts voice from agent segments, never supporter.
        
        Args:
            halo_id: Agent's Halo ID
            agent_name: Agent's name
            audio_path: Path to audio sample
            segments: Transcript segments with speaker labels (to isolate agent voice)
            
        Returns:
            True if successful
        """
        try:
            # If we have segments, only extract agent voice
            if segments:
                embedding = self._extract_agent_only_embedding(audio_path, segments)
            else:
                # Fallback to full audio (not recommended - may include supporter)
                logger.warning("No segments provided - fingerprint may include supporter voice")
                embedding = self.extract_embedding(audio_path)
            
            if embedding is None:
                logger.warning("Could not extract agent embedding - skipping fingerprint update")
                return False
            
            # Check for existing fingerprint
            existing = VoiceFingerprint.get_by_halo_id(halo_id)
            
            if existing and existing.sample_count > 0:
                # Average with existing embedding
                old_embedding = np.frombuffer(existing.fingerprint_data, dtype=np.float32)
                new_count = existing.sample_count + 1
                
                # Weighted average
                combined = (old_embedding * existing.sample_count + embedding) / new_count
                combined = combined.astype(np.float32)
                
                fp = VoiceFingerprint(
                    halo_id=halo_id,
                    agent_name=agent_name,
                    fingerprint_data=combined.tobytes(),
                    sample_count=new_count,
                )
            else:
                fp = VoiceFingerprint(
                    halo_id=halo_id,
                    agent_name=agent_name,
                    fingerprint_data=embedding.tobytes(),
                    sample_count=1,
                )
            
            fp.save()
            
            # Update cache
            self._fingerprint_cache[halo_id] = np.frombuffer(
                fp.fingerprint_data, dtype=np.float32
            )
            
            logger.info(f"Updated fingerprint for agent {agent_name} (halo_id={halo_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update fingerprint: {e}")
            return False
    
    def detect_call_transfer(
        self,
        segments: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Detect if a call was transferred between agents.
        
        Returns transfer info if detected, None otherwise.
        """
        # Count unique agents
        agents = set()
        transfer_points = []
        
        prev_speaker = None
        for seg in segments:
            speaker_id = seg.get("speaker_id", "")
            
            if speaker_id.startswith("agent_"):
                agent_id = speaker_id
                agents.add(agent_id)
                
                if prev_speaker and prev_speaker != agent_id and prev_speaker.startswith("agent_"):
                    transfer_points.append({
                        "timestamp": seg.get("start", 0),
                        "from_agent": prev_speaker,
                        "to_agent": agent_id,
                    })
                
                prev_speaker = agent_id
        
        if len(agents) > 1:
            return {
                "transfer_detected": True,
                "agent_count": len(agents),
                "agents": list(agents),
                "transfer_points": transfer_points,
            }
        
        return None
    
    def is_available(self) -> bool:
        """Check if voice fingerprinting is available."""
        if self.use_mock:
            return True
        return RESEMBLYZER_AVAILABLE
