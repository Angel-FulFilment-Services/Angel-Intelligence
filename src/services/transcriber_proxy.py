"""
Transcription Proxy - HTTP client for shared WhisperX transcription service.

This allows multiple workers to share a single GPU-loaded WhisperX model
instead of each worker loading its own model (which wastes VRAM).

Usage:
    proxy = TranscriptionProxy("http://localhost:8000")
    result = await proxy.transcribe("/path/to/audio.wav", diarize=True)
"""

import httpx
import logging
import base64
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    """Result from transcription service."""
    text: str
    segments: list
    language: str
    duration: float
    word_count: int
    processing_time: float


class TranscriptionProxy:
    """HTTP client for shared transcription service."""
    
    def __init__(self, base_url: str, timeout: float = 300.0):
        """
        Initialize proxy client.
        
        Args:
            base_url: URL of transcription service (e.g., http://localhost:8000)
            timeout: Request timeout in seconds (default 5 minutes for long audio)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=30.0)
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def health_check(self) -> bool:
        """Check if transcription service is available."""
        try:
            client = await self._get_client()
            response = await client.get("/internal/health")
            return response.status_code == 200
        except Exception as e:
            logger.debug(f"Transcription service health check failed: {e}")
            return False
    
    async def transcribe(
        self,
        audio_path: str,
        diarize: bool = True,
        language: Optional[str] = None,
        num_speakers: Optional[int] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio via shared service.
        
        Args:
            audio_path: Path to audio file
            diarize: Whether to perform speaker diarization
            language: Language code (auto-detect if None)
            num_speakers: Expected number of speakers (optional)
            
        Returns:
            TranscriptionResult with text, segments, and metadata
        """
        client = await self._get_client()
        
        # Read audio file and encode as base64
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        audio_data = audio_file.read_bytes()
        audio_base64 = base64.b64encode(audio_data).decode("utf-8")
        
        # Build request
        request_data = {
            "audio_base64": audio_base64,
            "filename": audio_file.name,
            "diarize": diarize
        }
        
        if language:
            request_data["language"] = language
        if num_speakers:
            request_data["num_speakers"] = num_speakers
        
        logger.info(f"Sending transcription request to {self.base_url} for {audio_file.name}")
        
        response = await client.post(
            "/internal/transcribe",
            json=request_data
        )
        
        if response.status_code != 200:
            error_detail = response.text
            logger.error(f"Transcription service error: {response.status_code} - {error_detail}")
            raise RuntimeError(f"Transcription failed: {response.status_code} - {error_detail}")
        
        data = response.json()
        
        return TranscriptionResult(
            text=data.get("text", ""),
            segments=data.get("segments", []),
            language=data.get("language", "unknown"),
            duration=data.get("duration", 0.0),
            word_count=data.get("word_count", 0),
            processing_time=data.get("processing_time", 0.0)
        )
    
    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        diarize: bool = True,
        language: Optional[str] = None,
        num_speakers: Optional[int] = None
    ) -> TranscriptionResult:
        """
        Transcribe audio bytes via shared service.
        
        Args:
            audio_bytes: Raw audio data
            filename: Original filename (for format detection)
            diarize: Whether to perform speaker diarization
            language: Language code (auto-detect if None)
            num_speakers: Expected number of speakers (optional)
            
        Returns:
            TranscriptionResult with text, segments, and metadata
        """
        client = await self._get_client()
        
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        request_data = {
            "audio_base64": audio_base64,
            "filename": filename,
            "diarize": diarize
        }
        
        if language:
            request_data["language"] = language
        if num_speakers:
            request_data["num_speakers"] = num_speakers
        
        response = await client.post(
            "/internal/transcribe",
            json=request_data
        )
        
        if response.status_code != 200:
            error_detail = response.text
            raise RuntimeError(f"Transcription failed: {response.status_code} - {error_detail}")
        
        data = response.json()
        
        return TranscriptionResult(
            text=data.get("text", ""),
            segments=data.get("segments", []),
            language=data.get("language", "unknown"),
            duration=data.get("duration", 0.0),
            word_count=data.get("word_count", 0),
            processing_time=data.get("processing_time", 0.0)
        )
