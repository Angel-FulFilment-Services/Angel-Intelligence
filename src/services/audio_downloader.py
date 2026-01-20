"""
Angel Intelligence - Audio Downloader Service

Downloads audio files from PBX sources or R2 storage.
Converts GSM files to WAV format using SoX.
Follows the same URL patterns as the Laravel CallRecordings.php helper.
"""

import logging
import os
import subprocess
import tempfile
from datetime import datetime
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
import boto3
from botocore.config import Config

from src.config import get_settings

logger = logging.getLogger(__name__)


class AudioDownloader:
    """
    Downloads and converts audio files from various sources.
    
    Supports:
    - Live PBX recordings
    - Archive PBX recordings
    - R2 cloud storage
    - Local file storage (development mode)
    
    Converts GSM format to WAV using SoX for Whisper compatibility.
    """
    
    def __init__(self):
        """Initialise the audio downloader."""
        settings = get_settings()
        
        # PBX URLs
        self.pbx_live_url = settings.pbx_live_url
        self.pbx_archive_url = settings.pbx_archive_url
        
        # R2 client
        self.r2_client = None
        if settings.r2_endpoint and settings.r2_access_key_id:
            self.r2_client = boto3.client(
                's3',
                endpoint_url=settings.r2_endpoint,
                aws_access_key_id=settings.r2_access_key_id,
                aws_secret_access_key=settings.r2_secret_access_key,
                config=Config(signature_version='s3v4')
            )
        
        self.r2_bucket = settings.r2_bucket
        self.local_storage_path = settings.local_storage_path
    
    def download_recording(
        self, 
        apex_id: str, 
        call_date: datetime,
        r2_path: Optional[str] = None,
        r2_bucket: Optional[str] = None
    ) -> Tuple[str, bool]:
        """
        Download a recording and convert to WAV format.
        
        Tries sources in this order (matching Laravel CallRecordings.php):
        1. Local storage path (development mode)
        2. R2 storage (if r2_path is provided)
        3. Live PBX: https://pbx.angelfs.co.uk/callrec/{apex_id}.gsm
        4. Archive: https://afs-pbx-callarchive.angelfs.co.uk/monitor-{year}/{month}/{apex_id}.gsm
        5. Archive alt: https://afs-pbx-callarchive.angelfs.co.uk/monitor-{year}/{month}/monitor/{apex_id}.gsm
        
        Args:
            apex_id: Unique call identifier
            call_date: Date of the call (for archive URL construction)
            r2_path: Optional R2 storage path
            r2_bucket: Optional R2 bucket name
            
        Returns:
            Tuple of (wav_file_path, is_local_file)
            is_local_file indicates if the file should NOT be deleted after processing
        """
        gsm_path = None
        is_local = False
        
        try:
            # 1. Try local storage first (development mode)
            if self.local_storage_path:
                local_gsm = os.path.join(self.local_storage_path, f"{apex_id}.gsm")
                local_wav = os.path.join(self.local_storage_path, f"{apex_id}.wav")
                
                if os.path.exists(local_wav):
                    logger.info(f"Using local WAV file: {local_wav}")
                    return local_wav, True
                
                if os.path.exists(local_gsm):
                    logger.info(f"Using local GSM file: {local_gsm}")
                    gsm_path = local_gsm
                    is_local = True
            
            # 2. Try R2 storage
            if not gsm_path and r2_path and self.r2_client:
                try:
                    gsm_path = self._download_from_r2(r2_path, r2_bucket or self.r2_bucket)
                    logger.info(f"Downloaded from R2: {r2_path}")
                except Exception as e:
                    logger.warning(f"R2 download failed: {e}")
            
            # Log the call_date being used for archive lookup
            logger.info(f"Looking for recording {apex_id}, call_date={call_date} (year={call_date.year}, month={call_date.month})")
            
            # 3. Try live PBX
            if not gsm_path:
                url = f"{self.pbx_live_url}{apex_id}.gsm"
                logger.info(f"Trying live PBX: {url}")
                gsm_path = self._download_from_url(url)
                if gsm_path:
                    logger.info(f"Downloaded from live PBX: {url}")
                else:
                    logger.info(f"Not found at live PBX")
            
            # 4. Try archive (primary path)
            if not gsm_path:
                year = call_date.year
                month = str(call_date.month).zfill(2)
                url = f"{self.pbx_archive_url}monitor-{year}/{month}/{apex_id}.gsm"
                logger.info(f"Trying archive: {url}")
                gsm_path = self._download_from_url(url)
                if gsm_path:
                    logger.info(f"Downloaded from archive: {url}")
                else:
                    logger.info(f"Not found at archive")
            
            # 5. Try archive (alternative path with /monitor/ subdirectory)
            if not gsm_path:
                year = call_date.year
                month = str(call_date.month).zfill(2)
                url = f"{self.pbx_archive_url}monitor-{year}/{month}/monitor/{apex_id}.gsm"
                logger.info(f"Trying archive alt: {url}")
                gsm_path = self._download_from_url(url)
                if gsm_path:
                    logger.info(f"Downloaded from archive (alt): {url}")
                else:
                    logger.info(f"Not found at archive alt")
            
            if not gsm_path:
                logger.error(f"Recording not found at any location for apex_id: {apex_id}")
                raise FileNotFoundError(f"Could not find recording for apex_id: {apex_id}")
            
            # Convert GSM to WAV
            wav_path = self._convert_gsm_to_wav(gsm_path)
            
            # Clean up GSM temp file if it's not a local file
            if not is_local and gsm_path != wav_path:
                try:
                    os.unlink(gsm_path)
                except Exception:
                    pass
            
            return wav_path, is_local
            
        except Exception as e:
            logger.error(f"Failed to download recording {apex_id}: {e}")
            raise
    
    def _download_from_url(self, url: str) -> Optional[str]:
        """
        Download file from HTTP URL.
        
        Args:
            url: URL to download from
            
        Returns:
            Path to downloaded temp file, or None if download failed
        """
        try:
            response = requests.get(url, timeout=30, stream=True)
            
            if response.status_code == 200:
                # Create temp file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.gsm')
                
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                
                temp_file.close()
                logger.debug(f"Downloaded {url} to {temp_file.name}")
                return temp_file.name
            else:
                logger.debug(f"URL returned {response.status_code}: {url}")
                return None
                
        except requests.RequestException as e:
            logger.debug(f"Failed to download {url}: {e}")
            return None
    
    def _download_from_r2(self, r2_path: str, bucket: str) -> str:
        """
        Download file from R2 storage.
        
        Args:
            r2_path: Path within the bucket
            bucket: R2 bucket name
            
        Returns:
            Path to downloaded temp file
        """
        # Handle URL-style paths
        if r2_path.startswith('http://') or r2_path.startswith('https://'):
            parsed = urlparse(r2_path)
            r2_path = parsed.path.lstrip('/')
        
        # Determine file extension
        ext = os.path.splitext(r2_path)[1] or '.audio'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_file.close()
        
        self.r2_client.download_file(bucket, r2_path, temp_file.name)
        return temp_file.name
    
    def _convert_gsm_to_wav(self, gsm_path: str) -> str:
        """
        Convert GSM audio to WAV format using SoX.
        
        Command: sox input.gsm -r 8000 -b 32 -c 1 output.wav
        
        Args:
            gsm_path: Path to GSM file
            
        Returns:
            Path to WAV file
        """
        # Check if already WAV
        if gsm_path.endswith('.wav'):
            return gsm_path
        
        # Check if it's a local file - convert in place
        if self.local_storage_path and gsm_path.startswith(self.local_storage_path):
            wav_path = gsm_path.replace('.gsm', '.wav')
        else:
            # Create temp file for converted audio
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_file.close()
            wav_path = temp_file.name
        
        try:
            # Use SoX for conversion
            # GSM is 8kHz mono, we convert to 32-bit WAV
            cmd = [
                'sox',
                gsm_path,
                '-r', '8000',  # Sample rate
                '-b', '32',    # Bit depth
                '-c', '1',     # Mono
                wav_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            
            logger.info(f"Converted GSM to WAV: {gsm_path} -> {wav_path}")
            return wav_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"SoX conversion failed: {e.stderr}")
            
            # Try ffmpeg as fallback
            try:
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-i', gsm_path,
                    '-ar', '16000',
                    '-ac', '1',
                    '-f', 'wav',
                    wav_path
                ]
                
                subprocess.run(cmd, capture_output=True, check=True)
                logger.info(f"Converted GSM to WAV using ffmpeg: {gsm_path} -> {wav_path}")
                return wav_path
                
            except subprocess.CalledProcessError as e2:
                logger.error(f"ffmpeg conversion also failed: {e2}")
                raise RuntimeError(f"Failed to convert GSM to WAV: {e.stderr}")
        
        except FileNotFoundError:
            logger.error("SoX not found. Please install SoX: apt-get install sox libsox-fmt-all")
            raise
    
    def upload_to_r2(self, local_path: str, r2_path: str, bucket: Optional[str] = None) -> str:
        """
        Upload a file to R2 storage.
        
        Args:
            local_path: Path to local file
            r2_path: Destination path in R2
            bucket: Optional bucket name (uses default if not specified)
            
        Returns:
            R2 path of uploaded file
        """
        if not self.r2_client:
            raise RuntimeError("R2 client not configured")
        
        bucket = bucket or self.r2_bucket
        self.r2_client.upload_file(local_path, bucket, r2_path)
        logger.info(f"Uploaded {local_path} to R2: {bucket}/{r2_path}")
        return r2_path
    
    def cleanup_temp_file(self, path: str, is_local: bool = False) -> None:
        """
        Clean up a temporary file.
        
        Args:
            path: File path to clean up
            is_local: If True, file is in local storage and should NOT be deleted
        """
        if is_local:
            logger.debug(f"Keeping local file: {path}")
            return
        
        if path and os.path.exists(path):
            try:
                os.unlink(path)
                logger.debug(f"Cleaned up temp file: {path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {path}: {e}")
