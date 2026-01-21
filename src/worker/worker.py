"""
Angel Intelligence - Worker

Autonomous background worker that polls the database for pending recordings
and processes them through the AI pipeline.

Runs independently - no Laravel queue dependency.
"""

# Set GPU selection BEFORE importing torch/transformers
# This MUST happen before any CUDA initialization
import os
if cuda_devices := os.environ.get("CUDA_VISIBLE_DEVICES"):
    pass  # Already set in environment
else:
    # On dual-GPU systems (Intel iGPU + NVIDIA dGPU), the iGPU may be device 0
    # Default to device 1 if not specified and we detect an NVIDIA GPU
    # User can override by setting CUDA_VISIBLE_DEVICES in .env
    pass  # Let PyTorch auto-detect

import logging
import signal
import sys
import time
from typing import Optional

from src.config import get_settings
from src.database import CallRecording
from src.worker.processor import CallProcessor

logger = logging.getLogger(__name__)


class Worker:
    """
    Background worker for processing call recordings.
    
    Features:
    - Polls database for pending recordings
    - Processes recordings through the AI pipeline
    - Handles retry logic for failed jobs
    - Graceful shutdown on SIGTERM/SIGINT
    """
    
    def __init__(self):
        """Initialise the worker."""
        settings = get_settings()
        
        self.poll_interval = settings.poll_interval
        self.max_concurrent_jobs = settings.max_concurrent_jobs
        self.worker_id = settings.worker_id
        
        self.processor = CallProcessor()
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)
        
        logger.info(f"Worker '{self.worker_id}' initialised (poll interval: {self.poll_interval}s)")
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def run(self):
        """
        Main worker loop.
        
        Continuously polls for pending recordings and processes them.
        """
        self.running = True
        logger.info(f"Worker '{self.worker_id}' starting...")
        
        while self.running:
            try:
                # Get pending recordings
                recordings = CallRecording.get_pending_recordings(limit=self.max_concurrent_jobs)
                
                if recordings:
                    logger.info(f"Found {len(recordings)} recording(s) to process")
                    
                    for recording in recordings:
                        if not self.running:
                            break
                        
                        self._process_recording(recording)
                else:
                    logger.debug("No pending recordings found")
                
                # Wait before next poll
                if self.running:
                    time.sleep(self.poll_interval)
                    
            except KeyboardInterrupt:
                logger.info("Worker interrupted by user")
                break
                
            except Exception as e:
                logger.error(f"Worker error: {e}", exc_info=True)
                
                # Wait before retrying
                if self.running:
                    time.sleep(self.poll_interval)
        
        logger.info(f"Worker '{self.worker_id}' stopped")
    
    def _process_recording(self, recording: CallRecording) -> bool:
        """
        Process a single recording.
        
        Args:
            recording: CallRecording to process
            
        Returns:
            True if processing succeeded
        """
        try:
            logger.info(f"[{self.worker_id}] Processing recording {recording.id}: {recording.apex_id}")
            
            start_time = time.time()
            success = self.processor.process(recording)
            elapsed = time.time() - start_time
            
            if success:
                logger.info(f"[{self.worker_id}] Completed recording {recording.id} in {elapsed:.1f}s")
            else:
                logger.warning(f"[{self.worker_id}] Failed recording {recording.id} after {elapsed:.1f}s")
            
            return success
            
        except Exception as e:
            logger.error(f"[{self.worker_id}] Exception processing recording {recording.id}: {e}", exc_info=True)
            recording.mark_failed(str(e))
            return False
    
    def process_single(self, recording_id: int) -> bool:
        """
        Process a single recording by ID.
        
        Useful for testing or manual processing.
        
        Args:
            recording_id: Database ID of recording to process
            
        Returns:
            True if processing succeeded
        """
        recording = CallRecording.get_by_id(recording_id)
        
        if not recording:
            logger.error(f"Recording {recording_id} not found")
            return False
        
        return self._process_recording(recording)


def main():
    """Main entry point for the worker."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    settings = get_settings()
    
    logger.info("=" * 60)
    logger.info("Angel Intelligence Worker")
    logger.info("=" * 60)
    logger.info(f"Environment: {settings.angel_env}")
    logger.info(f"Worker ID: {settings.worker_id}")
    logger.info(f"Analysis mode: {settings.analysis_mode}")
    logger.info(f"Mock mode: {settings.use_mock_models}")
    logger.info("=" * 60)
    
    # Create and run worker
    worker = Worker()
    worker.run()


if __name__ == "__main__":
    main()
