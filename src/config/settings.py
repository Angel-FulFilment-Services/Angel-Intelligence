"""
Angel Intelligence - Configuration Settings

Environment-based configuration supporting both production (Kubernetes/Jetson cluster)
and local development modes.
"""

import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # ==========================================================================
    # ENVIRONMENT MODE
    # ==========================================================================
    angel_env: str = Field(default="development", description="Environment: 'production' or 'development'")
    worker_id: str = Field(default="local-dev", description="Unique worker identifier")
    log_level: str = Field(default="INFO", description="Logging level: DEBUG, INFO, WARNING, ERROR")
    
    # ==========================================================================
    # API AUTHENTICATION
    # ==========================================================================
    api_auth_token: str = Field(default="", description="Bearer token for API authentication (min 64 chars)")
    
    # ==========================================================================
    # DATABASE CONFIGURATION (MySQL - 'ai' connection)
    # ==========================================================================
    ai_db_host: str = Field(default="localhost", description="MySQL host")
    ai_db_port: int = Field(default=3306, description="MySQL port")
    ai_db_database: str = Field(default="ai", description="Database name")
    ai_db_username: str = Field(default="root", description="Database username")
    ai_db_password: str = Field(default="", description="Database password")
    
    @field_validator('ai_db_port', mode='before')
    @classmethod
    def parse_port(cls, v):
        """Convert string port to integer."""
        if isinstance(v, str):
            return int(v)
        return v
    
    # ==========================================================================
    # R2 STORAGE CONFIGURATION
    # ==========================================================================
    r2_endpoint: str = Field(default="", description="Cloudflare R2 endpoint URL")
    r2_access_key_id: str = Field(default="", description="R2 access key ID")
    r2_secret_access_key: str = Field(default="", description="R2 secret access key")
    r2_bucket: str = Field(default="call-recordings", description="R2 bucket name")
    
    # ==========================================================================
    # LOCAL DEVELOPMENT STORAGE
    # ==========================================================================
    local_storage_path: str = Field(default="", description="Local path for audio files in development mode")
    
    # ==========================================================================
    # PBX RECORDING SOURCES
    # ==========================================================================
    pbx_live_url: str = Field(default="https://pbx.angelfs.co.uk/callrec/", description="Live PBX recording URL")
    pbx_archive_url: str = Field(default="https://afs-pbx-callarchive.angelfs.co.uk/", description="Archive PBX URL")
    
    # ==========================================================================
    # MODEL CONFIGURATION
    # ==========================================================================
    models_base_path: str = Field(default="./models", description="Base path for model storage")
    
    # Whisper model for transcription
    whisper_model: str = Field(default="medium", description="Whisper model size: tiny, base, small, medium, large")
    whisper_model_path: str = Field(default="", description="Custom Whisper model path")
    
    # Analysis model (fine-tunable for call analysis)
    analysis_model: str = Field(default="Qwen/Qwen2.5-Omni-7B", description="HuggingFace model ID or local path")
    analysis_model_path: str = Field(default="", description="Local path for fine-tuned analysis model")
    analysis_model_quantization: str = Field(default="", description="Quantization: int4, int8, or empty for none")
    max_transcript_length: int = Field(default=0, description="Max transcript chars for analysis (0=unlimited, use 8000 for 3B models)")
    
    # Chat model (base model, not fine-tuned)
    chat_model: str = Field(default="Qwen/Qwen2.5-Omni-7B", description="HuggingFace model ID for chat")
    chat_model_path: str = Field(default="", description="Local path for chat model")
    chat_model_quantization: str = Field(default="", description="Quantization for chat model")
    
    # vLLM/External API support (optional - for shared model serving)
    # When set, uses HTTP API instead of loading models locally
    # Format: http://vllm-server:8000/v1 (OpenAI-compatible endpoint)
    llm_api_url: str = Field(default="", description="External LLM API URL (vLLM/TGI). When set, uses API for text analysis and chat")
    llm_api_key: str = Field(default="", description="API key for external LLM service (if required)")
    
    # Audio analysis API URL (vLLM serving Qwen2.5-Omni for audio mode)
    # When set and analysis_mode='audio', uses this API instead of local Qwen2.5-Omni
    # Format: http://audio-vllm:8000/v1 (OpenAI-compatible multimodal endpoint)
    audio_analysis_api_url: str = Field(default="", description="vLLM API URL for audio analysis (Qwen2.5-Omni)")
    
    # LoRA adapter configuration (for fine-tuned models via vLLM)
    analysis_adapter_name: str = Field(default="call-analysis", description="LoRA adapter name for analysis (folder name under /models/adapters/)")
    
    # Analysis mode
    analysis_mode: str = Field(default="audio", description="Analysis mode: 'audio' or 'transcript'")
    
    # ==========================================================================
    # PROCESSING CONFIGURATION
    # ==========================================================================
    poll_interval: int = Field(default=30, description="Seconds between polling for new recordings")
    max_concurrent_jobs: int = Field(default=1, description="Maximum concurrent processing jobs")
    max_retry_attempts: int = Field(default=3, description="Maximum retry attempts for failed jobs")
    retry_delay_hours: int = Field(default=1, description="Hours to wait before retrying failed jobs")
    
    # Worker mode: 'batch' for call processing, 'interactive' for chat/summaries, 'api' for API only
    worker_mode: str = Field(default="batch", description="Worker mode: 'batch', 'interactive', 'api', or 'both'")
    
    # Interactive service URL for proxying (used when worker_mode=api)
    # In K8s, this is the internal service URL: http://angel-intelligence-interactive:8000
    interactive_service_url: str = Field(default="", description="URL of interactive service for AI proxy")
    
    # Preload models on startup (eliminates first-request delay)
    preload_chat_model: bool = Field(default=True, description="Preload chat model on API startup")
    
    # Transcription settings
    transcript_segmentation: str = Field(default="word", description="Segmentation: 'word' or 'sentence'")
    
    # Transcription service URL for shared WhisperX (optional)
    # When set, workers call this service instead of loading WhisperX locally
    # This allows multiple workers to share a single GPU-loaded model
    transcription_service_url: str = Field(default="", description="URL of transcription service (e.g., http://localhost:8000)")
    
    # HuggingFace token for pyannote speaker diarization
    # Get from https://huggingface.co/settings/tokens (read access)
    # Accept agreements at: https://huggingface.co/pyannote/speaker-diarization-3.1
    huggingface_token: str = Field(default="", description="HuggingFace token for pyannote diarization")
    
    # PII redaction
    enable_pii_redaction: bool = Field(default=True, description="Enable PII detection and redaction")
    
    # ==========================================================================
    # MODEL HOT RELOAD (Production only)
    # ==========================================================================
    enable_model_hot_reload: bool = Field(default=False, description="Enable hot-swapping models in production")
    
    # ==========================================================================
    # MOCK MODE (Development only)
    # ==========================================================================
    use_mock_models: bool = Field(default=False, description="Use mock responses instead of real LLM inference")
    
    # ==========================================================================
    # GPU CONFIGURATION
    # ==========================================================================
    cuda_visible_devices: str = Field(default="0", description="CUDA device IDs, or -1 for CPU")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.angel_env.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.angel_env.lower() == "development"
    
    @property
    def use_gpu(self) -> bool:
        """Check if GPU should be used."""
        return self.cuda_visible_devices != "-1"
    
    def get_analysis_model_path(self) -> str:
        """Get the path to the analysis model."""
        if self.analysis_model_path:
            return self.analysis_model_path
        return self.analysis_model
    
    def get_chat_model_path(self) -> str:
        """Get the path to the chat model."""
        if self.chat_model_path:
            return self.chat_model_path
        return self.chat_model


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
