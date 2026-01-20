"""Services module for Angel Intelligence."""

from .audio_downloader import AudioDownloader
from .transcriber import TranscriptionService
from .pii_detector import PIIDetector
# Lazy import analyzer to avoid loading qwen_omni_utils on Python 3.9
# from .analyzer import AnalysisService
from .voice_fingerprint import VoiceFingerprintService
from .interactive import InteractiveService, get_interactive_service
from .interactive_proxy import InteractiveServiceProxy, get_interactive_proxy
from .sql_agent import (
    execute_safe_query,
    validate_sql,
    handle_function_call,
    get_sql_agent_system_prompt,
    DATABASE_SCHEMA,
    SQL_AGENT_FUNCTIONS,
)

def get_analysis_service():
    """Lazy import for AnalysisService."""
    from .analyzer import AnalysisService
    return AnalysisService

__all__ = [
    "AudioDownloader",
    "TranscriptionService", 
    "PIIDetector",
    "get_analysis_service",
    "VoiceFingerprintService",
    "InteractiveService",
    "get_interactive_service",
    "InteractiveServiceProxy",
    "get_interactive_proxy",
    "execute_safe_query",
    "validate_sql",
    "handle_function_call",
    "get_sql_agent_system_prompt",
    "DATABASE_SCHEMA",
    "SQL_AGENT_FUNCTIONS",
]
