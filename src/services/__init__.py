"""
Services module for Angel Intelligence.

Uses lazy imports to avoid loading heavy dependencies (pyannote, transformers, etc.)
until they're actually needed. This allows lightweight pods (transcription, worker)
to start without pulling in unnecessary ML libraries.
"""

from typing import TYPE_CHECKING

# Lightweight imports - these have minimal dependencies
from .audio_downloader import AudioDownloader
from .pii_detector import PIIDetector

# Type hints only - not imported at runtime
if TYPE_CHECKING:
    from .transcriber import TranscriptionService
    from .transcriber_proxy import TranscriptionProxy
    from .voice_fingerprint import VoiceFingerprintService
    from .interactive import InteractiveService
    from .interactive_proxy import InteractiveServiceProxy
    from .enquiry_context import EnquiryContextService
    from .order_context import OrderContextService
    from .live_session import LiveSession, LiveSessionManager, CanvasUpdate, TranscriptSegment
    from .analyzer import AnalysisService


def get_analysis_service():
    """Lazy import for AnalysisService."""
    from .analyzer import AnalysisService
    return AnalysisService


def get_interactive_service():
    """Lazy import for InteractiveService singleton."""
    from .interactive import get_interactive_service as _get
    return _get()


def get_interactive_proxy():
    """Lazy import for InteractiveServiceProxy singleton."""
    from .interactive_proxy import get_interactive_proxy as _get
    return _get()


def get_enquiry_context_service():
    """Lazy import for EnquiryContextService singleton."""
    from .enquiry_context import get_enquiry_context_service as _get
    return _get()


def get_order_context_service():
    """Lazy import for OrderContextService singleton."""
    from .order_context import get_order_context_service as _get
    return _get()


# Lazy attribute access for classes
def __getattr__(name: str):
    """
    Lazy import heavy dependencies only when accessed.
    
    This prevents pyannote.audio, transformers, etc. from being loaded
    just because something did `from src.services import AudioDownloader`.
    """
    # Transcription services
    if name == "TranscriptionService":
        from .transcriber import TranscriptionService
        return TranscriptionService
    if name == "TranscriptionProxy":
        from .transcriber_proxy import TranscriptionProxy
        return TranscriptionProxy
    
    # Voice fingerprinting (requires pyannote.audio)
    if name == "VoiceFingerprintService":
        from .voice_fingerprint import VoiceFingerprintService
        return VoiceFingerprintService
    
    # Interactive services (requires transformers)
    if name == "InteractiveService":
        from .interactive import InteractiveService
        return InteractiveService
    if name == "InteractiveServiceProxy":
        from .interactive_proxy import InteractiveServiceProxy
        return InteractiveServiceProxy
    
    # Context services
    if name == "EnquiryContextService":
        from .enquiry_context import EnquiryContextService
        return EnquiryContextService
    if name == "OrderContextService":
        from .order_context import OrderContextService
        return OrderContextService
    
    # Live session (requires various deps)
    if name in ("LiveSession", "LiveSessionManager", "CanvasUpdate", "TranscriptSegment"):
        from . import live_session
        return getattr(live_session, name)
    
    # SQL agent functions
    if name in ("execute_safe_query", "validate_sql", "handle_function_call", 
                "get_sql_agent_system_prompt", "DATABASE_SCHEMA", "SQL_AGENT_FUNCTIONS"):
        from . import sql_agent
        return getattr(sql_agent, name)
    
    raise AttributeError(f"module 'src.services' has no attribute '{name}'")


__all__ = [
    # Eagerly loaded (lightweight)
    "AudioDownloader",
    "PIIDetector",
    # Lazy loaded via __getattr__
    "TranscriptionService",
    "TranscriptionProxy",
    "VoiceFingerprintService",
    "InteractiveService",
    "InteractiveServiceProxy",
    "EnquiryContextService",
    "OrderContextService",
    "LiveSession",
    "LiveSessionManager",
    "CanvasUpdate",
    "TranscriptSegment",
    "execute_safe_query",
    "validate_sql",
    "handle_function_call",
    "get_sql_agent_system_prompt",
    "DATABASE_SCHEMA",
    "SQL_AGENT_FUNCTIONS",
    # Getter functions
    "get_analysis_service",
    "get_interactive_service",
    "get_interactive_proxy",
    "get_enquiry_context_service",
    "get_order_context_service",
]
