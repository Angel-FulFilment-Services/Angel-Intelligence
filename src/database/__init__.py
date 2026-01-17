"""Database module for Angel Intelligence."""

from .connection import DatabaseConnection, get_db_connection
from .models import (
    CallRecording,
    CallTranscription,
    CallAnalysis,
    CallAnnotation,
    MonthlySummary,
    ChatConversation,
    ChatMessage,
    VoiceFingerprint,
    ClientConfig,
    ProcessingStatus,
)

__all__ = [
    "DatabaseConnection",
    "get_db_connection",
    "CallRecording",
    "CallTranscription",
    "CallAnalysis",
    "CallAnnotation",
    "MonthlySummary",
    "ChatConversation",
    "ChatMessage",
    "VoiceFingerprint",
    "ClientConfig",
    "ProcessingStatus",
]
