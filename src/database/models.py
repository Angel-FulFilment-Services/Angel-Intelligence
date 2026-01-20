"""
Angel Intelligence - Database Models

Data models matching the MySQL database schema for the 'ai' database.
All tables are prefixed with 'ai_'.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any

from src.database.connection import get_db_connection

logger = logging.getLogger(__name__)


class ProcessingStatus(Enum):
    """Processing status for call recordings."""
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class CallRecording:
    """
    Model for ai_call_recordings table.
    
    Represents a call recording that needs to be processed.
    """
    id: int
    apex_id: str
    call_date: datetime
    orderref: Optional[str] = None            # Order reference number
    enqref: Optional[str] = None              # Enquiry reference number
    obref: Optional[str] = None               # Outbound reference number
    client_ref: Optional[str] = None          # Client reference code
    campaign: Optional[str] = None            # Campaign name
    halo_id: Optional[int] = None             # Agent ID from Halo system
    agent_name: Optional[str] = None          # Agent display name
    creative: Optional[str] = None            # Creative name
    direction: str = "inbound"                # 'inbound' or 'outbound'
    invoicing: Optional[str] = None           # Invoicing category
    duration_seconds: int = 0
    file_size_bytes: Optional[int] = None
    file_format: str = "gsm"
    r2_path: Optional[str] = None
    r2_bucket: Optional[str] = None
    retain_audio: bool = False
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    processing_error: Optional[str] = None
    retry_count: int = 0
    next_retry_at: Optional[datetime] = None
    uploaded_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "CallRecording":
        """Create CallRecording from database row."""
        return cls(
            id=row["id"],
            apex_id=row["apex_id"],
            call_date=row.get("call_date"),
            orderref=row.get("orderref"),
            enqref=row.get("enqref"),
            obref=row.get("obref"),
            client_ref=row.get("client_ref"),
            campaign=row.get("campaign"),
            halo_id=row.get("halo_id"),
            agent_name=row.get("agent_name"),
            creative=row.get("creative"),
            direction=row.get("direction", "inbound"),
            invoicing=row.get("invoicing"),
            duration_seconds=row.get("duration_seconds", 0),
            file_size_bytes=row.get("file_size_bytes"),
            file_format=row.get("file_format", "gsm"),
            r2_path=row.get("r2_path"),
            r2_bucket=row.get("r2_bucket"),
            retain_audio=bool(row.get("retain_audio", False)),
            processing_status=ProcessingStatus(row.get("processing_status", "pending")),
            processing_started_at=row.get("processing_started_at"),
            processing_completed_at=row.get("processing_completed_at"),
            processing_error=row.get("processing_error"),
            retry_count=row.get("retry_count", 0),
            next_retry_at=row.get("next_retry_at"),
            uploaded_at=row.get("uploaded_at"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )
    
    @staticmethod
    def get_pending_recordings(limit: int = 10) -> List["CallRecording"]:
        """
        Get recordings that are pending or ready for retry.
        
        Returns recordings where:
        - Status is 'pending' or 'queued'
        - OR status is 'failed' AND retry_count < 3 AND next_retry_at <= NOW()
        """
        db = get_db_connection()
        rows = db.fetch_all("""
            SELECT * FROM ai_call_recordings
            WHERE processing_status IN ('pending', 'queued')
               OR (processing_status = 'failed' 
                   AND retry_count < 3 
                   AND (next_retry_at IS NULL OR next_retry_at <= NOW()))
            ORDER BY 
                CASE processing_status 
                    WHEN 'queued' THEN 0 
                    WHEN 'pending' THEN 1 
                    ELSE 2 
                END,
                created_at ASC
            LIMIT %s
        """, (limit,))
        return [CallRecording.from_row(row) for row in rows]
    
    @staticmethod
    def get_by_id(recording_id: int) -> Optional["CallRecording"]:
        """Get recording by ID."""
        db = get_db_connection()
        row = db.fetch_one(
            "SELECT * FROM ai_call_recordings WHERE id = %s",
            (recording_id,)
        )
        return CallRecording.from_row(row) if row else None
    
    def mark_processing(self) -> None:
        """Mark this recording as processing."""
        db = get_db_connection()
        db.execute("""
            UPDATE ai_call_recordings 
            SET processing_status = 'processing',
                processing_started_at = NOW(),
                updated_at = NOW()
            WHERE id = %s
        """, (self.id,))
        logger.info(f"Marked recording {self.id} ({self.apex_id}) as processing")
    
    def mark_completed(self, r2_path: Optional[str] = None) -> None:
        """Mark this recording as completed."""
        db = get_db_connection()
        if r2_path:
            db.execute("""
                UPDATE ai_call_recordings 
                SET processing_status = 'completed',
                    processing_completed_at = NOW(),
                    r2_path = %s,
                    updated_at = NOW()
                WHERE id = %s
            """, (r2_path, self.id))
        else:
            db.execute("""
                UPDATE ai_call_recordings 
                SET processing_status = 'completed',
                    processing_completed_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
            """, (self.id,))
        logger.info(f"Marked recording {self.id} ({self.apex_id}) as completed")
    
    def mark_failed(self, error: str) -> None:
        """
        Mark this recording as failed with retry logic.
        
        Sets next_retry_at to 1 hour from now if retry_count < 3.
        """
        db = get_db_connection()
        new_retry_count = self.retry_count + 1
        
        # Only set next retry if we haven't exceeded max retries
        if new_retry_count < 3:
            db.execute("""
                UPDATE ai_call_recordings 
                SET processing_status = 'failed',
                    processing_error = %s,
                    processing_completed_at = NOW(),
                    retry_count = %s,
                    next_retry_at = DATE_ADD(NOW(), INTERVAL 1 HOUR),
                    updated_at = NOW()
                WHERE id = %s
            """, (error[:500], new_retry_count, self.id))
            logger.warning(f"Recording {self.id} failed (attempt {new_retry_count}/3), will retry in 1 hour: {error}")
        else:
            db.execute("""
                UPDATE ai_call_recordings 
                SET processing_status = 'failed',
                    processing_error = %s,
                    processing_completed_at = NOW(),
                    retry_count = %s,
                    next_retry_at = NULL,
                    updated_at = NOW()
                WHERE id = %s
            """, (error[:500], new_retry_count, self.id))
            logger.error(f"Recording {self.id} permanently failed after {new_retry_count} attempts: {error}")
    
    def get_call_year_month(self) -> tuple:
        """Get year and month from call_date for archive URL construction."""
        if self.call_date:
            return self.call_date.year, self.call_date.month
        return datetime.now().year, datetime.now().month


@dataclass
class CallTranscription:
    """
    Model for ai_call_transcriptions table.
    
    Stores transcription results including PII-redacted versions.
    """
    id: Optional[int] = None
    ai_call_recording_id: int = 0
    full_transcript: str = ""
    segments: List[Dict[str, Any]] = field(default_factory=list)
    redacted_transcript: Optional[str] = None
    pii_detected: Optional[List[Dict[str, Any]]] = None
    language_detected: str = "en"
    confidence_score: float = 0.95
    model_used: str = "whisperx-medium"
    processing_time_seconds: int = 0
    created_at: Optional[datetime] = None
    
    def save(self) -> int:
        """Save transcription to database and return the ID."""
        db = get_db_connection()
        
        # Prepare JSON fields
        segments_json = json.dumps(self.segments)
        pii_json = json.dumps(self.pii_detected) if self.pii_detected else None
        
        self.id = db.insert("""
            INSERT INTO ai_call_transcriptions 
            (ai_call_recording_id, full_transcript, segments, redacted_transcript,
             pii_detected, language_detected, confidence_score, model_used, 
             processing_time_seconds, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            self.ai_call_recording_id,
            self.full_transcript,
            segments_json,
            self.redacted_transcript,
            pii_json,
            self.language_detected,
            self.confidence_score,
            self.model_used,
            self.processing_time_seconds,
        ))
        
        logger.info(f"Saved transcription {self.id} for recording {self.ai_call_recording_id}")
        return self.id


@dataclass
class CallAnalysis:
    """
    Model for ai_call_analysis table.
    
    Stores AI analysis results including sentiment, topics, and quality scores.
    """
    id: Optional[int] = None
    ai_call_recording_id: int = 0
    summary: str = ""
    sentiment_score: float = 0.0
    sentiment_label: str = "neutral"
    key_topics: List[Dict[str, Any]] = field(default_factory=list)
    agent_actions_performed: List[Dict[str, Any]] = field(default_factory=list)
    performance_scores: Dict[str, int] = field(default_factory=dict)
    quality_score: float = 50.0
    action_items: List[Dict[str, Any]] = field(default_factory=list)
    compliance_flags: List[Dict[str, Any]] = field(default_factory=list)
    improvement_areas: List[Dict[str, Any]] = field(default_factory=list)  # Key areas for agent improvement/coaching
    speaker_metrics: Dict[str, Any] = field(default_factory=dict)
    audio_analysis: Optional[Dict[str, Any]] = None
    model_used: str = ""
    model_version: Optional[str] = None
    processing_time_seconds: int = 0
    created_at: Optional[datetime] = None
    
    def save(self) -> int:
        """Save analysis to database and return the ID."""
        db = get_db_connection()
        
        # Map sentiment score to label
        self.sentiment_label = self._score_to_label(self.sentiment_score)
        
        # Prepare JSON fields
        key_topics_json = json.dumps(self.key_topics)
        agent_actions_json = json.dumps(self.agent_actions_performed)
        performance_scores_json = json.dumps(self.performance_scores)
        action_items_json = json.dumps(self.action_items)
        compliance_flags_json = json.dumps(self.compliance_flags)
        improvement_areas_json = json.dumps(self.improvement_areas)
        speaker_metrics_json = json.dumps(self.speaker_metrics)
        audio_analysis_json = json.dumps(self.audio_analysis) if self.audio_analysis else None
        
        self.id = db.insert("""
            INSERT INTO ai_call_analysis
            (ai_call_recording_id, summary, sentiment_score, sentiment_label,
             key_topics, agent_actions_performed, performance_scores, quality_score,
             action_items, compliance_flags, improvement_areas, speaker_metrics, audio_analysis,
             model_used, model_version, processing_time_seconds, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            self.ai_call_recording_id,
            self.summary,
            self.sentiment_score,
            self.sentiment_label,
            key_topics_json,
            agent_actions_json,
            performance_scores_json,
            self.quality_score,
            action_items_json,
            compliance_flags_json,
            improvement_areas_json,
            speaker_metrics_json,
            audio_analysis_json,
            self.model_used,
            self.model_version,
            self.processing_time_seconds,
        ))
        
        logger.info(f"Saved analysis {self.id} for recording {self.ai_call_recording_id}")
        return self.id
    
    @staticmethod
    def _score_to_label(score: float) -> str:
        """Convert sentiment score (-10 to +10) to label."""
        if score >= 6:
            return "very_positive"
        elif score >= 2:
            return "positive"
        elif score >= -2:
            return "neutral"
        elif score >= -6:
            return "negative"
        else:
            return "very_negative"


@dataclass
class CallAnnotation:
    """
    Model for ai_call_annotations table.
    
    Human corrections/annotations for training the analysis model.
    """
    id: Optional[int] = None
    ai_call_analysis_id: int = 0
    user_id: int = 0
    annotation_type: str = ""  # 'sentiment', 'quality', 'compliance', 'custom_tag', 'segment_flag'
    field_name: Optional[str] = None
    original_value: Optional[str] = None
    corrected_value: Optional[str] = None
    timestamp_start: Optional[float] = None
    timestamp_end: Optional[float] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    is_training_data: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def save(self) -> int:
        """Save annotation to database and return the ID."""
        db = get_db_connection()
        
        tags_json = json.dumps(self.tags) if self.tags else None
        
        self.id = db.insert("""
            INSERT INTO ai_call_annotations
            (ai_call_analysis_id, user_id, annotation_type, field_name,
             original_value, corrected_value, timestamp_start, timestamp_end,
             tags, notes, is_training_data, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
        """, (
            self.ai_call_analysis_id,
            self.user_id,
            self.annotation_type,
            self.field_name,
            self.original_value,
            self.corrected_value,
            self.timestamp_start,
            self.timestamp_end,
            tags_json,
            self.notes,
            self.is_training_data,
        ))
        
        return self.id
    
    @staticmethod
    def get_training_data(
        since: Optional[datetime] = None,
        annotation_type: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Export training data for model fine-tuning.
        
        Returns annotations joined with analysis and transcription data.
        """
        db = get_db_connection()
        
        query = """
            SELECT 
                a.id,
                a.annotation_type,
                a.field_name,
                a.original_value,
                a.corrected_value,
                a.notes,
                a.created_at,
                an.summary,
                an.sentiment_score,
                an.sentiment_label,
                an.quality_score,
                an.key_topics,
                t.full_transcript,
                t.redacted_transcript
            FROM ai_call_annotations a
            JOIN ai_call_analysis an ON a.ai_call_analysis_id = an.id
            JOIN ai_call_recordings r ON an.ai_call_recording_id = r.id
            LEFT JOIN ai_call_transcriptions t ON r.id = t.ai_call_recording_id
            WHERE a.is_training_data = TRUE
        """
        params = []
        
        if since:
            query += " AND a.created_at >= %s"
            params.append(since)
        
        if annotation_type:
            query += " AND a.annotation_type = %s"
            params.append(annotation_type)
        
        query += " ORDER BY a.created_at DESC LIMIT %s"
        params.append(limit)
        
        return db.fetch_all(query, tuple(params))


@dataclass
class Summary:
    """
    Model for ai_summaries table.
    
    AI-generated summaries for call quality and other features over date ranges.
    """
    id: Optional[int] = None
    feature: str = "call_quality"
    start_date: Optional[str] = None  # YYYY-MM-DD
    end_date: Optional[str] = None    # YYYY-MM-DD
    client_ref: Optional[str] = None
    campaign: Optional[str] = None
    agent_id: Optional[int] = None
    summary_data: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = None
    generated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def save(self) -> int:
        """Save or update summary."""
        db = get_db_connection()
        
        summary_json = json.dumps(self.summary_data)
        metrics_json = json.dumps(self.metrics) if self.metrics else None
        
        # Use INSERT ... ON DUPLICATE KEY UPDATE
        self.id = db.insert("""
            INSERT INTO ai_summaries
            (feature, start_date, end_date, client_ref, campaign, agent_id,
             summary_data, metrics, generated_at, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), NOW())
            ON DUPLICATE KEY UPDATE
             summary_data = VALUES(summary_data),
             metrics = VALUES(metrics),
             generated_at = NOW(),
             updated_at = NOW()
        """, (
            self.feature,
            self.start_date,
            self.end_date,
            self.client_ref,
            self.campaign,
            self.agent_id,
            summary_json,
            metrics_json,
        ))
        
        return self.id
    
    @staticmethod
    def get(
        feature: str,
        start_date: str,
        end_date: str,
        client_ref: Optional[str] = None,
        campaign: Optional[str] = None,
        agent_id: Optional[int] = None
    ) -> Optional["Summary"]:
        """Retrieve a summary for a date range."""
        db = get_db_connection()
        
        query = """
            SELECT * FROM ai_summaries
            WHERE feature = %s AND start_date = %s AND end_date = %s
        """
        params = [feature, start_date, end_date]
        
        if client_ref:
            query += " AND client_ref = %s"
            params.append(client_ref)
        else:
            query += " AND client_ref IS NULL"
        
        if campaign:
            query += " AND campaign = %s"
            params.append(campaign)
        else:
            query += " AND campaign IS NULL"
        
        if agent_id:
            query += " AND agent_id = %s"
            params.append(agent_id)
        else:
            query += " AND agent_id IS NULL"
        
        row = db.fetch_one(query, tuple(params))
        if not row:
            return None
        
        return Summary(
            id=row["id"],
            feature=row["feature"],
            start_date=row["start_date"],
            end_date=row["end_date"],
            client_ref=row.get("client_ref"),
            campaign=row.get("campaign"),
            agent_id=row.get("agent_id"),
            summary_data=json.loads(row["summary_data"]) if row["summary_data"] else {},
            metrics=json.loads(row["metrics"]) if row.get("metrics") else None,
            generated_at=row.get("generated_at"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )


@dataclass
class ChatConversation:
    """
    Model for ai_chat_conversations table.
    
    Stores chat conversation sessions.
    """
    id: Optional[int] = None
    user_id: int = 0
    feature: str = "call_quality"
    filters: Optional[Dict[str, Any]] = None
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def save(self) -> int:
        """Create a new conversation."""
        db = get_db_connection()
        
        filters_json = json.dumps(self.filters) if self.filters else None
        
        self.id = db.insert("""
            INSERT INTO ai_chat_conversations
            (user_id, feature, filters, title, created_at, updated_at)
            VALUES (%s, %s, %s, %s, NOW(), NOW())
        """, (
            self.user_id,
            self.feature,
            filters_json,
            self.title,
        ))
        
        return self.id
    
    @staticmethod
    def get_by_id(conversation_id: int) -> Optional["ChatConversation"]:
        """Get conversation by ID."""
        db = get_db_connection()
        row = db.fetch_one(
            "SELECT * FROM ai_chat_conversations WHERE id = %s",
            (conversation_id,)
        )
        if not row:
            return None
        
        return ChatConversation(
            id=row["id"],
            user_id=row["user_id"],
            feature=row["feature"],
            filters=json.loads(row["filters"]) if row.get("filters") else None,
            title=row.get("title"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )


@dataclass
class ChatMessage:
    """
    Model for ai_chat_messages table.
    
    Stores individual messages within a chat conversation.
    """
    id: Optional[int] = None
    ai_chat_conversation_id: int = 0
    role: str = "user"  # 'user', 'assistant', 'system'
    content: str = ""
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    
    def save(self) -> int:
        """Save message to database."""
        db = get_db_connection()
        
        metadata_json = json.dumps(self.metadata) if self.metadata else None
        
        self.id = db.insert("""
            INSERT INTO ai_chat_messages
            (ai_chat_conversation_id, role, content, metadata, created_at, updated_at)
            VALUES (%s, %s, %s, %s, NOW(), NOW())
        """, (
            self.ai_chat_conversation_id,
            self.role,
            self.content,
            metadata_json,
        ))
        
        return self.id
    
    @staticmethod
    def get_conversation_history(conversation_id: int, limit: int = 50) -> List["ChatMessage"]:
        """Get message history for a conversation."""
        db = get_db_connection()
        rows = db.fetch_all("""
            SELECT * FROM ai_chat_messages
            WHERE ai_chat_conversation_id = %s
            ORDER BY created_at ASC
            LIMIT %s
        """, (conversation_id, limit))
        
        return [
            ChatMessage(
                id=row["id"],
                ai_chat_conversation_id=row["ai_chat_conversation_id"],
                role=row["role"],
                content=row["content"],
                metadata=json.loads(row["metadata"]) if row.get("metadata") else None,
                created_at=row.get("created_at"),
            )
            for row in rows
        ]


@dataclass
class VoiceFingerprint:
    """
    Model for ai_voice_fingerprints table.
    
    Stores voice embeddings for agent identification.
    """
    id: Optional[int] = None
    halo_id: int = 0
    agent_name: str = ""
    fingerprint_data: bytes = b""
    sample_count: int = 0
    confidence_threshold: float = 0.85
    last_updated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def save(self) -> int:
        """Save or update voice fingerprint."""
        db = get_db_connection()
        
        # Use INSERT ... ON DUPLICATE KEY UPDATE
        self.id = db.insert("""
            INSERT INTO ai_voice_fingerprints
            (halo_id, agent_name, fingerprint_data, sample_count,
             confidence_threshold, last_updated_at, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW(), NOW(), NOW())
            ON DUPLICATE KEY UPDATE
             agent_name = VALUES(agent_name),
             fingerprint_data = VALUES(fingerprint_data),
             sample_count = VALUES(sample_count),
             last_updated_at = NOW(),
             updated_at = NOW()
        """, (
            self.halo_id,
            self.agent_name,
            self.fingerprint_data,
            self.sample_count,
            self.confidence_threshold,
        ))
        
        return self.id
    
    @staticmethod
    def get_by_halo_id(halo_id: int) -> Optional["VoiceFingerprint"]:
        """Get fingerprint by agent Halo ID."""
        db = get_db_connection()
        row = db.fetch_one(
            "SELECT * FROM ai_voice_fingerprints WHERE halo_id = %s",
            (halo_id,)
        )
        if not row:
            return None
        
        return VoiceFingerprint(
            id=row["id"],
            halo_id=row["halo_id"],
            agent_name=row["agent_name"],
            fingerprint_data=row["fingerprint_data"],
            sample_count=row.get("sample_count", 0),
            confidence_threshold=row.get("confidence_threshold", 0.85),
            last_updated_at=row.get("last_updated_at"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )
    
    @staticmethod
    def get_all() -> List["VoiceFingerprint"]:
        """Get all voice fingerprints for matching."""
        db = get_db_connection()
        rows = db.fetch_all("SELECT * FROM ai_voice_fingerprints")
        
        return [
            VoiceFingerprint(
                id=row["id"],
                halo_id=row["halo_id"],
                agent_name=row["agent_name"],
                fingerprint_data=row["fingerprint_data"],
                sample_count=row.get("sample_count", 0),
                confidence_threshold=row.get("confidence_threshold", 0.85),
                last_updated_at=row.get("last_updated_at"),
                created_at=row.get("created_at"),
                updated_at=row.get("updated_at"),
            )
            for row in rows
        ]


@dataclass
class ClientConfig:
    """
    Model for ai_client_configs table.
    
    Client-specific configuration overrides.
    """
    id: Optional[int] = None
    client_ref: Optional[str] = None  # NULL = global/default config
    config_type: str = ""  # 'topics', 'agent_actions', 'performance_rubric', 'prompt', 'analysis_mode'
    config_data: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def save(self) -> int:
        """Save or update client config."""
        db = get_db_connection()
        
        config_json = json.dumps(self.config_data)
        
        self.id = db.insert("""
            INSERT INTO ai_client_configs
            (client_ref, config_type, config_data, is_active, created_at, updated_at)
            VALUES (%s, %s, %s, %s, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
             config_data = VALUES(config_data),
             is_active = VALUES(is_active),
             updated_at = NOW()
        """, (
            self.client_ref,
            self.config_type,
            config_json,
            self.is_active,
        ))
        
        return self.id
    
    @staticmethod
    def get_config(client_ref: Optional[str], config_type: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a client, falling back to global if not found.
        
        Lookup order:
        1. Client-specific config (client_ref = 'XXX')
        2. Global config (client_ref = NULL)
        3. None (use file-based defaults)
        """
        db = get_db_connection()
        
        # Try client-specific first
        if client_ref:
            row = db.fetch_one("""
                SELECT config_data FROM ai_client_configs
                WHERE client_ref = %s AND config_type = %s AND is_active = TRUE
            """, (client_ref, config_type))
            
            if row:
                return json.loads(row["config_data"])
        
        # Fall back to global
        row = db.fetch_one("""
            SELECT config_data FROM ai_client_configs
            WHERE client_ref IS NULL AND config_type = %s AND is_active = TRUE
        """, (config_type,))
        
        if row:
            return json.loads(row["config_data"])
        
        return None
    
    @staticmethod
    def get_analysis_mode(client_ref: Optional[str]) -> Optional[str]:
        """Get analysis mode override for client ('audio' or 'transcript')."""
        config = ClientConfig.get_config(client_ref, "analysis_mode")
        if config and "mode" in config:
            return config["mode"]
        return None
