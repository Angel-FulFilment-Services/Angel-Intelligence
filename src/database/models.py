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


# ============================================================================
# Quality Score Calculation
# ============================================================================

def calculate_quality_score(score_impacts: List[Dict[str, Any]]) -> float:
    """
    Calculate quality score from score impacts using asymmetric curve.
    
    - Base score: 65% (neutral, meets expectations)
    - Positive impacts: Diminishing returns (hard to reach 100%)
    - Negative impacts: Linear penalty (easy to fall)
    
    Args:
        score_impacts: List of impact dictionaries with 'impact' field (-5 to +5)
        
    Returns:
        Quality score from 0 to 100
        
    Zones:
    - Excellent: 90-100%
    - Good: 75-89%
    - Satisfactory: 65-74%
    - Below Average: 50-64%
    - Poor: 0-49%
    """
    if not score_impacts:
        return 65.0  # Baseline for no impacts
    
    # Extract impact values
    impacts = []
    for item in score_impacts:
        impact = item.get("impact", 0)
        if isinstance(impact, (int, float)):
            # Clamp to valid range
            impacts.append(max(-5, min(5, impact)))
    
    if not impacts:
        return 65.0
    
    avg_impact = sum(impacts) / len(impacts)
    
    if avg_impact >= 0:
        # Diminishing returns for positives: gain = 35 * (avg/5)^0.6
        gain = 35 * (avg_impact / 5) ** 0.6
        score = 65 + gain
    else:
        # Linear penalty for negatives: 13 points per -1 avg impact
        score = 65 + (avg_impact * 13)
    
    # Clamp to 0-100
    return max(0, min(100, round(score, 1)))


def get_quality_zone(score: float) -> str:
    """
    Get the quality zone label for a given score.
    
    Args:
        score: Quality score from 0-100
        
    Returns:
        Zone label: Excellent, Good, Satisfactory, Below Average, or Poor
    """
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 65:
        return "Satisfactory"
    elif score >= 50:
        return "Below Average"
    else:
        return "Poor"


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
    ddi: Optional[str] = None                 # DDI (phone number) for calltype group lookup
    campaign: Optional[str] = None            # Campaign name
    campaign_type: Optional[str] = None       # Campaign type for config lookup
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
            ddi=row.get("ddi"),
            campaign=row.get("campaign"),
            campaign_type=row.get("campaign_type"),
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
                id ASC
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


class TranscriptionStatus:
    """Status values for transcription generation."""
    PENDING = "pending"       # Not yet started
    PROCESSING = "processing" # Currently being transcribed
    COMPLETED = "completed"   # Successfully transcribed
    FAILED = "failed"         # Transcription failed


@dataclass
class CallTranscription:
    """
    Model for ai_call_transcriptions table.
    
    Stores transcription results including PII-redacted versions.
    
    Note: apex_id allows storing transcription without a recording (Dojo training).
    When full analysis runs, ai_call_recording_id is linked via link_to_recording().
    """
    id: Optional[int] = None
    ai_call_recording_id: Optional[int] = None  # Nullable - may not have recording yet
    apex_id: Optional[str] = None               # Call identifier - can exist without recording
    full_transcript: str = ""
    segments: List[Dict[str, Any]] = field(default_factory=list)
    redacted_transcript: Optional[str] = None
    pii_detected: Optional[List[Dict[str, Any]]] = None
    language_detected: str = "en"
    confidence_score: float = 0.95
    model_used: str = "whisperx-medium"
    processing_time_seconds: int = 0
    status: str = TranscriptionStatus.COMPLETED  # pending, processing, completed, failed
    error_message: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def save(self) -> int:
        """Save transcription to database and return the ID."""
        db = get_db_connection()
        
        # Prepare JSON fields
        segments_json = json.dumps(self.segments)
        pii_json = json.dumps(self.pii_detected) if self.pii_detected else None
        
        if self.id:
            # Update existing record
            db.execute("""
                UPDATE ai_call_transcriptions SET
                    ai_call_recording_id = %s, full_transcript = %s, segments = %s,
                    redacted_transcript = %s, pii_detected = %s, language_detected = %s,
                    confidence_score = %s, model_used = %s, processing_time_seconds = %s,
                    status = %s, error_message = %s, updated_at = NOW()
                WHERE id = %s
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
                self.status,
                self.error_message,
                self.id,
            ))
        else:
            # Insert new record
            self.id = db.insert("""
                INSERT INTO ai_call_transcriptions 
                (ai_call_recording_id, apex_id, full_transcript, segments, redacted_transcript,
                 pii_detected, language_detected, confidence_score, model_used, 
                 processing_time_seconds, status, error_message, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            """, (
                self.ai_call_recording_id,
                self.apex_id,
                self.full_transcript,
                segments_json,
                self.redacted_transcript,
                pii_json,
                self.language_detected,
                self.confidence_score,
                self.model_used,
                self.processing_time_seconds,
                self.status,
                self.error_message,
            ))
        
        logger.info(f"Saved transcription {self.id} for apex_id {self.apex_id}")
        return self.id
    
    def update_status(self, status: str, error_message: Optional[str] = None) -> None:
        """Update just the status field."""
        db = get_db_connection()
        self.status = status
        self.error_message = error_message
        
        if self.id:
            db.execute("""
                UPDATE ai_call_transcriptions 
                SET status = %s, error_message = %s, updated_at = NOW()
                WHERE id = %s
            """, (status, error_message, self.id))
    
    @staticmethod
    def acquire_lock(
        apex_id: str,
        timeout_minutes: int = 5
    ) -> Optional["CallTranscription"]:
        """
        Try to acquire a transcription lock for an apex_id.
        
        Returns CallTranscription with 'processing' status if lock acquired,
        or None if already being processed.
        """
        db = get_db_connection()
        
        # Check if there's an existing transcription
        existing = CallTranscription.get_by_apex_id(apex_id)
        
        if existing:
            # If completed, return it (no need to re-transcribe)
            if existing.status == TranscriptionStatus.COMPLETED:
                return existing
            
            # If processing and not timed out, return None (already in progress)
            if existing.status == TranscriptionStatus.PROCESSING:
                if existing.updated_at:
                    from datetime import timedelta
                    timeout_threshold = datetime.now() - timedelta(minutes=timeout_minutes)
                    if existing.updated_at > timeout_threshold:
                        # Still within timeout, processing in progress
                        return None
                    # Timed out, allow retry
                    logger.warning(f"Transcription processing timed out, allowing retry: {existing.id}")
            
            # Update existing to processing status
            existing.update_status(TranscriptionStatus.PROCESSING)
            return existing
        
        # Create new transcription record with processing status
        new_transcription = CallTranscription(
            apex_id=apex_id,
            status=TranscriptionStatus.PROCESSING,
        )
        new_transcription.save()
        return new_transcription
    
    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "CallTranscription":
        """Create CallTranscription from database row."""
        segments = row.get("segments")
        if isinstance(segments, str):
            segments = json.loads(segments)
        
        pii_detected = row.get("pii_detected")
        if isinstance(pii_detected, str):
            pii_detected = json.loads(pii_detected)
        
        return cls(
            id=row.get("id"),
            ai_call_recording_id=row.get("ai_call_recording_id"),
            apex_id=row.get("apex_id"),
            full_transcript=row.get("full_transcript", ""),
            segments=segments or [],
            redacted_transcript=row.get("redacted_transcript"),
            pii_detected=pii_detected,
            language_detected=row.get("language_detected", "en"),
            confidence_score=float(row.get("confidence_score", 0.95)),
            model_used=row.get("model_used", "whisperx-medium"),
            processing_time_seconds=row.get("processing_time_seconds", 0),
            status=row.get("status", TranscriptionStatus.COMPLETED),
            error_message=row.get("error_message"),
            created_at=row.get("created_at"),
            updated_at=row.get("updated_at"),
        )
    
    @staticmethod
    def get_by_apex_id(apex_id: str) -> Optional["CallTranscription"]:
        """Find existing transcription by apex_id."""
        db = get_db_connection()
        row = db.fetch_one(
            "SELECT * FROM ai_call_transcriptions WHERE apex_id = %s ORDER BY id DESC LIMIT 1",
            (apex_id,)
        )
        return CallTranscription.from_row(row) if row else None
    
    @staticmethod
    def get_by_recording_id(recording_id: int) -> Optional["CallTranscription"]:
        """Find transcription by recording ID."""
        db = get_db_connection()
        row = db.fetch_one(
            "SELECT * FROM ai_call_transcriptions WHERE ai_call_recording_id = %s",
            (recording_id,)
        )
        return CallTranscription.from_row(row) if row else None
    
    def link_to_recording(self, recording_id: int) -> None:
        """
        Link this transcription to a recording.
        
        Called when full analysis runs on a call that was previously
        transcribed-only via Dojo.
        """
        if not self.id:
            raise ValueError("Cannot link unsaved transcription")
        
        db = get_db_connection()
        db.execute("""
            UPDATE ai_call_transcriptions 
            SET ai_call_recording_id = %s
            WHERE id = %s
        """, (recording_id, self.id))
        
        self.ai_call_recording_id = recording_id
        logger.info(f"Linked transcription {self.id} to recording {recording_id}")


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
    quality_score: float = 65.0  # Baseline score
    quality_zone: str = "Satisfactory"  # Excellent/Good/Satisfactory/Below Average/Poor
    key_topics: List[Dict[str, Any]] = field(default_factory=list)
    agent_actions: List[Dict[str, Any]] = field(default_factory=list)
    score_impacts: List[Dict[str, Any]] = field(default_factory=list)
    performance_scores: Dict[str, int] = field(default_factory=dict)
    action_items: List[Dict[str, Any]] = field(default_factory=list)
    compliance_flags: List[Dict[str, Any]] = field(default_factory=list)
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
        agent_actions_json = json.dumps(self.agent_actions)
        score_impacts_json = json.dumps(self.score_impacts)
        performance_scores_json = json.dumps(self.performance_scores)
        action_items_json = json.dumps(self.action_items)
        compliance_flags_json = json.dumps(self.compliance_flags)
        speaker_metrics_json = json.dumps(self.speaker_metrics)
        audio_analysis_json = json.dumps(self.audio_analysis) if self.audio_analysis else None
        
        self.id = db.insert("""
            INSERT INTO ai_call_analysis
            (ai_call_recording_id, summary, sentiment_score, sentiment_label,
             key_topics, agent_actions, score_impacts, 
             performance_scores, quality_score, quality_zone,
             action_items, compliance_flags, speaker_metrics, audio_analysis,
             model_used, model_version, processing_time_seconds, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """, (
            self.ai_call_recording_id,
            self.summary,
            self.sentiment_score,
            self.sentiment_label,
            key_topics_json,
            agent_actions_json,
            score_impacts_json,
            performance_scores_json,
            self.quality_score,
            self.quality_zone,
            action_items_json,
            compliance_flags_json,
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
        """Convert sentiment score (0 to 10) to label."""
        if score >= 8:
            return "very_positive"
        elif score >= 6:
            return "positive"
        elif score >= 4:
            return "neutral"
        elif score >= 2:
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


class SummaryStatus:
    """Status values for summary generation."""
    PENDING = "pending"       # Not yet generated
    GENERATING = "generating" # Currently being generated
    COMPLETED = "completed"   # Successfully generated
    FAILED = "failed"         # Generation failed


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
    status: str = SummaryStatus.PENDING  # pending, generating, completed, failed
    error_message: Optional[str] = None
    generated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def save(self) -> int:
        """Save or update summary."""
        db = get_db_connection()
        
        # Always serialize to valid JSON (empty dict if None)
        summary_json = json.dumps(self.summary_data or {})
        metrics_json = json.dumps(self.metrics) if self.metrics else None
        
        if self.id:
            # Update existing record
            db.execute("""
                UPDATE ai_summaries SET
                    summary_data = %s,
                    metrics = %s,
                    status = %s,
                    error_message = %s,
                    generated_at = NOW(),
                    updated_at = NOW()
                WHERE id = %s
            """, (
                summary_json,
                metrics_json,
                self.status,
                self.error_message,
                self.id,
            ))
        else:
            # Insert new record
            self.id = db.insert("""
                INSERT INTO ai_summaries
                (feature, start_date, end_date, client_ref, campaign, agent_id,
                 summary_data, metrics, status, error_message, generated_at, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW(), NOW())
            """, (
                self.feature,
                self.start_date,
                self.end_date,
                self.client_ref,
                self.campaign,
                self.agent_id,
                summary_json,
                metrics_json,
                self.status,
                self.error_message,
            ))
        
        return self.id
    
    def update_status(self, status: str, error_message: Optional[str] = None) -> None:
        """Update just the status field."""
        db = get_db_connection()
        self.status = status
        self.error_message = error_message
        
        if self.id:
            db.execute("""
                UPDATE ai_summaries 
                SET status = %s, error_message = %s, updated_at = NOW()
                WHERE id = %s
            """, (status, error_message, self.id))
    
    @staticmethod
    def acquire_lock(
        feature: str,
        start_date: str,
        end_date: str,
        client_ref: Optional[str] = None,
        campaign: Optional[str] = None,
        agent_id: Optional[int] = None,
        timeout_minutes: int = 5
    ) -> Optional["Summary"]:
        """
        Try to acquire a generation lock for this summary.
        
        Returns Summary with 'generating' status if lock acquired,
        or None if already being generated.
        """
        db = get_db_connection()
        
        # Check if there's an existing summary being generated
        existing = Summary.get(feature, start_date, end_date, client_ref, campaign, agent_id)
        
        if existing:
            # If currently generating, check if it's timed out
            if existing.status == SummaryStatus.GENERATING:
                from datetime import timedelta
                timeout_threshold = datetime.now() - timedelta(minutes=timeout_minutes)
                
                # Check if still within timeout (generation in progress)
                if existing.updated_at and existing.updated_at > timeout_threshold:
                    # Still within timeout, block retry
                    return None
                elif not existing.updated_at:
                    # No updated_at means just started, assume in progress
                    return None
                else:
                    # Timed out, allow retry
                    logger.warning(f"Summary generation timed out, allowing retry: {existing.id}")
            
            # For FAILED, COMPLETED, or timed-out GENERATING - allow retry
            # Update existing to generating status
            existing.update_status(SummaryStatus.GENERATING)
            return existing
        
        # Create new summary with generating status
        new_summary = Summary(
            feature=feature,
            start_date=start_date,
            end_date=end_date,
            client_ref=client_ref,
            campaign=campaign,
            agent_id=agent_id,
            status=SummaryStatus.GENERATING,
        )
        new_summary.save()
        return new_summary
    
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
        
        # Order by id DESC to get the most recent if duplicates exist
        query += " ORDER BY id DESC LIMIT 1"
        
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
            status=row.get("status", SummaryStatus.COMPLETED),
            error_message=row.get("error_message"),
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
    Model for ai_configs table.
    
    Client-specific configuration overrides.
    """
    id: Optional[int] = None
    client_ref: Optional[str] = None  # NULL = global/default config
    campaign: Optional[str] = None    # NULL = applies to all campaigns
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
            INSERT INTO ai_configs
            (client_ref, campaign, config_type, config_data, is_active, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
            ON DUPLICATE KEY UPDATE
             config_data = VALUES(config_data),
             is_active = VALUES(is_active),
             updated_at = NOW()
        """, (
            self.client_ref,
            self.campaign,
            self.config_type,
            config_json,
            self.is_active,
        ))
        
        return self.id
    
    @staticmethod
    def get_config(
        client_ref: Optional[str], 
        config_type: str,
        campaign: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a client/campaign, with fallback hierarchy.
        
        Lookup order:
        1. Campaign-specific config (client_ref + campaign)
        2. Client-specific config (client_ref, campaign = NULL)
        3. Global config (client_ref = NULL, campaign = NULL)
        4. None (use file-based defaults)
        """
        db = get_db_connection()
        
        # Try campaign-specific first (if both client and campaign provided)
        if client_ref and campaign:
            row = db.fetch_one("""
                SELECT config_data FROM ai_configs
                WHERE client_ref = %s AND campaign = %s AND config_type = %s AND is_active = TRUE
            """, (client_ref, campaign, config_type))
            
            if row:
                return json.loads(row["config_data"])
        
        # Try client-specific (campaign = NULL)
        if client_ref:
            row = db.fetch_one("""
                SELECT config_data FROM ai_configs
                WHERE client_ref = %s AND campaign IS NULL AND config_type = %s AND is_active = TRUE
            """, (client_ref, config_type))
            
            if row:
                return json.loads(row["config_data"])
        
        # Fall back to global (client_ref = NULL, campaign = NULL)
        row = db.fetch_one("""
            SELECT config_data FROM ai_configs
            WHERE client_ref IS NULL AND campaign IS NULL AND config_type = %s AND is_active = TRUE
        """, (config_type,))
        
        if row:
            return json.loads(row["config_data"])
        
        return None
    
    @staticmethod
    def get_analysis_mode(client_ref: Optional[str], campaign: Optional[str] = None) -> Optional[str]:
        """Get analysis mode override for client/campaign ('audio' or 'transcript')."""
        config = ClientConfig.get_config(client_ref, "analysis_mode", campaign)
        if config and "mode" in config:
            return config["mode"]
        return None
