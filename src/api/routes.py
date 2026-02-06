"""
Angel Intelligence - API Routes

All API endpoints for the Angel Intelligence service.
Endpoints require Bearer token authentication.
"""

import json
import logging
import os
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from pydantic import BaseModel, Field

from src.api.auth import verify_token
from src.config import get_settings
from src.database import CallRecording, CallTranscription, CallAnalysis, get_db_connection

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Request/Response Models
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    version: str = "1.0.0"
    worker_id: str
    environment: str
    device: str
    cuda_available: bool
    models_loaded: dict = {}
    worker_mode: str = "batch"


class RecordingStatusResponse(BaseModel):
    """Recording processing status response."""
    id: int
    apex_id: str
    status: str
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None
    error: Optional[str] = None
    retry_count: int = 0


class TranscriptionResponse(BaseModel):
    """Transcription result response."""
    id: int
    recording_id: int
    full_transcript: str
    segments: List[dict]
    redacted_transcript: Optional[str] = None
    pii_detected: Optional[List[dict]] = None
    language: str
    confidence: float
    model: str
    processing_time_seconds: int


class AnalysisResponse(BaseModel):
    """Analysis result response."""
    id: int
    recording_id: int
    summary: str
    sentiment_score: float
    sentiment_label: str
    quality_score: float
    quality_zone: Optional[str] = None  # Excellent/Good/Satisfactory/Below Average/Poor
    key_topics: List[dict]
    agent_actions: List[dict]  # Neutral actions (what the agent did)
    score_impacts: List[dict]  # Quality impacts (-5 to +5)
    performance_scores: dict
    action_items: List[dict]
    compliance_flags: List[dict]
    speaker_metrics: dict
    audio_observations: Optional[dict] = None
    model: str
    model_version: Optional[str] = None
    processing_time_seconds: int


class SubmitRecordingRequest(BaseModel):
    """Request to submit a recording for processing."""
    apex_id: str
    call_date: datetime
    client_ref: Optional[str] = None
    campaign: Optional[str] = None
    halo_id: Optional[int] = None
    agent_name: Optional[str] = None
    direction: str = "outbound"
    duration_seconds: int = 0
    retain_audio: bool = False
    # Additional reference fields
    orderref: Optional[str] = None
    enqref: Optional[str] = None
    obref: Optional[str] = None
    creative: Optional[str] = None
    invoicing: Optional[str] = None


class SubmitRecordingResponse(BaseModel):
    """Response after submitting a recording."""
    id: int
    apex_id: str
    status: str
    message: str


class ConfigResponse(BaseModel):
    """Configuration response."""
    topics: List[str]
    agent_actions: List[str]
    performance_rubric: List[str]


class ChatUser(BaseModel):
    """User information for personalized chat responses."""
    id: int
    name: str
    email: Optional[str] = None


class ChatRequest(BaseModel):
    """Chat conversation request."""
    message: str
    recording_id: Optional[int] = None
    conversation_id: Optional[str] = None
    # User object for personalization and conversation creation
    user: Optional[ChatUser] = None
    # Legacy field - use user.id instead
    user_id: Optional[int] = None
    feature: Optional[str] = "general"
    filters: Optional[dict] = None


class ChatResponse(BaseModel):
    """Chat conversation response."""
    success: bool = True
    response: str
    conversation_id: str
    message_id: Optional[int] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


# =============================================================================
# Health and Info Endpoints
# =============================================================================

@router.get("/", response_model=dict)
async def root():
    """Root endpoint - service info."""
    import torch
    settings = get_settings()
    
    return {
        "service": "Angel Intelligence",
        "description": "AI-powered call transcription and analysis",
        "version": "1.0.0",
        "environment": settings.angel_env,
        "status": "running",
    }


@router.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint - no auth required."""
    import os
    
    # PyTorch is optional for API pod
    cuda_available = False
    device = "cpu"
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        device = "cuda" if cuda_available else "cpu"
    except ImportError:
        pass
    
    settings = get_settings()
    
    # Check interactive service status
    interactive_status = {"available": False, "model_loaded": False}
    try:
        from src.services.interactive import get_interactive_service
        service = get_interactive_service()
        interactive_status = service.get_status()
    except Exception:
        pass
    
    # Check which models are configured
    # Models will auto-download from HuggingFace on first use
    models_loaded = {
        "analysis": {
            "version": os.getenv("ANALYSIS_MODEL_VERSION", "v1.0.0"),
            "source": "local" if settings.analysis_model_path else "huggingface",
            "path_or_id": settings.analysis_model_path or settings.analysis_model,
            "will_auto_download": not settings.analysis_model_path,
        },
        "chat": {
            "version": "base",
            "loaded": interactive_status.get("model_loaded", False),
            "source": "local" if settings.chat_model_path else "huggingface",
            "path_or_id": settings.chat_model_path or settings.chat_model,
            "will_auto_download": not settings.chat_model_path,
        },
        "whisper": {
            "version": settings.whisper_model or "medium",
            "source": "huggingface",
            "will_auto_download": True,
        },
        "interactive": interactive_status,
    }
    
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        worker_id=settings.worker_id,
        environment=settings.angel_env,
        device=device,
        cuda_available=cuda_available,
        models_loaded=models_loaded,
        worker_mode=settings.worker_mode,
    )


# =============================================================================
# Recording Management Endpoints
# =============================================================================

@router.post(
    "/recordings/submit",
    response_model=SubmitRecordingResponse,
    dependencies=[Depends(verify_token)]
)
async def submit_recording(request: SubmitRecordingRequest):
    """
    Submit a new recording for processing.
    
    The recording will be queued and processed by the worker.
    Audio will be fetched from PBX sources based on apex_id and call_date.
    """
    db = get_db_connection()
    
    # Check if already exists
    existing = db.fetch_one(
        "SELECT id, processing_status FROM ai_call_recordings WHERE apex_id = %s",
        (request.apex_id,)
    )
    
    if existing:
        return SubmitRecordingResponse(
            id=existing["id"],
            apex_id=request.apex_id,
            status=existing["processing_status"],
            message=f"Recording already exists with status: {existing['processing_status']}"
        )
    
    # Insert new recording
    recording_id = db.insert("""
        INSERT INTO ai_call_recordings 
        (apex_id, call_date, client_ref, campaign, halo_id, agent_name, direction,
         duration_seconds, retain_audio, orderref, enqref, obref, creative, invoicing,
         processing_status, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', NOW())
    """, (
        request.apex_id,
        request.call_date,
        request.client_ref,
        request.campaign,
        request.halo_id,
        request.agent_name,
        request.direction,
        request.duration_seconds,
        request.retain_audio,
        request.orderref,
        request.enqref,
        request.obref,
        request.creative,
        request.invoicing,
    ))
    
    logger.info(f"Submitted recording {recording_id}: {request.apex_id}")
    
    return SubmitRecordingResponse(
        id=recording_id,
        apex_id=request.apex_id,
        status="pending",
        message="Recording submitted for processing"
    )


@router.get(
    "/recordings/{recording_id}/status",
    response_model=RecordingStatusResponse,
    dependencies=[Depends(verify_token)]
)
async def get_recording_status(recording_id: int):
    """Get the processing status of a recording."""
    recording = CallRecording.get_by_id(recording_id)
    
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    
    return RecordingStatusResponse(
        id=recording.id,
        apex_id=recording.apex_id,
        status=recording.processing_status.value,
        processing_started_at=recording.processing_started_at,
        processing_completed_at=recording.processing_completed_at,
        error=recording.processing_error,
        retry_count=recording.retry_count,
    )


@router.get(
    "/recordings/{recording_id}/transcription",
    response_model=TranscriptionResponse,
    dependencies=[Depends(verify_token)]
)
async def get_transcription(recording_id: int):
    """Get the transcription for a recording."""
    db = get_db_connection()
    
    row = db.fetch_one("""
        SELECT * FROM ai_call_transcriptions 
        WHERE ai_call_recording_id = %s
        ORDER BY created_at DESC LIMIT 1
    """, (recording_id,))
    
    if not row:
        raise HTTPException(status_code=404, detail="Transcription not found")
    
    import json
    
    return TranscriptionResponse(
        id=row["id"],
        recording_id=row["ai_call_recording_id"],
        full_transcript=row["full_transcript"],
        segments=json.loads(row["segments"]) if row["segments"] else [],
        redacted_transcript=row.get("redacted_transcript"),
        pii_detected=json.loads(row["pii_detected"]) if row.get("pii_detected") else None,
        language=row.get("language_detected", "en"),
        confidence=float(row.get("confidence_score", 0.95)),
        model=row.get("model_used", "whisperx"),
        processing_time_seconds=int(row.get("processing_time_seconds", 0)),
    )


@router.get(
    "/recordings/{recording_id}/analysis",
    response_model=AnalysisResponse,
    dependencies=[Depends(verify_token)]
)
async def get_analysis(recording_id: int):
    """Get the analysis for a recording."""
    db = get_db_connection()
    
    row = db.fetch_one("""
        SELECT * FROM ai_call_analysis 
        WHERE ai_call_recording_id = %s
        ORDER BY created_at DESC LIMIT 1
    """, (recording_id,))
    
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    import json
    
    return AnalysisResponse(
        id=row["id"],
        recording_id=row["ai_call_recording_id"],
        summary=row["summary"],
        sentiment_score=float(row["sentiment_score"]),
        sentiment_label=row["sentiment_label"],
        quality_score=float(row["quality_score"]),
        quality_zone=row.get("quality_zone"),
        key_topics=json.loads(row["key_topics"]) if row["key_topics"] else [],
        agent_actions=json.loads(row.get("agent_actions", "[]")) if row.get("agent_actions") else [],
        score_impacts=json.loads(row.get("score_impacts", "[]")) if row.get("score_impacts") else [],
        performance_scores=json.loads(row.get("performance_scores", "{}")),
        action_items=json.loads(row["action_items"]) if row["action_items"] else [],
        compliance_flags=json.loads(row["compliance_flags"]) if row["compliance_flags"] else [],
        speaker_metrics=json.loads(row["speaker_metrics"]) if row["speaker_metrics"] else {},
        audio_observations=json.loads(row["audio_analysis"]) if row.get("audio_analysis") else None,
        model=row.get("model_used", "unknown"),
        model_version=row.get("model_version"),
        processing_time_seconds=int(row.get("processing_time_seconds", 0)),
    )


# =============================================================================
# Configuration Endpoints
# =============================================================================

@router.get(
    "/config/analysis",
    response_model=ConfigResponse,
    dependencies=[Depends(verify_token)]
)
async def get_analysis_config():
    """Get the analysis configuration (topics, actions, rubric)."""
    import json
    import os
    
    config_path = "call_analysis_config.json"
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            return ConfigResponse(
                topics=config.get("topics", []),
                agent_actions=config.get("agent_actions", []),
                performance_rubric=config.get("performance_rubric", []),
            )
    
    return ConfigResponse(topics=[], agent_actions=[], performance_rubric=[])


# =============================================================================
# Chat Endpoints
# =============================================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(verify_token)],
    deprecated=True
)
async def chat(request: ChatRequest):
    """
    Chat with the AI about call data.
    
    .. deprecated::
        Use POST /api/chat instead. This endpoint redirects to the enhanced
        chat endpoint to prevent duplicate message saves.
    """
    # Redirect to enhanced chat to prevent duplicate saves
    from src.api.routes import enhanced_chat, EnhancedChatRequest, ChatUser as EnhancedChatUser
    
    enhanced_request = EnhancedChatRequest(
        message=request.message,
        recording_id=request.recording_id,
        conversation_id=int(request.conversation_id) if request.conversation_id else None,
        user=EnhancedChatUser(
            id=request.user.id if request.user else request.user_id or 0,
            name=request.user.name if request.user else "User",
            email=request.user.email if request.user else None,
        ) if (request.user or request.user_id) else None,
        user_id=request.user_id,
        feature=request.feature or "general",
        filters=request.filters,
    )
    
    enhanced_response = await enhanced_chat(enhanced_request)
    
    return ChatResponse(
        success=enhanced_response.success,
        response=enhanced_response.response,
        conversation_id=str(enhanced_response.conversation_id) if enhanced_response.conversation_id else "",
        message_id=enhanced_response.message_id,
        error=enhanced_response.error,
        error_type=enhanced_response.error_type,
    )


# Legacy chat implementation removed - all chat now goes through enhanced_chat (/api/chat)


# =============================================================================
# Batch Operations
# =============================================================================

@router.post(
    "/recordings/reprocess/{recording_id}",
    dependencies=[Depends(verify_token)]
)
async def reprocess_recording(recording_id: int):
    """
    Requeue a recording for reprocessing.
    
    Resets status to 'queued' so the worker will pick it up again.
    """
    db = get_db_connection()
    
    # Check recording exists
    recording = CallRecording.get_by_id(recording_id)
    if not recording:
        raise HTTPException(status_code=404, detail="Recording not found")
    
    # Reset to queued
    db.execute("""
        UPDATE ai_call_recordings 
        SET processing_status = 'queued',
            processing_error = NULL,
            retry_count = 0,
            next_retry_at = NULL,
            updated_at = NOW()
        WHERE id = %s
    """, (recording_id,))
    
    logger.info(f"Requeued recording {recording_id} for reprocessing")
    
    return {"message": f"Recording {recording_id} queued for reprocessing"}


@router.get(
    "/recordings/pending",
    dependencies=[Depends(verify_token)]
)
async def get_pending_recordings(
    limit: int = Query(default=50, le=200)
):
    """Get list of pending recordings."""
    db = get_db_connection()
    
    rows = db.fetch_all("""
        SELECT id, apex_id, processing_status, retry_count, created_at
        FROM ai_call_recordings
        WHERE processing_status IN ('pending', 'queued')
        ORDER BY created_at ASC
        LIMIT %s
    """, (limit,))
    
    return {
        "count": len(rows),
        "recordings": rows
    }


@router.get(
    "/recordings/failed",
    dependencies=[Depends(verify_token)]
)
async def get_failed_recordings(
    limit: int = Query(default=50, le=200)
):
    """Get list of failed recordings."""
    db = get_db_connection()
    
    rows = db.fetch_all("""
        SELECT id, apex_id, processing_error, retry_count, 
               processing_completed_at, next_retry_at
        FROM ai_call_recordings
        WHERE processing_status = 'failed'
        ORDER BY processing_completed_at DESC
        LIMIT %s
    """, (limit,))
    
    return {
        "count": len(rows),
        "recordings": rows
    }


# =============================================================================
# Transcription Only (for Dojo Training)
# =============================================================================

def _normalize_speaker_labels(segments: List[Dict[str, Any]]) -> List["TranscribeSegment"]:
    """
    Normalize speaker labels to agent/supporter using talk-time heuristics.
    
    Uses the same logic as the full analysis pipeline:
    1. Agent typically speaks MORE (guiding the conversation)
    2. First speaker as tiebreaker if talk times are close
    
    Returns list of TranscribeSegment for API response.
    """
    if not segments:
        return []
    
    # Calculate talk time per speaker
    speaker_talk_time = {}
    first_speaker = None
    
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_00")
        
        # If already labelled as agent/supporter, use simple mapping
        if speaker in ["agent", "supporter"] or speaker.startswith("agent_"):
            return _simple_normalize_segments(segments)
        
        duration = seg.get("end", 0) - seg.get("start", 0)
        speaker_talk_time[speaker] = speaker_talk_time.get(speaker, 0) + duration
        
        if first_speaker is None:
            first_speaker = speaker
    
    if not speaker_talk_time:
        return _simple_normalize_segments(segments)
    
    # Determine agent: speaker with most talk time
    sorted_speakers = sorted(speaker_talk_time.items(), key=lambda x: x[1], reverse=True)
    most_talkative = sorted_speakers[0][0]
    
    # Use first speaker as tiebreaker if talk times are close (within 20%)
    if len(sorted_speakers) > 1:
        ratio = sorted_speakers[1][1] / sorted_speakers[0][1] if sorted_speakers[0][1] > 0 else 0
        if ratio > 0.8:
            agent_speaker = first_speaker
            logger.debug(f"Talk times close ({ratio:.2f}), using first speaker as agent")
        else:
            agent_speaker = most_talkative
            logger.debug(f"Using most talkative speaker as agent ({sorted_speakers[0][1]:.1f}s)")
    else:
        agent_speaker = most_talkative
    
    # Build normalized segment list
    normalized = []
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_00")
        speaker_label = "agent" if speaker == agent_speaker else "supporter"
        
        normalized.append(TranscribeSegment(
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            text=seg.get("text", ""),
            speaker=speaker_label
        ))
    
    return normalized


def _simple_normalize_segments(segments: List[Dict[str, Any]]) -> List["TranscribeSegment"]:
    """Simple fallback normalization using SPEAKER_00 = agent."""
    normalized = []
    for seg in segments:
        speaker = seg.get("speaker", "SPEAKER_00")
        if speaker in ["SPEAKER_00", "agent"] or speaker.startswith("agent_"):
            speaker_label = "agent"
        else:
            speaker_label = "supporter"
        
        normalized.append(TranscribeSegment(
            start=seg.get("start", 0.0),
            end=seg.get("end", 0.0),
            text=seg.get("text", ""),
            speaker=speaker_label
        ))
    
    return normalized


class TranscribeRequest(BaseModel):
    """Request to transcribe a call without full analysis."""
    apex_id: str
    call_date: Optional[str] = None  # YYYY-MM-DD format, used for archive lookup


class TranscribeSegment(BaseModel):
    """A transcription segment with speaker and timing."""
    start: float
    end: float
    text: str
    speaker: str


class TranscribeResponse(BaseModel):
    """Response from transcription request."""
    success: bool
    full_transcript: Optional[str] = None
    segments: Optional[List[TranscribeSegment]] = None
    language: Optional[str] = None
    confidence_score: Optional[float] = None
    message: Optional[str] = None


@router.post(
    "/api/transcribe",
    response_model=TranscribeResponse,
    dependencies=[Depends(verify_token)]
)
async def transcribe_call(request: TranscribeRequest):
    """
    Transcribe a call without full analysis.
    
    Used by Dojo training feature when users need to label unanalysed calls.
    - Fetches audio for the given apex_id
    - Runs speech-to-text with speaker diarization
    - Stores transcript in database for later reuse
    - Does NOT run sentiment/quality analysis
    
    If transcription already exists for this apex_id, returns cached result.
    Returns 429 if transcription is already in progress by another user.
    """
    from src.services import TranscriptionService, AudioDownloader
    from src.database import CallTranscription, TranscriptionStatus, get_db_connection
    
    db = get_db_connection()
    apex_id = request.apex_id
    
    # Try to acquire database lock for this transcription
    transcription_lock = CallTranscription.acquire_lock(apex_id, timeout_minutes=5)
    
    if transcription_lock is None:
        # Already being processed by another node
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Transcription in progress",
                "message": "This call is being transcribed by another user, please wait.",
                "retry_after": 30
            }
        )
    
    # Check if it's already completed (acquire_lock returns completed transcriptions)
    if transcription_lock.status == TranscriptionStatus.COMPLETED and transcription_lock.full_transcript:
        logger.info(f"Returning cached transcription for apex_id {apex_id}")
        
        # Normalize speaker labels using talk-time heuristics
        segments = _normalize_speaker_labels(transcription_lock.segments)
        
        return TranscribeResponse(
            success=True,
            full_transcript=transcription_lock.full_transcript,
            segments=segments,
            language=transcription_lock.language_detected,
            confidence_score=transcription_lock.confidence_score
        )
    
    try:
            # Need to transcribe - first find call date for audio download
            # Check if there's a recording record (may not exist for unanalysed calls)
            recording_row = db.fetch_one(
                "SELECT call_date, r2_path, r2_bucket FROM ai_call_recordings WHERE apex_id = %s",
                (apex_id,)
            )
            
            # Determine call_date from: 1) request, 2) database, 3) apex_id timestamp, 4) fallback to now
            from datetime import datetime as dt
            call_date = None
            r2_path = None
            r2_bucket = None
            
            # 1. Use call_date from request if provided
            if request.call_date:
                try:
                    call_date = dt.strptime(request.call_date, "%Y-%m-%d")
                except ValueError:
                    logger.warning(f"Invalid call_date format: {request.call_date}, expected YYYY-MM-DD")
            
            # 2. Use database values if available
            if recording_row:
                if not call_date:
                    call_date = recording_row.get("call_date")
                r2_path = recording_row.get("r2_path")
                r2_bucket = recording_row.get("r2_bucket")
            
            # 3. Parse timestamp from apex_id
            if not call_date:
                try:
                    timestamp = float(apex_id.split('.')[0])
                    call_date = dt.fromtimestamp(timestamp)
                except (ValueError, IndexError):
                    pass
            
            # 4. Fallback to now (unlikely to find file, but won't crash)
            if not call_date:
                call_date = dt.now()
                logger.warning(f"Could not determine call_date for apex_id {apex_id}, using current date")
            
            # Download audio
            downloader = AudioDownloader()
            audio_path, is_local = downloader.download_recording(
                apex_id=apex_id,
                call_date=call_date,
                r2_path=r2_path,
                r2_bucket=r2_bucket
            )
            
            try:
                # Transcribe
                transcriber = TranscriptionService()
                result = transcriber.transcribe(
                    audio_path=audio_path,
                    language="en",
                    diarize=True
                )
                
                if not result or not result.get("full_transcript"):
                    transcription_lock.update_status(TranscriptionStatus.FAILED, "Transcription produced no results")
                    return TranscribeResponse(
                        success=False,
                        message="Transcription produced no results"
                    )
                
                # Update the transcription lock with completed data
                transcription_lock.full_transcript = result["full_transcript"]
                transcription_lock.segments = result["segments"]
                transcription_lock.language_detected = result.get("language_detected", "en")
                transcription_lock.confidence_score = result.get("confidence", 0.95)
                transcription_lock.model_used = result.get("model_used", "whisperx-medium")
                transcription_lock.processing_time_seconds = int(result.get("processing_time", 0))
                transcription_lock.status = TranscriptionStatus.COMPLETED
                transcription_lock.error_message = None
                transcription_lock.save()
                
                logger.info(f"Saved new transcription for apex_id {apex_id}")
                
                # Normalize speaker labels using talk-time heuristics
                # (same logic as full analysis pipeline)
                segments = _normalize_speaker_labels(result["segments"])
                
                return TranscribeResponse(
                    success=True,
                    full_transcript=result["full_transcript"],
                    segments=segments,
                    language=result.get("language_detected", "en"),
                    confidence_score=result.get("confidence", 0.95)
                )
                
            finally:
                # Clean up audio file if not local
                if not is_local and audio_path and os.path.exists(audio_path):
                    try:
                        os.remove(audio_path)
                    except Exception:
                        pass
                    
    except HTTPException:
        # Re-raise HTTP exceptions (including 429)
        raise
    except FileNotFoundError as e:
        error_str = str(e)
        logger.warning(f"Audio not found for apex_id {apex_id}: {e}")
        # Mark as failed so retry is allowed
        transcription_lock.update_status(TranscriptionStatus.FAILED, error_str)
        
        # Distinguish between missing audio file vs missing conversion tools
        if "conversion tool" in error_str.lower() or "sox" in error_str.lower() or "ffmpeg" in error_str.lower():
            raise HTTPException(
                status_code=500,
                detail=f"Audio conversion tool missing: {error_str}"
            )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Audio file not found for apex_id {apex_id}. Tried all storage locations (local, R2, PBX live, PBX archive)."
            )
    except Exception as e:
        logger.error(f"Transcription failed for apex_id {apex_id}: {e}", exc_info=True)
        # Mark as failed so retry is allowed
        transcription_lock.update_status(TranscriptionStatus.FAILED, str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Transcription failed: {str(e)}"
        )


# =============================================================================
# Process Single Call (Manual Trigger)
# =============================================================================

class ProcessRequest(BaseModel):
    """Request to process a single call."""
    apex_id: str
    force_reprocess: bool = False


class ProcessResponse(BaseModel):
    """Response from process request."""
    success: bool
    recording_id: Optional[int] = None
    status: str
    message: str


@router.post(
    "/api/process",
    response_model=ProcessResponse,
    dependencies=[Depends(verify_token)]
)
async def process_call(request: ProcessRequest):
    """
    Manually trigger processing of a single call.
    
    This queues the call for immediate processing by the worker.
    """
    db = get_db_connection()
    
    # Find the recording
    row = db.fetch_one(
        "SELECT id, processing_status FROM ai_call_recordings WHERE apex_id = %s",
        (request.apex_id,)
    )
    
    if not row:
        raise HTTPException(status_code=404, detail=f"Recording with apex_id {request.apex_id} not found")
    
    recording_id = row["id"]
    current_status = row["processing_status"]
    
    # Check if already completed and not forcing reprocess
    if current_status == "completed" and not request.force_reprocess:
        return ProcessResponse(
            success=True,
            recording_id=recording_id,
            status=current_status,
            message="Recording already processed. Use force_reprocess=true to reprocess."
        )
    
    # Queue for processing
    db.execute("""
        UPDATE ai_call_recordings 
        SET processing_status = 'queued',
            processing_error = NULL,
            retry_count = 0,
            next_retry_at = NULL,
            updated_at = NOW()
        WHERE id = %s
    """, (recording_id,))
    
    logger.info(f"Manually queued recording {recording_id} ({request.apex_id}) for processing")
    
    return ProcessResponse(
        success=True,
        recording_id=recording_id,
        status="queued",
        message="Call queued for processing"
    )


# =============================================================================
# Summary Generation
# =============================================================================

class SummaryGenerateRequest(BaseModel):
    """Request to generate a summary for a date range."""
    feature: str = "call_quality"
    start_date: str  # YYYY-MM-DD
    end_date: str    # YYYY-MM-DD
    client_ref: Optional[str] = None
    campaign: Optional[str] = None
    agent_id: Optional[int] = None


class SummaryGenerateResponse(BaseModel):
    """Response from summary generation."""
    success: bool
    summary_id: Optional[int] = None
    summary_data: Optional[dict] = None
    message: str


@router.post(
    "/api/summary/generate",
    response_model=SummaryGenerateResponse,
    dependencies=[Depends(verify_token)]
)
async def generate_summary(request: SummaryGenerateRequest):
    """
    Generate a summary using the interactive service.
    
    Aggregates call data for the specified date range and generates
    AI-powered insights and recommendations.
    
    Returns 429 if a summary is already being generated for the same period.
    """
    from src.database import Summary, SummaryStatus
    from src.services.interactive import get_interactive_service
    
    db = get_db_connection()
    
    # Try to acquire generation lock
    summary_lock = Summary.acquire_lock(
        feature=request.feature,
        start_date=request.start_date,
        end_date=request.end_date,
        client_ref=request.client_ref,
        campaign=request.campaign,
        agent_id=request.agent_id,
        timeout_minutes=5
    )
    
    if summary_lock is None:
        # Already being generated
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Summary generation already in progress",
                "message": "A summary for this period is currently being generated. Please wait and try again.",
                "retry_after": 30
            }
        )
    
    try:
        # Build the filter query
        query = """
            SELECT 
                COUNT(*) as call_count,
                AVG(a.quality_score) as avg_quality,
                AVG(a.sentiment_score) as avg_sentiment,
                SUM(r.duration_seconds) as total_duration
            FROM ai_call_analysis a
            JOIN ai_call_recordings r ON a.ai_call_recording_id = r.id
            WHERE r.call_date >= %s AND r.call_date <= %s
        """
        params = [request.start_date, request.end_date]
        
        filters = {"start_date": request.start_date, "end_date": request.end_date}
        
        if request.client_ref:
            query += " AND r.client_ref = %s"
            params.append(request.client_ref)
            filters["client_ref"] = request.client_ref
        
        if request.campaign:
            query += " AND r.campaign = %s"
            params.append(request.campaign)
            filters["campaign"] = request.campaign
        
        if request.agent_id:
            query += " AND r.halo_id = %s"
            params.append(request.agent_id)
            filters["agent_id"] = request.agent_id
        
        metrics_row = db.fetch_one(query, tuple(params))
    
        if not metrics_row or metrics_row["call_count"] == 0:
            summary_lock.update_status(SummaryStatus.FAILED, "No call data found")
            return SummaryGenerateResponse(
                success=False,
                message="No call data found for the specified period"
            )
        
        # Prepare basic metrics for summary generation
        data = {
            "call_count": metrics_row["call_count"],
            "avg_quality_score": float(metrics_row["avg_quality"] or 0),
            "avg_sentiment_score": float(metrics_row["avg_sentiment"] or 0),
            "total_duration_seconds": int(metrics_row["total_duration"] or 0),
        }
        
        # Fetch richer data for better AI insights
        
        # Top performing agents
        top_agents_query = """
            SELECT r.agent_name, r.halo_id, COUNT(*) as call_count,
                   AVG(a.quality_score) as avg_quality, AVG(a.sentiment_score) as avg_sentiment
            FROM ai_call_recordings r
            JOIN ai_call_analysis a ON r.id = a.ai_call_recording_id
            WHERE r.call_date >= %s AND r.call_date <= %s AND r.agent_name IS NOT NULL
        """
        top_params = [request.start_date, request.end_date]
        if request.client_ref:
            top_agents_query += " AND r.client_ref = %s"
            top_params.append(request.client_ref)
        if request.campaign:
            top_agents_query += " AND r.campaign = %s"
            top_params.append(request.campaign)
        top_agents_query += " GROUP BY r.agent_name, r.halo_id HAVING call_count >= 1 ORDER BY avg_quality DESC LIMIT 5"
        
        top_agents = db.fetch_all(top_agents_query, tuple(top_params))
        data["top_agents"] = [
            {"name": row["agent_name"], "calls": row["call_count"], 
             "quality": float(row["avg_quality"] or 0), "sentiment": float(row["avg_sentiment"] or 0)}
            for row in top_agents
        ]
        
        # Bottom performing agents (needing coaching)
        bottom_agents_query = top_agents_query.replace("ORDER BY avg_quality DESC", "ORDER BY avg_quality ASC")
        bottom_agents = db.fetch_all(bottom_agents_query, tuple(top_params))
        data["agents_needing_coaching"] = [
            {"name": row["agent_name"], "calls": row["call_count"],
             "quality": float(row["avg_quality"] or 0), "sentiment": float(row["avg_sentiment"] or 0)}
            for row in bottom_agents
        ]
        
        # Quality distribution
        quality_dist_query = """
            SELECT 
                SUM(CASE WHEN a.quality_score >= 80 THEN 1 ELSE 0 END) as excellent,
                SUM(CASE WHEN a.quality_score >= 60 AND a.quality_score < 80 THEN 1 ELSE 0 END) as good,
                SUM(CASE WHEN a.quality_score >= 40 AND a.quality_score < 60 THEN 1 ELSE 0 END) as average,
                SUM(CASE WHEN a.quality_score < 40 THEN 1 ELSE 0 END) as poor
            FROM ai_call_recordings r
            JOIN ai_call_analysis a ON r.id = a.ai_call_recording_id
            WHERE r.call_date >= %s AND r.call_date <= %s
        """
        qual_params = [request.start_date, request.end_date]
        if request.client_ref:
            quality_dist_query += " AND r.client_ref = %s"
            qual_params.append(request.client_ref)
        if request.campaign:
            quality_dist_query += " AND r.campaign = %s"
            qual_params.append(request.campaign)
        
        qual_dist = db.fetch_one(quality_dist_query, tuple(qual_params))
        if qual_dist:
            data["quality_distribution"] = {
                "excellent_80_plus": int(qual_dist["excellent"] or 0),
                "good_60_79": int(qual_dist["good"] or 0),
                "average_40_59": int(qual_dist["average"] or 0),
                "poor_below_40": int(qual_dist["poor"] or 0),
            }
        
        # Common improvement areas (from the new improvement_areas field)
        improvement_query = """
            SELECT a.improvement_areas
            FROM ai_call_recordings r
            JOIN ai_call_analysis a ON r.id = a.ai_call_recording_id
            WHERE r.call_date >= %s AND r.call_date <= %s
              AND a.improvement_areas IS NOT NULL AND a.improvement_areas != '[]'
        """
        imp_params = [request.start_date, request.end_date]
        if request.client_ref:
            improvement_query += " AND r.client_ref = %s"
            imp_params.append(request.client_ref)
        if request.campaign:
            improvement_query += " AND r.campaign = %s"
            imp_params.append(request.campaign)
        improvement_query += " LIMIT 100"
        
        improvement_rows = db.fetch_all(improvement_query, tuple(imp_params))
        area_counts = {}
        for row in improvement_rows:
            try:
                import json
                areas = json.loads(row["improvement_areas"]) if isinstance(row["improvement_areas"], str) else row["improvement_areas"]
                if isinstance(areas, list):
                    for area in areas:
                        area_name = area.get("area") if isinstance(area, dict) else str(area)
                        if area_name:
                            area_counts[area_name] = area_counts.get(area_name, 0) + 1
            except:
                pass
        
        # Sort by frequency and take top 10
        sorted_areas = sorted(area_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        data["common_improvement_areas"] = [{"area": area, "count": count} for area, count in sorted_areas]
        
        # Compliance issue count
        compliance_query = """
            SELECT COUNT(*) as issue_count
            FROM ai_call_recordings r
            JOIN ai_call_analysis a ON r.id = a.ai_call_recording_id
            WHERE r.call_date >= %s AND r.call_date <= %s
              AND a.compliance_flags IS NOT NULL AND a.compliance_flags != '[]'
        """
        comp_params = [request.start_date, request.end_date]
        if request.client_ref:
            compliance_query += " AND r.client_ref = %s"
            comp_params.append(request.client_ref)
        if request.campaign:
            compliance_query += " AND r.campaign = %s"
            comp_params.append(request.campaign)
        
        compliance_row = db.fetch_one(compliance_query, tuple(comp_params))
        data["calls_with_compliance_issues"] = int(compliance_row["issue_count"] or 0) if compliance_row else 0
        
        # Topic breakdown
        topic_query = """
            SELECT a.key_topics
            FROM ai_call_recordings r
            JOIN ai_call_analysis a ON r.id = a.ai_call_recording_id
            WHERE r.call_date >= %s AND r.call_date <= %s
              AND a.key_topics IS NOT NULL AND a.key_topics != '[]'
        """
        topic_params = [request.start_date, request.end_date]
        if request.client_ref:
            topic_query += " AND r.client_ref = %s"
            topic_params.append(request.client_ref)
        if request.campaign:
            topic_query += " AND r.campaign = %s"
            topic_params.append(request.campaign)
        topic_query += " LIMIT 100"
        
        topic_rows = db.fetch_all(topic_query, tuple(topic_params))
        topic_counts = {}
        for row in topic_rows:
            try:
                import json
                topics = json.loads(row["key_topics"]) if isinstance(row["key_topics"], str) else row["key_topics"]
                if isinstance(topics, list):
                    for topic in topics:
                        topic_name = topic.get("name") if isinstance(topic, dict) else str(topic)
                        if topic_name:
                            topic_counts[topic_name] = topic_counts.get(topic_name, 0) + 1
            except:
                pass
        
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        data["top_topics"] = [{"topic": topic, "count": count} for topic, count in sorted_topics]

        # Use interactive service for AI summary
        # In API mode, proxy to interactive workers; otherwise run locally
        settings = get_settings()
        
        if settings.worker_mode == "api" and settings.interactive_service_url:
            from src.services.interactive_proxy import get_interactive_proxy
            proxy = get_interactive_proxy()
            result = proxy.generate_summary(
                transcript=str(data),  # Pass data as transcript
                summary_type="monthly",
                custom_prompt=f"Generate a summary for: {filters}",
            )
        else:
            from src.services.interactive import get_interactive_service
            service = get_interactive_service()
            result = service.generate_summary(
                data=data,
                summary_type="monthly",
                filters=filters,
            )
        
        # Check for proxy errors and raise HTTPException
        if result.get("error"):
            error_type = result.get("error_type", "unknown")
            error_msg = result.get("error_message") or result.get("summary", "AI service error")
            logger.error(f"Summary error: {error_type} - {error_msg}")
            # Mark as failed so retry is allowed
            summary_lock.update_status(SummaryStatus.FAILED, error_msg)
            raise HTTPException(
                status_code=503 if error_type in ["timeout", "connection_error"] else 500,
                detail={
                    "error": error_msg,
                    "error_type": error_type,
                }
            )
        
        summary_data = {
            "summary": result["summary"],
            "key_insights": result["key_insights"],
            "recommendations": result["recommendations"],
        }
        
        metrics = {
            "call_count": data["call_count"],
            "avg_quality_score": data["avg_quality_score"],
            "avg_sentiment_score": data["avg_sentiment_score"],
            "total_duration_seconds": data["total_duration_seconds"],
        }
        
        # Update the locked summary with completed data
        summary_lock.summary_data = summary_data
        summary_lock.metrics = metrics
        summary_lock.status = SummaryStatus.COMPLETED
        summary_lock.error_message = None
        summary_id = summary_lock.save()
        
        return SummaryGenerateResponse(
            success=True,
            summary_id=summary_id,
            summary_data=summary_data,
            message="Summary generated successfully"
        )

    except HTTPException:
        # Re-raise HTTP exceptions (already handled above)
        raise
    except Exception as e:
        logger.error(f"Summary generation failed: {e}", exc_info=True)
        # Mark as failed so retry is allowed
        summary_lock.update_status(SummaryStatus.FAILED, str(e))
        raise HTTPException(
            status_code=500,
            detail={"error": f"Summary generation failed: {str(e)}"}
        )


# =============================================================================
# Training Data Export/Import
# =============================================================================

class TrainingDataExportResponse(BaseModel):
    """Response for training data export."""
    success: bool
    count: int
    data: List[dict]


@router.get(
    "/api/training-data",
    response_model=TrainingDataExportResponse,
    dependencies=[Depends(verify_token)]
)
async def get_training_data(
    since: Optional[str] = Query(default=None, description="ISO date string, e.g., 2026-01-01"),
    annotation_type: Optional[str] = Query(default=None, description="Filter by annotation type"),
    limit: int = Query(default=1000, le=5000)
):
    """
    Export training data for model fine-tuning.
    
    Returns annotations joined with original analysis and transcripts.
    """
    from src.database import CallAnnotation
    from datetime import datetime
    
    since_date = None
    if since:
        try:
            since_date = datetime.fromisoformat(since)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format: YYYY-MM-DD")
    
    data = CallAnnotation.get_training_data(
        since=since_date,
        annotation_type=annotation_type,
        limit=limit
    )
    
    # Format for training
    training_data = []
    for row in data:
        training_data.append({
            "id": f"annotation_{row['id']}",
            "transcript": row.get("full_transcript") or row.get("redacted_transcript", ""),
            "original_analysis": {
                "summary": row.get("summary", ""),
                "sentiment_score": float(row.get("sentiment_score") or 0),
                "sentiment_label": row.get("sentiment_label", "neutral"),
                "quality_score": float(row.get("quality_score") or 0),
            },
            "corrected_analysis": {
                "field": row.get("field_name"),
                "original_value": row.get("original_value"),
                "corrected_value": row.get("corrected_value"),
            },
            "annotation_metadata": {
                "type": row.get("annotation_type"),
                "notes": row.get("notes"),
                "created_at": row.get("created_at").isoformat() if row.get("created_at") else None,
            }
        })
    
    return TrainingDataExportResponse(
        success=True,
        count=len(training_data),
        data=training_data
    )


class TrainingImportRequest(BaseModel):
    """Request to import training data."""
    model_type: str = "call_analysis"
    training_data: List[dict]
    options: Optional[dict] = None


class TrainingImportResponse(BaseModel):
    """Response from training import."""
    success: bool
    job_id: Optional[str] = None
    status: str
    message: str


@router.post(
    "/api/training/import",
    response_model=TrainingImportResponse,
    dependencies=[Depends(verify_token)]
)
async def import_training_data(request: TrainingImportRequest):
    """
    Import training data and optionally trigger fine-tuning.
    
    Note: This endpoint is for importing external data. The primary training
    flow uses annotations from ai_call_annotations via the nightly CronJob.
    
    Only the call_analysis model supports fine-tuning.
    The chat model uses base training only.
    """
    import uuid
    
    if request.model_type != "call_analysis":
        raise HTTPException(
            status_code=400, 
            detail="Only 'call_analysis' model supports fine-tuning. Chat model uses base training only."
        )
    
    if not request.training_data:
        raise HTTPException(status_code=400, detail="No training data provided")
    
    # Generate job ID
    job_id = f"training_{uuid.uuid4().hex[:12]}"
    
    # Log the import - actual training runs via nightly CronJob using database annotations
    # This endpoint is primarily for data import, not triggering training
    logger.info(f"Training data import {job_id}: Received {len(request.training_data)} samples")
    
    # Import data into ai_call_annotations if needed
    # For now, just acknowledge - training uses existing annotations from the UI
    
    return TrainingImportResponse(
        success=True,
        job_id=job_id,
        status="acknowledged",
        message=f"Received {len(request.training_data)} samples. Training runs nightly at 2 AM using annotations from ai_call_annotations table. Use the UI to create annotations, or POST to /api/annotations."
    )


class TrainingStatusResponse(BaseModel):
    """Response with training status."""
    adapter_exists: bool
    adapter_name: str
    current_version: Optional[str] = None
    trained_at: Optional[str] = None
    samples_used: Optional[int] = None
    training_loss: Optional[float] = None
    base_model: Optional[str] = None
    new_annotations_since_training: int = 0
    ready_for_training: bool = False
    ready_reason: str = ""
    available_versions: List[Dict[str, Any]] = []
    # Training in-progress tracking
    training_in_progress: bool = False
    training_started_at: Optional[str] = None
    training_adapter_name: Optional[str] = None


@router.get(
    "/api/training/status",
    response_model=TrainingStatusResponse,
    dependencies=[Depends(verify_token)]
)
async def get_training_status(
    adapter_name: str = Query(default="call-analysis", description="Adapter name to check")
):
    """
    Get current training/adapter status.
    
    Shows when the adapter was last trained and if new annotations are available.
    Includes list of all available versions for rollback.
    Also checks if training is currently in progress.
    """
    from src.services.trainer import TrainingService
    
    try:
        trainer = TrainingService(adapter_name=adapter_name)
        metadata = trainer.get_adapter_info()
        new_count = trainer.count_new_annotations()
        should_train, reason = trainer.should_train(min_new_annotations=1)
        versions = trainer.list_versions()
        current_version = trainer.get_current_version_name()
        
        # Check if training is in progress
        in_progress, lock_info = TrainingService.is_training_in_progress()
        
        if metadata:
            return TrainingStatusResponse(
                adapter_exists=True,
                adapter_name=adapter_name,
                current_version=current_version,
                trained_at=metadata.get("trained_at"),
                samples_used=metadata.get("samples_used"),
                training_loss=metadata.get("training_loss"),
                base_model=metadata.get("base_model"),
                new_annotations_since_training=new_count,
                ready_for_training=should_train and not in_progress,
                ready_reason=reason if not in_progress else "Training already in progress",
                available_versions=versions,
                training_in_progress=in_progress,
                training_started_at=lock_info.get("started_at") if lock_info else None,
                training_adapter_name=lock_info.get("adapter_name") if lock_info else None,
            )
        else:
            return TrainingStatusResponse(
                adapter_exists=False,
                adapter_name=adapter_name,
                new_annotations_since_training=new_count,
                ready_for_training=should_train and not in_progress,
                ready_reason=reason if not in_progress else "Training already in progress",
                available_versions=versions,
                training_in_progress=in_progress,
                training_started_at=lock_info.get("started_at") if lock_info else None,
                training_adapter_name=lock_info.get("adapter_name") if lock_info else None,
            )
    except Exception as e:
        logger.error(f"Failed to get training status: {e}")
        return TrainingStatusResponse(
            adapter_exists=False,
            adapter_name=adapter_name,
            ready_reason=f"Error: {str(e)}",
        )


class TrainingTriggerRequest(BaseModel):
    """Request to trigger training."""
    adapter_name: str = "call-analysis"
    force: bool = False  # If True, train even if no new annotations
    max_samples: int = 2500
    epochs: int = 3


class TrainingTriggerResponse(BaseModel):
    """Response from training trigger."""
    success: bool
    message: str
    adapter_name: Optional[str] = None
    version: Optional[str] = None  # New versioned name
    samples_used: Optional[int] = None
    training_loss: Optional[float] = None
    training_time_minutes: Optional[float] = None
    error: Optional[str] = None


@router.post(
    "/api/training/start",
    response_model=TrainingTriggerResponse,
    dependencies=[Depends(verify_token)]
)
async def trigger_training(request: TrainingTriggerRequest, background_tasks: BackgroundTasks):
    """
    Trigger model fine-tuning immediately.
    
    This runs training synchronously and may take 30-60+ minutes.
    For production, use the nightly CronJob instead.
    
    Set force=true to train even if there are no new annotations.
    """
    from src.services.trainer import TrainingService, TRAINING_AVAILABLE
    
    if not TRAINING_AVAILABLE:
        return TrainingTriggerResponse(
            success=False,
            message="Training dependencies not installed",
            error="Install peft and datasets: pip install peft datasets"
        )
    
    try:
        trainer = TrainingService(adapter_name=request.adapter_name)
        
        # Check if training is needed
        if not request.force:
            should_train, reason = trainer.should_train(min_new_annotations=1)
            if not should_train:
                return TrainingTriggerResponse(
                    success=False,
                    message=f"Training skipped: {reason}",
                    adapter_name=request.adapter_name,
                )
        
        logger.info(f"Starting training for adapter '{request.adapter_name}' (force={request.force})")
        
        # Run training (this blocks - may take a long time)
        result = trainer.train(
            max_samples=request.max_samples,
            epochs=request.epochs,
        )
        
        if result["success"]:
            return TrainingTriggerResponse(
                success=True,
                message="Training completed successfully",
                adapter_name=result.get("adapter_name"),
                version=result.get("version"),
                samples_used=result.get("samples_used"),
                training_loss=result.get("training_loss"),
                training_time_minutes=result.get("training_time_minutes"),
            )
        else:
            return TrainingTriggerResponse(
                success=False,
                message="Training failed",
                adapter_name=request.adapter_name,
                error=result.get("error"),
            )
            
    except Exception as e:
        logger.exception(f"Training failed: {e}")
        return TrainingTriggerResponse(
            success=False,
            message="Training failed with exception",
            adapter_name=request.adapter_name,
            error=str(e),
        )


class PromoteVersionRequest(BaseModel):
    """Request to promote an adapter version."""
    adapter_name: str = "call-analysis"
    version: str  # Version name to promote (e.g., "call-analysis-20260121-0200")


class PromoteVersionResponse(BaseModel):
    """Response from promoting a version."""
    success: bool
    message: str
    adapter_name: str
    version: str


@router.post(
    "/api/training/promote",
    response_model=PromoteVersionResponse,
    dependencies=[Depends(verify_token)]
)
async def promote_adapter_version(request: PromoteVersionRequest):
    """
    Promote a specific adapter version to be the current active version.
    
    Use this for:
    - Zero-downtime switchover after training completes
    - Rolling back to a previous version if issues are detected
    
    Note: After promoting, you may need to restart vLLM or update ANALYSIS_ADAPTER_NAME
    in the ConfigMap to point to the new version for it to take effect.
    """
    from src.services.trainer import TrainingService
    
    try:
        trainer = TrainingService(adapter_name=request.adapter_name)
        
        success = trainer.promote_version(request.version)
        
        if success:
            return PromoteVersionResponse(
                success=True,
                message=f"Version '{request.version}' is now the current active adapter",
                adapter_name=request.adapter_name,
                version=request.version,
            )
        else:
            return PromoteVersionResponse(
                success=False,
                message=f"Version '{request.version}' not found",
                adapter_name=request.adapter_name,
                version=request.version,
            )
    except Exception as e:
        logger.error(f"Failed to promote version: {e}")
        return PromoteVersionResponse(
            success=False,
            message=f"Error: {str(e)}",
            adapter_name=request.adapter_name,
            version=request.version,
        )


class CleanupVersionsRequest(BaseModel):
    """Request to cleanup old adapter versions."""
    adapter_name: str = "call-analysis"
    keep: int = 5  # Number of versions to keep


class CleanupVersionsResponse(BaseModel):
    """Response from cleanup operation."""
    success: bool
    message: str
    versions_removed: int


@router.post(
    "/api/training/cleanup",
    response_model=CleanupVersionsResponse,
    dependencies=[Depends(verify_token)]
)
async def cleanup_old_versions(request: CleanupVersionsRequest):
    """
    Remove old adapter versions, keeping the newest N.
    
    The current active version is never removed, even if it's older.
    This helps manage disk space on the NFS storage.
    """
    from src.services.trainer import TrainingService
    
    try:
        trainer = TrainingService(adapter_name=request.adapter_name)
        removed = trainer.cleanup_old_versions(keep=request.keep)
        
        return CleanupVersionsResponse(
            success=True,
            message=f"Cleaned up {removed} old version(s), kept newest {request.keep}",
            versions_removed=removed,
        )
    except Exception as e:
        logger.error(f"Failed to cleanup versions: {e}")
        return CleanupVersionsResponse(
            success=False,
            message=f"Error: {str(e)}",
            versions_removed=0,
        )


# =============================================================================
# Client Configuration Management
# =============================================================================

class ClientConfigRequest(BaseModel):
    """Request to create/update client config."""
    client_ref: Optional[str] = None
    campaign: Optional[str] = None
    config_type: str  # 'topics', 'agent_actions', 'performance_rubric', 'prompt', 'analysis_mode'
    config_data: dict
    is_active: bool = True


class ClientConfigResponse(BaseModel):
    """Response with client config."""
    id: int
    client_ref: Optional[str]
    campaign: Optional[str] = None
    config_type: str
    config_data: dict
    is_active: bool


@router.get(
    "/api/config",
    dependencies=[Depends(verify_token)]
)
async def get_client_config(
    client_ref: Optional[str] = Query(default=None),
    campaign: Optional[str] = Query(default=None),
    config_type: Optional[str] = Query(default=None)
):
    """
    Get client configuration with fallback hierarchy.
    
    Lookup order:
    1. Campaign-specific (client_ref + campaign)
    2. Client-specific (client_ref only)
    3. Global (no client_ref or campaign)
    4. File-based defaults (call_analysis_config.json)
    """
    from src.database import ClientConfig
    import os
    
    # Load file-based defaults as final fallback
    def get_file_defaults() -> dict:
        config_path = "call_analysis_config.json"
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}
    
    file_defaults = get_file_defaults()
    
    if config_type:
        config_data = ClientConfig.get_config(client_ref, config_type, campaign)
        if config_data is None:
            # Fall back to file-based config
            config_data = file_defaults.get(config_type)
        if config_data is None:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        return {"config_type": config_type, "config_data": config_data, "source": "database" if ClientConfig.get_config(client_ref, config_type, campaign) else "file"}
    
    # Return all config types
    config_types = ["topics", "agent_actions", "performance_rubric", "prompt", "analysis_mode"]
    result = {}
    
    for ct in config_types:
        config_data = ClientConfig.get_config(client_ref, ct, campaign)
        if config_data:
            result[ct] = {"data": config_data, "source": "database"}
        elif file_defaults.get(ct):
            result[ct] = {"data": file_defaults[ct], "source": "file"}
    
    return {"client_ref": client_ref, "campaign": campaign, "configs": result}


@router.post(
    "/api/config",
    response_model=ClientConfigResponse,
    dependencies=[Depends(verify_token)]
)
async def create_client_config(request: ClientConfigRequest):
    """Create or update a client/campaign configuration."""
    from src.database import ClientConfig
    
    valid_types = ["topics", "agent_actions", "performance_rubric", "prompt", "analysis_mode"]
    if request.config_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid config_type. Must be one of: {valid_types}"
        )
    
    config = ClientConfig(
        client_ref=request.client_ref,
        campaign=request.campaign,
        config_type=request.config_type,
        config_data=request.config_data,
        is_active=request.is_active,
    )
    config_id = config.save()
    
    return ClientConfigResponse(
        id=config_id,
        client_ref=request.client_ref,
        campaign=request.campaign,
        config_type=request.config_type,
        config_data=request.config_data,
        is_active=request.is_active,
    )


@router.delete(
    "/api/config/{config_id}",
    dependencies=[Depends(verify_token)]
)
async def delete_client_config(config_id: int):
    """Deactivate a client configuration."""
    db = get_db_connection()
    
    result = db.execute("""
        UPDATE ai_configs
        SET is_active = FALSE, updated_at = NOW()
        WHERE id = %s
    """, (config_id,))
    
    if result == 0:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    return {"message": "Configuration deactivated"}


# =============================================================================
# Three-Tier Configuration API (v2)
# =============================================================================

class UniversalConfigRequest(BaseModel):
    """Request to save universal config."""
    topics: List[str] = []
    agent_actions: List[str] = []
    performance_rubric: List[str] = []


class UniversalConfigResponse(BaseModel):
    """Response with universal config."""
    topics: List[str]
    agent_actions: List[str]
    performance_rubric: List[str]
    quality_signals: dict = {}
    source: str = "database"


class ClientContextRequest(BaseModel):
    """Request to save client context config."""
    client_ref: str
    config_data: dict  # Dynamic structure


class ClientContextResponse(BaseModel):
    """Response with client context."""
    client_ref: str
    config_data: dict
    prompt_context: str  # How it appears in prompts
    source: str = "database"  # "database" or "universal_fallback"
    # Effective arrays (with overrides applied)
    effective_topics: List[str] = []
    effective_agent_actions: List[str] = []
    effective_performance_rubric: List[str] = []


class CampaignTypeRequest(BaseModel):
    """Request to save campaign type config."""
    campaign_type: str
    config_data: dict  # Dynamic structure


class CampaignTypeResponse(BaseModel):
    """Response with campaign type config."""
    campaign_type: str
    config_data: dict
    prompt_context: str  # How it appears in prompts
    source: str = "database"  # "database" or "universal_fallback"
    # Effective arrays (with overrides applied)
    effective_topics: List[str] = []
    effective_agent_actions: List[str] = []
    effective_performance_rubric: List[str] = []


class DirectionRequest(BaseModel):
    """Request to save direction config."""
    direction: str  # inbound, outbound, sms_handraiser
    config_data: dict  # Dynamic structure


class DirectionResponse(BaseModel):
    """Response with direction config."""
    direction: str
    config_data: dict = {}
    prompt_context: str = ""  # How it appears in prompts


class MergedConfigResponse(BaseModel):
    """Response with merged four-tier config."""
    universal: UniversalConfigResponse  # Also called 'global' in new terminology
    client: Optional[ClientContextResponse] = None
    campaign_type: Optional[CampaignTypeResponse] = None
    direction: Optional[DirectionResponse] = None
    prompt_context: str  # Combined context for LLM prompts
    quality_signals: dict = {}  # Quality assessment framework from global config
    # Effective arrays (with overrides applied)
    effective_topics: List[str] = []
    effective_agent_actions: List[str] = []
    effective_performance_rubric: List[str] = []


@router.post(
    "/api/v2/config/universal",
    response_model=UniversalConfigResponse,
    dependencies=[Depends(verify_token)],
    tags=["Configuration v2"]
)
async def save_universal_config(request: UniversalConfigRequest):
    """
    Save universal configuration.
    
    This updates the standard analysis framework used for all calls.
    """
    from src.services.config import get_config_service, UniversalConfig
    
    config_service = get_config_service()
    
    config = UniversalConfig(
        topics=request.topics,
        agent_actions=request.agent_actions,
        performance_rubric=request.performance_rubric
    )
    
    config_service.save_universal_config(config)
    
    return UniversalConfigResponse(
        topics=config.topics,
        agent_actions=config.agent_actions,
        performance_rubric=config.performance_rubric,
        source="database"
    )


@router.get(
    "/api/v2/config/clients",
    dependencies=[Depends(verify_token)],
    tags=["Configuration v2"]
)
async def list_clients_with_config():
    """List all clients that have configurations."""
    from src.services.config import get_config_service
    
    config_service = get_config_service()
    clients = config_service.list_clients()
    
    return {"clients": clients, "count": len(clients)}


@router.post(
    "/api/v2/config/client/{client_ref}",
    response_model=ClientContextResponse,
    dependencies=[Depends(verify_token)],
    tags=["Configuration v2"]
)
async def save_client_context(client_ref: str, request: ClientContextRequest):
    """
    Save client-specific context configuration.
    
    Structure is flexible - include any fields that help contextualise analysis.
    
    Common fields:
    - organisation_name: Name of the charity/organisation
    - organisation_type: Type (e.g., "charity", "nonprofit")
    - mission: Mission statement
    - tone_guidelines: How agents should communicate
    - compliance_notes: Special compliance requirements
    """
    from src.services.config import get_config_service, ClientConfig
    
    if request.client_ref != client_ref:
        raise HTTPException(status_code=400, detail="client_ref in path and body must match")
    
    config_service = get_config_service()
    config_service.save_client_config(client_ref, request.config_data)
    
    # Retrieve saved config
    client = config_service.get_client_config(client_ref, use_cache=False)
    
    return ClientContextResponse(
        client_ref=client.client_ref,
        config_data=client.config_data,
        prompt_context=client.to_prompt_context()
    )


@router.delete(
    "/api/v2/config/client/{client_ref}",
    dependencies=[Depends(verify_token)],
    tags=["Configuration v2"]
)
async def delete_client_context(client_ref: str):
    """Deactivate client configuration."""
    db = get_db_connection()
    
    result = db.execute("""
        UPDATE ai_configs
        SET is_active = FALSE, updated_at = NOW()
        WHERE config_tier = 'client' AND client_ref = %s
    """, (client_ref,))
    
    if result == 0:
        raise HTTPException(status_code=404, detail=f"No configuration found for client '{client_ref}'")
    
    return {"message": f"Configuration for client '{client_ref}' deactivated"}


@router.get(
    "/api/v2/config/campaign-types",
    dependencies=[Depends(verify_token)],
    tags=["Configuration v2"]
)
async def list_campaign_types():
    """List all campaign types with configurations."""
    from src.services.config import get_config_service
    
    config_service = get_config_service()
    campaign_types = config_service.list_campaign_types()
    
    return {"campaign_types": campaign_types, "count": len(campaign_types)}


@router.post(
    "/api/v2/config/campaign-type/{campaign_type}",
    response_model=CampaignTypeResponse,
    dependencies=[Depends(verify_token)],
    tags=["Configuration v2"]
)
async def save_campaign_type_config(campaign_type: str, request: CampaignTypeRequest):
    """
    Save campaign type configuration.
    
    Campaign types are reusable templates that define call expectations.
    
    Common fields:
    - description: What this campaign type is
    - goals: List of goals for this type of call
    - success_criteria: What defines success
    - key_metrics: What to measure
    - agent_expectations: Expectations for agent behaviour
    """
    from src.services.config import get_config_service
    
    if request.campaign_type != campaign_type:
        raise HTTPException(status_code=400, detail="campaign_type in path and body must match")
    
    config_service = get_config_service()
    config_service.save_campaign_type_config(campaign_type, request.config_data)
    
    # Retrieve saved config
    campaign = config_service.get_campaign_type_config(campaign_type, use_cache=False)
    
    return CampaignTypeResponse(
        campaign_type=campaign.campaign_type,
        config_data=campaign.config_data,
        prompt_context=campaign.to_prompt_context()
    )


@router.delete(
    "/api/v2/config/campaign-type/{campaign_type}",
    dependencies=[Depends(verify_token)],
    tags=["Configuration v2"]
)
async def delete_campaign_type_config(campaign_type: str):
    """Deactivate campaign type configuration."""
    db = get_db_connection()
    
    result = db.execute("""
        UPDATE ai_configs
        SET is_active = FALSE, updated_at = NOW()
        WHERE config_tier IN ('campaign', 'campaign_type') AND campaign_type = %s
    """, (campaign_type,))
    
    if result == 0:
        raise HTTPException(status_code=404, detail=f"No configuration found for campaign type '{campaign_type}'")
    
    return {"message": f"Configuration for campaign type '{campaign_type}' deactivated"}


# =============================================================================
# Direction Configuration Endpoints (NEW - Four Tier System)
# =============================================================================


@router.get(
    "/api/v2/config/directions",
    response_model=List[str],
    dependencies=[Depends(verify_token)],
    tags=["Configuration v2"]
)
async def list_directions_with_config():
    """List all directions that have configuration."""
    from src.services.config import get_config_service
    config_service = get_config_service()
    return config_service.list_directions()

@router.get(
    "/api/v2/config/direction/{direction}",
    response_model=DirectionResponse,
    dependencies=[Depends(verify_token)],
    tags=["Configuration v2"]
)
async def get_direction_config(direction: str):
    """Get configuration for a specific direction."""
    from src.services.config import get_config_service
    config_service = get_config_service()
    config = config_service.get_direction_config(direction)
    
    if not config:
        raise HTTPException(status_code=404, detail=f"No configuration found for direction '{direction}'")
    
    return DirectionResponse(
        direction=config.direction,
        config_data=config.config_data,
        prompt_context=config.to_prompt_context()
    )


@router.post(
    "/api/v2/config/direction/{direction}",
    response_model=DirectionResponse,
    dependencies=[Depends(verify_token)],
    tags=["Configuration v2"]
)
async def save_direction_config(direction: str, request: DirectionRequest):
    """Save or update direction configuration."""
    from src.services.config import get_config_service
    config_service = get_config_service()
    
    # Validate direction value
    valid_directions = ["inbound", "outbound", "sms_handraiser"]
    if direction not in valid_directions:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid direction '{direction}'. Must be one of: {', '.join(valid_directions)}"
        )
    
    config_service.save_direction_config(direction, request.config_data)
    
    # Fetch the saved config
    config = config_service.get_direction_config(direction, use_cache=False)
    
    return DirectionResponse(
        direction=config.direction,
        config_data=config.config_data,
        prompt_context=config.to_prompt_context()
    )


@router.delete(
    "/api/v2/config/direction/{direction}",
    dependencies=[Depends(verify_token)],
    tags=["Configuration v2"]
)
async def delete_direction_config(direction: str):
    """Deactivate direction configuration."""
    db = get_db_connection()
    
    result = db.execute("""
        UPDATE ai_configs
        SET is_active = FALSE, updated_at = NOW()
        WHERE config_tier = 'direction' AND direction = %s
    """, (direction,))
    
    if result == 0:
        raise HTTPException(status_code=404, detail=f"No configuration found for direction '{direction}'")
    
    return {"message": f"Configuration for direction '{direction}' deactivated"}


@router.get(
    "/api/v2/config/merged",
    response_model=MergedConfigResponse,
    dependencies=[Depends(verify_token)],
    tags=["Configuration v2"]
)
async def get_merged_config(
    client_ref: Optional[str] = Query(default=None),
    campaign_type: Optional[str] = Query(default=None),
    direction: Optional[str] = Query(default=None)
):
    """
    Get merged four-tier configuration for a specific context.
    
    Combines all applicable tiers:
    1. Global config (always included, fallback to file)
    2. Campaign config (if campaign_type provided)
    3. Direction config (if direction provided)
    4. Client config (if client_ref provided)
    
    Returns the combined prompt_context that would be used in LLM prompts.
    Override priority: client > direction > campaign > global
    """
    from src.services.config import get_config_service
    
    config_service = get_config_service()
    merged = config_service.get_merged_config(
        campaign_type=campaign_type,
        direction=direction,
        client_ref=client_ref
    )
    
    global_cfg = merged["global_config"]
    campaign_cfg = merged.get("campaign_config")
    direction_cfg = merged.get("direction_config")
    client_cfg = merged.get("client_config")
    
    return MergedConfigResponse(
        universal=UniversalConfigResponse(
            topics=global_cfg.topics,
            agent_actions=global_cfg.agent_actions,
            performance_rubric=global_cfg.performance_rubric,
            quality_signals=global_cfg.quality_signals,
            source="database" if global_cfg.topics else "file"
        ),
        quality_signals=global_cfg.quality_signals,
        campaign_type=CampaignTypeResponse(
            campaign_type=campaign_cfg.campaign_type,
            config_data=campaign_cfg.config_data,
            prompt_context=campaign_cfg.to_prompt_context()
        ) if campaign_cfg else None,
        direction=DirectionResponse(
            direction=direction_cfg.direction,
            config_data=direction_cfg.config_data,
            prompt_context=direction_cfg.to_prompt_context()
        ) if direction_cfg else None,
        client=ClientContextResponse(
            client_ref=client_cfg.client_ref,
            config_data=client_cfg.config_data,
            prompt_context=client_cfg.to_prompt_context()
        ) if client_cfg else None,
        prompt_context=merged.get("prompt_context", ""),
        # Effective arrays (with overrides applied)
        effective_topics=merged.get("topics", []),
        effective_agent_actions=merged.get("agent_actions", []),
        effective_performance_rubric=merged.get("performance_rubric", [])
    )


# =============================================================================
# Enhanced Chat Endpoint with Full Features
# =============================================================================

class EnhancedChatRequest(BaseModel):
    """Enhanced chat request matching specification."""
    message: str
    conversation_id: Optional[int] = None
    # User object for personalization and conversation creation
    user: Optional[ChatUser] = None
    # Legacy field - use user.id instead
    user_id: Optional[int] = None
    feature: str = "call_quality"
    filters: Optional[dict] = None
    history: Optional[List[dict]] = None


class EnhancedChatResponse(BaseModel):
    """Enhanced chat response."""
    success: bool
    response: str
    conversation_id: Optional[int] = None
    message_id: Optional[int] = None
    metadata: Optional[dict] = None
    error: Optional[str] = None
    error_type: Optional[str] = None


# Request deduplication cache to prevent double saves
# Key: (user_id, message_hash, conversation_id) -> (timestamp, response)
import time as _time
import hashlib as _hashlib
_chat_request_cache: Dict[str, tuple] = {}
_CHAT_DEDUP_WINDOW = 5  # seconds


@router.post(
    "/api/chat",
    response_model=EnhancedChatResponse,
    dependencies=[Depends(verify_token)]
)
async def enhanced_chat(request: EnhancedChatRequest):
    """
    Chat with the AI about call data.
    
    Handles both new conversations and existing ones atomically:
    - New conversation: Creates conversation + messages in a transaction
    - Existing conversation: Appends messages to existing conversation
    
    If AI processing fails, the transaction rolls back - no orphaned records.
    Supports user personalization when user object is provided.
    """
    from src.database import ChatConversation, ChatMessage
    from src.services.interactive import get_interactive_service
    import json
    import uuid
    
    # Generate unique request ID for tracing
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[CHAT-{request_id}] enhanced_chat called with message: {request.message[:50]}...")
    
    # Deduplication check - prevent double processing of same message within window
    user_id = request.user.id if request.user else request.user_id
    msg_hash = _hashlib.md5(request.message.encode()).hexdigest()[:16]
    dedup_key = f"{user_id}:{msg_hash}:{request.conversation_id or 'new'}"
    
    current_time = _time.time()
    if dedup_key in _chat_request_cache:
        cached_time, cached_response = _chat_request_cache[dedup_key]
        if current_time - cached_time < _CHAT_DEDUP_WINDOW:
            logger.warning(f"[CHAT-{request_id}] Duplicate request detected (key={dedup_key}), returning cached response")
            return cached_response
    
    # Clean old entries from cache (simple cleanup)
    expired_keys = [k for k, (t, _) in _chat_request_cache.items() if current_time - t > 60]
    for k in expired_keys:
        del _chat_request_cache[k]
    
    db = get_db_connection()
    conversation_id = None
    is_new_conversation = False
    assistant_message_id = None
    context_calls = 0
    
    # Extract user info for personalization
    user_id = request.user.id if request.user else request.user_id
    user_name = request.user.name if request.user else None
    
    try:
        # Start transaction for atomic operations
        db.execute("START TRANSACTION")
        
        # Get or create conversation
        if request.conversation_id:
            conversation = ChatConversation.get_by_id(request.conversation_id)
            if not conversation:
                db.execute("ROLLBACK")
                raise HTTPException(status_code=404, detail="Conversation not found")
            conversation_id = conversation.id
        else:
            # New conversation - create atomically with messages
            if not user_id:
                db.execute("ROLLBACK")
                raise HTTPException(
                    status_code=400,
                    detail="user or user_id is required when creating a new conversation"
                )
            
            is_new_conversation = True
            conversation = ChatConversation(
                user_id=user_id,
                feature=request.feature,
                filters=request.filters,
                title="New Conversation",  # Pulse will update this later
            )
            conversation_id = conversation.save()
            logger.info(f"[CHAT-{request_id}] Created new conversation {conversation_id} for user {user_id} ({user_name})")
        
        # Save user message
        logger.info(f"[CHAT-{request_id}] Saving user message to conversation {conversation_id}")
        user_message = ChatMessage(
            ai_chat_conversation_id=conversation_id,
            role="user",
            content=request.message,
        )
        user_message.save()
        logger.info(f"[CHAT-{request_id}] User message saved with id {user_message.id}")
        
        # Build context from filtered data
        context_parts = []
        
        # Add user context for personalization
        if user_name:
            context_parts.append(f"You are speaking with {user_name}. Address them by name to make responses personal and friendly.")
        
        if request.filters:
            # Get aggregated metrics and recent calls
            query = """
                SELECT 
                    COUNT(*) as cnt, 
                    AVG(a.quality_score) as avg_quality, 
                    AVG(a.sentiment_score) as avg_sentiment,
                    SUM(r.duration_seconds) as total_duration
                FROM ai_call_recordings r 
                LEFT JOIN ai_call_analysis a ON r.id = a.ai_call_recording_id 
                WHERE 1=1
            """
            params = []
            
            if request.filters.get("client_ref"):
                query += " AND r.client_ref = %s"
                params.append(request.filters["client_ref"])
            
            if request.filters.get("start_date"):
                query += " AND r.call_date >= %s"
                params.append(request.filters["start_date"])
            
            if request.filters.get("end_date"):
                query += " AND r.call_date <= %s"
                params.append(request.filters["end_date"])
            
            result = db.fetch_one(query, tuple(params))
            if result:
                context_calls = result["cnt"] or 0
                avg_quality = result["avg_quality"] or 0
                avg_sentiment = result["avg_sentiment"] or 0
                total_duration = result["total_duration"] or 0
                
                context_parts.append(f"**Available Data:** {context_calls} calls analysed")
                if request.filters.get("start_date") or request.filters.get("end_date"):
                    from datetime import datetime
                    start_uk = datetime.strptime(request.filters.get("start_date", ""), "%Y-%m-%d").strftime("%d/%m/%Y") if request.filters.get("start_date") else "start"
                    end_uk = datetime.strptime(request.filters.get("end_date", ""), "%Y-%m-%d").strftime("%d/%m/%Y") if request.filters.get("end_date") else "now"
                    context_parts.append(f"Period: {start_uk} to {end_uk}")
                if request.filters.get("client_ref"):
                    context_parts.append(f"Client: {request.filters['client_ref']}")
                context_parts.append(f"Average Quality Score: {avg_quality:.1f}%")
                context_parts.append(f"Average Sentiment: {avg_sentiment:.1f}/10")
                context_parts.append(f"Total Duration: {total_duration // 60:.0f} minutes")
                
                # Get topic breakdown if available
                topic_query = """
                    SELECT key_topics
                    FROM ai_call_analysis a
                    JOIN ai_call_recordings r ON a.ai_call_recording_id = r.id
                    WHERE 1=1 AND a.key_topics IS NOT NULL
                """
                if request.filters.get("start_date"):
                    topic_query += " AND r.call_date >= %s"
                if request.filters.get("end_date"):
                    topic_query += " AND r.call_date <= %s"
                
                topic_query += " LIMIT 50"
                
                topics_results = db.fetch_all(topic_query, tuple([f for f in [request.filters.get("start_date"), request.filters.get("end_date")] if f]))
                if topics_results:
                    all_topics = {}
                    for row in topics_results:
                        if row.get("key_topics"):
                            try:
                                topics = json.loads(row["key_topics"]) if isinstance(row["key_topics"], str) else row["key_topics"]
                                if isinstance(topics, list):
                                    for topic in topics:
                                        topic_name = topic.get("name") if isinstance(topic, dict) else str(topic)
                                        all_topics[topic_name] = all_topics.get(topic_name, 0) + 1
                            except:
                                pass
                    
                    if all_topics:
                        top_topics = sorted(all_topics.items(), key=lambda x: x[1], reverse=True)[:5]
                        context_parts.append(f"\n**Common Topics:** {', '.join([f'{t[0]} ({t[1]} calls)' for t in top_topics])}")
        
        # Use interactive service for AI response
        context = "\n".join(context_parts) if context_parts else None
        settings = get_settings()
        
        if settings.worker_mode == "api" and settings.interactive_service_url:
            from src.services.interactive_proxy import get_interactive_proxy
            proxy = get_interactive_proxy()
            result = proxy.chat_with_functions(
                message=request.message,
                user_name=user_name,
                filters=request.filters,
            )
        else:
            service = get_interactive_service()
            result = service.chat_with_functions(
                message=request.message,
                user_name=user_name,
                filters=request.filters,
            )
        
        # Check for AI errors - rollback if failed
        if result.get("error"):
            db.execute("ROLLBACK")
            error_type = result.get("error_type", "unknown")
            error_msg = result.get("error_message") or result.get("response", "AI service error")
            logger.error(f"Chat error: {error_type} - {error_msg}")
            
            return EnhancedChatResponse(
                success=False,
                response="",
                conversation_id=conversation_id if not is_new_conversation else None,
                error=error_msg,
                error_type=error_type,
            )
        
        response_text = result["response"]
        
        # Save assistant message
        logger.info(f"[CHAT-{request_id}] Saving assistant message to conversation {conversation_id}")
        assistant_message = ChatMessage(
            ai_chat_conversation_id=conversation_id,
            role="assistant",
            content=response_text,
            metadata={
                "context_calls": context_calls,
                "function_calls": result.get("function_calls", []),
                "request_id": request_id,  # Track which request saved this
            },
        )
        assistant_message_id = assistant_message.save()
        logger.info(f"[CHAT-{request_id}] Assistant message saved with id {assistant_message_id}")
        
        # Commit transaction - all records saved atomically
        db.execute("COMMIT")
        
        response = EnhancedChatResponse(
            success=True,
            response=response_text,
            conversation_id=conversation_id,
            message_id=assistant_message_id,
            metadata={
                "model": result.get("model", "unknown"),
                "tokens_used": result.get("tokens_used", 0),
                "context_calls_analysed": context_calls,
                "processing_time": result.get("processing_time", 0),
                "function_calls": len(result.get("function_calls", [])),
                "format": "markdown",
            }
        )
        
        # Cache the response for deduplication
        _chat_request_cache[dedup_key] = (current_time, response)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        try:
            db.execute("ROLLBACK")
        except:
            pass
        logger.error(f"Enhanced chat endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "error_type": "internal_error",
            }
        )


# =============================================================================
# Internal Endpoints (for worker-to-worker proxying)
# =============================================================================
# These endpoints are called by the API pod proxy to delegate AI workloads
# to interactive worker pods. They bypass the database operations that
# the public endpoints do.

class InternalChatRequest(BaseModel):
    """Internal chat request (used for worker proxying)."""
    message: str
    context: Optional[str] = None
    conversation_history: Optional[List[dict]] = None
    max_tokens: int = 512


class InternalChatResponse(BaseModel):
    """Internal chat response."""
    response: str = ""
    generation_time: float = 0.0
    model: str = ""
    error: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class InternalSummaryRequest(BaseModel):
    """Internal summary request."""
    transcript: str
    summary_type: str = "brief"
    custom_prompt: Optional[str] = None


class InternalSummaryResponse(BaseModel):
    """Internal summary response."""
    summary: str = ""
    generation_time: float = 0.0
    error: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class InternalChatFunctionsRequest(BaseModel):
    """Internal chat with functions request (SQL Agent mode)."""
    message: str
    user_name: Optional[str] = None
    filters: Optional[dict] = None
    conversation_history: Optional[List[dict]] = None
    max_tokens: int = 512
    use_functions: bool = True


class InternalChatFunctionsResponse(BaseModel):
    """Internal chat with functions response."""
    response: str = ""
    generation_time: float = 0.0
    model: str = ""
    function_calls: List[dict] = []
    error: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None


class InternalHealthResponse(BaseModel):
    """Internal health response for interactive service."""
    status: str
    model_loaded: bool = False
    device: str = ""
    worker_id: str = ""


class InternalTranscribeRequest(BaseModel):
    """Request for internal transcription service."""
    audio_base64: str = Field(..., description="Base64-encoded audio data")
    filename: str = Field(default="audio.wav", description="Original filename for format detection")
    diarize: bool = Field(default=True, description="Whether to perform speaker diarization")
    language: Optional[str] = Field(default=None, description="Language code (auto-detect if None)")
    num_speakers: Optional[int] = Field(default=None, description="Expected number of speakers")


class InternalTranscribeResponse(BaseModel):
    """Response from internal transcription service."""
    text: str = ""
    segments: List[Dict[str, Any]] = []
    language: str = "unknown"
    duration: float = 0.0
    word_count: int = 0
    processing_time: float = 0.0
    error: bool = False
    error_type: Optional[str] = None
    error_message: Optional[str] = None


@router.post(
    "/internal/chat",
    response_model=InternalChatResponse,
    deprecated=True,
)
async def internal_chat(request: InternalChatRequest):
    """
    Internal chat endpoint for worker-to-worker communication.
    
    .. deprecated::
        Use /internal/chat-functions instead. This endpoint does not have
        SQL Agent capabilities. Kept for backwards compatibility.
    
    This endpoint is called by the API pod proxy to delegate chat
    requests to interactive workers. It runs the AI inference directly
    without database operations.
    """
    from src.services.interactive import get_interactive_service
    
    settings = get_settings()
    
    # Only allow this endpoint on interactive workers
    if settings.worker_mode not in ["interactive", "both"]:
        raise HTTPException(
            status_code=403,
            detail=f"This endpoint is only available on interactive workers (current mode: {settings.worker_mode})"
        )
    
    try:
        service = get_interactive_service()
        result = service.chat(
            message=request.message,
            context=request.context,
            conversation_history=request.conversation_history,
            max_tokens=request.max_tokens,
        )
        
        return InternalChatResponse(
            response=result.get("response", ""),
            generation_time=result.get("generation_time", 0.0),
            model=result.get("model", ""),
            error=False,
        )
    except Exception as e:
        logger.error(f"Internal chat error: {e}")
        return InternalChatResponse(
            response="",
            error=True,
            error_type="inference_error",
            error_message=str(e),
        )


@router.post(
    "/internal/chat-functions",
    response_model=InternalChatFunctionsResponse,
)
async def internal_chat_functions(request: InternalChatFunctionsRequest):
    """
    Internal chat with SQL Agent functions endpoint.
    
    This endpoint enables AI to query the database to answer questions.
    Called by the API pod proxy to delegate to interactive workers.
    """
    from src.services.interactive import get_interactive_service
    
    settings = get_settings()
    
    if settings.worker_mode not in ["interactive", "both"]:
        raise HTTPException(
            status_code=403,
            detail=f"This endpoint is only available on interactive workers"
        )
    
    try:
        service = get_interactive_service()
        result = service.chat_with_functions(
            message=request.message,
            user_name=request.user_name,
            filters=request.filters,
            conversation_history=request.conversation_history,
            max_tokens=request.max_tokens,
        )
        
        return InternalChatFunctionsResponse(
            response=result.get("response", ""),
            generation_time=result.get("processing_time", 0.0),
            model=result.get("model", ""),
            function_calls=result.get("function_calls", []),
            error=False,
        )
    except Exception as e:
        logger.error(f"Internal chat-functions error: {e}")
        return InternalChatFunctionsResponse(
            response="",
            error=True,
            error_type="inference_error",
            error_message=str(e),
        )


@router.post(
    "/internal/summary",
    response_model=InternalSummaryResponse,
)
async def internal_summary(request: InternalSummaryRequest):
    """
    Internal summary endpoint for worker-to-worker communication.
    """
    from src.services.interactive import get_interactive_service
    
    settings = get_settings()
    
    if settings.worker_mode not in ["interactive", "both"]:
        raise HTTPException(
            status_code=403,
            detail=f"This endpoint is only available on interactive workers"
        )
    
    try:
        service = get_interactive_service()
        result = service.generate_summary(
            transcript=request.transcript,
            summary_type=request.summary_type,
            custom_prompt=request.custom_prompt,
        )
        
        return InternalSummaryResponse(
            summary=result.get("summary", ""),
            generation_time=result.get("generation_time", 0.0),
            error=False,
        )
    except Exception as e:
        logger.error(f"Internal summary error: {e}")
        return InternalSummaryResponse(
            summary="",
            error=True,
            error_type="inference_error",
            error_message=str(e),
        )


@router.get(
    "/internal/health",
    response_model=InternalHealthResponse,
)
async def internal_health():
    """
    Internal health check for interactive workers.
    
    Returns model status for load balancing decisions.
    """
    settings = get_settings()
    
    model_loaded = False
    device = "unknown"
    
    if settings.worker_mode in ["interactive", "both"]:
        try:
            from src.services.interactive import get_interactive_service
            service = get_interactive_service()
            model_loaded = service._model is not None
            device = str(service._device) if hasattr(service, '_device') else "unknown"
        except Exception:
            pass
    
    return InternalHealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        device=device,
        worker_id=settings.worker_id,
    )


# =============================================================================
# Internal Transcription Endpoint (for shared WhisperX service)
# =============================================================================

# Singleton transcription service for shared access
_transcription_service = None
_transcription_service_lock = None


def get_transcription_service():
    """Get or create the shared transcription service."""
    global _transcription_service, _transcription_service_lock
    import asyncio
    
    if _transcription_service_lock is None:
        _transcription_service_lock = asyncio.Lock()
    
    if _transcription_service is None:
        from src.services.transcriber import TranscriptionService
        _transcription_service = TranscriptionService()
    
    return _transcription_service


@router.post(
    "/internal/transcribe",
    response_model=InternalTranscribeResponse,
)
async def internal_transcribe(request: InternalTranscribeRequest):
    """
    Internal transcription endpoint for shared WhisperX service.
    
    This endpoint allows multiple workers to share a single GPU-loaded
    WhisperX model instead of each loading their own (which wastes VRAM).
    
    The audio is sent as base64-encoded data to avoid file system
    dependencies between services.
    """
    import base64
    import tempfile
    import time
    from pathlib import Path
    
    start_time = time.time()
    
    try:
        # Decode audio data
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Write to temp file (WhisperX needs file path)
        suffix = Path(request.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        try:
            # Get transcription service
            service = get_transcription_service()
            
            # Transcribe
            result = service.transcribe(
                audio_path=tmp_path,
                diarize=request.diarize,
            )
            
            processing_time = time.time() - start_time
            
            # Format response
            return InternalTranscribeResponse(
                text=result.get("text", ""),
                segments=result.get("segments", []),
                language=result.get("language", "unknown"),
                duration=result.get("duration", 0.0),
                word_count=len(result.get("text", "").split()),
                processing_time=processing_time,
                error=False,
            )
            
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
                
    except Exception as e:
        logger.error(f"Internal transcription error: {e}")
        return InternalTranscribeResponse(
            error=True,
            error_type="transcription_error",
            error_message=str(e),
        )

