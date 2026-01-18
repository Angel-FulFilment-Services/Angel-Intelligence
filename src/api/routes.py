"""
Angel Intelligence - API Routes

All API endpoints for the Angel Intelligence service.
Endpoints require Bearer token authentication.
"""

import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query
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
    key_topics: List[dict]
    agent_actions_performed: List[dict]
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


class ChatRequest(BaseModel):
    """Chat conversation request."""
    message: str
    recording_id: Optional[int] = None
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat conversation response."""
    response: str
    conversation_id: str


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
    import torch
    import os
    settings = get_settings()
    
    # Check interactive service status
    interactive_status = {"available": False, "model_loaded": False}
    try:
        from src.services import get_interactive_service
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
        device="cuda" if torch.cuda.is_available() else "cpu",
        cuda_available=torch.cuda.is_available(),
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
         duration_seconds, retain_audio, processing_status, created_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'pending', NOW())
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
        key_topics=json.loads(row["key_topics"]) if row["key_topics"] else [],
        agent_actions_performed=json.loads(row.get("agent_actions_performed", "[]")),
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
    dependencies=[Depends(verify_token)]
)
async def chat(request: ChatRequest):
    """
    Chat with the AI about call data.
    
    Can reference a specific recording or ask general questions.
    Uses the interactive service on dedicated node(s).
    """
    import time
    import uuid
    from src.services import get_interactive_service
    from src.database import ChatConversation, ChatMessage
    
    request_start = time.time()
    logger.info(f"[TIMING] Chat request received at {request_start}")
    
    db = get_db_connection()
    
    # Generate or use existing conversation ID
    if request.conversation_id:
        conversation_id = request.conversation_id
        # Verify conversation exists
        conversation = ChatConversation.get_by_id(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Create new conversation
        conversation = ChatConversation(
            user_id=0,  # TODO: Get from auth
            feature="call_quality",
            title=request.message[:100] if len(request.message) > 100 else request.message,
        )
        conversation_id = conversation.save()
    
    conv_time = time.time()
    logger.info(f"[TIMING] Conversation setup took {conv_time - request_start:.2f}s")
    
    # Save user message immediately
    user_message = ChatMessage(
        ai_chat_conversation_id=conversation_id,
        role="user",
        content=request.message,
    )
    user_message.save()
    
    save_time = time.time()
    logger.info(f"[TIMING] Message save took {save_time - conv_time:.2f}s")
    
    # Build context if recording specified
    context = None
    if request.recording_id:
        # Get transcription
        trans = db.fetch_one(
            "SELECT full_transcript FROM ai_call_transcriptions WHERE ai_call_recording_id = %s",
            (request.recording_id,)
        )
        
        # Get analysis
        analysis = db.fetch_one(
            "SELECT summary, sentiment_label, quality_score FROM ai_call_analysis WHERE ai_call_recording_id = %s",
            (request.recording_id,)
        )
        
        context_parts = []
        if trans:
            context_parts.append(f"Transcript: {trans['full_transcript'][:2000]}")
        if analysis:
            context_parts.append(f"Summary: {analysis['summary']}")
            context_parts.append(f"Sentiment: {analysis['sentiment_label']}, Quality: {analysis['quality_score']}")
        
        if context_parts:
            context = "\n".join(context_parts)
    
    context_time = time.time()
    logger.info(f"[TIMING] Context build took {context_time - save_time:.2f}s")
    
    # Use interactive service for AI response
    logger.info(f"[TIMING] Starting AI generation...")
    service = get_interactive_service()
    result = service.chat(
        message=request.message,
        context=context,
    )
    
    ai_time = time.time()
    logger.info(f"[TIMING] AI generation took {ai_time - context_time:.2f}s")
    
    # Save assistant response
    assistant_message = ChatMessage(
        ai_chat_conversation_id=conversation_id,
        role="assistant",
        content=result["response"],
    )
    assistant_message.save()
    
    total_time = time.time()
    logger.info(f"[TIMING] Total request took {total_time - request_start:.2f}s")
    
    return ChatResponse(
        response=result["response"],
        conversation_id=str(conversation_id),
    )


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
    """
    from src.database import Summary
    from src.services import get_interactive_service
    
    db = get_db_connection()
    
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
        return SummaryGenerateResponse(
            success=False,
            message="No call data found for the specified period"
        )
    
    # Prepare data for summary generation
    data = {
        "call_count": metrics_row["call_count"],
        "avg_quality_score": float(metrics_row["avg_quality"] or 0),
        "avg_sentiment_score": float(metrics_row["avg_sentiment"] or 0),
        "total_duration_seconds": int(metrics_row["total_duration"] or 0),
    }
    
    # Use interactive service for AI summary
    service = get_interactive_service()
    result = service.generate_summary(
        data=data,
        summary_type="monthly",
        filters=filters,
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
    
    # Save to database
    summary = Summary(
        feature=request.feature,
        start_date=request.start_date,
        end_date=request.end_date,
        client_ref=request.client_ref,
        campaign=request.campaign,
        agent_id=request.agent_id,
        summary_data=summary_data,
        metrics=metrics,
    )
    summary_id = summary.save()
    
    return SummaryGenerateResponse(
        success=True,
        summary_id=summary_id,
        summary_data=summary_data,
        message="Summary generated successfully"
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
    
    # TODO: Implement actual training pipeline
    # For now, just acknowledge the request
    logger.info(f"Training job {job_id}: Received {len(request.training_data)} training samples")
    
    return TrainingImportResponse(
        success=True,
        job_id=job_id,
        status="queued",
        message=f"Training job queued with {len(request.training_data)} samples. Fine-tuning pipeline not yet implemented."
    )


# =============================================================================
# Client Configuration Management
# =============================================================================

class ClientConfigRequest(BaseModel):
    """Request to create/update client config."""
    client_ref: Optional[str] = None
    config_type: str  # 'topics', 'agent_actions', 'performance_rubric', 'prompt', 'analysis_mode'
    config_data: dict
    is_active: bool = True


class ClientConfigResponse(BaseModel):
    """Response with client config."""
    id: int
    client_ref: Optional[str]
    config_type: str
    config_data: dict
    is_active: bool


@router.get(
    "/api/config",
    dependencies=[Depends(verify_token)]
)
async def get_client_config(
    client_ref: Optional[str] = Query(default=None),
    config_type: Optional[str] = Query(default=None)
):
    """Get client configuration, falling back to global if not found."""
    from src.database import ClientConfig
    
    if config_type:
        config_data = ClientConfig.get_config(client_ref, config_type)
        if config_data is None:
            raise HTTPException(status_code=404, detail="Configuration not found")
        
        return {"config_type": config_type, "config_data": config_data}
    
    # Return all config types
    config_types = ["topics", "agent_actions", "performance_rubric", "prompt", "analysis_mode"]
    result = {}
    
    for ct in config_types:
        config_data = ClientConfig.get_config(client_ref, ct)
        if config_data:
            result[ct] = config_data
    
    return {"client_ref": client_ref, "configs": result}


@router.post(
    "/api/config",
    response_model=ClientConfigResponse,
    dependencies=[Depends(verify_token)]
)
async def create_client_config(request: ClientConfigRequest):
    """Create or update a client configuration."""
    from src.database import ClientConfig
    
    valid_types = ["topics", "agent_actions", "performance_rubric", "prompt", "analysis_mode"]
    if request.config_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid config_type. Must be one of: {valid_types}"
        )
    
    config = ClientConfig(
        client_ref=request.client_ref,
        config_type=request.config_type,
        config_data=request.config_data,
        is_active=request.is_active,
    )
    config_id = config.save()
    
    return ClientConfigResponse(
        id=config_id,
        client_ref=request.client_ref,
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
        UPDATE ai_client_configs
        SET is_active = FALSE, updated_at = NOW()
        WHERE id = %s
    """, (config_id,))
    
    if result == 0:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    return {"message": "Configuration deactivated"}


# =============================================================================
# Enhanced Chat Endpoint with Full Features
# =============================================================================

class EnhancedChatRequest(BaseModel):
    """Enhanced chat request matching specification."""
    message: str
    conversation_id: Optional[int] = None
    feature: str = "call_quality"
    filters: Optional[dict] = None
    history: Optional[List[dict]] = None


class EnhancedChatResponse(BaseModel):
    """Enhanced chat response."""
    success: bool
    response: str
    metadata: dict


@router.post(
    "/api/chat",
    response_model=EnhancedChatResponse,
    dependencies=[Depends(verify_token)]
)
async def enhanced_chat(request: EnhancedChatRequest):
    """
    Chat with the AI about call data.
    
    Uses the interactive service on dedicated node(s).
    Can filter by client, campaign, agent, and date range.
    """
    from src.database import ChatConversation, ChatMessage
    from src.services import get_interactive_service
    import json
    
    db = get_db_connection()
    
    # Get or create conversation
    if request.conversation_id:
        conversation = ChatConversation.get_by_id(request.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        conversation_id = conversation.id
    else:
        # Create new conversation
        conversation = ChatConversation(
            user_id=0,  # TODO: Get from auth
            feature=request.feature,
            filters=request.filters,
            title=request.message[:100] if len(request.message) > 100 else request.message,
        )
        conversation_id = conversation.save()
    
    # Save user message
    user_message = ChatMessage(
        ai_chat_conversation_id=conversation_id,
        role="user",
        content=request.message,
    )
    user_message.save()
    
    # Build context from filtered data
    context_parts = []
    context_calls = 0
    
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
                # Convert dates to UK format (DD/MM/YYYY)
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
                        import json
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
    service = get_interactive_service()
    context = "\n".join(context_parts) if context_parts else None
    
    result = service.chat(
        message=request.message,
        context=context,
    )
    
    response_text = result["response"]
    
    # Save assistant message
    assistant_message = ChatMessage(
        ai_chat_conversation_id=conversation_id,
        role="assistant",
        content=response_text,
        metadata={"context_calls": context_calls},
    )
    assistant_message.save()
    
    return EnhancedChatResponse(
        success=True,
        response=response_text,
        metadata={
            "conversation_id": conversation_id,
            "model": result.get("model", "unknown"),
            "tokens_used": result.get("tokens_used", 0),
            "context_calls_analysed": context_calls,
            "processing_time": result.get("processing_time", 0),
            "format": "markdown",
        }
    )
