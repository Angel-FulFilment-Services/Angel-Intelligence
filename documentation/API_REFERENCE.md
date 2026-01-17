# API Reference

Complete documentation for all Angel Intelligence API endpoints.

## Base URL

```
Production:  https://ai.angelfs.co.uk
Development: http://localhost:8000
```

## Authentication

All endpoints (except `/health`) require Bearer token authentication:

```http
Authorization: Bearer <API_AUTH_TOKEN>
```

The token must be at least 64 characters and match the `API_AUTH_TOKEN` environment variable.

### Error Response

```json
{
  "detail": "Invalid authentication token"
}
```

---

## Health & Status

### GET /health

Health check endpoint. **No authentication required.**

**Response 200:**

```json
{
  "status": "healthy",
  "version": "1.0.0",
  "worker_id": "worker-abc123",
  "environment": "production",
  "device": "cuda",
  "cuda_available": true,
  "models_loaded": {
    "analysis": {
      "version": "v1.0.0",
      "loaded": true,
      "path": "/models/analysis"
    },
    "chat": {
      "version": "base",
      "loaded": true,
      "path": "/models/chat"
    },
    "whisper": {
      "version": "medium",
      "loaded": true
    }
  }
}
```

### GET /

Service information endpoint.

**Response 200:**

```json
{
  "name": "Angel Intelligence",
  "version": "1.0.0",
  "environment": "production"
}
```

---

## Recordings

### POST /recordings/submit

Submit a call recording for processing.

**Request Body:**

```json
{
  "apex_id": "REF-2026-001234",
  "call_date": "2026-01-17",
  "client_ref": "CLIENT001",
  "campaign": "Q1-APPEAL",
  "halo_id": 123,
  "agent_name": "Jane Smith",
  "direction": "outbound",
  "duration_seconds": 420,
  "retain_audio": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| apex_id | string | Yes | Unique recording identifier from PBX |
| call_date | string | Yes | Call date (YYYY-MM-DD) |
| client_ref | string | No | Client reference code |
| campaign | string | No | Campaign name |
| halo_id | int | No | Agent's Halo system ID |
| agent_name | string | No | Agent's display name |
| direction | string | No | "inbound" or "outbound" (default: outbound) |
| duration_seconds | int | No | Call duration in seconds |
| retain_audio | bool | No | Keep audio after processing (default: false) |

**Response 201:**

```json
{
  "id": 12345,
  "apex_id": "REF-2026-001234",
  "status": "pending",
  "message": "Recording submitted for processing"
}
```

---

### GET /recordings/{recording_id}/status

Get processing status of a recording.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| recording_id | int | Database recording ID |

**Response 200:**

```json
{
  "id": 12345,
  "apex_id": "REF-2026-001234",
  "status": "completed",
  "processing_started_at": "2026-01-17T14:30:00Z",
  "processing_completed_at": "2026-01-17T14:31:15Z",
  "error": null,
  "retry_count": 0
}
```

**Status Values:**

| Status | Description |
|--------|-------------|
| pending | Waiting in queue |
| processing | Currently being processed |
| completed | Successfully processed |
| failed | Processing failed |
| queued | Manually queued for reprocessing |

---

### GET /recordings/{recording_id}/transcription

Get the transcription for a recording.

**Response 200:**

```json
{
  "id": 67890,
  "recording_id": 12345,
  "full_transcript": "Hello, thank you for calling...",
  "segments": [
    {
      "text": "Hello, thank you for calling.",
      "start": 0.0,
      "end": 2.5,
      "speaker": "agent",
      "speaker_id": "agent_123",
      "confidence": 0.95,
      "words": [
        {"word": "Hello", "start": 0.0, "end": 0.4, "confidence": 0.98}
      ]
    }
  ],
  "redacted_transcript": "Hello, thank you for calling. My postcode is [POSTCODE].",
  "pii_detected": [
    {
      "type": "postcode",
      "original": "SW1A 1AA",
      "redacted": "[POSTCODE]",
      "timestamp_start": 45.2,
      "timestamp_end": 47.1,
      "confidence": 0.92
    }
  ],
  "language": "en",
  "confidence": 0.95,
  "model": "whisperx-medium",
  "processing_time_seconds": 28
}
```

---

### GET /recordings/{recording_id}/analysis

Get the analysis for a recording.

**Response 200:**

```json
{
  "id": 11111,
  "recording_id": 12345,
  "summary": "The supporter enquired about regular giving options...",
  "sentiment_score": 7.5,
  "sentiment_label": "positive",
  "quality_score": 85.0,
  "key_topics": [
    {"name": "Regular giving signup", "confidence": 0.95, "timestamp_start": 45.5, "timestamp_end": 78.2}
  ],
  "agent_actions_performed": [
    {"action": "Greeted supporter", "timestamp_start": 0.0, "quality": 5}
  ],
  "performance_scores": {
    "Clarity of speech": 8,
    "Tone control": 9
  },
  "action_items": [
    {"description": "Send confirmation email", "priority": "high", "due_date": "2026-01-20"}
  ],
  "compliance_flags": [],
  "speaker_metrics": {
    "agent": {"talk_time_seconds": 245, "talk_time_percentage": 58}
  },
  "audio_observations": {
    "call_quality": "good",
    "background_noise": "low",
    "agent_tone": "friendly"
  },
  "model": "Qwen2.5-Omni-7B",
  "model_version": "v1.0.0",
  "processing_time_seconds": 45
}
```

---

### POST /recordings/reprocess/{recording_id}

Requeue a recording for processing.

**Response 200:**

```json
{
  "id": 12345,
  "status": "queued",
  "message": "Recording queued for reprocessing"
}
```

---

### GET /recordings/pending

List recordings pending processing.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | int | 50 | Maximum results (max: 200) |

**Response 200:**

```json
{
  "count": 15,
  "recordings": [
    {
      "id": 12346,
      "apex_id": "REF-2026-001235",
      "created_at": "2026-01-17T15:00:00Z"
    }
  ]
}
```

---

### GET /recordings/failed

List failed recordings.

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | int | 50 | Maximum results (max: 200) |

**Response 200:**

```json
{
  "count": 3,
  "recordings": [
    {
      "id": 12340,
      "apex_id": "REF-2026-001230",
      "processing_error": "DOWNLOAD_FAILED: Recording not found",
      "retry_count": 3,
      "processing_completed_at": "2026-01-17T14:00:00Z",
      "next_retry_at": null
    }
  ]
}
```

---

### POST /api/process

Manually trigger processing of a single call.

**Request Body:**

```json
{
  "apex_id": "REF-2026-001234",
  "force_reprocess": false
}
```

**Response 200:**

```json
{
  "success": true,
  "recording_id": 12345,
  "status": "queued",
  "message": "Call queued for processing"
}
```

---

## Chat

### POST /chat

Simple chat about call data.

**Request Body:**

```json
{
  "message": "What was the average quality score last week?",
  "feature": "call_quality",
  "conversation_id": null
}
```

**Response 200:**

```json
{
  "response": "Based on the call data, the average quality score last week was 78.5%...",
  "conversation_id": 456
}
```

---

### POST /api/chat

Enhanced chat with filters.

**Request Body:**

```json
{
  "message": "Show me trends for this client",
  "conversation_id": 456,
  "feature": "call_quality",
  "filters": {
    "client_ref": "CLIENT001",
    "start_date": "2026-01-01",
    "end_date": "2026-01-17"
  },
  "history": []
}
```

**Response 200:**

```json
{
  "success": true,
  "response": "Based on 245 calls for CLIENT001 in January 2026...",
  "metadata": {
    "conversation_id": 456,
    "model": "Qwen2.5-Omni-7B",
    "tokens_used": 1250,
    "context_calls_analysed": 245
  }
}
```

---

## Summaries

### POST /api/summary/generate

Generate a monthly summary.

**Request Body:**

```json
{
  "feature": "call_quality",
  "month": 1,
  "year": 2026,
  "client_ref": "CLIENT001",
  "campaign": null,
  "agent_id": null
}
```

**Response 200:**

```json
{
  "success": true,
  "summary_id": 789,
  "summary_data": {
    "summary": "In January 2026, 1,247 calls were analysed...",
    "key_insights": [
      "Call quality improved 5% compared to December"
    ],
    "recommendations": [
      "Continue objection handling training programme"
    ]
  },
  "message": "Summary generated successfully"
}
```

---

## Configuration

### GET /config/analysis

Get analysis configuration (topics, actions, rubric).

**Response 200:**

```json
{
  "topics": ["One-off donation request", "Regular giving signup", ...],
  "agent_actions": ["Greeted supporter", "Verified identity", ...],
  "performance_rubric": ["Clarity of speech", "Tone control", ...]
}
```

---

### GET /api/config

Get client-specific configuration.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| client_ref | string | Client reference (null for global) |
| config_type | string | Specific type or null for all |

**Response 200:**

```json
{
  "client_ref": "CLIENT001",
  "configs": {
    "topics": [...],
    "agent_actions": [...],
    "performance_rubric": [...],
    "prompt": {...},
    "analysis_mode": {"mode": "audio"}
  }
}
```

---

### POST /api/config

Create or update client configuration.

**Request Body:**

```json
{
  "client_ref": "CLIENT001",
  "config_type": "topics",
  "config_data": ["Custom topic 1", "Custom topic 2"],
  "is_active": true
}
```

**Config Types:**

| Type | Description |
|------|-------------|
| topics | List of call topics |
| agent_actions | List of agent actions |
| performance_rubric | Performance criteria |
| prompt | Custom analysis prompt |
| analysis_mode | Audio or transcript mode |

**Response 200:**

```json
{
  "id": 123,
  "client_ref": "CLIENT001",
  "config_type": "topics",
  "config_data": [...],
  "is_active": true
}
```

---

### DELETE /api/config/{config_id}

Deactivate a configuration.

**Response 200:**

```json
{
  "message": "Configuration deactivated"
}
```

---

## Training Data

### GET /api/training-data

Export training annotations for model fine-tuning.

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| since | string | ISO date filter (e.g., 2026-01-01) |
| annotation_type | string | Filter by type |
| limit | int | Maximum results (default: 1000, max: 5000) |

**Response 200:**

```json
{
  "success": true,
  "count": 156,
  "data": [
    {
      "id": "annotation_123",
      "transcript": "Full call transcript...",
      "original_analysis": {...},
      "corrected_analysis": {...},
      "annotation_metadata": {...}
    }
  ]
}
```

---

### POST /api/training/import

Import training data for model fine-tuning.

**Request Body:**

```json
{
  "model_type": "call_analysis",
  "training_data": [...],
  "options": {
    "trigger_training": false
  }
}
```

**Response 200:**

```json
{
  "success": true,
  "job_id": "training_abc123def456",
  "status": "queued",
  "message": "Training job queued with 156 samples"
}
```

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| VALIDATION_ERROR | 400 | Invalid request data |
| AUTHENTICATION_ERROR | 401 | Invalid or missing token |
| NOT_FOUND | 404 | Resource not found |
| RECORDING_NOT_FOUND | 404 | Recording ID doesn't exist |
| DOWNLOAD_FAILED | 500 | Failed to download audio |
| TRANSCRIPTION_FAILED | 500 | Transcription error |
| ANALYSIS_FAILED | 500 | Analysis error |
| DATABASE_ERROR | 500 | Database operation failed |

---

## Rate Limits

No rate limits are currently enforced. Workers self-regulate based on `MAX_CONCURRENT_JOBS`.

---

## Pagination

Endpoints returning lists support `limit` parameter. Large datasets return partial results with count indicators.

---

## Webhook Support

Webhook notifications are not currently implemented. Poll `/recordings/{id}/status` for status updates.
