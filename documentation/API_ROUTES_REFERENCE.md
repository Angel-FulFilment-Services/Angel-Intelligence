# Angel Intelligence - API Routes Reference

All endpoints require Bearer token authentication (header: `Authorization: Bearer <token>`) unless noted otherwise.

---

## Health & Info

### `GET /`
Root endpoint - service info. **No auth required.**

**Response:**
```json
{
  "service": "Angel Intelligence",
  "description": "AI-powered call transcription and analysis",
  "version": "1.0.0",
  "environment": "production",
  "status": "running"
}
```

---

### `GET /health`
Health check endpoint. **No auth required.**

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "worker_id": "worker-abc123",
  "environment": "production",
  "device": "cuda",
  "cuda_available": true,
  "models_loaded": {
    "analysis": { "version": "v1.0.0", "source": "huggingface" },
    "chat": { "loaded": true },
    "whisper": { "version": "medium" }
  },
  "worker_mode": "batch"
}
```

---

## Recording Management

### `POST /recordings/submit`
Submit a new recording for processing.

**Request Body:**
```json
{
  "apex_id": "ABC123",
  "call_date": "2026-01-21T10:30:00Z",
  "client_ref": "CLIENT001",
  "campaign": "Sales Q1",
  "halo_id": 12345,
  "agent_name": "John Smith",
  "direction": "outbound",
  "duration_seconds": 300,
  "retain_audio": false,
  "orderref": "ORD-12345",
  "enqref": "ENQ-67890",
  "obref": "OB-11111",
  "creative": "Summer Sale 2026",
  "invoicing": "INV-99999"
}
```

**Response:**
```json
{
  "id": 1,
  "apex_id": "ABC123",
  "status": "pending",
  "message": "Recording submitted for processing"
}
```

---

### `GET /recordings/{recording_id}/status`
Get the processing status of a recording.

**Path Parameters:**
- `recording_id` (int): Recording ID

**Response:**
```json
{
  "id": 1,
  "apex_id": "ABC123",
  "status": "completed",
  "processing_started_at": "2026-01-21T10:31:00Z",
  "processing_completed_at": "2026-01-21T10:35:00Z",
  "error": null,
  "retry_count": 0
}
```

---

### `GET /recordings/{recording_id}/transcription`
Get the transcription for a recording.

**Path Parameters:**
- `recording_id` (int): Recording ID

**Response:**
```json
{
  "id": 1,
  "recording_id": 1,
  "full_transcript": "Agent: Hello, how can I help you today?...",
  "segments": [
    { "start": 0.0, "end": 2.5, "text": "Hello, how can I help you today?", "speaker": "AGENT" }
  ],
  "redacted_transcript": "Agent: Hello, how can I help you today?...",
  "pii_detected": [
    { "type": "PHONE_NUMBER", "start": 45, "end": 57 }
  ],
  "language": "en",
  "confidence": 0.95,
  "model": "whisperx",
  "processing_time_seconds": 45
}
```

---

### `GET /recordings/{recording_id}/analysis`
Get the analysis for a recording.

**Path Parameters:**
- `recording_id` (int): Recording ID

**Response:**
```json
{
  "id": 1,
  "recording_id": 1,
  "summary": "Customer called to enquire about...",
  "sentiment_score": 0.75,
  "sentiment_label": "positive",
  "quality_score": 0.85,
  "key_topics": [
    { "topic": "Product Inquiry", "confidence": 0.9 }
  ],
  "agent_actions_performed": [
    { "action": "Greeted customer", "performed": true }
  ],
  "performance_scores": {
    "professionalism": 0.9,
    "empathy": 0.8
  },
  "action_items": [],
  "compliance_flags": [],
  "speaker_metrics": {
    "agent_talk_time": 120,
    "customer_talk_time": 180
  },
  "audio_observations": null,
  "model": "Qwen2.5-7B",
  "model_version": "v1.0.0",
  "processing_time_seconds": 30
}
```

---

### `POST /recordings/reprocess/{recording_id}`
Requeue a recording for reprocessing.

**Path Parameters:**
- `recording_id` (int): Recording ID

**Response:**
```json
{
  "message": "Recording 1 queued for reprocessing"
}
```

---

### `GET /recordings/pending`
Get list of pending recordings.

**Query Parameters:**
- `limit` (int, optional, max 200, default 50): Maximum records to return

**Response:**
```json
{
  "count": 5,
  "recordings": [
    { "id": 1, "apex_id": "ABC123", "processing_status": "pending", "retry_count": 0, "created_at": "..." }
  ]
}
```

---

### `GET /recordings/failed`
Get list of failed recordings.

**Query Parameters:**
- `limit` (int, optional, max 200, default 50): Maximum records to return

**Response:**
```json
{
  "count": 2,
  "recordings": [
    { "id": 5, "apex_id": "XYZ789", "processing_error": "Timeout", "retry_count": 3, "processing_completed_at": "...", "next_retry_at": null }
  ]
}
```

---

## Processing

### `POST /api/process`
Manually trigger processing of a single call.

**Request Body:**
```json
{
  "apex_id": "ABC123",
  "force_reprocess": false
}
```

**Response:**
```json
{
  "success": true,
  "recording_id": 1,
  "status": "queued",
  "message": "Call queued for processing"
}
```

---

### `POST /api/transcribe`
Transcribe a call without full analysis (for Dojo training).

**Request Body:**
```json
{
  "apex_id": "ABC123",
  "call_date": "2026-01-21"
}
```

**Response:**
```json
{
  "success": true,
  "full_transcript": "Agent: Hello...",
  "segments": [
    { "start": 0.0, "end": 2.5, "text": "Hello", "speaker": "AGENT" }
  ],
  "language": "en",
  "confidence_score": 0.95,
  "message": null
}
```

---

## Chat

### `POST /chat`
Chat with the AI about call data (legacy endpoint).

**Request Body:**
```json
{
  "message": "What was the sentiment of the last call?",
  "recording_id": 1,
  "conversation_id": "conv_abc123",
  "user": {
    "id": 1,
    "name": "John",
    "email": "john@example.com"
  },
  "feature": "general",
  "filters": {}
}
```

**Response:**
```json
{
  "success": true,
  "response": "The last call had a positive sentiment score of 0.75...",
  "conversation_id": "conv_abc123",
  "message_id": 42,
  "error": null
}
```

---

### `POST /api/chat`
Enhanced chat endpoint with full features and SQL Agent capabilities.

**Request Body:**
```json
{
  "message": "Show me calls with low quality scores this week",
  "conversation_id": 1,
  "user": {
    "id": 1,
    "name": "John",
    "email": "john@example.com"
  },
  "feature": "call_quality",
  "filters": {
    "client_ref": "CLIENT001",
    "date_from": "2026-01-14",
    "date_to": "2026-01-21"
  },
  "history": []
}
```

**Response:**
```json
{
  "success": true,
  "response": "I found 5 calls with quality scores below 0.5...",
  "conversation_id": 1,
  "message_id": 43,
  "metadata": {},
  "error": null
}
```

---

## Summary Generation

### `POST /api/summary/generate`
Generate a summary for a date range.

**Request Body:**
```json
{
  "feature": "call_quality",
  "start_date": "2026-01-01",
  "end_date": "2026-01-21",
  "client_ref": "CLIENT001",
  "campaign": "Sales Q1",
  "agent_id": 5
}
```

**Response:**
```json
{
  "success": true,
  "summary_id": 1,
  "summary_data": { ... },
  "message": "Summary generated successfully"
}
```

---

## Configuration

### `GET /config/analysis`
Get the analysis configuration (topics, actions, rubric).

**Response:**
```json
{
  "topics": ["Product Inquiry", "Complaint", "Billing"],
  "agent_actions": ["Greeted customer", "Verified identity"],
  "performance_rubric": ["Professionalism", "Empathy"]
}
```

---

### `GET /api/config`
Get client configuration, falling back to global if not found.

**Query Parameters:**
- `client_ref` (string, optional): Client reference
- `config_type` (string, optional): One of `topics`, `agent_actions`, `performance_rubric`, `prompt`, `analysis_mode`

**Response (single type):**
```json
{
  "config_type": "topics",
  "config_data": { "topics": ["..."] }
}
```

**Response (all types):**
```json
{
  "client_ref": "CLIENT001",
  "configs": {
    "topics": { ... },
    "agent_actions": { ... }
  }
}
```

---

### `POST /api/config`
Create or update a client configuration.

**Request Body:**
```json
{
  "client_ref": "CLIENT001",
  "config_type": "topics",
  "config_data": { "topics": ["Sales", "Support"] },
  "is_active": true
}
```

**Response:**
```json
{
  "id": 1,
  "client_ref": "CLIENT001",
  "config_type": "topics",
  "config_data": { "topics": ["Sales", "Support"] },
  "is_active": true
}
```

---

### `DELETE /api/config/{config_id}`
Deactivate a client configuration.

**Path Parameters:**
- `config_id` (int): Configuration ID

**Response:**
```json
{
  "message": "Configuration deactivated"
}
```

---

## Training

### `GET /api/training-data`
Export training data for model fine-tuning.

**Query Parameters:**
- `since` (string, optional): ISO date, e.g., `2026-01-01`
- `annotation_type` (string, optional): Filter by annotation type
- `limit` (int, optional, max 5000, default 1000): Maximum records

**Response:**
```json
{
  "success": true,
  "count": 100,
  "data": [
    {
      "id": "annotation_1",
      "transcript": "...",
      "original_analysis": { ... },
      "corrected_analysis": { ... },
      "annotation_metadata": { ... }
    }
  ]
}
```

---

### `POST /api/training/import`
Import training data for fine-tuning.

**Request Body:**
```json
{
  "model_type": "call_analysis",
  "training_data": [
    { "input": "...", "output": "..." }
  ],
  "options": {}
}
```

**Response:**
```json
{
  "success": true,
  "job_id": "training_abc123",
  "status": "acknowledged",
  "message": "Received 10 samples. Training runs nightly at 2 AM..."
}
```

---

### `GET /api/training/status`
Get current training/adapter status.

**Query Parameters:**
- `adapter_name` (string, optional, default "call-analysis"): Adapter name

**Response:**
```json
{
  "adapter_exists": true,
  "adapter_name": "call-analysis",
  "current_version": "v20260121_143052",
  "trained_at": "2026-01-21T14:30:52Z",
  "samples_used": 500,
  "training_loss": 0.45,
  "base_model": "Qwen/Qwen2.5-7B-Instruct",
  "new_annotations_since_training": 25,
  "ready_for_training": true,
  "ready_reason": "25 new annotations available",
  "available_versions": [...]
}
```

---

### `POST /api/training/start`
Trigger model fine-tuning immediately.

**Request Body:**
```json
{
  "adapter_name": "call-analysis",
  "force": false,
  "max_samples": 2500,
  "epochs": 3
}
```

**Response:**
```json
{
  "success": true,
  "message": "Training completed successfully",
  "adapter_name": "call-analysis",
  "version": "v20260121_150000",
  "samples_used": 500,
  "training_loss": 0.42,
  "training_time_minutes": 45.5,
  "error": null
}
```

---

### `POST /api/training/promote`
Promote a specific adapter version to be active.

**Request Body:**
```json
{
  "adapter_name": "call-analysis",
  "version": "v20260120_100000"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Version 'v20260120_100000' is now the current active adapter",
  "adapter_name": "call-analysis",
  "version": "v20260120_100000"
}
```

---

### `POST /api/training/cleanup`
Remove old adapter versions, keeping newest N.

**Request Body:**
```json
{
  "adapter_name": "call-analysis",
  "keep": 5
}
```

**Response:**
```json
{
  "success": true,
  "message": "Cleaned up 3 old version(s), kept newest 5",
  "versions_removed": 3
}
```

---

## Internal Endpoints (Worker-to-Worker)

These are for internal service communication and typically don't require external access.

### `POST /internal/chat`
*(Deprecated)* Internal chat endpoint for worker-to-worker communication.

---

### `POST /internal/chat-functions`
Internal chat with SQL Agent functions.

**Request Body:**
```json
{
  "message": "...",
  "user_name": "John",
  "filters": {},
  "conversation_history": [],
  "max_tokens": 2000
}
```

---

### `POST /internal/summary`
Internal summary generation endpoint.

**Request Body:**
```json
{
  "transcript": "...",
  "summary_type": "brief",
  "custom_prompt": null
}
```

---

### `GET /internal/health`
Internal health check for interactive workers.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0",
  "worker_id": "interactive-worker-1"
}
```

---

## Error Responses

All endpoints return standard error responses:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**Common HTTP Status Codes:**
- `400` - Bad Request (invalid input)
- `401` - Unauthorized (missing or invalid token)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found (resource doesn't exist)
- `500` - Internal Server Error
