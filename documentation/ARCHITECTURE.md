# System Architecture

High-level architecture documentation for Angel Intelligence.

## Overview

Angel Intelligence is an AI-powered call analysis system designed to:
- Transcribe charity support call recordings
- Detect and redact PII (Personally Identifiable Information)
- Analyse call quality, sentiment, and agent performance
- Provide conversational AI interface for data insights
- Support fine-tuning through human annotations

---

## System Components

```
┌───────────────────────────────────────────────────────────────────────────┐
│                                PULSE (Laravel)                            │
│                         Frontend Web Application                          │
│    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│    │ Call Quality│    │   Reports   │    │    Chat     │                  │
│    │  Dashboard  │    │  Generator  │    │  Interface  │                  │
│    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                  │
└───────────┼──────────────────┼──────────────────┼─────────────────────────┘
            │                  │                  │
            └──────────────────┼──────────────────┘
                               │ HTTPS REST API
                               │ (Bearer Token Auth)
                               ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        ANGEL INTELLIGENCE API                             │
│                          (FastAPI + Uvicorn)                              │
│                                                                           │
│    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│    │  Recordings │    │   Analysis  │    │    Chat     │                  │
│    │   Routes    │    │   Routes    │    │   Routes    │                  │
│    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                  │
└───────────┼──────────────────┼──────────────────┼─────────────────────────┘
            │                  │                  │
            ▼                  ▼                  ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         DATABASE LAYER                                    │
│                    (MySQL + SQLAlchemy ORM)                               │
│                                                                           │
│    ┌────────────────────────────────────────────────────────────────┐     │
│    │                      ai Database                               │     │
│    │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐   │     │
│    │  │ Recordings │ │Transcripts │ │  Analysis  │ │    Chat    │   │     │
│    │  └────────────┘ └────────────┘ └────────────┘ └────────────┘   │     │
│    └────────────────────────────────────────────────────────────────┘     │
└───────────────────────────────────────────────────────────────────────────┘
            │
            │ Queue Polling
            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                          WORKER CLUSTER                                   │
│                    (K3s on NVIDIA Jetson Orin)                            │
│                                                                           │
│    ┌─────────────────────────────────────────────────────────────────┐    │
│    │                     Worker Pods (x4)                            │    │
│    │                                                                 │    │
│    │   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │    │
│    │   │ Transcriber │   │ PII Detector│   │  Analyser   │           │    │
│    │   │  (WhisperX) │   │  (Presidio) │   │ (Qwen Omni) │           │    │
│    │   └─────────────┘   └─────────────┘   └─────────────┘           │    │
│    │         │                  │                  │                 │    │
│    │         └──────────────────┼──────────────────┘                 │    │
│    │                            │                                    │    │
│    │                      ┌─────▼─────┐                              │    │
│    │                      │   GPU     │                              │    │
│    │                      │ (8GB VRAM)│                              │    │
│    │                      └───────────┘                              │    │
│    └─────────────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────────────┘
            │
            │ Download/Upload
            ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                         EXTERNAL SERVICES                                 │
│                                                                           │
│    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                  │
│    │  PBX Server │    │Cloudflare R2│    │  HuggingFace│                  │
│    │  (Recordings)│    │  (Storage)  │    │   (Models)  │                 │
│    └─────────────┘    └─────────────┘    └─────────────┘                  │
└───────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Call Processing Pipeline

```
1. SUBMISSION
   Pulse → API → Database (status: pending)

2. PICKUP
   Worker polls → Claims recording → Database (status: processing)

3. DOWNLOAD
   Worker → PBX/R2 → Local temp file (.gsm/.wav)

4. CONVERSION
   SoX → Convert to WAV (8kHz, 32-bit, mono)

5. TRANSCRIPTION
   WhisperX → Transcribe → Word timestamps → Speaker labels

6. PII DETECTION
   Presidio → Detect → Redact transcript → Timestamp mapping

7. ANALYSIS
   Qwen2.5-Omni → Audio/Transcript analysis → Structured JSON

8. STORAGE
   Results → Database → R2 (optional audio retention)

9. COMPLETION
   Database (status: completed) → Worker releases

10. DISPLAY
    Pulse polls → Displays results → User interaction
```

---

## Component Details

### API Server (FastAPI)

**Responsibilities:**
- RESTful endpoint handlers
- Authentication/authorisation
- Request validation
- Response formatting
- Health monitoring

**Technology:**
- FastAPI framework
- Uvicorn ASGI server
- Pydantic validation
- OpenAPI documentation

**Endpoints:**
- `/health` - System health
- `/recordings/*` - Recording management
- `/chat` - Conversational AI
- `/config/*` - Configuration
- `/api/*` - Extended API

---

### Worker Service

**Responsibilities:**
- Queue polling and job claiming
- Audio download and conversion
- Transcription orchestration
- PII detection and redaction
- AI analysis coordination
- Result persistence

**Technology:**
- Python multiprocessing
- Concurrent job handling
- Graceful shutdown
- Automatic retries

**Configuration:**
- Poll interval: 30 seconds
- Max concurrent jobs: 4
- Max retries: 3
- Retry delay: 1 hour

---

### Transcription Service (WhisperX)

**Responsibilities:**
- Audio-to-text transcription
- Word-level timestamp alignment
- Speaker diarisation
- Language detection

**Technology:**
- WhisperX (OpenAI Whisper variant)
- faster-whisper backend
- pyannote.audio (optional diarisation)

**Models:**
- tiny (1GB VRAM) - fastest, lowest accuracy
- small (2GB VRAM) - balanced
- medium (5GB VRAM) - recommended
- large-v3 (10GB VRAM) - best accuracy

---

### PII Detection Service (Presidio)

**Responsibilities:**
- Identify UK-specific PII patterns
- Redact PII from transcripts
- Map PII to audio timestamps
- Generate redacted audio (optional)

**Technology:**
- Microsoft Presidio
- Custom UK pattern recognisers
- spaCy NLP backend

**Detected PII Types:**
- National Insurance Number
- NHS Number
- UK Postcodes
- Phone Numbers
- Bank Details (Sort Code, Account)
- Credit Card Numbers
- Date of Birth
- Email Addresses
- Driving Licence

---

### Analysis Service (Qwen2.5-Omni)

**Responsibilities:**
- Call quality assessment
- Sentiment analysis
- Topic extraction
- Agent action identification
- Performance scoring
- Compliance checking

**Technology:**
- Qwen2.5-Omni-7B (audio mode)
- Qwen2.5-7B (transcript mode)
- Transformers library
- GPU acceleration (CUDA)

**Output:**
- Summary (British English)
- Sentiment score (-10 to +10)
- Quality score (0-100)
- Key topics
- Agent actions performed
- Performance scores
- Compliance flags
- Action items

---

### Chat Service

**Responsibilities:**
- Natural language queries
- Data retrieval and aggregation
- Insight generation
- Conversation history

**Technology:**
- Qwen2.5-7B-Instruct
- Context-aware prompting
- Database query generation

---

## Infrastructure

### Kubernetes (K3s)

**Cluster Configuration:**
- 4x NVIDIA Jetson Orin Nano 8GB
- ARM64 architecture
- K3s lightweight Kubernetes
- Local container registry

**Resources per Worker:**
- 4 CPU cores
- 8GB RAM
- 1 GPU (8GB VRAM)
- Shared NFS for models

**Scaling:**
- Horizontal pod autoscaling
- One worker per node (GPU limit)
- Pod anti-affinity rules

---

### Storage

**MySQL Database:**
- Call recordings queue
- Transcriptions
- Analysis results
- Chat conversations
- Training annotations
- Configuration

**Cloudflare R2:**
- Audio file storage (optional)
- Retained recordings
- Archive storage

**NFS:**
- Shared model storage
- Hot-reload support
- 50GB+ capacity

---

## Security

### Authentication

- Bearer token authentication
- Minimum 64-character tokens
- Environment-based token storage
- No hardcoded credentials

### Data Protection

- PII detection and redaction
- Redacted transcripts stored
- Audio deletion after processing
- GDPR compliance

### Network Security

- TLS/HTTPS encryption
- Internal cluster networking
- Firewall rules
- API rate limiting (future)

---

## Scalability

### Horizontal Scaling

```
Current: 4 workers × 1 GPU = 4 parallel jobs
Maximum: N workers × 4 jobs = N×4 parallel jobs
```

### Bottlenecks

1. **GPU Memory** - Limits model size and concurrency
2. **Network** - Audio download bandwidth
3. **Database** - Connection pooling
4. **Storage** - NFS throughput

### Optimisations

- Model caching across workers
- Audio streaming (future)
- Connection pooling
- SSD for temp storage

---

## Monitoring

### Health Checks

- API `/health` endpoint
- Worker heartbeat
- GPU status
- Model load status

### Metrics

- Queue depth
- Processing time
- Error rates
- GPU utilisation

### Logging

- Structured JSON logs
- Log aggregation
- Error alerting (future)

---

## Failure Handling

### Retry Logic

```python
if error and retry_count < MAX_RETRIES:
    next_retry = now + RETRY_DELAY_HOURS
    status = 'queued'
else:
    status = 'failed'
```

### Error Categories

| Category | Retry | Example |
|----------|-------|---------|
| RECORDING_NOT_FOUND | No | Invalid apex_id |
| DOWNLOAD_FAILED | Yes | Network timeout |
| CONVERSION_FAILED | No | Corrupt audio |
| TRANSCRIPTION_FAILED | Yes | GPU error |
| ANALYSIS_FAILED | Yes | Model error |
| DATABASE_ERROR | Yes | Connection lost |

### Recovery

- Automatic retry with backoff
- Manual requeue via API
- Dead letter queue (failed)
- Admin notification (future)

---

## Future Enhancements

### Planned Features

1. **Voice Fingerprinting** - Automatic agent identification
2. **Real-time Processing** - Streaming analysis
3. **Custom Model Training** - Fine-tuning pipeline
4. **Webhook Notifications** - Status callbacks
5. **Multi-language Support** - Beyond English

### Technical Debt

1. Connection pooling improvement
2. Async database operations
3. Streaming audio processing
4. Prometheus metrics
5. OpenTelemetry tracing
