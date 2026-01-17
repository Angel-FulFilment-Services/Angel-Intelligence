# Angel Intelligence

**AI platform for Angel Fulfilment Services**

Angel Intelligence is the centralised AI backend powering intelligent automation across Angel Fulfilment Services. Built on a scalable Python architecture, it provides AI capabilities to Pulse and other internal systems.

## Current Capabilities

### 📞 Call Quality Analysis (Live)

The first module processes charity call recordings to provide:

- 🎙️ **Transcription** - Word-level timestamps with speaker diarisation using WhisperX
- 🔒 **PII Protection** - UK-specific PII detection and redaction (NI numbers, postcodes, bank details)
- 🧠 **AI Analysis** - Sentiment, quality scoring, topic detection using Qwen2.5-Omni-7B
- 💬 **Chat Interface** - Query call data using natural language
- 🇬🇧 **British English** - All outputs use UK English spelling and conventions

### 🔮 Future Modules (Planned)

| Module | Description | Status |
|--------|-------------|--------|
| Live Call Transcription | Live in call call transcripts, ready for analysis | Planned |
| Agent Assistance | Real-time suggestions, knowledge base search, script guidance | Planned |

---

## Call Quality Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Pulse (Laravel)                         │
│                     Frontend Application                        │
└─────────────────────┬───────────────────────────────────────────┘
                      │ REST API (Bearer Token Auth)
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Angel Intelligence API                      │
│                      (FastAPI Gateway)                          │
└─────────────────────┬───────────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌─────────┐  ┌─────────┐  ┌─────────┐
    │ Worker 1│  │ Worker 2│  │ Worker 3│  (Jetson Nano Cluster)
    └─────────┘  └─────────┘  └─────────┘
         │            │            │
         └────────────┼────────────┘
                      ▼
              ┌───────────────┐
              │  MySQL (ai)   │
              │   R2 Storage  │
              │   NFS Models  │
              └───────────────┘
```

## Call Quality Features

### Transcription
- **WhisperX** for accurate speech-to-text
- Word-level timestamps for karaoke-style playback
- Speaker diarisation (agent vs supporter identification)
- Multiple languages supported (defaults to English)

### PII Detection
UK-specific patterns:
- National Insurance Numbers (AB123456C)
- NHS Numbers (123 456 7890)
- UK Postcodes (SW1A 1AA)
- UK Phone Numbers (07700 900123, +44 7700 900123)
- Bank Sort Codes (12-34-56)
- Bank Account Numbers (12345678)
- Credit/Debit Card Numbers
- Dates of Birth (DD/MM/YYYY)
- Email Addresses
- UK Driving Licence Numbers

### AI Analysis
Two operating modes:
1. **Audio Mode** - Direct audio analysis with Qwen2.5-Omni for tone/emotion detection
2. **Transcript Mode** - Text-based analysis for resource-constrained environments

Analysis outputs:
- Call summary (British English)
- Sentiment score (-10 to +10)
- Quality score (0-100)
- Topic classification (from configured topics list)
- Agent actions performed
- Performance scores (rubric-based)
- Action items
- Compliance flags

### Separate Models
- **Analysis Model** - Fine-tunable on human annotations
- **Chat Model** - Base model for conversations and queries

---

## Quick Start

Get started with the Call Quality module:

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- MySQL 8.0+
- SoX (for GSM to WAV conversion)
- ffmpeg

### Installation

```bash
# Clone repository
git clone https://github.com/Angel-FulFilment-Services/angel-intelligence.git
cd angel-intelligence

# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\Activate.ps1

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install SoX (Ubuntu/Debian)
sudo apt-get install sox libsox-fmt-all

# Install SoX (Windows - via chocolatey)
choco install sox

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings
```

### Configuration

```env
# Environment mode
ANGEL_ENV=development

# API Authentication (min 64 characters)
API_AUTH_TOKEN=your-secure-token-here-minimum-64-characters-long-for-security

# Database
AI_DB_HOST=localhost
AI_DB_DATABASE=ai
AI_DB_USERNAME=root
AI_DB_PASSWORD=

# PBX Recording Sources
PBX_LIVE_URL=https://pbx.angelfs.co.uk/callrec/
PBX_ARCHIVE_URL=https://afs-pbx-callarchive.angelfs.co.uk/

# Models
ANALYSIS_MODE=audio
WHISPER_MODEL=medium
ANALYSIS_MODEL=Qwen/Qwen2.5-Omni-7B
```

### Running

**Start API Server:**
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

**Start Worker:**
```bash
python -m src.worker.worker
```

## API Endpoints

All endpoints require Bearer token authentication:
```
Authorization: Bearer <API_AUTH_TOKEN>
```

### Health & Info
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check with model status (no auth) |
| GET | `/` | Service info |

### Recordings
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/recordings/submit` | Submit recording for processing |
| GET | `/recordings/{id}/status` | Get processing status |
| GET | `/recordings/{id}/transcription` | Get transcription |
| GET | `/recordings/{id}/analysis` | Get analysis |
| POST | `/recordings/reprocess/{id}` | Requeue for processing |
| GET | `/recordings/pending` | List pending recordings |
| GET | `/recordings/failed` | List failed recordings |
| POST | `/api/process` | Manually trigger single call processing |

### Chat & Summaries
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/chat` | Simple chat about call data |
| POST | `/api/chat` | Enhanced chat with filters |
| POST | `/api/summary/generate` | Generate monthly summary |

### Training & Configuration
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/config/analysis` | Get topics/actions/rubric |
| GET | `/api/config` | Get client configuration |
| POST | `/api/config` | Create/update client config |
| DELETE | `/api/config/{id}` | Deactivate client config |
| GET | `/api/training-data` | Export training annotations |
| POST | `/api/training/import` | Import training data |

## Processing Pipeline

1. **Load Config** - Load client-specific or default configuration
2. **Fetch Recording** - Download GSM from PBX (live or archive fallback)
3. **Convert Audio** - GSM to WAV using SoX (`sox input.gsm -r 8000 -b 32 -c 1 output.wav`)
4. **Transcribe** - WhisperX with word-level alignment and speaker diarisation
5. **Identify Speakers** - Voice fingerprinting to identify agents
6. **Detect PII** - Presidio + UK-specific patterns with timestamps
7. **Redact** - Replace PII in transcript with placeholders
8. **Analyse** - Qwen2.5-Omni for sentiment/quality/topics (audio or transcript mode)
9. **Store** - Save transcription and analysis to MySQL database
10. **Update Fingerprint** - Build/update agent voice fingerprint
11. **Handle Retention** - Delete audio or upload PII-redacted version to R2
12. **Clean Up** - Remove temporary files

## Development Mode

Set `ANGEL_ENV=development` for:
- Swagger UI at `/docs`
- ReDoc at `/redoc`
- Relaxed CORS
- Local file storage support
- Mock models (set `USE_MOCK_MODELS=true`)

### Mock Mode

For testing without GPU:
```env
USE_MOCK_MODELS=true
WHISPER_MODEL=tiny
```

Returns realistic mock responses for transcription and analysis.

## Production Deployment

### Kubernetes (Jetson Nano Cluster)

See [documentation/PRODUCTION_DEPLOYMENT.md](documentation/PRODUCTION_DEPLOYMENT.md) for full cluster setup.

```bash
# Configure secrets
cp k8s/secret.yaml.example k8s/secret.yaml
vim k8s/secret.yaml

# Deploy
kubectl apply -f k8s/

# Scale workers
kubectl scale deployment angel-intelligence-worker --replicas=4
```

### Docker

```bash
# Build
docker build -t angel-intelligence .

# Run API
docker run -d --gpus all -p 8000:8000 \
  --env-file .env \
  angel-intelligence uvicorn src.api:app --host 0.0.0.0

# Run Worker
docker run -d --gpus all \
  --env-file .env \
  angel-intelligence python -m src.worker.worker
```

## Database Schema

Tables in the `ai` database:

- `ai_call_recordings` - Recording queue and status
- `ai_call_transcriptions` - Transcription results with segments
- `ai_call_analysis` - Analysis results with sentiment/quality/topics
- `ai_call_annotations` - Human annotations for model fine-tuning
- `ai_monthly_summaries` - AI-generated monthly reports
- `ai_chat_conversations` - Chat session records
- `ai_chat_messages` - Individual chat messages
- `ai_voice_fingerprints` - Agent voice embeddings for speaker ID
- `ai_client_configs` - Client-specific configuration overrides

## Project Structure

```
angel-intelligence/
├── src/
│   ├── api/                      # FastAPI application
│   │   ├── app.py               # App factory with lifespan
│   │   ├── auth.py              # Bearer token authentication
│   │   └── routes.py            # All API endpoints
│   ├── config/                   # Configuration
│   │   ├── settings.py          # Pydantic environment settings
│   │   └── models.py            # Model configuration
│   ├── database/                 # Database layer
│   │   ├── connection.py        # MySQL connection pooling
│   │   └── models.py            # ORM models with retry logic
│   ├── services/                 # Core services
│   │   ├── audio_downloader.py  # PBX/R2 download + GSM conversion
│   │   ├── transcriber.py       # WhisperX transcription
│   │   ├── pii_detector.py      # UK PII detection + audio redaction
│   │   ├── analyzer.py          # AI analysis (audio/transcript modes)
│   │   └── voice_fingerprint.py # Speaker identification
│   └── worker/                   # Background processing
│       ├── processor.py         # Complete processing pipeline
│       └── worker.py            # Database polling worker
├── k8s/                          # Kubernetes manifests
│   ├── configmap.yaml           # Processing configuration
│   ├── deployment.yaml          # API + Worker deployments
│   └── secret.yaml.example      # Secret template
├── scripts/                      # Convenience scripts
│   ├── run_api.ps1/.sh          # Run API locally
│   ├── run_worker.ps1/.sh       # Run worker locally
│   ├── setup.ps1/.sh            # K8s initial setup
│   └── deploy.ps1/.sh           # Build and deploy
├── call_analysis_config.json     # Topics/actions/rubric
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Documentation

Comprehensive documentation is available in the [documentation/](documentation/) folder:

- [Quick Start Guide](documentation/QUICK_START.md)
- [Installation Guide](documentation/INSTALLATION.md)
- [API Reference](documentation/API_REFERENCE.md)
- [Production Deployment](documentation/PRODUCTION_DEPLOYMENT.md)
- [Environment Variables](documentation/ENVIRONMENT_VARIABLES.md)
- [Troubleshooting](documentation/TROUBLESHOOTING.md)

## Licence

Copyright © 2026 Angel Fulfilment Services. All rights reserved.

## Testing

### Upload Test Recording (Laravel)

```php
$service = app(\App\Services\CallRecordingService::class);
$recording = $service->uploadFromPath('/path/to/test.wav', 'TEST-001');
// Worker will automatically process it
```

### Monitor Processing

```bash
# Watch worker logs
tail -f worker.log

# Check database
mysql> SELECT id, apex_id, processing_status FROM ai_call_recordings;
```

## Performance

### Single Worker (Jetson Orin Nano 8GB):
- **Capacity: ~300 calls/day**

### 4-Worker Cluster:
- **Capacity: ~1,200 calls/day**
- Automatic load distribution

## Platform Benefits

✅ **Modular Architecture** - Add new AI capabilities without disrupting existing services  
✅ **Zero Laravel Load** - All AI processing happens in dedicated workers  
✅ **Fully Scalable** - Add more workers = more capacity  
✅ **GPU Optimised** - Runs on NVIDIA Jetson cluster for cost-effective inference  
✅ **Fault Tolerant** - Worker crashes auto-recover, jobs auto-retry  
✅ **Fine-Tunable** - Models can be trained on human annotations  
✅ **UK Focused** - British English, UK data patterns, GDPR compliant
