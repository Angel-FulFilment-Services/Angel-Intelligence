# Environment Variables Reference

Complete reference for all Angel Intelligence environment variables.

## Quick Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| ANGEL_ENV | Yes | - | Environment mode |
| API_AUTH_TOKEN | Yes | - | Bearer token (min 64 chars) |
| AI_DB_HOST | Yes | - | MySQL host |
| AI_DB_DATABASE | Yes | - | Database name |
| AI_DB_USERNAME | Yes | - | Database user |
| AI_DB_PASSWORD | Yes | - | Database password |
| USE_MOCK_MODELS | No | false | Enable mock mode |
| ANALYSIS_MODE | No | audio | audio or transcript |

---

## Core Settings

### ANGEL_ENV

**Required**: Yes  
**Type**: String  
**Values**: `development`, `staging`, `production`

Controls environment-specific behaviour:

| Environment | Swagger UI | Debug Logs | CORS | Mock Models |
|-------------|------------|------------|------|-------------|
| development | ✅ Enabled | ✅ Enabled | Permissive | Allowed |
| staging | ✅ Enabled | ⚠️ Warning | Restricted | Allowed |
| production | ❌ Disabled | ❌ Error only | Strict | ❌ Ignored |

```env
ANGEL_ENV=production
```

---

### API_AUTH_TOKEN

**Required**: Yes  
**Type**: String  
**Minimum Length**: 64 characters

Bearer token for API authentication. Generate a secure random string.

```bash
# Generate secure token
python -c "import secrets; print(secrets.token_urlsafe(64))"
```

```env
API_AUTH_TOKEN=your-secure-64-character-minimum-token-here-keep-this-secret
```

⚠️ **Security**: Never commit this to version control.

---

## Database Settings

### AI_DB_HOST

**Required**: Yes  
**Type**: String

MySQL server hostname or IP address.

```env
AI_DB_HOST=mysql.internal.angelfs.co.uk
```

---

### AI_DB_PORT

**Required**: No  
**Type**: Integer  
**Default**: 3306

MySQL server port.

```env
AI_DB_PORT=3306
```

---

### AI_DB_DATABASE

**Required**: Yes  
**Type**: String

Database name (must exist).

```env
AI_DB_DATABASE=ai
```

---

### AI_DB_USERNAME

**Required**: Yes  
**Type**: String

MySQL username with read/write access to the database.

```env
AI_DB_USERNAME=angel_ai
```

---

### AI_DB_PASSWORD

**Required**: Yes  
**Type**: String

MySQL password.

```env
AI_DB_PASSWORD=secure-password-here
```

---

## Model Settings

### USE_MOCK_MODELS

**Required**: No  
**Type**: Boolean  
**Default**: false

Enable mock mode for testing without GPU. Returns deterministic test data.

```env
USE_MOCK_MODELS=true
```

---

### USE_GPU

**Required**: No  
**Type**: Boolean  
**Default**: true (auto-detected)

Force CPU usage even if GPU is available.

```env
USE_GPU=false
```

---

### ANALYSIS_MODE

**Required**: No  
**Type**: String  
**Default**: audio  
**Values**: `audio`, `transcript`

| Mode | Description | Requirements |
|------|-------------|--------------|
| audio | Direct audio analysis with tone detection | GPU with 16GB+ VRAM |
| transcript | Text-based analysis only | Lower memory usage |

```env
ANALYSIS_MODE=audio
```

---

### WHISPER_MODEL

**Required**: No  
**Type**: String  
**Default**: medium  
**Values**: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`

| Model | VRAM | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | 1GB | Fastest | Lowest |
| base | 1GB | Fast | Low |
| small | 2GB | Medium | Good |
| medium | 5GB | Slow | Better |
| large-v3 | 10GB | Slowest | Best |

```env
WHISPER_MODEL=medium
```

---

### TRANSCRIPT_SEGMENTATION

**Required**: No  
**Type**: String  
**Default**: sentence  
**Values**: `word`, `sentence`

How to segment transcripts:
- `word`: Individual word segments with timestamps (for karaoke)
- `sentence`: Sentence-level segments (for display)

```env
TRANSCRIPT_SEGMENTATION=word
```

---

### MODELS_BASE_PATH

**Required**: No  
**Type**: String  
**Default**: ./models

Base path for model storage.

```env
MODELS_BASE_PATH=/nfs/models
```

---

### ANALYSIS_MODEL_PATH

**Required**: No  
**Type**: String  
**Default**: (auto from HuggingFace)

Path to fine-tuned analysis model.

```env
ANALYSIS_MODEL_PATH=/models/analysis/v1.0.0
```

---

### CHAT_MODEL_PATH

**Required**: No  
**Type**: String  
**Default**: (auto from HuggingFace)

Path to chat model.

```env
CHAT_MODEL_PATH=/models/chat/base
```

---

## Audio Sources

### PBX_LIVE_URL

**Required**: No  
**Type**: String  
**Default**: (none)

URL for live PBX recording downloads.

```env
PBX_LIVE_URL=https://pbx.angelfs.co.uk/callrec/
```

---

### PBX_ARCHIVE_URL

**Required**: No  
**Type**: String  
**Default**: (none)

URL for archived PBX recordings (fallback).

```env
PBX_ARCHIVE_URL=https://afs-pbx-callarchive.angelfs.co.uk/
```

---

## Cloudflare R2 Storage

### R2_ENDPOINT

**Required**: No  
**Type**: String

Cloudflare R2 endpoint URL.

```env
R2_ENDPOINT=https://abc123.r2.cloudflarestorage.com
```

---

### R2_ACCESS_KEY

**Required**: No  
**Type**: String

R2 access key ID.

```env
R2_ACCESS_KEY=your-access-key
```

---

### R2_SECRET_KEY

**Required**: No  
**Type**: String

R2 secret access key.

```env
R2_SECRET_KEY=your-secret-key
```

---

### R2_BUCKET

**Required**: No  
**Type**: String  
**Default**: angel-call-recordings

R2 bucket name for audio storage.

```env
R2_BUCKET=angel-call-recordings
```

---

## Worker Settings

### WORKER_MODE

**Required**: No  
**Type**: String  
**Default**: batch  
**Values**: `batch`, `interactive`, `both`

Determines what workload this worker handles:

- **`batch`**: Call processing (transcription, analysis) - long-running, GPU-heavy
- **`interactive`**: Real-time AI requests (chat, summaries) - fast response, user-facing
- **`both`**: All workloads (for development or single-node setups)

**Production Setup:**
- Deploy 3+ workers with `WORKER_MODE=batch` for call processing
- Deploy 1+ workers with `WORKER_MODE=interactive` for chat/summaries
- Adjust ratio based on workload

```env
# Batch worker
WORKER_MODE=batch

# Interactive worker
WORKER_MODE=interactive
```

---

### POLL_INTERVAL_SECONDS

**Required**: No  
**Type**: Integer  
**Default**: 30

How often workers check for new recordings.

```env
POLL_INTERVAL_SECONDS=30
```

---

### MAX_CONCURRENT_JOBS

**Required**: No  
**Type**: Integer  
**Default**: 4

Maximum parallel processing jobs per worker.

```env
MAX_CONCURRENT_JOBS=4
```

---

### MAX_RETRIES

**Required**: No  
**Type**: Integer  
**Default**: 3

Maximum retry attempts for failed recordings.

```env
MAX_RETRIES=3
```

---

### RETRY_DELAY_HOURS

**Required**: No  
**Type**: Integer  
**Default**: 1

Hours to wait before retrying failed recordings.

```env
RETRY_DELAY_HOURS=1
```

---

### WORKER_ID

**Required**: No  
**Type**: String  
**Default**: (auto-generated)

Unique identifier for this worker instance.

```env
WORKER_ID=jetson-01
```

---

## PII Settings

### ENABLE_PII_REDACTION

**Required**: No  
**Type**: Boolean  
**Default**: true

Enable PII detection and redaction.

```env
ENABLE_PII_REDACTION=true
```

---

## HuggingFace Settings

### HUGGINGFACE_TOKEN

**Required**: No  
**Type**: String

HuggingFace API token for:
- Downloading gated models
- pyannote.audio speaker diarisation

```env
HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

Get token from: https://huggingface.co/settings/tokens

---

## Model Hot Reload

### ENABLE_MODEL_HOT_RELOAD

**Required**: No  
**Type**: Boolean  
**Default**: false

Enable automatic model reloading when files change.

```env
ENABLE_MODEL_HOT_RELOAD=true
```

---

### MODEL_RELOAD_CHECK_INTERVAL

**Required**: No  
**Type**: Integer  
**Default**: 60

Seconds between checking for model updates.

```env
MODEL_RELOAD_CHECK_INTERVAL=60
```

---

## Logging

### LOG_LEVEL

**Required**: No  
**Type**: String  
**Default**: INFO  
**Values**: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`

```env
LOG_LEVEL=INFO
```

---

### LOG_FORMAT

**Required**: No  
**Type**: String  
**Default**: text  
**Values**: `text`, `json`

```env
LOG_FORMAT=json
```

---

## Example Configurations

### Development

```env
ANGEL_ENV=development
API_AUTH_TOKEN=dev-token-64-characters-minimum-for-local-development-only

AI_DB_HOST=localhost
AI_DB_DATABASE=ai_dev
AI_DB_USERNAME=root
AI_DB_PASSWORD=

USE_MOCK_MODELS=true
WHISPER_MODEL=tiny
LOG_LEVEL=DEBUG
```

### Production

```env
ANGEL_ENV=production
API_AUTH_TOKEN=<secure-production-token>

AI_DB_HOST=mysql.internal.angelfs.co.uk
AI_DB_PORT=3306
AI_DB_DATABASE=ai
AI_DB_USERNAME=angel_ai
AI_DB_PASSWORD=<secure-password>

USE_GPU=true
ANALYSIS_MODE=audio
WHISPER_MODEL=medium

PBX_LIVE_URL=https://pbx.angelfs.co.uk/callrec/
PBX_ARCHIVE_URL=https://afs-pbx-callarchive.angelfs.co.uk/

R2_ENDPOINT=https://abc123.r2.cloudflarestorage.com
R2_ACCESS_KEY=<access-key>
R2_SECRET_KEY=<secret-key>
R2_BUCKET=angel-call-recordings

POLL_INTERVAL_SECONDS=30
MAX_CONCURRENT_JOBS=4
MAX_RETRIES=3

ENABLE_PII_REDACTION=true
ENABLE_MODEL_HOT_RELOAD=true
MODEL_RELOAD_CHECK_INTERVAL=60

LOG_LEVEL=INFO
LOG_FORMAT=json
```
