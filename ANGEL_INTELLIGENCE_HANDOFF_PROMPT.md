# Angel Intelligence Backend - Complete Integration Handoff Prompt

## Overview

You are working on **Angel Intelligence** (formerly called "Harmony"), a Python backend service that handles AI-powered call transcription and analysis. The service integrates with a Laravel/PHP frontend application called **Pulse**.

**IMPORTANT FIRST STEPS:**
1. Rename the repository from "Harmony" / "Voice" to "Angel Intelligence"
2. Update all references in code, README, documentation, and configuration files
3. Update the README.md to reflect the new name and full feature set
4. **All outputs must be in British English (en-GB)** - this includes transcripts, summaries, chat responses, and all user-facing text

---

## Language Requirements (CRITICAL)

All text output from Angel Intelligence MUST use **United Kingdom English (British English)**:

- **Transcripts**: Use British spelling when correcting/formatting (e.g., "organisation" not "organization")
- **Summaries**: All AI-generated summaries in British English
- **Chat responses**: AI Assistant must respond in British English
- **Analysis outputs**: All text fields use British spelling and conventions
- **Date formats**: Use DD/MM/YYYY or "17 January 2026" format
- **Currency**: Use £ (GBP) when referencing money

**Common spelling differences to enforce:**
- -ise not -ize (organise, recognise, analyse)
- -our not -or (colour, behaviour, favour)
- -re not -er (centre, metre)
- -ogue not -og (catalogue, dialogue)
- -ence not -ense (licence, defence)

---

## Authentication

All API requests from Pulse to Angel Intelligence must include a **Bearer token** in the Authorization header:

```
Authorization: Bearer <API_TOKEN>
```

- Token is configured via environment variable `API_AUTH_TOKEN`
- Validate token on every request
- Return `401 Unauthorized` for missing/invalid tokens
- Token should be a long random string (minimum 64 characters)

---

## Current State of the Python Backend

The existing Python backend:
- Watches the `ai_call_recordings` MySQL table every 30 seconds for pending calls
- Attempts to find recordings from R2 bucket or local file storage
- Uses **OpenAI Whisper** for transcription
- Uses **Qwen/Qwen2.5-Omni-7B** LLM for call analysis
- Has formatting issues where the analysis sometimes includes the raw prompt in the summary output

---

## Infrastructure Architecture

### Kubernetes Jetson Nano Cluster

Angel Intelligence runs on a **Kubernetes cluster of NVIDIA Jetson Nano devices**. This has specific architectural implications:

1. **API Gateway / Load Balancer**
   - All API requests come through a single endpoint
   - The gateway must route requests to the appropriate worker node
   - Implement health checks to only route to healthy workers
   - Use round-robin or least-connections for load balancing

2. **Worker Distribution**
   - Each Jetson Nano runs as a worker pod
   - Workers should be stateless - all state lives in the MySQL database
   - Workers claim jobs by updating `processing_status` with atomic operations
   - Use database locking to prevent multiple workers claiming the same job

3. **Shared Model Storage (CRITICAL)**
   - Models must be stored on a **shared NFS volume** or similar persistent storage
   - All worker nodes mount the same model directory
   - This ensures:
     - Training only needs to happen once
     - All workers use the same model version
     - Model updates are immediately available to all workers
   - Suggested mount path: `/models/` with subdirectories for each model type

4. **Model Loading Strategy**
   - Load models into memory on worker startup
   - Keep models loaded between requests (don't reload per-request)
   - Implement model hot-reloading when new versions are deployed
   - Consider model sharding if memory is constrained

### Kubernetes Configuration Recommendations

```yaml
# Shared model volume
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: angel-intelligence-models
spec:
  accessModes:
    - ReadWriteMany  # Multiple workers can read
  resources:
    requests:
      storage: 50Gi  # Adjust based on model sizes

---
# Worker deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: angel-intelligence-worker
spec:
  replicas: 4  # Number of Jetson Nano nodes
  template:
    spec:
      containers:
      - name: worker
        image: angel-intelligence:latest
        volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true  # Workers only read, training pod writes
        resources:
          limits:
            nvidia.com/gpu: 1  # Each Jetson has 1 GPU
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: angel-intelligence-models
```

---

## Local Development & Testing Environment

Angel Intelligence must support running in a **local development environment** without the Kubernetes/Jetson cluster. This enables end-to-end testing before deployment.

### Environment Modes

Set via environment variable `ANGEL_ENV`:
- `production` - Full Kubernetes cluster on Jetson Nanos
- `development` - Local single-instance mode

### Local Development Configuration

```env
# Environment mode
ANGEL_ENV=development

# Local model storage (not NFS)
MODELS_BASE_PATH=./models
ANALYSIS_MODEL_PATH=./models/analysis/current
CHAT_MODEL_PATH=./models/chat/current
WHISPER_MODEL_PATH=./models/whisper/current

# Use smaller/quantized models for local testing
WHISPER_MODEL=openai/whisper-small  # or whisper-tiny for faster testing
ANALYSIS_MODEL_QUANTIZATION=int4    # Lower memory usage
CHAT_MODEL_QUANTIZATION=int4

# Single worker mode
MAX_CONCURRENT_JOBS=1
WORKER_ID=local-dev

# Disable cluster features
ENABLE_MODEL_HOT_RELOAD=false

# Optional: Use mock/stub models for faster testing
USE_MOCK_MODELS=false  # Set true to skip actual LLM inference
```

### Local Development Features

1. **Single Worker Mode**
   - No load balancing needed
   - No database locking required (only one worker)
   - Models loaded once at startup

2. **Smaller Models**
   - Use `whisper-tiny` or `whisper-small` instead of `whisper-medium`
   - Use higher quantization (int4) to reduce memory
   - Accept lower accuracy for faster iteration

3. **Mock Mode** (optional)
   - Set `USE_MOCK_MODELS=true` to return deterministic test data
   - Useful for testing API integration without GPU
   - Returns realistic-looking fake transcripts/analysis

4. **Local File Storage**
   - Models stored in `./models/` directory
   - No NFS mount required
   - Audio temp files in `./temp/`

### Mock Model Responses

When `USE_MOCK_MODELS=true`, return consistent test data:

```python
MOCK_TRANSCRIPTION = {
    "full_transcript": "Agent: Hello, thank you for calling...",
    "segments": [...],  # Predefined test segments
    "confidence_score": 0.95,
    "language_detected": "en"
}

MOCK_ANALYSIS = {
    "summary": "The supporter enquired about regular giving options...",
    "sentiment_score": 5.5,
    "sentiment_label": "positive",
    "quality_score": 78.5,
    # ... full mock response
}
```

### Docker Compose for Local Development

```yaml
version: '3.8'
services:
  angel-intelligence:
    build: .
    environment:
      - ANGEL_ENV=development
      - USE_MOCK_MODELS=false
      - AI_DB_HOST=host.docker.internal
      - AI_DB_PORT=3306
      - AI_DB_DATABASE=pulse
    volumes:
      - ./models:/app/models
      - ./temp:/app/temp
    ports:
      - "8000:8000"
    # No GPU required for mock mode, optional for real inference
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - capabilities: [gpu]
```

### End-to-End Testing Checklist

1. **API Tests** (no GPU needed)
   - [ ] Health endpoint returns correctly
   - [ ] Auth token validation works
   - [ ] Invalid requests return proper errors
   - [ ] Config endpoints work

2. **Processing Tests** (mock mode)
   - [ ] Recording status updates correctly
   - [ ] Retry logic triggers after failures
   - [ ] Database records created properly
   - [ ] R2 upload simulated correctly

3. **Processing Tests** (real models, GPU optional)
   - [ ] GSM download from test URL
   - [ ] SoX conversion works
   - [ ] Whisper transcription runs
   - [ ] LLM analysis produces valid JSON
   - [ ] PII detection finds test patterns
   - [ ] Audio redaction works

4. **Integration Tests**
   - [ ] Pulse can call all endpoints
   - [ ] Polling picks up status changes
   - [ ] Chat conversations work end-to-end

---

## Model Architecture (IMPORTANT)

### Separate Models for Different Tasks

Angel Intelligence uses **TWO distinct model instances**:

#### 1. Call Analysis Model (`ANALYSIS_MODEL`)
- **Purpose**: Call analysis - sentiment detection, quality scoring, topic extraction
- **Base Model**: Qwen/Qwen2.5-Omni-7B (multi-modal: supports both audio and text input)
- **Fine-tuning**: YES - This model will be fine-tuned using human annotations
- **Training Data**: From `ai_call_annotations` table
- **Location**: `/models/analysis/`
- **Versioning**: Track versions in `/models/analysis/versions/`

#### 2. General LLM (`CHAT_MODEL`)
- **Purpose**: Chat conversations, monthly summaries, ad-hoc queries
- **Base Model**: Qwen/Qwen2.5-Omni-7B (or similar)
- **Fine-tuning**: NO - Uses base model only, no custom training
- **Location**: `/models/chat/`
- **Note**: This model remains static and uses the vendor's base training

### Analysis Mode: Audio vs Transcript

The system supports **two analysis modes**, configurable per-client or globally:

#### Mode 1: Direct Audio Analysis (`ANALYSIS_MODE=audio`)
- Feed the audio file directly to the multi-modal LLM (Qwen2.5-Omni)
- The model processes audio natively - no separate transcription step needed
- **Advantages**:
  - Captures tone, emotion, hesitation, speaking pace directly from audio
  - More accurate sentiment detection from vocal cues
  - Single model call instead of Whisper + LLM
- **Disadvantages**:
  - Requires multi-modal model with audio support
  - Higher compute requirements
  - Cannot use transcript-only models

#### Mode 2: Transcript Analysis (`ANALYSIS_MODE=transcript`)
- First transcribe audio using Whisper
- Then feed transcript text to the LLM for analysis
- **Advantages**:
  - Works with text-only LLMs
  - Lower compute requirements
  - Transcript is always available regardless of analysis
- **Disadvantages**:
  - Loses audio cues (tone, pace, hesitation)
  - Two-step process (Whisper + LLM)

#### Hybrid Approach (Recommended)
- **Always run Whisper** to generate transcript (needed for karaoke, search, PII detection)
- **Choose analysis mode** based on configuration:
  - Audio mode: Pass audio to multi-modal LLM for richer analysis
  - Transcript mode: Pass transcript text to LLM

```python
# Analysis mode configuration
ANALYSIS_MODE = os.getenv('ANALYSIS_MODE', 'audio')  # 'audio' or 'transcript'

def analyse_call(audio_path: str, transcript: dict, config: dict):
    if ANALYSIS_MODE == 'audio':
        # Multi-modal: feed audio directly to Qwen2.5-Omni
        return analyse_from_audio(audio_path, config)
    else:
        # Text-only: feed transcript to LLM
        return analyse_from_transcript(transcript['full_transcript'], config)
```

### Why Separate Models?

1. **Training Isolation**: Fine-tuning the analysis model won't affect chat responses
2. **Rollback Safety**: Can rollback analysis model without impacting chat
3. **Resource Optimization**: Can load only the model needed for each task
4. **Quality Control**: Analysis model improves with annotations; chat stays consistent

### Model Configuration

```python
# models/config.py
MODEL_CONFIG = {
    "analysis": {
        "base_model": "Qwen/Qwen2.5-Omni-7B",
        "path": "/models/analysis/current",
        "versions_path": "/models/analysis/versions",
        "supports_finetuning": True,
        "supports_audio": True,  # Multi-modal audio support
        "quantization": "int8",  # For Jetson memory constraints
    },
    "chat": {
        "base_model": "Qwen/Qwen2.5-Omni-7B", 
        "path": "/models/chat/current",
        "supports_finetuning": False,
        "supports_audio": False,  # Text-only for chat
        "quantization": "int8",
    },
    "whisper": {
        "model": "openai/whisper-medium",
        "path": "/models/whisper/current",
        "supports_finetuning": False,
        "always_run": True,  # Always generate transcript regardless of analysis mode
    }
}
```

---

## Call Analysis Configuration

The analysis model uses a configuration file to guide its output. This ensures consistent topic categorisation and scoring across all analyses.

### Configuration File: `call_analysis_config.json`

```json
{
  "topics": [
    "One-off donation request",
    "Regular giving signup",
    "Donation completion/payment processing",
    "Donation refusal/decline",
    "Upgrade/downgrade of monthly donation",
    "Fundraising campaign discussion",
    "Gift Aid explanation/enrolment",
    "Pledge follow-up",
    "High-value donor stewardship",
    "Corporate giving enquiry",
    "Emergency appeal update",
    "Donation receipt confirmation",
    "Address capture",
    "Address update/change of details",
    "Email/phone update",
    "Bank detail update",
    "Payment card update",
    "Consent/preferences management",
    "Identity verification",
    "Supporter account enquiry",
    "Membership/regular giver account changes",
    "Mission information request",
    "Charity impact/programme explanation",
    "Information about beneficiaries",
    "How donations are used",
    "Event participation enquiry",
    "Volunteering enquiry",
    "Educational resource request",
    "Complaint about agent behaviour",
    "Complaint about charity/services",
    "Cancellation of regular giving",
    "Retention attempt / save conversation",
    "Call-back request / follow-up needed",
    "PCI/Payment compliance discussion",
    "Safeguarding concern",
    "Vulnerable person indicators",
    "Legacy/bequest enquiry",
    "Free-gift fulfilment / merchandise query",
    "General admin/enquiry",
    "Wrong number / misdialled contact"
  ],
  "agent_actions": [
    "Greeted supporter",
    "Verified supporter identity",
    "Built rapport / established connection",
    "Listened actively / acknowledged concerns",
    "Asked clarifying questions",
    "Provided requested information",
    "Explained charity mission/programs",
    "Gave policy/compliance information",
    "Captured mailing address",
    "Updated supporter address",
    "Updated email/phone",
    "Updated payment method",
    "Recorded donation details",
    "Confirmed marketing consent preferences",
    "Made initial donation ask",
    "Requested donation (one-off)",
    "Requested regular giving signup",
    "Requested upgrade to existing donation",
    "Made secondary/follow-up ask after decline",
    "Handled objection or hesitation",
    "Overcame objection successfully",
    "Accepted decline gracefully",
    "Offered alternative giving option",
    "Processed donation payment",
    "Attempted retention/save during cancellation",
    "Attempted supporter conversion",
    "Arranged free-gift fulfilment/shipping",
    "Handled complaint",
    "Resolved supporter issue",
    "Escalated to supervisor/team",
    "Scheduled callback",
    "Closed the enquiry clearly",
    "Thanked supporter for their time"
  ],
  "performance_rubric": [
    "Clarity of speech",
    "Tone control",
    "Active listening",
    "Empathy & rapport",
    "Confidence & authority",
    "Accurate information delivery",
    "Script/protocol adherence",
    "Payment and data protection compliance",
    "Recording of mandatory information",
    "Call structure/flow control",
    "Quality of donation ask or conversion attempt",
    "Objection handling skill",
    "Engagement effectiveness",
    "Problem solving",
    "Effective closing"
  ]
}
```

### Using the Configuration in Analysis

The analysis prompt should incorporate this configuration to ensure:

1. **Topic Classification**: Only use topics from the predefined list
2. **Agent Action Tracking**: Identify which actions the agent performed (especially sales attempts and objection handling)
3. **Performance Scoring**: Score each rubric item for quality assessment
4. **British English**: All output text must use UK English spelling and conventions

---

## Critical Issues to Fix

### 1. Recording Retrieval Method (PRIORITY)
The current method of finding recordings needs to be updated to match the pattern used in the Laravel application's `CallRecordings.php` helper.

**The Laravel pattern for finding GSM recordings:**

```php
// Primary source - live PBX recordings
$url = 'https://pbx.angelfs.co.uk/callrec/' . $apex_id . '.gsm';

// Archive fallback 1
$url = 'https://afs-pbx-callarchive.angelfs.co.uk/monitor-' . $year . '/' . $month . '/' . $apex_id . '.gsm';

// Archive fallback 2
$url = 'https://afs-pbx-callarchive.angelfs.co.uk/monitor-' . $year . '/' . $month . '/monitor/' . $apex_id . '.gsm';
```

**Python implementation should:**
1. Try each URL in order until one returns HTTP 200
2. Download the `.gsm` file
3. Convert from GSM to WAV format using SoX (command: `sox input.gsm -r 8000 -b 32 -c 1 output.wav`)
4. Use the WAV file for transcription
5. **Delete the audio file after processing** unless `retain_audio = TRUE` in the database
6. If `retain_audio = TRUE`, store a **PII-redacted audio version** in R2 and update `r2_path`

**Required fields from the database to construct URLs:**
- `apex_id` - The unique call identifier
- `call_date` - To extract year and month for archive paths

### 2. Audio Retention and R2 Storage

**Default behaviour**: Delete all audio files after transcription and analysis.

**If `retain_audio = TRUE`**:
1. Create a PII-redacted version of the audio (bleep/silence PII segments)
2. Upload redacted audio to R2 bucket
3. Update `r2_path` and `r2_bucket` fields in the database
4. Delete the original unredacted audio

**Pulse will check `r2_path`**: If populated, it fetches audio from R2. If empty, it fetches from PBX/archive directly.

### 3. Analysis Output Formatting (PRIORITY)
The LLM analysis is producing malformed output that sometimes includes the prompt itself in the response. Fix this by:
- Using structured output/JSON mode if available
- Implementing strict output parsing with validation
- Adding retry logic when output doesn't match expected schema
- Using proper prompt engineering with clear delimiters and examples

### 4. Processing Retry Logic

When processing fails:
1. Mark as `failed` with error message
2. Automatically retry after **1 hour**
3. Maximum **3 retry attempts**
4. After 3 failures, leave as `failed` permanently (no notifications needed)

Track retries using `retry_count` field in database.

---

## Database Schema (MySQL - `ai` connection)

All tables use the `ai` database connection and are prefixed with `ai_`.

### Table: `ai_call_recordings`
```sql
CREATE TABLE ai_call_recordings (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    apex_id VARCHAR(255) UNIQUE NOT NULL,       -- Unique call ID from apex_data system
    client_ref VARCHAR(255) NULL,                -- Client reference code
    campaign VARCHAR(255) NULL,                  -- Campaign name
    halo_id BIGINT UNSIGNED NULL,                -- Agent ID from Halo system
    agent_name VARCHAR(255) NULL,                -- Agent display name
    creative VARCHAR(255) NULL,                  -- Creative name
    direction VARCHAR(255) NULL,                 -- 'inbound' or 'outbound'
    invoicing VARCHAR(255) NULL,                 -- Invoicing category
    call_date DATE NULL,                         -- Date of the call
    retain_audio BOOLEAN DEFAULT FALSE,          -- If TRUE, store redacted audio in R2
    r2_path VARCHAR(255) NULL,                   -- Path to redacted audio in R2 (if retained)
    r2_bucket VARCHAR(255) NULL,                 -- R2 bucket name
    duration_seconds INT NULL,                   -- Call duration
    file_size_bytes BIGINT NULL,                 -- File size
    file_format VARCHAR(255) DEFAULT 'wav',      -- Audio format
    processing_status ENUM('pending', 'queued', 'processing', 'completed', 'failed') DEFAULT 'pending',
    processing_error TEXT NULL,                  -- Error message if failed
    retry_count TINYINT UNSIGNED DEFAULT 0,      -- Number of processing retries (max 3)
    next_retry_at TIMESTAMP NULL,                -- When to retry failed processing
    uploaded_at TIMESTAMP NULL,
    processing_started_at TIMESTAMP NULL,
    processing_completed_at TIMESTAMP NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    deleted_at TIMESTAMP NULL,
    
    INDEX idx_apex_id (apex_id),
    INDEX idx_client_ref (client_ref),
    INDEX idx_campaign (campaign),
    INDEX idx_call_date (call_date),
    INDEX idx_processing_status (processing_status),
    INDEX idx_halo_id (halo_id),
    INDEX idx_agent_name (agent_name),
    INDEX idx_next_retry (next_retry_at)
);
```

**Processing Status Flow:**
- `pending` → Initial state, ready to be picked up
- `queued` → Submitted via UI for processing
- `processing` → Currently being transcribed/analysed
- `completed` → Successfully processed
- `failed` → Processing failed (check `processing_error`, will retry if `retry_count < 3`)

**Audio Retention:**
- If `retain_audio = FALSE` (default): Audio deleted after processing
- If `retain_audio = TRUE`: Redacted audio stored in R2, path saved to `r2_path`
- Pulse checks `r2_path`: if set, fetch from R2; if empty, fetch from PBX/archive

### Table: `ai_call_transcriptions`
```sql
CREATE TABLE ai_call_transcriptions (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    ai_call_recording_id BIGINT NOT NULL,        -- FK to ai_call_recordings
    full_transcript LONGTEXT NOT NULL,           -- Complete transcript text
    segments JSON NULL,                          -- Word-level segments with timestamps
    redacted_transcript LONGTEXT NULL,           -- PII-redacted version
    pii_detected JSON NULL,                      -- PII entities found (see UK PII patterns below)
    language_detected VARCHAR(255) NULL,         -- Detected language code
    confidence_score DECIMAL(5,4) NULL,          -- 0.0000 to 1.0000
    model_used VARCHAR(255) DEFAULT 'whisper-medium',
    processing_time_seconds INT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    
    FOREIGN KEY (ai_call_recording_id) REFERENCES ai_call_recordings(id) ON DELETE CASCADE,
    INDEX idx_recording (ai_call_recording_id)
);
```

**UK PII Detection Patterns (CRITICAL):**
The system must detect and redact UK-specific PII:

| PII Type | Pattern/Example | Regex Pattern |
|----------|-----------------|---------------|
| National Insurance Number | AB123456C | `[A-Z]{2}\d{6}[A-Z]` |
| NHS Number | 123 456 7890 | `\d{3}\s?\d{3}\s?\d{4}` |
| UK Postcode | SW1A 1AA | `[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}` |
| UK Phone Number | 07700 900123, +44 7700 900123 | `(\+44\s?|0)7\d{3}\s?\d{6}` |
| UK Bank Sort Code | 12-34-56 | `\d{2}-\d{2}-\d{2}` |
| UK Bank Account | 12345678 | `\d{8}` (in context of banking) |
| Credit/Debit Card | 4111 1111 1111 1111 | `\d{4}\s?\d{4}\s?\d{4}\s?\d{4}` |
| Card Expiry | 12/26, 12/2026 | `\d{2}/\d{2,4}` |
| CVV/CVC | 123 | `\d{3}` (in context of payment) |
| Date of Birth | 17/01/1990 | `\d{2}/\d{2}/\d{4}` |
| Email Address | user@example.com | Standard email regex |
| UK Driving Licence | SMITH901017AB1CD | `[A-Z]{5}\d{6}[A-Z]{2}\d[A-Z]{2}` |

**PII Detected JSON Structure:**
```json
[
    {
        "type": "national_insurance_number",
        "original": "AB123456C",
        "redacted": "[NI_NUMBER]",
        "timestamp_start": 45.2,
        "timestamp_end": 47.8,
        "confidence": 0.95
    },
    {
        "type": "credit_card",
        "original": "4111 1111 1111 1111",
        "redacted": "[CARD_NUMBER]",
        "timestamp_start": 120.5,
        "timestamp_end": 125.3,
        "confidence": 0.99
    }
]
```

**Audio PII Redaction:**
When `retain_audio = TRUE`, the redacted audio should have PII segments replaced with:
- A consistent tone/beep, OR
- Silence
Use the `timestamp_start` and `timestamp_end` from detected PII to identify audio segments to redact.

**Segments JSON Structure (required for transcript karaoke feature):**
```json
[
    {
        "text": "Hello, thank you for calling.",
        "start": 0.0,
        "end": 2.5,
        "speaker": "agent",
        "speaker_id": "agent_001",
        "confidence": 0.95,
        "words": [
            {"word": "Hello", "start": 0.0, "end": 0.4, "confidence": 0.98},
            {"word": "thank", "start": 0.5, "end": 0.7, "confidence": 0.96}
        ]
    }
]
```

### Table: `ai_call_analysis`
```sql
CREATE TABLE ai_call_analysis (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    ai_call_recording_id BIGINT NOT NULL,        -- FK to ai_call_recordings
    summary TEXT NULL,                           -- AI-generated call summary (2-4 sentences)
    sentiment_score DECIMAL(4,2) NULL,           -- Scale: -10.00 to +10.00
    sentiment_label ENUM('very_negative', 'negative', 'neutral', 'positive', 'very_positive') NULL,
    key_topics JSON NULL,                        -- Array of identified topics (from config)
    agent_actions_performed JSON NULL,           -- Actions agent took (from config)
    performance_scores JSON NULL,                -- Rubric scores (from config)
    action_items JSON NULL,                      -- Extracted action items / follow-ups
    quality_score DECIMAL(5,2) NULL,             -- 0.00 to 100.00 (derived from performance_scores)
    compliance_flags JSON NULL,                  -- Compliance issues detected
    speaker_metrics JSON NULL,                   -- Talk time, interruptions per speaker
    audio_analysis JSON NULL,                    -- Raw audio analysis data
    model_used VARCHAR(255) DEFAULT 'qwen2.5-omni-7b',
    model_version VARCHAR(255) NULL,             -- Specific model version used
    processing_time_seconds INT NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    
    FOREIGN KEY (ai_call_recording_id) REFERENCES ai_call_recordings(id) ON DELETE CASCADE,
    INDEX idx_recording (ai_call_recording_id),
    INDEX idx_sentiment (sentiment_label),
    INDEX idx_quality (quality_score)
);
```

**Key Topics JSON Structure:**
```json
[
    {
        "name": "Regular giving signup",
        "confidence": 0.92,
        "timestamp_start": 45.5,
        "timestamp_end": 78.2
    },
    {
        "name": "Gift Aid explanation/enrolment",
        "confidence": 0.88,
        "timestamp_start": 120.0,
        "timestamp_end": 180.5
    }
]
```

**Agent Actions Performed JSON Structure:**
```json
[
    {
        "action": "Greeted supporter",
        "timestamp_start": 0.0,
        "quality": 5
    },
    {
        "action": "Verified supporter identity",
        "timestamp_start": 15.2,
        "quality": 4
    },
    {
        "action": "Requested regular giving signup",
        "timestamp_start": 120.5,
        "quality": 5
    }
]
```

**Performance Scores JSON Structure:**
```json
{
    "Clarity of speech": 8,
    "Tone control": 9,
    "Active listening": 7,
    "Empathy & rapport": 8,
    "Confidence & authority": 7,
    "Accurate information delivery": 9,
    "Script/protocol adherence": 8,
    "Payment and data protection compliance": 10,
    "Recording of mandatory information": 9,
    "Call structure/flow control": 7,
    "Quality of donation ask or conversion attempt": 8,
    "Objection handling skill": 6,
    "Engagement effectiveness": 8,
    "Problem solving": 7,
    "Effective closing": 8
}
```

**Action Items JSON Structure:**
```json
[
    {
        "description": "Send follow-up email with pricing details",
        "priority": "high",
        "due_date": "2026-01-20"
    }
]
```

**Compliance Flags JSON Structure:**
```json
[
    {
        "type": "missing_disclosure",
        "issue": "Call recording disclosure not provided",
        "severity": "high",
        "timestamp_start": 0.0,
        "timestamp_end": 15.0
    }
]
```

**Speaker Metrics JSON Structure:**
```json
{
    "agent": {
        "talk_time_seconds": 245,
        "interruptions": 2,
        "average_pace_wpm": 145,
        "silence_percentage": 12
    },
    "supporter": {
        "talk_time_seconds": 180,
        "interruptions": 1,
        "average_pace_wpm": 130,
        "sentiment_trend": "positive"
    }
}
```

### Table: `ai_call_annotations` (Human Corrections for Training)
```sql
CREATE TABLE ai_call_annotations (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    ai_call_analysis_id BIGINT NOT NULL,         -- FK to ai_call_analysis
    user_id BIGINT UNSIGNED NOT NULL,            -- User who created annotation
    annotation_type VARCHAR(255) NOT NULL,       -- 'sentiment', 'quality', 'compliance', 'custom_tag', 'segment_flag'
    field_name VARCHAR(255) NULL,                -- Which field is being corrected
    original_value TEXT NULL,                    -- Original AI-generated value
    corrected_value TEXT NULL,                   -- Human-corrected value
    timestamp_start DECIMAL(10,3) NULL,          -- For segment annotations
    timestamp_end DECIMAL(10,3) NULL,
    tags JSON NULL,                              -- Custom tags/labels
    notes TEXT NULL,                             -- Annotator notes
    is_training_data BOOLEAN DEFAULT TRUE,       -- Whether to use for model training
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    
    FOREIGN KEY (ai_call_analysis_id) REFERENCES ai_call_analysis(id) ON DELETE CASCADE,
    INDEX idx_analysis (ai_call_analysis_id),
    INDEX idx_user (user_id),
    INDEX idx_type (annotation_type),
    INDEX idx_training (is_training_data)
);
```

### Table: `ai_monthly_summaries`
```sql
CREATE TABLE ai_monthly_summaries (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    feature VARCHAR(255) NOT NULL,               -- 'call_quality', etc.
    month SMALLINT UNSIGNED NOT NULL,            -- 1-12
    year SMALLINT UNSIGNED NOT NULL,             -- e.g., 2026
    client_ref VARCHAR(255) NULL,                -- Filter by client
    campaign VARCHAR(255) NULL,                  -- Filter by campaign
    agent_id BIGINT UNSIGNED NULL,               -- Filter by agent
    summary_data JSON NOT NULL,                  -- AI-generated summary
    metrics JSON NULL,                           -- Aggregated metrics
    generated_at TIMESTAMP NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    
    UNIQUE KEY unique_summary (feature, month, year, client_ref, campaign, agent_id),
    INDEX idx_feature (feature),
    INDEX idx_month_year (month, year)
);
```

**Summary Data JSON Structure:**
```json
{
    "summary": "This month saw strong call quality performance with an average satisfaction score of 78%...",
    "key_insights": [
        "Positive trend in supporter satisfaction compared to last month",
        "Technical support calls have longest average duration"
    ],
    "recommendations": [
        "Consider additional training for evening shift agents",
        "Update FAQ documentation for common billing questions"
    ]
}
```

### Table: `ai_chat_conversations`
```sql
CREATE TABLE ai_chat_conversations (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT UNSIGNED NOT NULL,
    feature VARCHAR(255) NOT NULL,               -- 'call_quality', etc.
    filters JSON NULL,                           -- Active filters when conversation started
    title VARCHAR(255) NULL,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    
    INDEX idx_user (user_id),
    INDEX idx_feature (feature)
);
```

### Table: `ai_chat_messages`
```sql
CREATE TABLE ai_chat_messages (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    ai_chat_conversation_id BIGINT NOT NULL,
    role ENUM('user', 'assistant', 'system') NOT NULL,
    content TEXT NOT NULL,
    metadata JSON NULL,                          -- tokens_used, model, etc.
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    
    FOREIGN KEY (ai_chat_conversation_id) REFERENCES ai_chat_conversations(id) ON DELETE CASCADE,
    INDEX idx_conversation (ai_chat_conversation_id)
);
```

### Table: `ai_voice_fingerprints` (Agent Voice Identification)
```sql
CREATE TABLE ai_voice_fingerprints (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    halo_id BIGINT UNSIGNED NOT NULL,            -- Agent ID from Halo system
    agent_name VARCHAR(255) NOT NULL,            -- Agent display name
    fingerprint_data BLOB NOT NULL,              -- Voice embedding/fingerprint vector
    sample_count INT DEFAULT 0,                  -- Number of samples used to build fingerprint
    confidence_threshold DECIMAL(3,2) DEFAULT 0.85, -- Min confidence to match
    last_updated_at TIMESTAMP NULL,              -- When fingerprint was last refined
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    
    UNIQUE KEY unique_agent (halo_id),
    INDEX idx_agent_name (agent_name)
);
```

**Voice Fingerprinting System:**

Angel Intelligence should implement voice fingerprinting to identify which specific agent is speaking:

1. **Building Fingerprints**
   - When processing calls, extract voice embeddings for each speaker
   - If `halo_id` is known for a call, associate embeddings with that agent
   - Accumulate samples to build robust agent fingerprints
   - Use speaker diarization to separate voices before fingerprinting

2. **Matching Speakers**
   - When processing new calls, compare voice embeddings against known fingerprints
   - If match confidence > threshold, identify speaker as that agent
   - Support multiple agents on one call (call transfers)
   - Label unknown speakers as "agent_unknown" or "supporter"

3. **Speaker Labels in Segments**
   - `speaker`: "agent" | "supporter" | "agent_unknown"
   - `speaker_id`: Specific identifier (e.g., "agent_1001" for halo_id 1001)
   - `speaker_confidence`: Confidence of speaker identification

4. **Handling Call Transfers**
   - Detect when voice characteristics change mid-call
   - Identify new agent if fingerprint matches
   - Track handover timestamp in analysis

**Recommended Libraries:**
- `resemblyzer` or `speechbrain` for speaker embeddings
- `pyannote.audio` for speaker diarization

### Table: `ai_client_configs` (Client-Specific Configuration)
```sql
CREATE TABLE ai_client_configs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    client_ref VARCHAR(255) NULL,                -- NULL = default/global config
    config_type VARCHAR(50) NOT NULL,            -- 'topics', 'agent_actions', 'performance_rubric', 'prompt'
    config_data JSON NOT NULL,                   -- The configuration data
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    
    UNIQUE KEY unique_config (client_ref, config_type),
    INDEX idx_client (client_ref),
    INDEX idx_type (config_type)
);
```

**Client Configuration System:**

The `call_analysis_config.json` values can be overridden per-client:

1. **Lookup Order**
   - Check for client-specific config (`client_ref = 'ACME001'`)
   - Fall back to global config (`client_ref = NULL`)
   - Fall back to hardcoded defaults

2. **Config Types**
   - `topics` - Custom topic list for this client
   - `agent_actions` - Custom agent action list
   - `performance_rubric` - Custom scoring rubric
   - `prompt` - Custom analysis prompt template
   - `analysis_mode` - Override analysis mode ('audio' or 'transcript')

3. **API to Manage Configs**
```
GET /api/config?client_ref=ACME001&type=topics
POST /api/config
PUT /api/config/{id}
DELETE /api/config/{id}
```

4. **Prompt Customisation**
   - Store full prompt templates in database
   - Support variable substitution: `{transcript}`, `{config}`, `{client_name}`
   - Validate prompts produce valid JSON before saving

5. **Analysis Mode Override**
   - Global default set via `ANALYSIS_MODE` environment variable
   - Per-client override via `ai_client_configs` with `config_type = 'analysis_mode'`
   - Config data: `{"mode": "audio"}` or `{"mode": "transcript"}`

---

## Required API Endpoints

The Python backend should expose these endpoints for the Laravel frontend to consume.

**Cluster Routing Notes:**
- All endpoints go through a single API gateway URL
- The gateway routes to available worker pods
- Each worker can handle any endpoint
- Workers use the appropriate model based on the endpoint:
  - `/api/process`, `/api/training/*` → Analysis Model
  - `/api/chat`, `/api/summary/*` → Chat Model
  - `/health` → No model needed

### 1. Health Check
```
GET /health
Response: { 
    "status": "healthy", 
    "version": "1.0.0",
    "worker_id": "angel-worker-abc123",
    "models_loaded": {
        "analysis": {"version": "v1.2.0", "loaded": true},
        "chat": {"version": "base", "loaded": true},
        "whisper": {"version": "medium", "loaded": true}
    }
}
```

### 2. Process Single Call (Manual Trigger)
```
POST /api/process
Content-Type: application/json

Request:
{
    "apex_id": "1737123456.12345",
    "force_reprocess": false
}

Response:
{
    "success": true,
    "recording_id": 123,
    "status": "processing",
    "message": "Call queued for processing"
}
```

### 3. Chat Completion (AI Assistant)
```
POST /api/chat
Content-Type: application/json

Request:
{
    "message": "What are the main compliance issues this month?",
    "conversation_id": 456,
    "feature": "call_quality",
    "filters": {
        "client_ref": "ACME001",
        "campaign": null,
        "agent_id": null,
        "start_date": "2026-01-01",
        "end_date": "2026-01-17"
    },
    "history": [
        {"role": "user", "content": "Show me the trends"},
        {"role": "assistant", "content": "Based on the data..."}
    ]
}

Response:
{
    "success": true,
    "response": "Based on the call quality data for ACME001 this month, the main compliance issues are...",
    "metadata": {
        "model": "qwen2.5-omni-7b",
        "tokens_used": 450,
        "context_calls_analysed": 45
    }
}
```

### 4. Generate Monthly Summary
```
POST /api/summary/generate
Content-Type: application/json

Request:
{
    "feature": "call_quality",
    "month": 1,
    "year": 2026,
    "client_ref": null,
    "campaign": null,
    "agent_id": null
}

Response:
{
    "success": true,
    "summary_id": 789,
    "summary_data": { ... }
}
```

### 5. Get Training Data (Annotations Export)
```
GET /api/training-data?since=2026-01-01&type=sentiment

Response:
{
    "success": true,
    "count": 150,
    "data": [
        {
            "transcript": "...",
            "original_sentiment": "positive",
            "corrected_sentiment": "neutral",
            "annotator_notes": "Supporter was actually frustrated"
        }
    ]
}
```

### 6. Fine-tune Model (Training Data Import)
```
POST /api/training/import
Content-Type: application/json

Request:
{
    "model_type": "call_analysis",
    "training_data": [...],
    "options": {
        "epochs": 3,
        "learning_rate": 0.0001
    }
}

Response:
{
    "success": true,
    "job_id": "training_abc123",
    "status": "queued"
}
```

---

## Training Data Facility (IMPORTANT)

Create a comprehensive training data management system for the **call analysis model** (NOT the chat or summarisation LLM - those will use a the base LLM training data and will not be extended).

### Requirements:

1. **Export Training Data**
   - Query `ai_call_annotations` where `is_training_data = TRUE`
   - Join with `ai_call_analysis` and `ai_call_transcriptions`
   - Format as JSONL for fine-tuning
   - Support filtering by annotation_type, date range, user

2. **Training Data Schema**
```json
{
    "id": "annotation_123",
    "transcript": "Full call transcript text...",
    "original_analysis": {
        "sentiment_label": "positive",
        "sentiment_score": 5.2,
        "quality_score": 78.5,
        "key_topics": [...]
    },
    "corrected_analysis": {
        "sentiment_label": "neutral",
        "quality_score": 65.0
    },
    "annotation_metadata": {
        "annotator_id": 42,
        "annotation_type": "sentiment",
        "notes": "Supporter showed frustration despite polite language",
        "created_at": "2026-01-15T14:30:00Z"
    }
}
```

3. **Training Pipeline**
   - Load annotations from database
   - Convert to fine-tuning format (instruction/response pairs)
   - Support LoRA fine-tuning for Qwen model
   - Track training runs and metrics
   - Ability to A/B test original vs fine-tuned models

4. **Model Versioning**
   - Store model versions with metadata
   - Track which training data was used
   - Easy rollback capability

---

## Analysis Prompt Engineering

The current prompts are causing the model to include raw prompt text in outputs. Here's the recommended structure.

**Note**: Qwen2.5-Omni is a multi-modal model that can analyse audio directly. The system supports both audio and transcript-based analysis.

### Analysis Mode Configuration

```python
ANALYSIS_MODE = os.getenv('ANALYSIS_MODE', 'audio')  # 'audio' or 'transcript'
```

### Prompt Template: Audio Analysis Mode (`ANALYSIS_MODE=audio`)

When using audio mode, the multi-modal model receives the audio file directly. This captures vocal cues like tone, pace, hesitation, and emotion.

**IMPORTANT**: The prompt must incorporate the `call_analysis_config.json` to constrain outputs to predefined categories. All output MUST be in British English.

```
You are a call quality analyst for a charity fundraising contact centre in the United Kingdom. You will be provided with an audio recording of a call. Listen carefully and provide structured analysis.

LANGUAGE: You MUST use British English (en-GB) spelling and conventions throughout your response. For example: analyse (not analyze), organisation (not organization), behaviour (not behavior), centre (not center).

<config>
{call_analysis_config_json}
</config>

<audio>
[Audio file is passed directly to the model via multi-modal input]
</audio>

Listen to this call and analyse it using ONLY the topics, agent_actions, and performance_rubric from the config above.

When analysing, pay attention to:
- Vocal tone and emotion (enthusiasm, frustration, hesitation)
- Speaking pace and clarity
- Pauses and silences (may indicate thinking or discomfort)
- Overlapping speech and interruptions
- Background noise or call quality issues

Provide your analysis in the following JSON format ONLY. Do not include any other text:

{
    "summary": "A 2-3 sentence summary of the call's purpose and outcome (in British English)",
    "sentiment_score": <number from -10 to 10>,
    "sentiment_label": "<very_negative|negative|neutral|positive|very_positive>",
    "quality_score": <number from 0 to 100>,
    "key_topics": [
        {"name": "<topic from config.topics>", "confidence": <0-1>}
    ],
    "agent_actions_performed": [
        {"action": "<action from config.agent_actions>", "timestamp_start": <seconds>, "quality": <1-5>}
    ],
    "performance_scores": {
        "<rubric item from config.performance_rubric>": <score 1-10>,
        ...
    },
    "action_items": [
        {"description": "<action in British English>", "priority": "<low|medium|high>"}
    ],
    "compliance_flags": [
        {"type": "<issue_type>", "issue": "<description in British English>", "severity": "<low|medium|high|critical>"}
    ],
    "speaker_metrics": {
        "agent": {"talk_time_percentage": <0-100>, "interruptions": <count>},
        "supporter": {"talk_time_percentage": <0-100>, "interruptions": <count>}
    },
    "audio_observations": {
        "call_quality": "<good|fair|poor>",
        "background_noise": "<none|low|moderate|high>",
        "agent_tone": "<professional|friendly|neutral|rushed|frustrated>",
        "supporter_tone": "<happy|neutral|confused|frustrated|angry>"
    }
}

RULES:
- ALL text output MUST use British English spelling (analyse, organisation, behaviour, colour, etc.)
- key_topics.name MUST be from the config.topics list only
- agent_actions_performed.action MUST be from the config.agent_actions list only
- performance_scores keys MUST be from the config.performance_rubric list
- Pay special attention to donation asks, objection handling, and sales conversion attempts
- Use audio cues (tone, hesitation, pace) to inform sentiment and quality scores
- Respond with ONLY the JSON object. No markdown, no explanations, no additional text.
```

### Prompt Template: Transcript Analysis Mode (`ANALYSIS_MODE=transcript`)

When using transcript mode, the model receives the text transcript (after Whisper transcription).

```
You are a call quality analyst for a charity fundraising contact centre in the United Kingdom. Analyse the following call transcript and provide structured analysis.

LANGUAGE: You MUST use British English (en-GB) spelling and conventions throughout your response. For example: analyse (not analyze), organisation (not organization), behaviour (not behavior), centre (not center).

<config>
{call_analysis_config_json}
</config>

<transcript>
{full_transcript}
</transcript>

Analyse this call using ONLY the topics, agent_actions, and performance_rubric from the config above.

Provide your analysis in the following JSON format ONLY. Do not include any other text:

{
    "summary": "A 2-3 sentence summary of the call's purpose and outcome (in British English)",
    "sentiment_score": <number from -10 to 10>,
    "sentiment_label": "<very_negative|negative|neutral|positive|very_positive>",
    "quality_score": <number from 0 to 100>,
    "key_topics": [
        {"name": "<topic from config.topics>", "confidence": <0-1>}
    ],
    "agent_actions_performed": [
        {"action": "<action from config.agent_actions>", "timestamp_start": <seconds>, "quality": <1-5>}
    ],
    "performance_scores": {
        "<rubric item from config.performance_rubric>": <score 1-10>,
        ...
    },
    "action_items": [
        {"description": "<action in British English>", "priority": "<low|medium|high>"}
    ],
    "compliance_flags": [
        {"type": "<issue_type>", "issue": "<description in British English>", "severity": "<low|medium|high|critical>"}
    ],
    "speaker_metrics": {
        "agent": {"talk_time_percentage": <0-100>, "interruptions": <count>},
        "supporter": {"talk_time_percentage": <0-100>, "interruptions": <count>}
    }
}

RULES:
- ALL text output MUST use British English spelling (analyse, organisation, behaviour, colour, etc.)
- key_topics.name MUST be from the config.topics list only
- agent_actions_performed.action MUST be from the config.agent_actions list only
- performance_scores keys MUST be from the config.performance_rubric list
- Pay special attention to donation asks, objection handling, and sales conversion attempts
- Respond with ONLY the JSON object. No markdown, no explanations, no additional text.
```

### Processing Flow by Mode

```python
def process_call(recording):
    # Step 1: Always download and convert audio
    audio_path = download_and_convert(recording.apex_id, recording.call_date)
    
    # Step 2: Always run Whisper for transcript (needed for search, karaoke, PII)
    transcript = transcribe_with_whisper(audio_path)
    transcript = detect_and_redact_pii(transcript)
    save_transcription(recording.id, transcript)
    
    # Step 3: Choose analysis method based on mode
    if ANALYSIS_MODE == 'audio':
        # Multi-modal: pass audio directly to Qwen2.5-Omni
        analysis = analyse_from_audio(audio_path, config)
    else:
        # Text-only: pass transcript to LLM
        analysis = analyse_from_transcript(transcript['full_transcript'], config)
    
    save_analysis(recording.id, analysis)
    
    # Step 4: Handle audio retention (after analysis, before cleanup)
    if recording.retain_audio:
        redacted_audio = redact_audio_pii(audio_path, transcript['pii_detected'])
        upload_to_r2(redacted_audio, recording.apex_id)
    
    # Step 5: Always cleanup local files
    cleanup(audio_path)
```

### Output Validation
Implement strict JSON validation:
1. Parse response as JSON
2. Validate against schema
3. Validate that topics, actions, and rubric items exist in config
4. If invalid, retry with more explicit instructions (max 3 retries)
5. If still invalid, log error and mark as failed with specific error message

---

## Configuration

### Environment Variables Required
```env
# Database
AI_DB_HOST=afs-db02.angelfs.co.uk
AI_DB_PORT=3306
AI_DB_DATABASE=pulse
AI_DB_USERNAME=afs_wings
AI_DB_PASSWORD=<password>

# Recording Sources
PBX_RECORDING_URL=https://pbx.angelfs.co.uk/callrec/
PBX_ARCHIVE_URL=https://afs-pbx-callarchive.angelfs.co.uk/

# R2 Storage (optional, for future use)
R2_ENDPOINT=<endpoint>
R2_ACCESS_KEY=<key>
R2_SECRET_KEY=<secret>
R2_BUCKET=angel-call-recordings

# Shared Model Storage (NFS mount for Kubernetes cluster)
MODELS_BASE_PATH=/models
ANALYSIS_MODEL_PATH=/models/analysis/current
CHAT_MODEL_PATH=/models/chat/current
WHISPER_MODEL_PATH=/models/whisper/current
MODEL_VERSIONS_PATH=/models/versions

# Analysis Model (fine-tuned, multi-modal)
ANALYSIS_MODEL_BASE=Qwen/Qwen2.5-Omni-7B
ANALYSIS_MODEL_QUANTIZATION=int8
ANALYSIS_MODE=audio  # 'audio' (multi-modal) or 'transcript' (text-only)

# Chat/Summary Model (base only, no fine-tuning)
CHAT_MODEL_BASE=Qwen/Qwen2.5-Omni-7B
CHAT_MODEL_QUANTIZATION=int8

# Whisper (always runs to generate transcript)
WHISPER_MODEL=openai/whisper-medium

# Processing
POLL_INTERVAL_SECONDS=30
MAX_CONCURRENT_JOBS=4
MAX_RETRIES=3
RETRY_DELAY_HOURS=1

# Kubernetes/Cluster
WORKER_ID=${HOSTNAME}
ENABLE_MODEL_HOT_RELOAD=true
MODEL_RELOAD_CHECK_INTERVAL=60

# Authentication
API_AUTH_TOKEN=<long-random-string-min-64-chars>

# Config
CALL_ANALYSIS_CONFIG_PATH=/config/call_analysis_config.json
```

---

## Processing Workflow

### 1. Main Processing Loop (Every 30 seconds)
```python
def process_pending_calls():
    # Find pending calls (including retries that are due)
    recordings = query("""
        SELECT * FROM ai_call_recordings 
        WHERE (processing_status = 'pending' OR processing_status = 'queued')
           OR (processing_status = 'failed' AND retry_count < 3 AND next_retry_at <= NOW())
        ORDER BY created_at ASC 
        LIMIT 10
    """)
    
    for recording in recordings:
        try:
            # Update status
            update_status(recording.id, 'processing')
            
            # Load client-specific config if exists
            config = load_client_config(recording.client_ref)
            
            # Download and convert audio
            audio_path = download_and_convert(recording.apex_id, recording.call_date)
            if not audio_path:
                raise Exception("Failed to find/download recording")
            
            # Transcribe with speaker diarization
            transcription = transcribe_audio(audio_path)
            
            # Identify speakers using voice fingerprints
            transcription = identify_speakers(transcription, recording.halo_id)
            
            # Detect and redact PII (UK patterns)
            transcription = detect_and_redact_pii(transcription)
            
            save_transcription(recording.id, transcription)
            
            # Analyse (using client-specific config if available)
            analysis = analyse_transcript(transcription, config)
            save_analysis(recording.id, analysis)
            
            # Handle audio retention
            if recording.retain_audio:
                redacted_audio = redact_audio_pii(audio_path, transcription.pii_detected)
                r2_path = upload_to_r2(redacted_audio, recording.apex_id)
                update_r2_path(recording.id, r2_path)
            
            # Always delete local audio files
            cleanup(audio_path)
            
            # Mark complete
            update_status(recording.id, 'completed')
            
        except Exception as e:
            handle_failure(recording, e)

def handle_failure(recording, error):
    retry_count = recording.retry_count + 1
    if retry_count >= 3:
        # Max retries reached, mark as permanently failed
        update_status(recording.id, 'failed', str(error), retry_count=retry_count)
    else:
        # Schedule retry in 1 hour
        next_retry = datetime.now() + timedelta(hours=1)
        update_status(recording.id, 'failed', str(error), 
                      retry_count=retry_count, next_retry_at=next_retry)
    log_error(recording.id, error)
```

### 2. Download and Convert
```python
def download_and_convert(apex_id: str, call_date: date) -> Optional[str]:
    year = call_date.year
    month = str(call_date.month).zfill(2)
    
    urls = [
        f"https://pbx.angelfs.co.uk/callrec/{apex_id}.gsm",
        f"https://afs-pbx-callarchive.angelfs.co.uk/monitor-{year}/{month}/{apex_id}.gsm",
        f"https://afs-pbx-callarchive.angelfs.co.uk/monitor-{year}/{month}/monitor/{apex_id}.gsm"
    ]
    
    for url in urls:
        if check_url_exists(url):
            gsm_path = download_file(url)
            wav_path = convert_gsm_to_wav(gsm_path)  # Uses: sox input.gsm -r 8000 -b 32 -c 1 output.wav
            cleanup(gsm_path)
            return wav_path
    
    return None
```

### 3. Load Client Configuration
```python
def load_client_config(client_ref: str) -> dict:
    """Load config, checking client-specific first, then global defaults"""
    config = {}
    
    for config_type in ['topics', 'agent_actions', 'performance_rubric', 'prompt']:
        # Try client-specific
        result = query("""
            SELECT config_data FROM ai_client_configs 
            WHERE client_ref = %s AND config_type = %s AND is_active = TRUE
        """, [client_ref, config_type])
        
        if not result:
            # Fall back to global
            result = query("""
                SELECT config_data FROM ai_client_configs 
                WHERE client_ref IS NULL AND config_type = %s AND is_active = TRUE
            """, [config_type])
        
        if result:
            config[config_type] = result.config_data
    
    # Fall back to file-based config if no DB config
    if not config:
        config = load_json_file(CALL_ANALYSIS_CONFIG_PATH)
    
    return config
```

---

## Frontend Integration Points

The Laravel frontend expects these specific response formats:

### When processing completes, update these fields:

**ai_call_recordings:**
- `processing_status` = 'completed'
- `processing_completed_at` = NOW()
- `duration_seconds` = <from audio file>
- `file_format` = 'wav'

**ai_call_transcriptions:**
- All fields as per schema
- `segments` must include word-level timestamps for karaoke feature
- `confidence_score` between 0 and 1

**ai_call_analysis:**
- `summary` should be 2-4 concise sentences
- `sentiment_score` between -10 and +10
- `sentiment_label` must be one of the ENUM values exactly
- `quality_score` between 0 and 100 (derived from performance_scores average)
- `key_topics` names MUST come from `call_analysis_config.json` topics list
- `agent_actions_performed` actions MUST come from `call_analysis_config.json` agent_actions list
- `performance_scores` keys MUST come from `call_analysis_config.json` performance_rubric list
- `model_version` should track which fine-tuned version was used

---

## Error Handling

Specific error messages to store in `processing_error`:

- `RECORDING_NOT_FOUND` - Could not find recording at any URL
- `DOWNLOAD_FAILED` - HTTP error downloading recording
- `CONVERSION_FAILED` - SoX conversion error
- `TRANSCRIPTION_FAILED` - Whisper error
- `ANALYSIS_FAILED` - LLM error
- `ANALYSIS_PARSE_ERROR` - Could not parse LLM response as JSON
- `ANALYSIS_VALIDATION_ERROR` - JSON valid but topics/actions not in config
- `DATABASE_ERROR` - Failed to save to database
- `MODEL_LOAD_ERROR` - Failed to load model from shared storage

---

## Summary of Required Changes

1. **Rename repository** from "Harmony" / "Voice" to "Angel Intelligence"
2. **Update README.md** with new name and complete documentation
3. **Fix recording retrieval** to use Laravel pattern (3 URL fallbacks)
4. **Fix GSM to WAV conversion** using SoX
5. **Fix analysis prompt** to prevent prompt leakage in output
6. **Incorporate call_analysis_config.json** for topic/action/rubric constraints
7. **Implement strict JSON validation** for LLM responses
8. **Validate outputs against config** (topics, actions, rubric items must exist)
9. **Add training data export** functionality
10. **Add training data import/fine-tuning** pipeline for call analysis model ONLY
11. **Add model versioning** system with shared NFS storage
12. **Implement separate model loading** - Analysis model (fine-tuned) vs Chat model (base)
13. **Add chat endpoint** for AI Assistant integration (uses Chat model)
14. **Add monthly summary generation** endpoint (uses Chat model)
15. **Configure for Kubernetes** - stateless workers, shared model storage
16. **Implement model hot-reload** for when new versions are deployed
17. **Update database schema** to include agent_actions_performed, performance_scores, model_version
18. **Ensure segments JSON includes word-level timestamps** for transcript karaoke
19. **Add proper error codes** in processing_error field
20. **Add worker health endpoint** that reports loaded models
21. **Implement Bearer token authentication** on all API endpoints
22. **Add UK-specific PII detection** (NI numbers, NHS numbers, UK postcodes, sort codes, etc.)
23. **Implement audio PII redaction** (beep/silence PII segments)
24. **Add `retain_audio` flag handling** - delete audio by default, store redacted in R2 if flagged
25. **Implement retry logic** - retry failed jobs 3 times with 1 hour delay
26. **Add voice fingerprinting** for agent identification
27. **Handle call transfers** - detect multiple agents on one call
28. **Create `ai_voice_fingerprints` table** and speaker matching logic
29. **Create `ai_client_configs` table** for client-specific configurations
30. **Add config management API** endpoints for topics, actions, rubrics, prompts
31. **Ensure all outputs use British English** spelling and conventions
32. **Implement dual analysis modes** - audio (multi-modal) and transcript (text-only)
33. **Add `ANALYSIS_MODE` environment variable** to switch between modes
34. **Include `audio_observations` in analysis** when using audio mode (tone, call quality, etc.)

---

## Testing

Create test cases for:
1. Recording download from each URL pattern
2. GSM to WAV conversion
3. Transcription with Whisper
4. Analysis JSON parsing and validation
5. **Validation against call_analysis_config.json**
6. Training data export format
7. Chat endpoint responses (verify uses Chat model, not Analysis model)
8. Monthly summary generation
9. Database operations on all tables
10. **Shared model loading from NFS**
11. **Model hot-reload functionality**
12. **Concurrent worker job claiming (no duplicate processing)**
13. **Bearer token authentication (valid/invalid/missing)**
14. **UK PII detection patterns** (NI, NHS, postcodes, bank details, cards)
15. **Audio PII redaction quality**
16. **R2 upload when retain_audio=TRUE**
17. **Retry logic** (verify 3 retries at 1 hour intervals)
18. **Voice fingerprint matching** accuracy
19. **Call transfer detection** (multiple agents)
20. **Client-specific config loading** (override vs fallback)
21. **British English output** verification
22. **Audio analysis mode** - verify multi-modal model receives audio correctly
23. **Transcript analysis mode** - verify text-only analysis works
24. **Mode switching** - verify ANALYSIS_MODE env var changes behaviour
25. **Audio observations** - verify tone/quality captured in audio mode

---

## Notes

- The Laravel frontend polls for status changes, so ensure database updates are atomic
- All timestamps should be in UTC
- The `apex_id` format is typically `timestamp.sequence` (e.g., `1737123456.12345`)
- Speaker diarization (identifying agent vs supporter) is important for the UI
- The training data facility is specifically for the **call analysis model ONLY**
- The chat and summarisation features use the **base Chat model with NO fine-tuning**
- Models are stored on shared NFS so training only happens once for the entire cluster
- Workers are stateless and can be scaled horizontally
- Use database-level locking when claiming jobs to prevent race conditions
- **Audio is deleted after processing** unless `retain_audio = TRUE`
- **Voice fingerprints** should be built incrementally as more calls are processed
- **Config can be customised per-client** via database, with fallback to global/file defaults
- All API requests require **Bearer token** in Authorization header
- **Qwen2.5-Omni is multi-modal** - it can analyse audio directly, not just transcripts
- **Always run Whisper** regardless of analysis mode (transcript needed for search, karaoke, PII)
- **Audio mode captures more nuance** (tone, hesitation, pace) but requires more compute
- **Transcript mode is lighter** but loses vocal cues
