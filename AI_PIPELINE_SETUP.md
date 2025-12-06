# AI Call Recording Pipeline - Setup Guide

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│         Kubernetes Cluster (4x Jetson)              │
│                                                      │
│  ┌────────────────────────────────────────────────┐ │
│  │  Python Worker (Autonomous)                    │ │
│  │  - Polls database for pending recordings       │ │
│  │  - Marks as in_progress                        │ │
│  │  - Downloads from R2                           │ │
│  │  - WhisperX transcription                      │ │
│  │  - Llama analysis                              │ │
│  │  - PII redaction (optional)                    │ │
│  │  - Uploads cleaned audio                       │ │
│  │  - Saves results to database                   │ │
│  └────────────────────────────────────────────────┘ │
│           ▲                            │            │
│           │ polls                      │ updates    │
└───────────┼────────────────────────────┼────────────┘
            │                            │
    ┌───────┴─────┐              ┌───────▼─────┐
    │  R2 Bucket  │              │ AI Database │
    │ (Recordings)│              │   (MySQL)   │
    └───────▲─────┘              └───────▲─────┘
            │                            │
            │ writes                     │ reads/writes
            │                            │
    ┌───────┴────────────────────────────┴─────┐
    │      Laravel (Web UI ONLY)               │
    │  - Mark calls for processing             │
    │  - Display transcriptions & analysis     │
    │  - NO queue jobs, NO processing          │
    └──────────────────────────────────────────┘
```

**Key Changes:**
- ❌ No Laravel queue jobs
- ❌ No FastAPI endpoints
- ✅ Autonomous Python worker polls database
- ✅ Worker handles entire pipeline
- ✅ Laravel is just UI + database access

## Setup Instructions

### 1. Laravel Database Configuration

Add to your `config/database.php`:

```php
'ai' => [
    'driver' => 'mysql',
    'host' => env('AI_DB_HOST', '127.0.0.1'),
    'port' => env('AI_DB_PORT', '3306'),
    'database' => env('AI_DB_DATABASE', 'ai_calls'),
    'username' => env('AI_DB_USERNAME', 'root'),
    'password' => env('AI_DB_PASSWORD', ''),
    'charset' => 'utf8mb4',
    'collation' => 'utf8mb4_unicode_ci',
    'prefix' => '',
    'strict' => true,
    'engine' => null,
],
```

Add to Laravel `.env`:

```env
# AI Database (same credentials will be used by Python worker)
AI_DB_HOST=127.0.0.1
AI_DB_PORT=3306
AI_DB_DATABASE=ai_calls
AI_DB_USERNAME=root
AI_DB_PASSWORD=your-password
```

### 2. Run Migrations

```bash
php artisan migrate --database=ai
```

### 3. Configure Python Worker

```bash
cd ai-service

# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
notepad .env  # Edit with your database & R2 credentials
```

Edit `ai-service/.env`:

```env
# Database (MUST match Laravel's AI database)
AI_DB_HOST=127.0.0.1
AI_DB_PORT=3306
AI_DB_DATABASE=ai_calls
AI_DB_USERNAME=root
AI_DB_PASSWORD=your-password

# R2 Storage (same as Laravel's R2 config)
R2_ENDPOINT=https://your-account-id.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-r2-key
R2_SECRET_ACCESS_KEY=your-r2-secret
R2_BUCKET=call-recordings

# Worker Settings
POLL_INTERVAL=30  # Check for new recordings every 30 seconds
```

### 4. Start Python Worker

```bash
# In ai-service directory
python worker.py
```

You'll see:
```
2025-12-05 10:00:00 - Initializing WhisperX on cuda
2025-12-05 10:00:00 - Worker initialized successfully
2025-12-05 10:00:00 - Starting worker (poll interval: 30s)
2025-12-05 10:00:00 - No pending recordings found
```

**That's it!** No queue workers, no API servers needed.

### 5. Laravel Usage (Upload Only)

```php
use App\Services\CallRecordingService;

$service = new CallRecordingService();

// Upload a recording - worker picks it up automatically
$recording = $service->uploadForProcessing(
    file: $request->file('recording'),
    apexId: '12345.67890',
    metadata: [
        'duration' => 300, // seconds
    ]
);

// Check status (polling)
$recording->refresh();
$recording->processing_status; // 'pending', 'processing', 'completed', 'failed'

// Get results (after worker completes)
$transcription = $recording->transcription;
$analysis = $recording->analysis;

// Access data
echo $transcription->full_transcript;
echo $analysis->summary;
echo $analysis->sentiment_label;
```

## Production Deployment

### For 4x Jetson Orin Nano Cluster:

**No load balancer needed!** Each worker independently polls the database.

1. **On each Jetson, build Docker image:**
```bash
cd ai-service
docker build -t ai-worker:latest .
```

2. **Configure `docker-compose.yml` on each Jetson:**

```yaml
version: '3.8'

services:
  ai-worker:
    build: .
    environment:
      # Database (shared across all workers)
      - AI_DB_HOST=your-central-db-host
      - AI_DB_PORT=3306
      - AI_DB_DATABASE=ai_calls
      - AI_DB_USERNAME=root
      - AI_DB_PASSWORD=your-password
      
      # R2 Storage
      - R2_ENDPOINT=https://your-account.r2.cloudflarestorage.com
      - R2_ACCESS_KEY_ID=your-key
      - R2_SECRET_ACCESS_KEY=your-secret
      - R2_BUCKET=call-recordings
      
      # Worker settings
      - POLL_INTERVAL=30
      - CUDA_VISIBLE_DEVICES=0
      
    volumes:
      - ./models:/root/.cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

3. **Deploy on each Jetson:**
```bash
docker-compose up -d
```

4. **Monitor all workers:**
```bash
# On each Jetson
docker logs -f ai-worker
```

**How it works:**
- All 4 workers connect to the same database
- They poll for `pending` recordings
- First worker to grab a recording marks it `processing`
- Other workers skip it (database locking prevents duplicates)
- Automatic load distribution!

### Systemd Service (Alternative to Docker)

Create `/etc/systemd/system/ai-worker.service`:

```ini
[Unit]
Description=AI Call Processing Worker
After=network.target

[Service]
Type=simple
User=aiworker
WorkingDirectory=/opt/ai-service
Environment="PATH=/opt/ai-service/venv/bin"
EnvironmentFile=/opt/ai-service/.env
ExecStart=/opt/ai-service/venv/bin/python worker.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable ai-worker
sudo systemctl start ai-worker
sudo systemctl status ai-worker
```

## Testing the Pipeline

### 1. Verify worker is running:
```bash
# Check worker logs
python worker.py
# Should see: "Worker initialized successfully"
```

### 2. Upload a test recording (Laravel):
```php
// In tinker or controller
$service = app(\App\Services\CallRecordingService::class);
$recording = $service->uploadFromPath('/path/to/test.wav', 'TEST-001');

echo "Recording ID: {$recording->id}, Status: {$recording->processing_status}";
// Recording ID: 1, Status: pending
```

### 3. Monitor processing:
```bash
# Watch worker logs (you'll see real-time processing)
# Worker output:
# 2025-12-05 10:01:00 - Found 1 recordings to process
# 2025-12-05 10:01:00 - Processing recording 1: TEST-001
# 2025-12-05 10:01:05 - Downloaded call-recordings/2025/12/05/test.wav
# 2025-12-05 10:01:10 - Starting transcription
# 2025-12-05 10:03:45 - Saved transcription 1 with 45 segments
# 2025-12-05 10:04:00 - Saved analysis for recording 1
# 2025-12-05 10:04:00 - Marked recording 1 as completed
```

### 4. Check database directly:
```sql
SELECT id, apex_id, processing_status, processing_started_at, processing_completed_at 
FROM ai_call_recordings;
```

### 5. Check results (Laravel):
```php
$recording = \App\Models\AI\CallRecording::find(1);

// Transcription
echo $recording->transcription->full_transcript;
echo "Segments: " . $recording->transcription->segments->count();

// Analysis
echo "Sentiment: " . $recording->analysis->sentiment_label;
echo "Quality: " . $recording->analysis->quality_score;
dd($recording->analysis->speaker_metrics);
```

## Performance Expectations

### Single Jetson Orin Nano (8GB):
- Whisper (medium): ~2-3 minutes per 5-min call
- Llama 8B: ~30-60 seconds per analysis
- **Capacity: ~250-300 calls/day**

### 4x Jetson Orin Nano Cluster:
- **Capacity: ~1,000-1,200 calls/day**
- Load balanced across 4 devices
- Failover support

## Monitoring

```php
// Get statistics
$service = app(CallRecordingService::class);
$stats = $service->getStatistics();
/*
[
    'total' => 1000,
    'pending' => 5,
    'processing' => 3,
    'completed' => 985,
    'failed' => 7,
]
*/
```

## Troubleshooting

### Worker not processing recordings:

1. **Check database connection:**
```bash
python -c "
from worker import CallProcessingWorker
worker = CallProcessingWorker()
recordings = worker.scan_for_pending_recordings()
print(f'Found {len(recordings)} pending recordings')
"
```

2. **Check R2 connection:**
```bash
python -c "
import boto3
from botocore.config import Config
import os

client = boto3.client(
    's3',
    endpoint_url=os.getenv('R2_ENDPOINT'),
    aws_access_key_id=os.getenv('R2_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('R2_SECRET_ACCESS_KEY')
)
print(client.list_buckets())
"
```

3. **Check CUDA:**
```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Worker crashes / Out of memory:

Edit `worker.py`:
```python
self.batch_size = 8  # Reduce from 16
# OR
self.whisperx_model = whisperx.load_model("small", ...)  # Use smaller model
```

### Failed recordings:

```php
// Laravel - mark for reprocessing
$recording = \App\Models\AI\CallRecording::find(123);
app(\App\Services\CallRecordingService::class)->reprocess($recording);
// Worker will automatically pick it up on next poll
```

### Duplicate processing:

This shouldn't happen due to database-level status checking, but if it does:
```sql
-- Check for stuck 'processing' records
SELECT * FROM ai_call_recordings 
WHERE processing_status = 'processing' 
AND processing_started_at < NOW() - INTERVAL 1 HOUR;

-- Reset them
UPDATE ai_call_recordings 
SET processing_status = 'pending' 
WHERE processing_status = 'processing' 
AND processing_started_at < NOW() - INTERVAL 1 HOUR;
```

## Next Steps / Improvements

### Current Implementation:
✅ WhisperX transcription (word-level timestamps)  
✅ Basic speaker diarization (gap-based)  
✅ Database polling architecture  
✅ Autonomous worker (no Laravel dependency)  
✅ Multi-worker support (4x Jetson ready)  

### To Do:
1. **Integrate real LLM** - Replace placeholder analysis with Llama/Groq
2. **Better speaker diarization** - Add pyannote.audio with HuggingFace token
3. **PII redaction** - Detect and redact sensitive information
4. **Model quantization** - Optimize for Jetson (4-bit/8-bit quantization)
5. **Build analytics dashboard** - Inertia/Vue components for viewing results
6. **Health monitoring** - Add Prometheus metrics, alerting
7. **Batch processing** - Process multiple recordings in parallel per worker

## Cost Analysis

### Cloud GPU Alternative (for comparison):
- 1000 calls/day × 4 min processing = 4000 GPU min/day
- RunPod RTX 4090: ~$46/day = **$1,400/month**
- Replicate/Modal: ~$30-50/day = **$900-1,500/month**

### 4x Jetson Orin Nano (Recommended):
- Upfront: ~$2,000 (4 × $499)
- Power: ~60W × 24hr × $0.12/kWh = **$5/month**
- **Break-even: ~1.5 months**
- **Year 1 savings: ~$15,000+**

### Why This Architecture?

✅ **Zero web server load** - Laravel just does UI  
✅ **Infinite scalability** - Add more Jetsons = more capacity  
✅ **Fault tolerant** - Workers can crash and restart  
✅ **No queue complexity** - Database IS the queue  
✅ **Cost effective** - One-time hardware investment  
✅ **Simple deployment** - Docker container on each Jetson  
✅ **Easy monitoring** - Just check database status
