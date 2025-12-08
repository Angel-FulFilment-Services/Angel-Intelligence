# Autonomous AI Call Processing Worker

**Completely independent from Laravel** - no queue jobs, no API calls.

## How It Works

1. **Worker polls database** every 30 seconds for `pending` recordings
2. **Marks as processing** to prevent duplicate work
3. **Downloads audio from R2**
4. **Transcribes with WhisperX** (word-level timestamps + speaker diarization)
5. **Analyzes with LLM** (sentiment, summary, quality scoring)
6. **Redacts PII** (optional - future feature)
7. **Uploads cleaned audio** back to R2 (optional)
8. **Saves results** to database
9. **Marks as completed**

Laravel only:
- Uploads recordings to R2
- Creates database records with `status='pending'`
- Displays results

## Local Development Setup

### 1. Install Dependencies

```bash
cd ai-service

# Create virtual environment
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate    # Linux/Mac

# Install packages
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example env
copy .env.example .env

# Edit .env with your settings
notepad .env
```

Required configuration:
- Database credentials (AI_DB_*)
- R2 credentials (R2_*)
- Poll interval

### 3. Run Worker

```bash
python worker.py
```

You'll see:
```
2025-12-05 10:00:00 - Worker initialized successfully
2025-12-05 10:00:00 - Starting worker (poll interval: 30s)
2025-12-05 10:00:00 - No pending recordings found
...
2025-12-05 10:01:00 - Found 3 recordings to process
2025-12-05 10:01:00 - Processing recording 123: TEST-001
```

## Kubernetes Deployment (K3s Cluster)

See **[K3S_SETUP.md](K3S_SETUP.md)** for complete setup instructions.

### Quick Start

1. **Setup control plane** on HyperV VM (see K3S_SETUP.md Part 1)
2. **Join Jetson workers** to cluster (see K3S_SETUP.md Part 2)
3. **Deploy workers:**

```bash
# Configure secrets
cp k8s/secret.yaml.example k8s/secret.yaml
# Edit k8s/secret.yaml with your credentials

# Deploy
./scripts/setup.sh   # Linux/Mac
.\scripts\setup.ps1  # Windows

# Scale workers (automatically distributes across available Jetsons)
kubectl scale deployment ai-worker --replicas=4
```

### Add New Jetson Node

```bash
# On new Jetson - join cluster (see K3S_SETUP.md Part 2.1)
# Then scale up:
kubectl scale deployment ai-worker --replicas=5
```

Workers automatically coordinate via database locking.

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

## Architecture Benefits

✅ **Zero Laravel load** - No queue jobs, no processing  
✅ **Fully scalable** - Add more workers = more capacity  
✅ **Fault tolerant** - Worker crash? Just restart it  
✅ **Simple** - Docker container, done  
✅ **No Redis needed** - Database is the queue
