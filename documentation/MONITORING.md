# Monitoring Guide

Observability and monitoring setup for Angel Intelligence.

## Overview

Monitoring covers:
- Health checks
- Metrics collection
- Log aggregation
- Alerting

---

## Health Checks

### API Health Endpoint

```bash
curl http://localhost:8000/health
```

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
    "analysis": {"version": "v1.0.0", "loaded": true},
    "chat": {"version": "base", "loaded": true},
    "whisper": {"version": "medium", "loaded": true}
  }
}
```

### Kubernetes Probes

```yaml
# k8s/deployment.yaml
spec:
  containers:
  - name: worker
    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 60
      periodSeconds: 30
      timeoutSeconds: 10
      failureThreshold: 3
    readinessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3
```

---

## Key Metrics

### Processing Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| Queue Depth | Pending recordings | < 100 |
| Processing Time | Seconds per recording | < 120s |
| Error Rate | Failed / Total | < 5% |
| Retry Rate | Retried / Total | < 10% |

### System Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| GPU Memory | VRAM usage | < 90% |
| CPU Usage | Per worker | < 80% |
| Memory Usage | RAM per worker | < 8GB |
| Disk Space | Temp storage | > 10GB free |

---

## Database Monitoring

### Queue Status Query

```sql
-- Current queue status
SELECT 
    processing_status,
    COUNT(*) as count
FROM ai_call_recordings
GROUP BY processing_status;
```

### Processing Time Analysis

```sql
-- Average processing time by day
SELECT 
    DATE(processing_completed_at) as date,
    COUNT(*) as completed,
    AVG(TIMESTAMPDIFF(SECOND, processing_started_at, processing_completed_at)) as avg_seconds,
    MAX(TIMESTAMPDIFF(SECOND, processing_started_at, processing_completed_at)) as max_seconds
FROM ai_call_recordings
WHERE processing_status = 'completed'
  AND processing_completed_at > DATE_SUB(NOW(), INTERVAL 7 DAY)
GROUP BY DATE(processing_completed_at)
ORDER BY date DESC;
```

### Error Analysis

```sql
-- Recent errors by type
SELECT 
    SUBSTRING_INDEX(processing_error, ':', 1) as error_type,
    COUNT(*) as count,
    MAX(processing_completed_at) as last_seen
FROM ai_call_recordings
WHERE processing_status = 'failed'
  AND processing_completed_at > DATE_SUB(NOW(), INTERVAL 24 HOUR)
GROUP BY error_type
ORDER BY count DESC;
```

### Stuck Recordings

```sql
-- Recordings processing for too long
SELECT id, apex_id, processing_started_at,
       TIMESTAMPDIFF(MINUTE, processing_started_at, NOW()) as minutes_processing
FROM ai_call_recordings
WHERE processing_status = 'processing'
  AND processing_started_at < DATE_SUB(NOW(), INTERVAL 30 MINUTE);
```

---

## Log Monitoring

### Log Formats

**Text Format (default):**
```
2026-01-17 14:30:00,123 - INFO - Processing recording 12345
2026-01-17 14:30:45,456 - INFO - Transcription complete: 120 segments
2026-01-17 14:31:15,789 - INFO - Analysis complete: quality=85.0
```

**JSON Format (production):**
```json
{
  "timestamp": "2026-01-17T14:30:00.123Z",
  "level": "INFO",
  "message": "Processing recording 12345",
  "worker_id": "worker-abc123",
  "recording_id": 12345
}
```

Enable JSON logging:
```env
LOG_FORMAT=json
```

### Log Aggregation

#### Kubernetes Logs

```bash
# All workers
kubectl logs -l app=angel-intelligence -f

# Specific worker
kubectl logs deployment/angel-intelligence-worker -f

# Previous container (after crash)
kubectl logs <pod-name> --previous
```

#### Docker Compose Logs

```bash
# Follow all logs
docker-compose logs -f

# Filter by service
docker-compose logs -f worker

# With timestamps
docker-compose logs -f -t worker
```

### Log Analysis

```bash
# Count errors in last hour
grep ERROR worker.log | grep "$(date +%Y-%m-%d\ %H)" | wc -l

# Find specific error
grep "TRANSCRIPTION_FAILED" worker.log | tail -10

# Processing times
grep "Processing complete" worker.log | awk '{print $NF}' | sort -n
```

---

## GPU Monitoring

### nvidia-smi

```bash
# Current status
nvidia-smi

# Watch mode
nvidia-smi -l 1

# Query specific metrics
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv -l 1
```

### In Container

```bash
kubectl exec -it <pod-name> -- nvidia-smi
```

### GPU Memory Alert

```python
# Check GPU memory usage
import torch
if torch.cuda.is_available():
    memory_used = torch.cuda.memory_allocated() / 1024**3
    memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    usage_percent = (memory_used / memory_total) * 100
    
    if usage_percent > 90:
        print(f"WARNING: GPU memory at {usage_percent:.1f}%")
```

---

## Alerting Rules

### Critical Alerts

| Condition | Action |
|-----------|--------|
| API health check fails 3x | Page on-call |
| Queue > 500 pending | Notify team |
| Error rate > 10% for 5 min | Page on-call |
| GPU OOM error | Notify team |
| Disk space < 5GB | Notify team |

### Warning Alerts

| Condition | Action |
|-----------|--------|
| Queue > 100 pending | Log warning |
| Processing time > 5 min | Log warning |
| Retry rate > 5% | Log warning |
| GPU memory > 80% | Log warning |

---

## Dashboard Queries

### Grafana/Prometheus (Future)

```promql
# Processing rate
rate(angel_recordings_processed_total[5m])

# Error rate
rate(angel_recordings_failed_total[5m]) / rate(angel_recordings_processed_total[5m])

# Queue depth
angel_recordings_pending

# Average processing time
histogram_quantile(0.95, rate(angel_processing_duration_seconds_bucket[5m]))
```

### Simple Monitoring Script

```bash
#!/bin/bash
# monitor.sh - Simple monitoring script

# Check API health
health=$(curl -s http://localhost:8000/health | jq -r '.status')
if [ "$health" != "healthy" ]; then
    echo "ALERT: API unhealthy"
fi

# Check queue depth
pending=$(mysql -N -e "SELECT COUNT(*) FROM ai_call_recordings WHERE processing_status='pending'" ai)
if [ "$pending" -gt 100 ]; then
    echo "WARNING: Queue depth $pending"
fi

# Check failed recordings
failed=$(mysql -N -e "SELECT COUNT(*) FROM ai_call_recordings WHERE processing_status='failed' AND processing_completed_at > DATE_SUB(NOW(), INTERVAL 1 HOUR)" ai)
if [ "$failed" -gt 10 ]; then
    echo "WARNING: $failed failures in last hour"
fi

# Check GPU
gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
gpu_total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
gpu_percent=$((gpu_mem * 100 / gpu_total))
if [ "$gpu_percent" -gt 90 ]; then
    echo "WARNING: GPU memory at ${gpu_percent}%"
fi

echo "Monitoring check complete"
```

---

## Performance Benchmarks

### Expected Performance

| Metric | Target | Acceptable |
|--------|--------|------------|
| Transcription (60s audio) | < 15s | < 30s |
| Analysis (transcript) | < 30s | < 60s |
| Analysis (audio) | < 45s | < 90s |
| Total processing | < 90s | < 180s |
| API response (health) | < 50ms | < 200ms |
| API response (status) | < 100ms | < 500ms |

### Load Testing

```bash
# Simple load test
for i in {1..10}; do
    curl -X POST http://localhost:8000/recordings/submit \
        -H "Authorization: Bearer $TOKEN" \
        -H "Content-Type: application/json" \
        -d "{\"apex_id\": \"LOAD-TEST-$i\", \"call_date\": \"2026-01-17\"}" &
done
wait
```

---

## Runbook

### High Queue Depth

1. Check worker pods are running:
   ```bash
   kubectl get pods -l app=angel-intelligence
   ```

2. Check worker logs for errors:
   ```bash
   kubectl logs -l app=angel-intelligence --tail=100
   ```

3. Scale up workers:
   ```bash
   kubectl scale deployment angel-intelligence-worker --replicas=6
   ```

### Worker Crash Loop

1. Check pod events:
   ```bash
   kubectl describe pod <pod-name>
   ```

2. Check recent logs:
   ```bash
   kubectl logs <pod-name> --previous
   ```

3. Check resource limits:
   - GPU memory
   - RAM usage
   - Disk space

4. Restart deployment:
   ```bash
   kubectl rollout restart deployment angel-intelligence-worker
   ```

### Database Issues

1. Check MySQL status:
   ```bash
   kubectl get pods -l app=mysql
   ```

2. Check connection from worker:
   ```bash
   kubectl exec <pod-name> -- python -c "from src.database import get_db_connection; get_db_connection()"
   ```

3. Check for long-running queries:
   ```sql
   SHOW PROCESSLIST;
   ```

4. Kill stuck query if needed:
   ```sql
   KILL <process_id>;
   ```
