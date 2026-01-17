# Troubleshooting Guide

Common issues and solutions for Angel Intelligence.

## Quick Diagnostics

Run these commands to diagnose common issues:

```bash
# Check Python environment
python --version
pip list | grep -E "torch|whisperx|presidio"

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Check database
python -c "from src.database import get_db_connection; print(get_db_connection())"

# Check API health
curl http://localhost:8000/health

# Check worker logs
tail -f worker.log
```

---

## Installation Issues

### CUDA Not Detected

**Symptoms:**
- `torch.cuda.is_available()` returns `False`
- GPU not showing in health check

**Solutions:**

1. Check NVIDIA driver:
   ```bash
   nvidia-smi
   ```

2. Reinstall PyTorch for your CUDA version:
   ```bash
   # For CUDA 11.8
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.4
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

3. Verify CUDA toolkit version matches:
   ```bash
   nvcc --version
   python -c "import torch; print(torch.version.cuda)"
   ```

---

### WhisperX Import Error

**Symptoms:**
```
ModuleNotFoundError: No module named 'whisperx'
```

**Solution:**
```bash
pip install git+https://github.com/m-bain/whisperx.git
```

---

### Presidio Import Error

**Symptoms:**
```
ModuleNotFoundError: No module named 'presidio_analyzer'
```

**Solution:**
```bash
pip install presidio-analyzer presidio-anonymizer
python -m spacy download en_core_web_lg
```

---

### SoX Not Found

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'sox'
```

**Solutions:**

**Ubuntu/Debian:**
```bash
sudo apt install sox libsox-fmt-all
```

**Windows:**
```powershell
choco install sox
# Or download from: https://sourceforge.net/projects/sox/
```

**macOS:**
```bash
brew install sox
```

Verify installation:
```bash
sox --version
```

---

## Database Issues

### Connection Refused

**Symptoms:**
```
mysql.connector.errors.InterfaceError: 2003: Can't connect to MySQL server
```

**Solutions:**

1. Check MySQL is running:
   ```bash
   sudo systemctl status mysql
   ```

2. Verify host and port in `.env`:
   ```env
   AI_DB_HOST=localhost
   AI_DB_PORT=3306
   ```

3. Check firewall rules:
   ```bash
   sudo ufw allow 3306
   ```

4. Test connection directly:
   ```bash
   mysql -h $AI_DB_HOST -u $AI_DB_USERNAME -p$AI_DB_PASSWORD $AI_DB_DATABASE
   ```

---

### Access Denied

**Symptoms:**
```
mysql.connector.errors.ProgrammingError: 1045 (28000): Access denied
```

**Solutions:**

1. Verify credentials in `.env`

2. Check user permissions:
   ```sql
   SHOW GRANTS FOR 'angel_ai'@'%';
   ```

3. Grant permissions:
   ```sql
   GRANT ALL PRIVILEGES ON ai.* TO 'angel_ai'@'%';
   FLUSH PRIVILEGES;
   ```

---

### Table Doesn't Exist

**Symptoms:**
```
mysql.connector.errors.ProgrammingError: 1146 (42S02): Table 'ai.ai_call_recordings' doesn't exist
```

**Solution:**

Run the schema creation script:
```bash
mysql -h $AI_DB_HOST -u $AI_DB_USERNAME -p$AI_DB_PASSWORD $AI_DB_DATABASE < documentation/schema.sql
```

---

## API Issues

### 401 Unauthorized

**Symptoms:**
```json
{"detail": "Invalid authentication token"}
```

**Solutions:**

1. Check token is at least 64 characters

2. Verify token matches exactly:
   ```bash
   echo $API_AUTH_TOKEN | wc -c  # Should be >= 64
   ```

3. Check Authorization header format:
   ```bash
   curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:8000/
   ```

---

### 500 Internal Server Error

**Symptoms:**
- API returns 500 errors
- Worker crashes

**Solutions:**

1. Check API logs:
   ```bash
   uvicorn src.api:app --reload  # See error in terminal
   ```

2. Check for missing environment variables:
   ```bash
   python -c "from src.config import get_settings; print(get_settings())"
   ```

3. Enable debug mode:
   ```env
   ANGEL_ENV=development
   LOG_LEVEL=DEBUG
   ```

---

## Worker Issues

### Worker Not Processing

**Symptoms:**
- Recordings stay in "pending" status
- Worker appears idle

**Solutions:**

1. Check worker is running:
   ```bash
   ps aux | grep worker
   ```

2. Check for errors in logs:
   ```bash
   tail -100 worker.log | grep -E "ERROR|Exception"
   ```

3. Check database connection:
   ```bash
   python -c "
   from src.database import CallRecording
   pending = CallRecording.get_pending()
   print(f'Pending: {len(pending)}')"
   ```

4. Check poll interval:
   ```env
   POLL_INTERVAL_SECONDS=30  # Reduce for testing
   ```

---

### CUDA Out of Memory

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions:**

1. Use smaller model:
   ```env
   WHISPER_MODEL=small  # Instead of medium/large
   ```

2. Switch to transcript mode:
   ```env
   ANALYSIS_MODE=transcript  # Instead of audio
   ```

3. Reduce concurrent jobs:
   ```env
   MAX_CONCURRENT_JOBS=1
   ```

4. Enable mock mode for testing:
   ```env
   USE_MOCK_MODELS=true
   ```

5. Clear GPU memory:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

---

### Recording Download Failed

**Symptoms:**
```
DOWNLOAD_FAILED: Recording not found
```

**Solutions:**

1. Check PBX URLs are configured:
   ```env
   PBX_LIVE_URL=https://pbx.angelfs.co.uk/callrec/
   PBX_ARCHIVE_URL=https://afs-pbx-callarchive.angelfs.co.uk/
   ```

2. Verify recording exists:
   ```bash
   curl -I "https://pbx.angelfs.co.uk/callrec/YOUR-APEX-ID.gsm"
   ```

3. Check network connectivity from worker

---

### Transcription Failed

**Symptoms:**
```
TRANSCRIPTION_FAILED: Audio file is invalid or corrupt
```

**Solutions:**

1. Check audio file:
   ```bash
   ffprobe /tmp/angel/recording.wav
   ```

2. Check SoX conversion:
   ```bash
   sox input.gsm -r 8000 -b 32 -c 1 output.wav
   ```

3. Check disk space:
   ```bash
   df -h /tmp
   ```

---

## Model Issues

### Model Loading Slow

**Symptoms:**
- First request takes several minutes
- High disk I/O during startup

**Solutions:**

1. Pre-download models:
   ```python
   import whisperx
   whisperx.load_model("medium", "cpu")
   ```

2. Use NFS for shared models:
   ```env
   MODELS_BASE_PATH=/nfs/models
   ```

3. Use smaller models initially:
   ```env
   WHISPER_MODEL=small
   ```

---

### Model Not Found

**Symptoms:**
```
OSError: Model not found: /models/analysis/v1.0.0
```

**Solutions:**

1. Check model path exists:
   ```bash
   ls -la /models/analysis/
   ```

2. Check NFS mount:
   ```bash
   mount | grep nfs
   ```

3. Download model manually:
   ```python
   from transformers import AutoModel
   model = AutoModel.from_pretrained("Qwen/Qwen2.5-Omni-7B")
   model.save_pretrained("/models/analysis/")
   ```

---

## Performance Issues

### Slow Processing

**Symptoms:**
- Processing takes longer than expected
- Queue builds up

**Solutions:**

1. Check GPU utilisation:
   ```bash
   nvidia-smi -l 1  # Watch GPU usage
   ```

2. Reduce model size:
   ```env
   WHISPER_MODEL=small
   ```

3. Increase worker count:
   ```bash
   kubectl scale deployment angel-intelligence-worker --replicas=4
   ```

4. Check for disk bottleneck:
   ```bash
   iostat -x 1
   ```

---

### High Memory Usage

**Symptoms:**
- OOM killer terminates worker
- Swap usage high

**Solutions:**

1. Reduce concurrent jobs:
   ```env
   MAX_CONCURRENT_JOBS=1
   ```

2. Use transcript mode:
   ```env
   ANALYSIS_MODE=transcript
   ```

3. Add memory limits:
   ```yaml
   resources:
     limits:
       memory: "8Gi"
   ```

---

## Kubernetes Issues

### Pod CrashLoopBackOff

**Symptoms:**
- Pod restarts repeatedly
- Status shows `CrashLoopBackOff`

**Solutions:**

1. Check pod logs:
   ```bash
   kubectl logs <pod-name> --previous
   ```

2. Check resource limits:
   ```bash
   kubectl describe pod <pod-name>
   ```

3. Check secrets are mounted:
   ```bash
   kubectl exec <pod-name> -- env | grep AI_DB
   ```

---

### ImagePullBackOff

**Symptoms:**
- Pod stuck in `ImagePullBackOff`

**Solutions:**

1. Check image exists:
   ```bash
   docker images | grep angel-intelligence
   ```

2. Check registry credentials:
   ```bash
   kubectl get secret regcred
   ```

3. Push image to registry:
   ```bash
   docker push localhost:5000/angel-intelligence:latest
   ```

---

## Getting Help

If these solutions don't resolve your issue:

1. Collect diagnostic information:
   ```bash
   # System info
   uname -a
   python --version
   pip freeze > requirements_installed.txt
   
   # GPU info
   nvidia-smi > gpu_info.txt
   
   # Logs
   tail -500 worker.log > worker_log.txt
   ```

2. Check environment:
   ```bash
   env | grep -E "AI_|ANGEL_|WHISPER|ANALYSIS" > env_info.txt
   ```

3. Create a minimal reproduction case

4. Contact the development team with:
   - Error message
   - Steps to reproduce
   - Environment details
   - Relevant logs
