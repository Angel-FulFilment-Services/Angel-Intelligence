# Training System Documentation

## Overview

The Pulse Call Intelligence system includes an automated training pipeline that fine-tunes the Qwen 2.5 72B model using LoRA (Low-Rank Adaptation) based on human-corrected call annotations. This allows the system to continuously improve its analysis quality over time.

## Architecture

### Components

1. **TrainingService** (`src/services/trainer.py`)
   - Manages LoRA fine-tuning on human annotations
   - Handles versioning and adapter management
   - Provides lock mechanism to prevent concurrent training

2. **Training API** (`src/api/routes.py`)
   - REST endpoints for triggering, monitoring, and managing training
   - Allows manual training runs and version rollbacks

3. **Training CronJob** (`k8s/training-cronjob.yaml`)
   - Scheduled nightly training at 2:00 AM UTC
   - Runs only if sufficient new annotations exist

4. **vLLM Integration**
   - Loads trained LoRA adapters dynamically
   - Supports hot-reloading of new adapter versions
   - Can serve multiple adapters concurrently (up to 4)

## Training Data

### Sources

Training data comes from two sources in the `ai_call_annotations` table:

**1. Legacy Annotations**
- Full call-level corrections
- Field-by-field adjustments (sentiment, quality, topics)
- All samples receive weight 1.0 (corrections)

**2. Dojo Session Annotations (Recommended)**
- Segment-level training on specific call moments
- Three segment types: `agent_action`, `score_impact`, `compliance_flag`
- Trainer agrees/disagrees with AI assessment
- Supports timestamp boundary adjustments
- Weighted training: 0.5 for agreements, 1.0 for corrections

### Dojo Session Data Structure

Each Dojo annotation includes:
```json
{
  "segment_key": "agent_action-greeting-123",
  "segment_type": "agent_action",
  "ai_assessment": "Good",
  "trainer_assessment": "Excellent",
  "disagreement_reason": "too_harsh",
  "timestamp_start": 15.5,
  "timestamp_end": 28.3,
  "original_timestamp_start": 14.0,
  "original_timestamp_end": 30.0,
  "segment_ids": [3, 4, 5, 6]
}
```

**Segment Types:**
- `agent_action` - Actions taken by the agent
- `score_impact` - Events that impact call quality score  
- `compliance_flag` - Compliance violations or concerns

**Disagreement Reasons:**
- `too_harsh` - AI assessment was too negative
- `too_lenient` - AI assessment was too positive
- `wrong_category` - Wrong type identified
- `not_quality_affecting` - Doesn't affect quality
- `false_positive` - AI incorrectly flagged this
- `incorrect_timestamp` - Timing is wrong
- `missed_context` - AI missed important context
- `other` - Other reason

### Training Sample Weights

The system uses weighted training to balance reinforcement and correction:

| Sample Type | Weight | Purpose |
|-------------|--------|---------|
| Agreement (trainer confirms AI) | 0.5 | Positive reinforcement - AI already doing well |
| Correction (trainer disagrees) | 1.0 | Full weight - AI needs to learn from mistake |
| False Positive | 1.0 | Special handling to reduce over-flagging |
| Timestamp Adjustment | 1.0 | Train for better boundary detection |

### Format

**Legacy Format (full call):**
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert call analyst..."},
    {"role": "user", "content": "Analyse this call transcript:\n\n[TRANSCRIPT]"},
    {"role": "assistant", "content": "{\"sentiment\": \"positive\", ...}"}
  ]
}
```

**Dojo Segment Format (segment-level):**
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert call analyst..."},
    {"role": "user", "content": "Analyse this call segment for AGENT ACTIONS.\n\n[SEGMENT]"},
    {"role": "assistant", "content": "{\"action_detected\": true, \"quality_impact\": \"Excellent\"}"}
  ]
}
```

### Requirements
- Minimum 50 new annotations required to trigger training (configurable)
- Dojo sessions automatically trigger training when session closes
- Supports mixed training from both legacy and Dojo annotations

## Training Process

### Configuration

**LoRA Parameters:**
- Rank: 16
- Alpha: 32
- Target modules: All attention and MLP layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- Dropout: 0.05

**Training Arguments:**
- Epochs: 3 (default, configurable)
- Batch size: 1 per device
- Gradient accumulation: 8 steps
- Learning rate: 1e-4 with cosine schedule
- Precision: FP16 mixed precision
- Quantization: 4-bit (BitsAndBytes) for memory efficiency
- Gradient checkpointing: Enabled

### Resource Requirements

**Memory:**
- Base model (4-bit): ~40GB
- Gradients and optimizer states: ~8GB
- Training batch: ~4GB
- **Total: ~52GB** (fits in 64GB system)

**Compute:**
- CPU: 4-8 cores
- Training time: 30-60 minutes for 500 samples, 3 epochs
- No GPU required (4-bit training runs efficiently on CPU)

### Execution Flow

1. **Lock Acquisition**
   - Checks for existing training lock (`/app/data/training.lock`)
   - Prevents concurrent training runs
   - Stale lock timeout: 4 hours

2. **Data Preparation**
   - Fetches new annotations from database
   - Filters for quality (non-empty corrections)
   - Formats as chat conversations
   - Tokenizes and pads to model's max length

3. **Model Loading**
   - Loads base model (Qwen2.5-72B-Instruct-AWQ) in 4-bit
   - Applies LoRA configuration
   - Freezes base model parameters
   - Enables gradient checkpointing

4. **Training Loop**
   - 3 epochs over training data
   - Gradient accumulation every 8 steps
   - Mixed precision (FP16)
   - Cosine learning rate schedule with warmup

5. **Adapter Saving**
   - Saves to versioned directory: `/models/adapters/call-analysis/call-analysis-{YYYYMMDD-HHMM}/`
   - Includes adapter weights, tokenizer, and training metadata
   - Creates `current.json` manifest pointing to new version
   - Creates `current` symlink for vLLM static loading

6. **Version Management**
   - Keeps last 5 versions automatically
   - Removes older versions to save storage
   - Maintains history for rollback capability

7. **Lock Release**
   - Always releases lock, even on failure
   - Logs completion time and metrics

## Versioning

### Directory Structure

```
/models/adapters/call-analysis/
├── current                              # Symlink → call-analysis-20260130-0200
├── current.json                         # Manifest with metadata
├── call-analysis-20260130-0200/        # Versioned adapter
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── training_metadata.json
│   ├── tokenizer_config.json
│   └── ...
├── call-analysis-20260129-0200/        # Previous version
└── call-analysis-20260128-0200/        # Older version
```

### current.json Format

```json
{
  "version": "call-analysis-20260130-0200",
  "promoted_at": "2026-01-30T02:45:23Z",
  "trained_at": "2026-01-30T02:15:00Z",
  "samples_used": 523,
  "training_loss": 0.342
}
```

### training_metadata.json Format

```json
{
  "adapter_name": "call-analysis",
  "version": "call-analysis-20260130-0200",
  "trained_at": "2026-01-30T02:15:00Z",
  "base_model": "/models/Qwen2.5-72B-Instruct-AWQ",
  "samples_used": 523,
  "epochs": 3,
  "training_loss": 0.342,
  "training_time_seconds": 1847.3,
  "lora_config": {
    "r": 16,
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", ...],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
  }
}
```

## API Endpoints

### POST /api/training/start

Triggers a new training run.

**Request:**
```json
{
  "force": false,           // Skip minimum annotation check
  "max_samples": 1000,      // Limit training data size
  "epochs": 3               // Number of training epochs
}
```

**Response:**
```json
{
  "success": true,
  "adapter_name": "call-analysis",
  "version": "call-analysis-20260130-0200",
  "adapter_path": "/models/adapters/call-analysis/call-analysis-20260130-0200",
  "samples_used": 523,
  "epochs": 3,
  "training_loss": 0.342,
  "training_time_minutes": 30.8
}
```

**Errors:**
- `409 Conflict` - Training already in progress
- `400 Bad Request` - Insufficient new annotations (use `force: true` to override)
- `500 Internal Server Error` - Training failed

### GET /api/training/status

Gets current training status and version information.

**Response:**
```json
{
  "is_training": false,
  "lock_acquired_at": null,
  "current_version": {
    "version": "call-analysis-20260130-0200",
    "promoted_at": "2026-01-30T02:45:23Z",
    "trained_at": "2026-01-30T02:15:00Z",
    "samples_used": 523,
    "training_loss": 0.342
  },
  "available_versions": [
    "call-analysis-20260130-0200",
    "call-analysis-20260129-0200",
    "call-analysis-20260128-0200"
  ],
  "new_annotations_count": 127,
  "min_annotations_required": 50
}
```

### POST /api/training/promote

Promotes (activates) a different adapter version.

**Request:**
```json
{
  "version": "call-analysis-20260129-0200"
}
```

**Response:**
```json
{
  "success": true,
  "version": "call-analysis-20260129-0200",
  "previous_version": "call-analysis-20260130-0200"
}
```

**Use Cases:**
- Rollback after bad training run
- A/B testing different adapter versions
- Revert to known-good version

### POST /api/training/cleanup

Removes old adapter versions.

**Request:**
```json
{
  "keep": 5  // Number of versions to retain
}
```

**Response:**
```json
{
  "success": true,
  "removed": [
    "call-analysis-20260120-0200",
    "call-analysis-20260119-0200"
  ],
  "kept": [
    "call-analysis-20260130-0200",
    "call-analysis-20260129-0200",
    "call-analysis-20260128-0200",
    "call-analysis-20260127-0200",
    "call-analysis-20260126-0200"
  ]
}
```

### GET /api/training-data

Exports training data in JSONL format.

**Query Parameters:**
- `limit`: Maximum samples to export (default: all)
- `offset`: Skip first N samples (default: 0)
- `format`: Export format - `jsonl` or `json` (default: `jsonl`)

**Response:**
```
{"messages": [{"role": "system", ...}, ...]}
{"messages": [{"role": "system", ...}, ...]}
...
```

## Automated Training (CronJob)

### Schedule

Training runs automatically at **2:00 AM UTC** every night.

### Prerequisites

Before training runs:
1. Checks for minimum 50 new annotations (configurable via `MIN_NEW_ANNOTATIONS` env var)
2. Checks that no training is currently in progress
3. Validates base model is available

### Monitoring

**Check if training is running:**
```bash
kubectl get jobs -n pulse -l component=training
```

**View training logs:**
```bash
kubectl logs -n pulse job/model-training-{timestamp} -f
```

**Check last training completion:**
```bash
kubectl get cronjob -n pulse model-training
```

### Manual Trigger

**Via kubectl:**
```bash
kubectl create job --from=cronjob/model-training manual-training-$(date +%Y%m%d-%H%M) -n pulse
```

**Via API:**
```bash
curl -X POST http://api.pulse.local/api/training/start \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"force": true, "epochs": 3, "max_samples": 1000}'
```

## vLLM Integration

### Static Loading (Startup)

vLLM loads adapters at startup using the `--lora-modules` flag:

```bash
vllm serve /models/Qwen2.5-72B-Instruct-AWQ \
  --enable-lora \
  --lora-modules call-analysis=/models/adapters/call-analysis/current \
  --max-loras 4 \
  --max-lora-rank 64
```

The `current` symlink always points to the active adapter version, updated after each training run.

### Dynamic Loading (Runtime)

Applications can request specific adapter versions dynamically via the API:

```python
payload = {
    "model": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    "messages": [...],
    "extra_body": {
        "lora_request": {
            "lora_name": "call-analysis",
            "lora_path": "/models/adapters/call-analysis/call-analysis-20260130-0200"
        }
    }
}
```

### Hot-Reloading

After training completes:

1. New adapter saved to versioned directory
2. `current.json` manifest updated
3. `current` symlink updated
4. vLLM automatically discovers new adapter on next request (no restart required)
5. Old adapters remain loaded for concurrent requests
6. vLLM caches up to 4 adapters simultaneously

### Version Rollback

To rollback to a previous version:

```bash
# Via API
curl -X POST http://api.pulse.local/api/training/promote \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"version": "call-analysis-20260129-0200"}'

# Or manually update symlink
cd /models/adapters/call-analysis
ln -sf call-analysis-20260129-0200 current
```

vLLM will use the rolled-back version on subsequent requests.

## Monitoring & Troubleshooting

### Training Metrics

**Check training status:**
```bash
curl http://api.pulse.local/api/training/status
```

**View adapter metadata:**
```bash
cat /models/adapters/call-analysis/current.json
cat /models/adapters/call-analysis/call-analysis-20260130-0200/training_metadata.json
```

### Common Issues

#### Training Stuck

**Symptom:** CronJob never completes, no new versions created

**Diagnosis:**
```bash
# Check if lock file exists and is stale
ls -la /app/data/training.lock

# View training logs
kubectl logs -n pulse job/model-training-{timestamp} -f
```

**Resolution:**
```bash
# Remove stale lock (only if training genuinely stuck)
rm /app/data/training.lock

# Or wait 4 hours for automatic stale lock timeout
```

#### Insufficient Annotations

**Symptom:** Training skipped with "Not enough new annotations"

**Diagnosis:**
```bash
curl http://api.pulse.local/api/training/status
# Check "new_annotations_count" vs "min_annotations_required"
```

**Resolution:**
```bash
# Lower threshold (in k8s/training-cronjob.yaml)
MIN_NEW_ANNOTATIONS: "25"

# Or force training with fewer samples
curl -X POST http://api.pulse.local/api/training/start \
  -d '{"force": true}'
```

#### vLLM Not Loading Adapter

**Symptom:** Analysis quality doesn't improve after training

**Diagnosis:**
```bash
# Check if current symlink points to new version
ls -la /models/adapters/call-analysis/current

# Check vLLM logs for LoRA loading
kubectl logs -n pulse deployment/vllm-server | grep -i lora
```

**Resolution:**
```bash
# Recreate symlink
cd /models/adapters/call-analysis
ln -sf call-analysis-20260130-0200 current

# Restart vLLM (if hot-reload not working)
kubectl rollout restart deployment/vllm-server -n pulse
```

#### Out of Memory During Training

**Symptom:** Training OOM killed

**Diagnosis:**
```bash
kubectl describe pod -n pulse {training-pod-name}
# Look for "OOMKilled" in container status
```

**Resolution:**
```bash
# Increase memory limit in k8s/training-cronjob.yaml
resources:
  limits:
    memory: "32Gi"  # Was 24Gi

# Or reduce training batch size (in trainer.py)
per_device_train_batch_size = 1
gradient_accumulation_steps = 4  # Was 8

# Or limit training samples
curl -X POST /api/training/start -d '{"max_samples": 500}'
```

#### Training Takes Too Long

**Symptom:** Training exceeds 90-minute timeout

**Diagnosis:**
```bash
# Check sample count and training time in metadata
cat /models/adapters/call-analysis/{version}/training_metadata.json
```

**Resolution:**
```bash
# Reduce epochs in k8s/training-cronjob.yaml or API call
curl -X POST /api/training/start -d '{"epochs": 2}'

# Or limit sample count
curl -X POST /api/training/start -d '{"max_samples": 800, "epochs": 3}'

# Or increase timeout in k8s/training-cronjob.yaml
activeDeadlineSeconds: 7200  # 2 hours instead of 90 min
```

## Best Practices

### Training Frequency

- **Nightly (default):** Good balance for most deployments
- **Weekly:** Sufficient if annotation volume is low (<50/week)
- **On-demand:** Trigger manually after significant annotation batches

### Annotation Quality

- Ensure annotations are genuine improvements, not just reformats
- Include diverse examples (different call types, issues, sentiments)
- Aim for 100-500 annotations per training run for best results
- More training data → better generalization

### Version Management

- Keep 5 versions for rollback capability (default)
- Test new versions on sample calls before full deployment
- Monitor analysis quality metrics after each training run
- Roll back immediately if quality degrades

### Resource Optimization

- Train during low-traffic hours (default: 2 AM)
- Use non-GPU nodes to avoid competing with vLLM
- Limit max_samples if training takes too long
- Consider reducing epochs to 2 for very large datasets

### Monitoring

- Set up alerts for:
  - Training failures
  - Training duration exceeds threshold
  - New version quality below baseline
  - Annotation backlog exceeds threshold
- Track metrics:
  - Training loss over time
  - Analysis quality scores pre/post training
  - User feedback on AI analysis accuracy

## Advanced Topics

### Custom Adapter Names

To train multiple specialized adapters (e.g., for different call types):

```python
# In training script
trainer = TrainingService(adapter_name="email-support")
trainer.train()

# In analysis code
analyzer.analysis_adapter_name = "email-support"
```

### Multi-Stage Training

For complex scenarios, train adapters sequentially:

1. Base adapter on all call types
2. Specialized adapter for complex cases
3. Load both adapters in vLLM with priority

### Training on Specific Data Subsets

Filter annotations by metadata before training:

```python
# In trainer.py prepare_training_data()
samples = CallAnnotation.get_training_data(
    min_date="2026-01-01",
    call_type="customer_complaint"
)
```

### Hyperparameter Tuning

Experiment with LoRA configuration:

```python
self.lora_config = {
    "r": 32,              # Higher rank for more expressiveness
    "lora_alpha": 64,     # Higher alpha for stronger adaptation
    "lora_dropout": 0.1,  # Higher dropout for regularization
}
```

### Distributed Training

For very large datasets, use multi-GPU training:

```bash
# In k8s/training-cronjob.yaml
resources:
  limits:
    nvidia.com/gpu: "2"

# In trainer.py
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    dataloader_num_workers=4,
)
```

## References

- [PEFT/LoRA Documentation](https://huggingface.co/docs/peft)
- [vLLM LoRA Adapters](https://docs.vllm.ai/en/latest/models/lora.html)
- [Qwen 2.5 Model Card](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct)
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)
