# ==============================================================================
# Angel Intelligence - AGX Thor 128GB Configuration
# ==============================================================================
# Optimised for NVIDIA AGX Thor with 128GB unified memory
# Supports 5-6 concurrent workers with high-quality models

# ==============================================================================
# ENVIRONMENT MODE
# ==============================================================================
ANGEL_ENV=production
WORKER_ID=thor-worker-01

# ==============================================================================
# API AUTHENTICATION
# ==============================================================================
# Generate with: python -c "import secrets; print(secrets.token_urlsafe(64))"
API_AUTH_TOKEN=CHANGE_ME_GENERATE_SECURE_TOKEN

# ==============================================================================
# DATABASE CONFIGURATION
# ==============================================================================
AI_DB_HOST=your-db-host
AI_DB_PORT=3306
AI_DB_DATABASE=ai
AI_DB_USERNAME=your-user
AI_DB_PASSWORD=your-password

# ==============================================================================
# R2 STORAGE CONFIGURATION
# ==============================================================================
R2_ENDPOINT=https://your-account.r2.cloudflarestorage.com
R2_ACCESS_KEY_ID=your-access-key
R2_SECRET_ACCESS_KEY=your-secret-key
R2_BUCKET=call-recordings

# ==============================================================================
# PBX RECORDING SOURCES
# ==============================================================================
PBX_LIVE_URL=https://your-pbx-server/callrec/
PBX_ARCHIVE_URL=https://your-archive-server/

# ==============================================================================
# MODEL CONFIGURATION - THOR 128GB OPTIMISED
# ==============================================================================
# Base path for model storage
MODELS_BASE_PATH=/models

# Whisper: Use large-v3 for best accuracy (Thor has headroom)
WHISPER_MODEL=large-v3

# Analysis model: 32B provides excellent JSON structure and insights
# ~18GB VRAM at INT4, well within Thor's capacity
ANALYSIS_MODEL=Qwen/Qwen2.5-32B-Instruct
ANALYSIS_MODEL_PATH=/models/analysis
ANALYSIS_MODEL_QUANTIZATION=int4

# Chat model: 14B for interactive queries (runs separately from analysis)
CHAT_MODEL=Qwen/Qwen2.5-14B-Instruct
CHAT_MODEL_PATH=/models/chat
CHAT_MODEL_QUANTIZATION=int4

# Analysis mode: transcript (text-based, works with any LLM)
ANALYSIS_MODE=transcript

# No transcript length limit - 32B handles full transcripts well
MAX_TRANSCRIPT_LENGTH=0

# ==============================================================================
# SPEAKER DIARIZATION
# ==============================================================================
# HuggingFace token for pyannote speaker diarization
HUGGINGFACE_TOKEN=hf_your_token_here

# ==============================================================================
# PROCESSING CONFIGURATION - THOR OPTIMISED
# ==============================================================================
# Poll interval (seconds)
POLL_INTERVAL=10

# Thor 128GB can handle 5-6 concurrent jobs with separate models
# Use higher values if implementing shared model server
MAX_CONCURRENT_JOBS=5

# Retry configuration
MAX_RETRY_ATTEMPTS=3
RETRY_DELAY_HOURS=1

# Worker mode
WORKER_MODE=batch

# Preload chat model for API responsiveness
PRELOAD_CHAT_MODEL=true

# Word-level timestamps for karaoke feature
TRANSCRIPT_SEGMENTATION=word

# ==============================================================================
# PII REDACTION
# ==============================================================================
ENABLE_PII_REDACTION=true

# ==============================================================================
# GPU CONFIGURATION
# ==============================================================================
# Thor has integrated GPU, use device 0
CUDA_VISIBLE_DEVICES=0

# ==============================================================================
# vLLM SHARED MODEL SERVER (Optional)
# ==============================================================================
# When using vLLM, uncomment to share a single 32B model for analysis and chat
# This allows running 10-12 workers instead of 5-6
# 
# Deploy vLLM first: docker run -p 8000:8000 vllm/vllm-openai --model Qwen/Qwen2.5-32B-Instruct
# Or use k8s/vllm-deployment.yaml for Kubernetes
#
# LLM_API_URL=http://localhost:8000/v1
# LLM_API_KEY=  # Only if vLLM requires authentication

# ==============================================================================
