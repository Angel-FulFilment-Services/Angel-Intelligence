# ==============================================================================
# Angel Intelligence - Requirements Structure
# ==============================================================================
#
# This folder contains modular requirements for different pod types.
# Use the appropriate file for each deployment scenario.
#
# ## Pod-Specific Requirements (Recommended for Production)
#
# | File | Pod Type | Image Size | Description |
# |------|----------|------------|-------------|
# | `api.txt` | API | ~200MB | Lightweight API gateway |
# | `worker.txt` | Worker | ~400MB | HTTP orchestrator (shared services) |
# | `transcription.txt` | Transcription | ~4GB | WhisperX + pyannote |
# | `worker-standalone.txt` | Worker | ~8GB | Full worker with local models |
#
# ## Base Requirements
#
# | File | Description |
# |------|-------------|
# | `base.txt` | Common dependencies (included by all) |
#
# ## Usage
#
# Each pod-specific file includes base.txt automatically:
# ```bash
# pip install -r requirements/api.txt
# pip install -r requirements/worker.txt
# pip install -r requirements/transcription.txt
# ```
#
# ## Corresponding Dockerfiles
#
# | Dockerfile | Requirements | Target |
# |------------|--------------|--------|
# | `Dockerfile.api` | api.txt | x86_64 |
# | `Dockerfile.worker` | worker.txt | x86_64 |
# | `Dockerfile.transcription` | transcription.txt | x86_64 |
# | `Dockerfile.worker-jetson` | worker.txt | ARM64 (Thor/Jetson) |
# | `Dockerfile.transcription-jetson` | transcription.txt | ARM64 (Thor/Jetson) |
#
# ## vLLM Pod
#
# The vLLM pod uses the official `vllm/vllm-openai:latest` image.
# No custom requirements needed - just mount model storage.
#
# ## Root Directory File
#
# The `requirements.txt` in the project root is for local development
# and installs ALL dependencies (equivalent to worker-standalone.txt).
