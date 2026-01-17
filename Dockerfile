# ==============================================================================
# Angel Intelligence - Docker Configuration
# ==============================================================================
# Supports both x86_64 (development) and ARM64 (Jetson Nano)

FROM python:3.10-slim

# Labels
LABEL org.opencontainers.image.title="Angel Intelligence"
LABEL org.opencontainers.image.description="AI-powered call transcription and analysis"
LABEL org.opencontainers.image.vendor="Angel Fulfilment Services"
LABEL org.opencontainers.image.version="1.0.0"

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ANGEL_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    sox \
    libsox-fmt-all \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download spaCy model for PII detection
RUN python -m spacy download en_core_web_lg

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd -m -u 1000 angel && \
    chown -R angel:angel /app
USER angel

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose API port
EXPOSE 8000

# Default command - run worker
# Override with: docker run ... uvicorn src.api:app --host 0.0.0.0
CMD ["python", "-m", "src.worker.worker"]
