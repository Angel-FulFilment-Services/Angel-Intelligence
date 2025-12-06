# Docker configuration for AI service
# Supports both x86_64 (development) and ARM64 (Jetson)

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port (for health checks)
EXPOSE 8000

# Run worker by default
CMD ["python", "worker.py"]
