# Quick Start Guide

Get Angel Intelligence running in 5 minutes.

## Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (recommended) or CPU
- MySQL 8.0+
- SoX audio tool
- ffmpeg

## 1. Clone and Install

```bash
# Clone repository
git clone https://github.com/Angel-FulFilment-Services/angel-intelligence.git
cd angel-intelligence

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Linux/macOS)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Configure Environment

```bash
# Copy example configuration
cp .env.example .env
```

Edit `.env` with minimum required settings:

```env
# Required
ANGEL_ENV=development
API_AUTH_TOKEN=your-secure-64-character-minimum-token-here-replace-this-text

# Database
AI_DB_HOST=localhost
AI_DB_DATABASE=ai
AI_DB_USERNAME=root
AI_DB_PASSWORD=

# Mock mode (no GPU required)
USE_MOCK_MODELS=true
```

## 3. Create Database Tables

```sql
CREATE DATABASE IF NOT EXISTS ai CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- See documentation/DATABASE_SCHEMA.md for full schema
```

## 4. Start the API Server

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Visit: http://localhost:8000/docs for Swagger UI

## 5. Start the Worker

In a separate terminal:

```bash
python -m src.worker.worker
```

## 6. Test the Installation

```bash
# Health check (no auth required)
curl http://localhost:8000/health

# Submit a test recording (with auth)
curl -X POST http://localhost:8000/recordings/submit \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{"apex_id": "TEST-001", "call_date": "2026-01-17"}'
```

## What's Next?

- [Local Development](LOCAL_DEVELOPMENT.md) - Full development setup
- [API Reference](API_REFERENCE.md) - All available endpoints
- [Production Deployment](PRODUCTION_DEPLOYMENT.md) - Deploy to production

## Quick Commands

| Task | Command |
|------|---------|
| Start API | `uvicorn src.api:app --reload` |
| Start Worker | `python -m src.worker.worker` |
| Run Tests | `pytest tests/` |
| View Logs | `tail -f worker.log` |
