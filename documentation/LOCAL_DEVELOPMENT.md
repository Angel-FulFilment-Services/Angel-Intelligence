# Local Development Guide

Set up a local development environment for Angel Intelligence.

## Prerequisites

- Python 3.10+
- MySQL 8.0+
- Git
- VS Code (recommended)
- NVIDIA GPU (optional, for full testing)

---

## Quick Setup

```bash
# Clone repository
git clone https://github.com/Angel-FulFilment-Services/angel-intelligence.git
cd angel-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1 on Windows

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Dev tools

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start services
uvicorn src.api:app --reload --port 8000
```

---

## Development Environment

### VS Code Setup

Install recommended extensions:

```json
// .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.black-formatter",
    "charliermarsh.ruff",
    "mtxr.sqltools",
    "mtxr.sqltools-driver-mysql"
  ]
}
```

Configure VS Code settings:

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.flake8Enabled": true,
  "editor.formatOnSave": true,
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  }
}
```

### Development Dependencies

```bash
# Install dev dependencies
pip install pytest pytest-cov black flake8 mypy pre-commit

# Set up pre-commit hooks
pre-commit install
```

---

## Mock Mode

For development without a GPU, enable mock mode:

```env
USE_MOCK_MODELS=true
```

Mock mode:
- Returns deterministic test data
- Doesn't require GPU or models
- Faster iteration during development
- Still tests full pipeline logic

---

## Local Database

### Using Docker

```bash
# Start MySQL container
docker run -d \
  --name angel-mysql \
  -e MYSQL_ROOT_PASSWORD=dev \
  -e MYSQL_DATABASE=ai \
  -p 3306:3306 \
  mysql:8.0

# Configure .env
AI_DB_HOST=localhost
AI_DB_DATABASE=ai
AI_DB_USERNAME=root
AI_DB_PASSWORD=dev
```

### Create Tables

```bash
mysql -h localhost -u root -pdev ai < documentation/schema.sql
```

### Database Tools

Connect with VS Code SQLTools or:

```bash
# MySQL CLI
mysql -h localhost -u root -pdev ai

# View tables
SHOW TABLES;

# Check pending recordings
SELECT id, apex_id, processing_status FROM ai_call_recordings;
```

---

## Running Services

### API Server (Development)

```bash
# With auto-reload
uvicorn src.api:app --reload --port 8000

# Access at:
# - API: http://localhost:8000
# - Swagger: http://localhost:8000/docs
# - ReDoc: http://localhost:8000/redoc
```

### Worker (Development)

In a separate terminal:

```bash
# Activate venv
source venv/bin/activate

# Run worker
python -m src.worker.worker
```

### Both with Docker Compose

```bash
docker-compose -f docker-compose.dev.yml up
```

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Run Specific Tests

```bash
# Single file
pytest tests/test_pii_detector.py -v

# Single test
pytest tests/test_pii_detector.py::TestPIIDetector::test_detect_ni_number -v

# Matching pattern
pytest tests/ -k "pii" -v
```

### Test with Mock Mode

```bash
USE_MOCK_MODELS=true pytest tests/ -v
```

---

## Code Quality

### Format Code

```bash
# Format with Black
black src/ tests/

# Check formatting
black src/ tests/ --check
```

### Lint Code

```bash
# Flake8
flake8 src/ tests/

# Ruff (faster)
ruff check src/ tests/
```

### Type Check

```bash
mypy src/ --ignore-missing-imports
```

---

## Debugging

### VS Code Debug Configuration

```json
// .vscode/launch.json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "API Server",
      "type": "python",
      "request": "launch",
      "module": "uvicorn",
      "args": ["src.api:app", "--reload", "--port", "8000"],
      "envFile": "${workspaceFolder}/.env"
    },
    {
      "name": "Worker",
      "type": "python",
      "request": "launch",
      "module": "src.worker.worker",
      "envFile": "${workspaceFolder}/.env"
    },
    {
      "name": "Current File",
      "type": "python",
      "request": "launch",
      "program": "${file}",
      "envFile": "${workspaceFolder}/.env"
    }
  ]
}
```

### Debug Logging

```env
LOG_LEVEL=DEBUG
```

### Interactive Debugging

```python
# Add breakpoint in code
import pdb; pdb.set_trace()

# Or use IPython
import IPython; IPython.embed()
```

---

## Working with Audio Files

### Sample Files

Place test audio files in `tests/fixtures/`:

```bash
tests/fixtures/
├── sample_audio.wav      # Clean call
├── sample_pii.wav        # Contains PII
├── sample_noisy.wav      # Poor quality
└── sample_long.wav       # 5+ minute call
```

### Convert Audio

```bash
# GSM to WAV
sox input.gsm -r 8000 -b 32 -c 1 output.wav

# Check audio info
ffprobe -hide_banner input.wav
```

---

## Git Workflow

### Branches

- `master` - Production releases
- `develop` - Integration branch
- `feature/*` - Feature branches
- `fix/*` - Bug fix branches

### Commits

Follow conventional commits:

```bash
git commit -m "feat: add voice fingerprinting service"
git commit -m "fix: handle missing audio file gracefully"
git commit -m "docs: update API reference"
```

### Pull Requests

1. Create feature branch from `develop`
2. Make changes
3. Run tests locally
4. Push and create PR
5. Wait for CI/CD checks
6. Request review

---

## Useful Commands

```bash
# Check GPU status
nvidia-smi

# Watch GPU usage
watch -n 1 nvidia-smi

# Check process
ps aux | grep python

# Kill process on port
lsof -ti:8000 | xargs kill

# View logs with colour
tail -f worker.log | ccze -A

# Database query
mysql -e "SELECT COUNT(*) FROM ai_call_recordings" ai

# Clear Python cache
find . -type d -name __pycache__ -exec rm -rf {} +

# Check disk space
df -h /tmp
```

---

## Environment Variables for Development

```env
# Development settings
ANGEL_ENV=development
API_AUTH_TOKEN=dev-token-64-chars-minimum-for-local-development-testing-only

# Database (Docker MySQL)
AI_DB_HOST=localhost
AI_DB_PORT=3306
AI_DB_DATABASE=ai
AI_DB_USERNAME=root
AI_DB_PASSWORD=dev

# Mock mode (no GPU needed)
USE_MOCK_MODELS=true
USE_GPU=false

# Faster polling for testing
POLL_INTERVAL_SECONDS=5
MAX_CONCURRENT_JOBS=1

# Verbose logging
LOG_LEVEL=DEBUG

# Small model for faster testing
WHISPER_MODEL=tiny

# Skip PII for faster testing (optional)
ENABLE_PII_REDACTION=true
```

---

## Next Steps

- [Testing Guide](TESTING.md) - Write and run tests
- [API Reference](API_REFERENCE.md) - Explore endpoints
- [Architecture](ARCHITECTURE.md) - Understand system design
