# Installation Guide

Complete installation instructions for Angel Intelligence.

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 22.04 LTS, Windows 10/11, or macOS 12+
- **Python**: 3.10 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 20GB free space (50GB+ for models)
- **GPU**: None (mock mode) or NVIDIA GPU with 8GB+ VRAM

### Recommended Production Setup
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.11
- **RAM**: 32GB
- **GPU**: NVIDIA Jetson Orin Nano 8GB or better
- **Storage**: 100GB SSD

## Step 1: Install System Dependencies

### Ubuntu/Debian

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install Python and pip
sudo apt install -y python3.11 python3.11-venv python3-pip

# Install audio tools
sudo apt install -y sox libsox-fmt-all ffmpeg

# Install MySQL client
sudo apt install -y mysql-client libmysqlclient-dev

# Install build tools (for some Python packages)
sudo apt install -y build-essential git curl
```

### Windows

1. **Python 3.11**: Download from https://www.python.org/downloads/
   - Check "Add Python to PATH" during installation

2. **SoX**: Install via Chocolatey
   ```powershell
   choco install sox
   ```

3. **ffmpeg**: Install via Chocolatey
   ```powershell
   choco install ffmpeg
   ```

4. **MySQL**: Download from https://dev.mysql.com/downloads/installer/

### macOS

```bash
# Install Homebrew if not present
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python@3.11 sox ffmpeg mysql-client
```

## Step 2: Install NVIDIA GPU Drivers (Optional)

Skip this section if using mock mode or CPU-only.

### Ubuntu (Desktop GPU)

```bash
# Check for existing NVIDIA driver
nvidia-smi

# If not installed, add NVIDIA repository
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt update

# Install driver (check latest version)
sudo apt install -y nvidia-driver-535

# Reboot
sudo reboot

# Install CUDA Toolkit
sudo apt install -y nvidia-cuda-toolkit
```

### NVIDIA Jetson

NVIDIA Jetson devices come with JetPack which includes CUDA. Ensure JetPack 5.0+ is installed.

```bash
# Check JetPack version
cat /etc/nv_tegra_release

# Check CUDA
nvcc --version
```

## Step 3: Clone Repository

```bash
# Clone from GitHub
git clone https://github.com/Angel-FulFilment-Services/angel-intelligence.git
cd angel-intelligence

# Or clone from internal GitLab
git clone git@gitlab.angelfs.co.uk:ai/angel-intelligence.git
cd angel-intelligence
```

## Step 4: Create Python Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate (Linux/macOS)
source venv/bin/activate

# Activate (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# Activate (Windows Command Prompt)
venv\Scripts\activate.bat
```

## Step 5: Install Python Dependencies

### Choose Your Installation Type

Angel Intelligence provides modular requirements for different deployment scenarios:

| File | Use Case | Size |
|------|----------|------|
| `requirements.txt` | Local development / standalone | Full |
| `requirements/api.txt` | API pod only | ~200MB |
| `requirements/worker.txt` | Worker pod (shared services) | ~400MB |
| `requirements/transcription.txt` | Transcription pod | ~4GB |

### Local Development (Full Installation)

```bash
pip install --upgrade pip wheel setuptools

# Install PyTorch first (CUDA 12.1)
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt
```

### CUDA 11.8 (Older GPUs)

```bash
pip install --upgrade pip wheel setuptools
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### CPU Only

```bash
pip install --upgrade pip wheel setuptools
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Install WhisperX

```bash
pip install git+https://github.com/m-bain/whisperx.git
```

### Install spaCy Language Model

```bash
python -m spacy download en_core_web_lg
```

### Optional: Install Qwen2.5-Omni Preview

For audio analysis mode (requires GPU with 16GB+ VRAM):

```bash
pip install git+https://github.com/huggingface/transformers@v4.51.3-Qwen2.5-Omni-preview
pip install qwen-omni-utils[decord]
```

## Step 6: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your settings
nano .env  # Linux/macOS
notepad .env  # Windows
```

Essential configuration:

```env
# Environment
ANGEL_ENV=development

# Security (REQUIRED - minimum 64 characters)
API_AUTH_TOKEN=generate-a-secure-random-string-here-minimum-64-characters

# Database
AI_DB_HOST=your-mysql-host
AI_DB_PORT=3306
AI_DB_DATABASE=ai
AI_DB_USERNAME=your-username
AI_DB_PASSWORD=your-password

# Processing
USE_MOCK_MODELS=false
ANALYSIS_MODE=audio
WHISPER_MODEL=medium
```

See [Environment Variables](ENVIRONMENT_VARIABLES.md) for complete reference.

## Step 7: Create Database

Connect to MySQL and create the database:

```sql
CREATE DATABASE IF NOT EXISTS ai 
  CHARACTER SET utf8mb4 
  COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS 'angel_ai'@'%' 
  IDENTIFIED BY 'your-secure-password';

GRANT ALL PRIVILEGES ON ai.* TO 'angel_ai'@'%';
FLUSH PRIVILEGES;
```

Create tables using the schema in [Database Schema](DATABASE_SCHEMA.md).

## Step 8: Verify Installation

### Check GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

### Check WhisperX

```bash
python -c "import whisperx; print('WhisperX installed successfully')"
```

### Check Database Connection

```bash
python -c "from src.database import get_db_connection; db = get_db_connection(); print('Database connected')"
```

### Run Health Check

```bash
# Start API temporarily
uvicorn src.api:app --host 0.0.0.0 --port 8000 &

# Check health
curl http://localhost:8000/health

# Stop API
kill %1
```

## Step 9: Download Models (First Run)

Models are downloaded automatically on first use. To pre-download:

```bash
# Download Whisper model
python -c "import whisperx; whisperx.load_model('medium', 'cpu')"

# Download spaCy model (if not done)
python -m spacy download en_core_web_lg
```

## Common Installation Issues

### CUDA Version Mismatch

```
RuntimeError: CUDA error: no kernel image is available for execution
```

Solution: Install PyTorch for your CUDA version:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Missing SoX

```
FileNotFoundError: [Errno 2] No such file or directory: 'sox'
```

Solution: Install SoX:
```bash
# Ubuntu
sudo apt install sox libsox-fmt-all

# Windows
choco install sox

# macOS
brew install sox
```

### MySQL Connection Failed

```
mysql.connector.errors.InterfaceError: 2003: Can't connect to MySQL server
```

Solution: 
1. Verify MySQL is running
2. Check firewall rules
3. Verify credentials in `.env`

### Memory Errors

```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

Solution:
- Use smaller model: `WHISPER_MODEL=small`
- Use transcript mode: `ANALYSIS_MODE=transcript`
- Enable mock mode for testing: `USE_MOCK_MODELS=true`

## Next Steps

- [Local Development](LOCAL_DEVELOPMENT.md) - Development workflow
- [Production Deployment](PRODUCTION_DEPLOYMENT.md) - Deploy to production
- [Testing Guide](TESTING.md) - Verify installation
