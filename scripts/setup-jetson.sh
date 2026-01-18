#!/bin/bash
# ==============================================================================
# Angel Intelligence - Jetson Setup Script
# ==============================================================================
# Quick setup script for NVIDIA Jetson devices (Python 3.8)

set -e  # Exit on error

echo "=========================================="
echo "Angel Intelligence - Jetson Setup"
echo "=========================================="

# Install system dependencies first
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y sox libsox-fmt-all ffmpeg
sudo apt install -y libopenblas-dev liblapack-dev libblas-dev gfortran
echo "System dependencies installed!"

# Check Python version
echo ""
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version)
echo "Found: $PYTHON_VERSION"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install PyTorch for Jetson
echo ""
echo "Installing PyTorch for Jetson..."
if [ ! -f "torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl" ]; then
    echo "Downloading PyTorch wheel..."
    wget https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
fi

echo "Installing PyTorch..."
pip install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl

echo "Installing torchaudio..."
pip install torchaudio==2.0.0

# Verify CUDA
echo ""
echo "Verifying CUDA availability..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Install Jetson-specific requirements
echo ""
echo "Installing Angel Intelligence dependencies..."
pip install -r requirements-jetson.txt

# Install spaCy model
echo ""
echo "Installing spaCy language model..."
python -m spacy download en_core_web_lg

# Install WhisperX (optional)
echo ""
echo "Installing WhisperX (this may take a while)..."
echo "Note: WhisperX may fail on Python 3.8 - if it fails, you can:"
echo "  1. Use mock mode (set USE_MOCK_MODELS=true)"
echo "  2. Use a Docker container with Python 3.10+"
echo ""
pip install git+https://github.com/m-bain/whisperx.git || {
    echo ""
    echo "⚠️  WhisperX installation failed (expected on Python 3.8)"
    echo "    The system will work in mock mode or you can:"
    echo "    - Use Docker with Python 3.10"
    echo "    - Upgrade to Python 3.10 on Jetson"
    echo ""
}

# Install system dependencies
echo ""
echo "Installing system dependencies..."
echo "System dependencies already installed (sox, ffmpeg, openblas, etc.)"

# Setup complete
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and configure:"
echo "   cp .env.example .env"
echo "   nano .env"
echo ""
echo "2. For quick testing with mock models (RECOMMENDED for Python 3.8):"
echo "   Set USE_MOCK_MODELS=true in .env"
echo "   This will skip model loading and return test data"
echo ""
echo "3. Start the API server:"
echo "   source venv/bin/activate"
echo "   uvicorn src.api:app --host 0.0.0.0 --port 8080"
echo ""
echo "4. Test from another machine:"
echo "   curl http://JETSON_IP:8080/health"
echo ""
