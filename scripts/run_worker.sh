#!/bin/bash
# Run Angel Intelligence Worker locally

set -e

echo "🚀 Starting Angel Intelligence Worker..."
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -d "venv" ]; then
        echo "📦 Activating virtual environment..."
        source venv/bin/activate
    else
        echo "⚠️  Virtual environment not found. Please run:"
        echo "   python -m venv venv"
        echo "   source venv/bin/activate"
        echo "   pip install -r requirements.txt"
        exit 1
    fi
fi

# Set development mode if not set
export ANGEL_ENV="${ANGEL_ENV:-development}"
export WORKER_ID="${WORKER_ID:-local-dev}"

echo "📍 Environment: $ANGEL_ENV"
echo "🆔 Worker ID: $WORKER_ID"
echo ""

# Run the worker
python -m src.worker.worker

