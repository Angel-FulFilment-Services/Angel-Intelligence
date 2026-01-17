#!/bin/bash
# Run Angel Intelligence API server locally

set -e

echo "🚀 Starting Angel Intelligence API..."
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

echo "📍 Environment: $ANGEL_ENV"
echo ""

# Run the API server
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload

