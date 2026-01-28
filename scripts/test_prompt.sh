#!/bin/bash
#
# Test LLM Prompt Generation (Bash)
#
# Tests the full LLM prompt generation for a call recording
# WITHOUT marking it as processing or completed.
#
# Usage:
#   ./scripts/test_prompt.sh                    # Test next pending recording
#   ./scripts/test_prompt.sh --id 123           # Test specific recording by ID
#   ./scripts/test_prompt.sh --apex ABC123      # Test specific recording by apex_id
#   ./scripts/test_prompt.sh --id 123 --save    # Save prompt to file

echo "Angel Intelligence - Prompt Tester"
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "venv/bin/activate" ]; then
        echo "Activating virtual environment..."
        source venv/bin/activate
    elif [ -f ".venv/bin/activate" ]; then
        echo "Activating virtual environment..."
        source .venv/bin/activate
    else
        echo "Virtual environment not found. Please run:"
        echo "   python -m venv venv"
        echo "   source venv/bin/activate"
        echo "   pip install -r requirements.txt"
        exit 1
    fi
fi

# Run the script with all arguments passed through
python scripts/test_prompt.py "$@"
