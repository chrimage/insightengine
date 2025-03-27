#!/bin/bash
#
# Script to generate summaries for indexed conversations
# This script now acts as a simple launcher for the Python module.
#

# Check for virtual environment and activate if possible
if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    # Only source if not already in the venv
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "📦 Activating virtual environment..."
        source venv/bin/activate
    fi
elif [ -d ".venv" ] && [ -f ".venv/bin/activate" ]; then
    # Check for alternate venv directory
    if [ -z "$VIRTUAL_ENV" ]; then
        echo "📦 Activating virtual environment..."
        source .venv/bin/activate
    fi
fi

# Execute the Python module, passing all arguments
python -m memory_ai.tools.summarize "$@"
