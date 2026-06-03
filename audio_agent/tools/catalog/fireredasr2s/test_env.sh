#!/bin/bash
# Test environment script for FireRedASR2S tool

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

echo "Testing FireRedASR2S tool environment..."
echo ""

# Run Python test script
.venv/bin/python test_env.py
