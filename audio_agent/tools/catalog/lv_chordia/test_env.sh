#!/bin/bash
# Environment test wrapper for lv-chordia tool

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

echo "Running environment tests with .venv/bin/python..."
.venv/bin/python test_env.py
