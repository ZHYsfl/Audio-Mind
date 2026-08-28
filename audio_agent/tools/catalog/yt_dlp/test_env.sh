#!/bin/bash
# Quick environment test wrapper

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$PYTHON_EXE" ]; then
    echo "Error: Python interpreter not found at $PYTHON_EXE"
    echo "Please run setup.sh first."
    exit 1
fi

echo "Using Python: $PYTHON_EXE"
echo "Python version: $($PYTHON_EXE --version)"
echo ""

$PYTHON_EXE test_env.py "$@"
