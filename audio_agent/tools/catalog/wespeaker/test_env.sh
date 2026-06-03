#!/bin/bash
# Environment smoke check for the WeSpeaker MCP tool.
# Thin shell wrapper around test_env.py.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"
if [ ! -x "$PYTHON_EXE" ]; then
    echo "Error: $PYTHON_EXE not found. Run ./setup.sh first." >&2
    exit 1
fi

echo "Using Python: $PYTHON_EXE ($($PYTHON_EXE --version))"
"$PYTHON_EXE" test_env.py "$@"
