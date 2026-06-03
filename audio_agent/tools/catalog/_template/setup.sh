#!/bin/bash
# Setup script template — copy and adapt for a new tool.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"

# Optionally use a repo-local persistent uv install at $REPO_ROOT/.uv.
if [ -f "$REPO_ROOT/.uv/activate.sh" ]; then
    source "$REPO_ROOT/.uv/activate.sh"
fi

# Find uv: prefer a repo-local persistent install, else system uv on PATH.
if [ -f "$REPO_ROOT/.uv/bin/uv" ]; then
    UV="$REPO_ROOT/.uv/bin/uv"
elif command -v uv &> /dev/null; then
    UV="uv"
else
    echo "Error: uv not found. Install via 'curl -LsSf https://astral.sh/uv/install.sh | sh' or place a uv binary at $REPO_ROOT/.uv/bin/uv." >&2
    exit 1
fi

echo "Using uv: $UV"
echo "Using UV_CACHE_DIR: $UV_CACHE_DIR"

# Recreate venv
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

echo "Installing this tool's package..."
$UV pip install --python .venv/bin/python -e .

echo ""
echo "Setup complete!"
echo "Python: $(.venv/bin/python --version)"
echo "Run ./test_env.sh to verify."
