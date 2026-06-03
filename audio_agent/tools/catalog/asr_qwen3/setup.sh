#!/bin/bash
# Setup script for ASR Qwen3 model

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"

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
echo "uv version: $($UV --version)"

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create virtual environment
echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

# Install PyTorch with CUDA (explicitly using .venv Python)
# echo "Installing PyTorch with CUDA support..."
# $UV pip install --python .venv/bin/python torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install package (explicitly using .venv Python)
echo "Installing ASR Qwen3 package..."
$UV pip install --python .venv/bin/python -e .

echo ""
echo "Setup complete!"
