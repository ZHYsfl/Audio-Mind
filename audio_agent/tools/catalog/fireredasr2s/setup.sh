#!/bin/bash
# Setup script for FireRedASR2S tool

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"

# Find uv - check persistent location first, then PATH
if [ -f "$REPO_ROOT/.uv/bin/uv" ]; then
    UV="$REPO_ROOT/.uv/bin/uv"
elif command -v uv &> /dev/null; then
    UV="uv"
else
    echo "Error: uv not found. Please install uv first."
    echo "Run: curl -LsSf https://astral.sh/uv/install.sh | sh"
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

# Create virtual environment with Python 3.11
echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

# Install package with index-strategy to handle PyTorch dependencies
echo "Installing FireRedASR2S package..."
$UV pip install --python .venv/bin/python --index-strategy unsafe-best-match -e .

echo ""
echo "Setup complete!"
echo "Python version: $(.venv/bin/python --version)"
echo ""
echo "To verify the installation, run: ./test_env.sh"
