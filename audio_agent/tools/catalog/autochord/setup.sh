#!/bin/bash
# Setup script for autochord tool

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
echo "Using UV_PYTHON_INSTALL_DIR: $UV_PYTHON_INSTALL_DIR"
echo "Using UV_CACHE_DIR: $UV_CACHE_DIR"

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create virtual environment
echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

# Install numpy first (vamp build dependency)
echo "Installing numpy first (required for vamp build)..."
$UV pip install --python .venv/bin/python numpy

# Install TensorFlow 2.15 (Keras 2 compatible) BEFORE autochord
echo "Installing TensorFlow 2.15.1 (Keras 2, required for autochord model format)..."
$UV pip install --python .venv/bin/python "tensorflow==2.15.1" "keras==2.15.0"

# Install autochord and soundfile with no-build-isolation
echo "Installing autochord dependencies..."
$UV pip install --python .venv/bin/python --no-build-isolation autochord==0.1.4 "soundfile>=0.12.1"

# Pin setuptools for pkg_resources compatibility
echo "Pinning setuptools for pkg_resources compatibility..."
$UV pip install --python .venv/bin/python "setuptools<82"

# Install local package as editable with no deps (already installed above)
echo "Installing local package..."
$UV pip install --python .venv/bin/python --no-deps -e .

echo ""
echo "Setup complete!"
echo "Python version: $(.venv/bin/python --version)"
