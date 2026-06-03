#!/bin/bash
# Setup script for Omni Captioner tool (API-based)
# Uses Qwen3-Omni via DashScope API - no local model needed

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.cache/uv}"
mkdir -p "$UV_CACHE_DIR"

echo "============================================================"
echo "Setting up Omni Captioner environment (API-based)"
echo "============================================================"
echo ""

# Find uv - check persistent location first, then PATH
if [ -f "$REPO_ROOT/.uv/bin/uv" ]; then
    UV="$REPO_ROOT/.uv/bin/uv"
elif command -v uv &> /dev/null; then
    UV="uv"
else
    echo "Error: uv not found. Please install uv first."
    exit 1
fi

echo "Using uv: $UV"
echo "Using UV_CACHE_DIR: $UV_CACHE_DIR"
echo "uv version: $($UV --version)"
echo ""

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create virtual environment
echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

# Install package and dependencies
echo "Installing Omni Captioner package..."
$UV pip install --python .venv/bin/python -e .

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "⚠️  IMPORTANT: Set your DashScope API key before use:"
echo ""
echo "    export DASHSCOPE_API_KEY='your-api-key'"
echo ""
echo "Or update config.yaml with your API key."
echo ""
echo "To verify installation:"
echo "  ./test_env.sh"
echo ""
echo "To test the server:"
echo "  .venv/bin/python server.py"
echo ""
