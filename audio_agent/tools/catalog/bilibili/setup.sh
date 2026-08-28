#!/bin/bash
# Setup script for bilibili tool

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
    exit 1
fi

echo "Using uv: $UV"
echo "Using UV_CACHE_DIR: $UV_CACHE_DIR"

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create virtual environment (proxy is stdlib-only)
echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

echo "Installing bilibili tool package..."
$UV pip install --python .venv/bin/python -e .

# Vendored Node server
if ! command -v npm &> /dev/null; then
    echo "Error: npm not found. Please install Node.js first."
    exit 1
fi
echo "Installing vendored Node server (@xzxzzx/bilibili-mcp)..."
npm install --ignore-scripts --no-audit --no-fund

if command -v node &> /dev/null; then
    echo "ok: node found ($(command -v node))"
else
    echo "warning: node not found on PATH - install it before using this tool" >&2
fi

cat <<'EOF'

Setup complete!
Next step (user action, interactive - credentials never pass through the agent):
  node node_modules/@xzxzzx/bilibili-mcp/dist/cli.js setup
  node node_modules/@xzxzzx/bilibili-mcp/dist/cli.js check
Then run ./test_env.sh to verify.
EOF
