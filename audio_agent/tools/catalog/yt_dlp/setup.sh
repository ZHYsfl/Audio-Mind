#!/bin/bash
# Setup script for yt_dlp tool

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

echo "Installing yt_dlp tool package..."
$UV pip install --python .venv/bin/python -e .

# Vendored Node server
if ! command -v npm &> /dev/null; then
    echo "Error: npm not found. Please install Node.js first."
    exit 1
fi
echo "Installing vendored Node server (@kevinwatt/yt-dlp-mcp)..."
npm install --ignore-scripts --no-audit --no-fund

# Patch vendored templates: truncate yt-dlp's %(title)s to 50 chars.
# Without this, videos with long CJK titles exceed the 255-byte filename
# limit and ffmpeg fails with ENAMETOOLONG ("exited with code 220").
PKG="node_modules/@kevinwatt/yt-dlp-mcp/lib/modules"
sed -i 's/%(title)s \[%/%(title).50s [%/' "$PKG/video.js" "$PKG/audio.js"
sed -i "s|path.join(tempDir, '%(title)s.%(ext)s')|path.join(tempDir, '%(title).50s.%(ext)s')|" "$PKG/subtitle.js"
grep -q '%(title)\.50s' "$PKG/video.js" || { echo "Error: title-truncation patch did not apply" >&2; exit 1; }
echo "ok: patched vendored title templates to %(title).50s"

# Host dependency checks (not installed by this script)
for cmd in node yt-dlp deno ffmpeg; do
    if command -v "$cmd" &> /dev/null; then
        echo "ok: $cmd found ($(command -v $cmd))"
    else
        echo "warning: $cmd not found on PATH - install it before using this tool" >&2
    fi
done

echo ""
echo "Setup complete!"
echo "Python version: $(.venv/bin/python --version)"
echo "Run ./test_env.sh to verify."
