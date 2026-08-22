#!/bin/bash
# Setup script for SortFormer speaker diarization tool
# Follows SERVER_SPECIFIC_UV_SETUP.md for persistent uv usage

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

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create virtual environment with Python 3.11
echo "Creating virtual environment with Python 3.11..."
$UV venv --python 3.11

# Install PyTorch with CUDA (must be before NeMo).
# Blackwell (RTX 50-series, sm_120) requires cu128 + torch>=2.7; we pin 2.8.0.
# Override TORCH_CUDA_VARIANT for a different host ABI (e.g. "cu121" on older GPUs).
TORCH_CUDA_VARIANT="${TORCH_CUDA_VARIANT:-cu128}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
echo "Installing PyTorch (torch==${TORCH_VERSION}+${TORCH_CUDA_VARIANT})..."
$UV pip install --python .venv/bin/python \
    "torch==${TORCH_VERSION}+${TORCH_CUDA_VARIANT}" "torchaudio==${TORCH_VERSION}+${TORCH_CUDA_VARIANT}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_VARIANT}"

# Install ModelScope SDK for checkpoint download
echo "Installing ModelScope SDK..."
$UV pip install --python .venv/bin/python modelscope>=1.15.0

# Install NeMo ASR toolkit
echo "Installing NVIDIA NeMo ASR toolkit..."
$UV pip install --python .venv/bin/python "nemo_toolkit[asr]>=2.0.0"

# Install remaining dependencies from pyproject.toml
echo "Installing remaining dependencies..."
$UV pip install --python .venv/bin/python -e .

# NeMo's transitive deps tend to upgrade torch/torchaudio. Re-pin to the same
# CUDA variant we picked above so the resulting wheels still match the host's
# libcudart.
echo "Re-pinning torch + torchaudio to ${TORCH_VERSION}+${TORCH_CUDA_VARIANT} (overriding any NeMo-driven upgrade)..."
$UV pip install --python .venv/bin/python --reinstall \
    "torch==${TORCH_VERSION}+${TORCH_CUDA_VARIANT}" "torchaudio==${TORCH_VERSION}+${TORCH_CUDA_VARIANT}" \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA_VARIANT}"

echo ""
echo "Setup complete!"
echo "Python version: $(.venv/bin/python --version)"
echo ""
echo "Run ./test_env.sh to verify the installation."
