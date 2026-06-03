#!/bin/bash
# Setup script for DiariZen speaker diarization tool

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate conda. DiariZen requires Python 3.10 (uv pins are too strict for the
# submodule install chain), so this is the one tool that genuinely needs conda.
# Search order: user-defined CONDA_SH, conda already on PATH, common installs.
_activate_conda() {
    # Explicit override wins.
    if [ -n "${CONDA_SH:-}" ] && [ -f "$CONDA_SH" ]; then
        # shellcheck source=/dev/null
        source "$CONDA_SH"
        return 0
    fi
    # Conda already exposed via shell hook?
    if command -v conda &> /dev/null; then
        local base
        base="$(conda info --base 2>/dev/null)"
        if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
            # shellcheck source=/dev/null
            source "$base/etc/profile.d/conda.sh"
            return 0
        fi
    fi
    # Probe a few common install roots without hardcoding any user's path.
    for candidate in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"; do
        if [ -f "$candidate" ]; then
            # shellcheck source=/dev/null
            source "$candidate"
            return 0
        fi
    done
    return 1
}

if ! _activate_conda; then
    echo "Error: conda not found. Diarizen needs conda to provision Python 3.10." >&2
    echo "Install miniconda/anaconda, or export CONDA_SH=/path/to/conda/etc/profile.d/conda.sh and retry." >&2
    exit 1
fi

# Remove old venv if exists
if [ -d ".venv" ]; then
    echo "Removing existing .venv..."
    rm -rf .venv
fi

# Create virtual environment with Python 3.10 (NOT 3.11!)
echo "Creating virtual environment with Python 3.10..."
conda create --prefix ./.venv python=3.10 -y

# Install PyTorch with CUDA (Step 2 - MUST be before DiariZen!)
echo "Installing PyTorch with CUDA support..."
.venv/bin/pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install DiariZen from submodule (Step 3)
echo "Installing DiariZen from submodule..."
git clone https://github.com/BUTSpeechFIT/DiariZen.git
mv DiariZen diarizen_src
cd diarizen_src
../.venv/bin/pip install -r requirements.txt && ../.venv/bin/pip install -e .

# Install pyannote-audio from submodule (Step 4 - CRITICAL!)
echo "Installing pyannote-audio from submodule..."
cd pyannote-audio
../../.venv/bin/pip install -e .
cd ../..

# Lock NumPy version (Step 5 - CRITICAL!)
echo "Locking NumPy to 1.26.4..."
.venv/bin/pip install numpy==1.26.4

# Install missing dependencies (Step 6)
echo "Installing missing dependencies..."
.venv/bin/pip install psutil accelerate

# Install remaining dependencies from pyproject.toml
echo "Installing remaining dependencies..."
.venv/bin/pip install huggingface-hub>=0.20.0

# Install the local package (server code)
echo "Installing server package..."
.venv/bin/pip install -e . --no-deps

echo ""
echo "Setup complete!"
echo ""
echo "Python version: $(.venv/bin/python --version)"
echo ""
echo "Run ./test_env.sh to verify the installation."
