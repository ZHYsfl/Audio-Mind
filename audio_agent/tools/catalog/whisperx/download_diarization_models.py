#!/usr/bin/env python3
"""Download pyannote speaker diarization models for offline use.

This script downloads the pyannote models needed for speaker diarization
to a local directory. After downloading, the models can be used
without an HF token.

Usage:
    export HF_TOKEN=hf_xxx
    # MODEL_DIR defaults to $AUDIO_AGENT_MODELS_DIR (or <repo>/models when unset).
    python download_diarization_models.py
"""

import os
import sys
from pathlib import Path

# Set model directory. Prefer explicit MODEL_DIR; else AUDIO_AGENT_MODELS_DIR;
# else fall back to <repo>/models.
_REPO_ROOT = Path(__file__).resolve().parents[4]
MODEL_DIR = (
    os.environ.get("MODEL_DIR")
    or os.environ.get("AUDIO_AGENT_MODELS_DIR")
    or str(_REPO_ROOT / "models")
)
os.environ["HF_HOME"] = MODEL_DIR


def download_models():
    """Download pyannote diarization models."""
    token = os.environ.get("HF_TOKEN")
    
    if not token:
        print("Error: HF_TOKEN environment variable not set")
        print("Get your token at: https://huggingface.co/settings/tokens")
        sys.exit(1)
    
    print("=" * 60)
    print("Downloading Pyannote Diarization Models")
    print("=" * 60)
    print(f"Model directory: {MODEL_DIR}")
    print()
    
    try:
        import torch
        from whisperx.diarize import DiarizationPipeline
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        print()
        
        # Download the diarization model
        print("Downloading pyannote speaker diarization model...")
        print("(This may take a few minutes...)")
        print()
        
        diarize_model = DiarizationPipeline(
            token=token,
            device=device,
        )
        
        print("✓ Model downloaded successfully!")
        print()
        
        # Verify model directory contents
        model_path = Path(MODEL_DIR) / "pyannote-speaker-diarization-3.1"
        if model_path.exists():
            print("Model files:")
            for item in model_path.rglob("*"):
                if item.is_file():
                    print(f"  - {item.relative_to(model_path)}")
        
        print()
        print("=" * 60)
        print("Download Complete!")
        print("=" * 60)
        print()
        print(f"Models saved to: {model_path}")
        print("You can use transcribe_with_diarization without HF_TOKEN.")
        
    except Exception as e:
        print(f"Error downloading models: {e}")
        sys.exit(1)


if __name__ == "__main__":
    download_models()
