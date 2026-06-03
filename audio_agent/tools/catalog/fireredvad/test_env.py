#!/usr/bin/env python3
"""Environment smoke check for the FireRedVAD MCP tool.

What this script verifies:
  1. Core deps (fireredvad, torch, numpy, soundfile) import.
  2. The local wrapper module imports.
  3. VAD/AED inference works on a synthetic WAV — only if model weights
     are present at $MODEL_PATH / $AUDIO_AGENT_MODELS_DIR/FireRedVAD;
     skipped otherwise so the test is informative without weights.

Run via `./test_env.sh` (the shell wrapper sets up paths).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))


def test_imports() -> bool:
    print("Testing imports...")
    try:
        import fireredvad  # noqa: F401
        print("  ✓ fireredvad imported")
    except Exception as exc:
        print(f"  ✗ fireredvad import failed: {exc}")
        return False
    try:
        import torch
        print(f"  ✓ torch {torch.__version__} imported")
    except Exception as exc:
        print(f"  ✗ torch import failed: {exc}")
        return False
    try:
        import numpy  # noqa: F401
        import soundfile  # noqa: F401
        print("  ✓ numpy + soundfile imported")
    except Exception as exc:
        print(f"  ✗ numpy/soundfile import failed: {exc}")
        return False
    try:
        from model import ModelWrapper, VADResult, AEDResult  # noqa: F401
        print("  ✓ ModelWrapper / VADResult / AEDResult imported")
    except Exception as exc:
        print(f"  ✗ wrapper import failed: {exc}")
        return False
    return True


def _synthetic_wav() -> str:
    """Write a tiny silence+tone+silence WAV; override via AUDIO_AGENT_TEST_WAV."""
    override = os.environ.get("AUDIO_AGENT_TEST_WAV")
    if override and Path(override).exists():
        return override
    import numpy as np
    import soundfile as sf
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32")
    audio = np.concatenate([
        np.zeros(int(sr * 0.5), dtype="float32"),
        tone,
        np.zeros(int(sr * 1.0), dtype="float32"),
    ])
    fh = tempfile.NamedTemporaryFile(prefix="fireredvad_test_", suffix=".wav", delete=False)
    fh.close()
    sf.write(fh.name, audio, sr)
    print(f"  wrote synthetic WAV: {fh.name}")
    return fh.name


def _model_dir() -> Path:
    if os.environ.get("MODEL_PATH"):
        return Path(os.environ["MODEL_PATH"])
    models_root = os.environ.get("AUDIO_AGENT_MODELS_DIR")
    if not models_root:
        models_root = str(THIS_DIR.parents[3] / "models")
    return Path(models_root) / "FireRedVAD"


def test_inference() -> bool:
    model_dir = _model_dir()
    if not (model_dir / "VAD").is_dir() or not (model_dir / "AED").is_dir():
        print(f"⚠ Model weights not found at {model_dir}.")
        print("  Skipping VAD/AED inference. Run: audio-agent-download-models --models fireredvad")
        return True

    from model import ModelWrapper
    wav = _synthetic_wav()
    wrapper = ModelWrapper()

    print("Testing VAD...")
    vad_result = wrapper.predict(wav)
    print(f"  ✓ VAD: {len(vad_result.timestamps)} speech segments")

    print("Testing AED...")
    aed_result = wrapper.predict_aed(wav)
    detected = [k for k, v in aed_result.event2timestamps.items() if v]
    print(f"  ✓ AED: detected events — {detected}")
    return True


def main() -> int:
    if not test_imports():
        return 1
    if not test_inference():
        return 1
    print()
    print("All environment tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
