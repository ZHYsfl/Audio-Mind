#!/usr/bin/env python3
"""Environment validation script for tempo-cnn tool."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def test_imports() -> None:
    """Test that required packages can be imported."""
    print("Testing imports...")
    try:
        import tempocnn
        from tempocnn.version import __version__ as tempocnn_version
        print(f"  ✓ tempocnn {tempocnn_version}")
    except Exception as e:
        print(f"  ✗ tempocnn import failed: {e}")
        sys.exit(1)

    try:
        import soundfile
        print(f"  ✓ soundfile {soundfile.__version__}")
    except Exception as e:
        print(f"  ✗ soundfile import failed: {e}")
        sys.exit(1)

    try:
        import tensorflow as tf
        print(f"  ✓ tensorflow {tf.__version__}")
    except Exception as e:
        print(f"  ✗ tensorflow import failed: {e}")
        sys.exit(1)


def test_model_wrapper() -> None:
    """Test that ModelWrapper can be instantiated and healthchecked."""
    print("\nTesting ModelWrapper...")
    try:
        from model import ModelWrapper

        wrapper = ModelWrapper()
        health = wrapper.healthcheck()
        print(f"  status: {health['status']}")
        print(f"  message: {health['message']}")
        print(f"  shared_models_dir: {health['shared_models_dir']}")
        print(f"  shared_models_dir_exists: {health['shared_models_dir_exists']}")
        if health["status"] != "ready":
            print(f"  ✗ Healthcheck failed: {health['message']}")
            sys.exit(1)
        print("  ✓ ModelWrapper healthcheck passed")
    except Exception as e:
        print(f"  ✗ ModelWrapper test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def test_inference() -> None:
    """Run a minimal inference on a sample audio file if available."""
    print("\nTesting inference...")

    # Look for sample audio in framework assets
    repo_root = Path(__file__).resolve().parent.parent.parent.parent.parent
    candidates = [
        repo_root / "assets" / "sample_music.mp3",
        repo_root / "assets" / "music_2.mp3",
        repo_root / "assets" / "jaychou.mp3",
    ]
    audio_path = None
    for c in candidates:
        if c.exists():
            audio_path = str(c)
            break

    if audio_path is None:
        print("  ⚠ No sample audio found, skipping inference test")
        return

    print(f"  Using audio: {audio_path}")

    try:
        from model import ModelWrapper

        wrapper = ModelWrapper()
        result = wrapper.estimate_tempo(audio_path)
        print(f"  ✓ Single-tempo inference passed")
        print(f"    tempo: {result.tempo} BPM")
        print(f"    model: {result.model_name}")

        # Test dual-tempo (mirex) mode
        dual = wrapper.estimate_mirex(audio_path)
        print(f"  ✓ Dual-tempo (MIREX) inference passed")
        print(f"    tempo1: {dual.tempo1} BPM")
        print(f"    tempo2: {dual.tempo2} BPM")
        print(f"    salience: {dual.salience:.2f}")
        print(f"    note: {dual.note}")
    except Exception as e:
        print(f"  ✗ Inference test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main() -> int:
    print("=" * 50)
    print("Tempo-CNN Tool Environment Validation")
    print("=" * 50)

    test_imports()
    test_model_wrapper()
    test_inference()

    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
