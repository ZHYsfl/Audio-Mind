#!/usr/bin/env python3
"""Quick environment test for snakers4_silero-vad tool."""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False
    try:
        from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
        print("  ✓ silero_vad imported successfully")
    except ImportError as e:
        print(f"  ✗ silero_vad: {e}")
        return False
    return True


def test_model_wrapper():
    """Test that the model wrapper can be imported and healthchecked."""
    print("\nTesting model wrapper...")
    try:
        from model import VADModel
        model = VADModel(device="cpu")
        health = model.healthcheck()
        print(f"  ✓ VADModel healthcheck: {health}")
        return True
    except Exception as e:
        print(f"  ✗ VADModel failed: {e}")
        return False


def test_inference():
    """Test minimal inference on fixture."""
    print("\nTesting inference...")
    fixture_path = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "shared" / "mir" / "rhythm_22k_15s.wav"
    if not fixture_path.exists():
        print(f"  ⚠ Fixture not found at {fixture_path}, skipping inference test")
        return True
    try:
        from model import VADModel
        model = VADModel(device="cpu")
        result = model.predict(str(fixture_path), sampling_rate=16000)
        print(f"  ✓ Inference passed: segments={len(result.speech_segments)}, duration={result.audio_duration:.2f}s")
        return True
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Silero VAD Tool Environment Test")
    print("=" * 60)
    print()

    results = [
        ("Imports", test_imports()),
        ("Model Wrapper", test_model_wrapper()),
        ("Inference", test_inference()),
    ]

    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(passed for _, passed in results)
    print()
    print("✓ All tests passed!" if all_passed else "✗ Some tests failed.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
