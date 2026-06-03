#!/usr/bin/env python3
"""Quick environment test for librosa tool."""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import librosa
        print(f"  ✓ librosa {librosa.__version__}")
    except ImportError as e:
        print(f"  ✗ librosa: {e}")
        return False
    return True


def test_model_wrapper():
    """Test that the model wrapper can be imported and healthchecked."""
    print("\nTesting model wrapper...")
    try:
        from model import ModelWrapper
        wrapper = ModelWrapper()
        health = wrapper.healthcheck()
        print(f"  ✓ ModelWrapper healthcheck: {health}")
        return True
    except Exception as e:
        print(f"  ✗ ModelWrapper failed: {e}")
        return False


def test_inference():
    """Test minimal inference on fixture."""
    print("\nTesting inference...")
    fixture_path = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "shared" / "mir" / "rhythm_22k_15s.wav"
    if not fixture_path.exists():
        print(f"  ⚠ Fixture not found at {fixture_path}, skipping inference test")
        return True
    try:
        from model import ModelWrapper
        wrapper = ModelWrapper()
        result = wrapper.predict(str(fixture_path))
        print(f"  ✓ Inference passed: tempo={result.tempo:.2f}, beats={len(result.beats)}")
        return True
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Librosa Tool Environment Test")
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
