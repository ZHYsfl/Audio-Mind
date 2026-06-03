#!/usr/bin/env python3
"""Quick environment test for autochord tool."""

import sys
from pathlib import Path


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    try:
        import autochord
        print("  ✓ autochord imported")
    except ImportError as e:
        print(f"  ✗ autochord: {e}")
        return False
    try:
        import soundfile
        print("  ✓ soundfile imported")
    except ImportError:
        print("  ⚠ soundfile not available (optional)")
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
    """Test minimal inference on fixture or generated sine wave."""
    print("\nTesting inference...")
    # Try shared fixture first
    fixture_path = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "shared" / "mir" / "chord_progression_16k_10s.wav"
    if fixture_path.exists():
        audio_path = str(fixture_path)
        print(f"  Using fixture: {audio_path}")
    else:
        # Generate a simple test audio file
        try:
            import numpy as np
            import soundfile as sf
            audio_path = "/tmp/autochord_test_440hz.wav"
            sr = 22050
            t = np.linspace(0, 3.0, int(sr * 3.0))
            # C major chord: C4+E4+G4 (261.63, 329.63, 392.00 Hz)
            y = (
                0.3 * np.sin(2 * np.pi * 261.63 * t)
                + 0.3 * np.sin(2 * np.pi * 329.63 * t)
                + 0.3 * np.sin(2 * np.pi * 392.00 * t)
            )
            y = y * 0.5
            sf.write(audio_path, y, sr)
            print(f"  Generated test audio: {audio_path}")
        except Exception as e:
            print(f"  ⚠ Could not generate test audio: {e}")
            return True  # Skip if we can't generate

    try:
        from model import ModelWrapper
        wrapper = ModelWrapper()
        result = wrapper.predict(audio_path)
        print(f"  ✓ Inference passed: {len(result.segments)} segments, duration={result.duration:.2f}s")
        for seg in result.segments[:3]:
            print(f"    {seg.start_time:.3f}-{seg.end_time:.3f}: {seg.chord}")
        if len(result.segments) > 3:
            print(f"    ... and {len(result.segments) - 3} more")
        return True
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("autochord Tool Environment Test")
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
        status = "PASS" if passed else "FAIL"
        print(f"  {'✓' if passed else '✗'} {status}: {name}")

    all_passed = all(passed for _, passed in results)
    print()
    print("All tests passed!" if all_passed else "Some tests failed.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
