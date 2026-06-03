#!/usr/bin/env python3
"""Quick environment test for ffmpeg tool."""

import sys
from pathlib import Path


def test_imports():
    """Test that model module can be imported."""
    print("Testing imports...")
    try:
        from model import FFmpegWrapper
        print("  ✓ FFmpegWrapper imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ FFmpegWrapper: {e}")
        return False


def test_ffmpeg_available():
    """Test that ffmpeg and ffprobe are available."""
    print("\nTesting FFmpeg availability...")
    try:
        import subprocess
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        print("  ✓ ffmpeg available")
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
        print("  ✓ ffprobe available")
        return True
    except Exception as e:
        print(f"  ✗ FFmpeg tools not available: {e}")
        return False


def test_model_wrapper():
    """Test that the model wrapper can be healthchecked."""
    print("\nTesting model wrapper...")
    try:
        from model import FFmpegWrapper
        wrapper = FFmpegWrapper()
        health = wrapper.healthcheck()
        print(f"  ✓ FFmpegWrapper healthcheck: {health}")
        return True
    except Exception as e:
        print(f"  ✗ FFmpegWrapper failed: {e}")
        return False


def test_inference():
    """Test minimal inference on fixture."""
    print("\nTesting inference...")
    fixture_path = Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "shared" / "mir" / "rhythm_22k_15s.wav"
    if not fixture_path.exists():
        print(f"  ⚠ Fixture not found at {fixture_path}, skipping inference test")
        return True
    try:
        from model import FFmpegWrapper
        wrapper = FFmpegWrapper()
        output_path = "/tmp/ffmpeg_test_output.wav"
        result = wrapper.predict(
            str(fixture_path),
            output_path,
            start_time=0,
            duration=3,
            sample_rate=16000,
            channels=1,
        )
        print(f"  ✓ Inference passed: output={result.output_path}, sr={result.sample_rate}, ch={result.channels}")
        return True
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("FFmpeg Tool Environment Test")
    print("=" * 60)
    print()

    results = [
        ("Imports", test_imports()),
        ("FFmpeg Available", test_ffmpeg_available()),
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
