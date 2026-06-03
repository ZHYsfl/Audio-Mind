#!/usr/bin/env python3
"""
Quick environment test for SortFormer diarization tool.
Verifies that all required dependencies are properly installed.
"""

import sys


def test_python_version():
    """Test that we're using Python 3.11."""
    print("Testing Python version...")

    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 11:
        print("  ✓ Python 3.11+ (correct)")
        return True
    else:
        print(f"  ⚠ Python {version.major}.{version.minor} (expected 3.11+)")
        return False


def test_imports():
    """Test that all required packages can be imported."""
    print("\nTesting imports...")

    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ torch: {e}")
        return False

    try:
        import torchaudio
        print(f"  ✓ torchaudio {torchaudio.__version__}")
    except ImportError as e:
        print(f"  ✗ torchaudio: {e}")
        return False

    try:
        import numpy
        print(f"  ✓ numpy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False

    try:
        import nemo
        print(f"  ✓ nemo {nemo.__version__}")
    except ImportError as e:
        print(f"  ✗ nemo: {e}")
        return False

    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
        print("  ✓ SortformerEncLabelModel imported successfully")
    except ImportError as e:
        print(f"  ✗ SortformerEncLabelModel: {e}")
        return False

    try:
        import modelscope
        print(f"  ✓ modelscope {modelscope.__version__}")
    except ImportError as e:
        print(f"  ✗ modelscope: {e}")
        return False

    try:
        import soundfile
        print(f"  ✓ soundfile {soundfile.__version__}")
    except ImportError as e:
        print(f"  ✗ soundfile: {e}")
        return False

    return True


def test_torch_cuda():
    """Test PyTorch CUDA availability."""
    print("\nTesting PyTorch CUDA...")

    import torch

    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
    else:
        print(f"  ⚠ CUDA not available (CPU only)")

    return True


def test_model_load():
    """Test that the SortFormer model class can be imported."""
    print("\nTesting SortFormer model class loading...")

    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
        print("  ✓ SortformerEncLabelModel class available")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import SortformerEncLabelModel: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("SortFormer Diarization Environment Test")
    print("=" * 60)
    print()

    results = []

    results.append(("Python Version", test_python_version()))
    results.append(("Imports", test_imports()))
    results.append(("PyTorch CUDA", test_torch_cuda()))
    results.append(("Model Class Loading", test_model_load()))

    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(passed for _, passed in results)

    print()
    if all_passed:
        print("✓ All tests passed!")
        print()
        print("Note: For full functionality, ensure:")
        print("  1. Model weights (.nemo file) are downloaded")
        print("  2. Test audio files are available")
        return 0
    else:
        print("✗ Some tests failed.")
        print()
        print("Troubleshooting:")
        print("  - Ensure ./setup.sh completed successfully")
        print("  - Check that PyTorch with CUDA was installed")
        print("  - Verify NeMo ASR toolkit is installed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
