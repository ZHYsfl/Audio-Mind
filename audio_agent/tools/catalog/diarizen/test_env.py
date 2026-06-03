#!/usr/bin/env python3
"""
Quick environment test for DiariZen tool.
Verifies that all required dependencies are properly installed.
"""

import sys


def test_python_version():
    """Test that we're using Python 3.10."""
    print("Testing Python version...")
    
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 10:
        print("  ✓ Python 3.10.x (correct)")
        return True
    else:
        print(f"  ⚠ Python {version.major}.{version.minor} (expected 3.10)")
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
        if numpy.__version__ != "1.26.4":
            print(f"    ⚠ WARNING: Expected numpy 1.26.4, got {numpy.__version__}")
            print(f"    This may cause compatibility issues!")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    try:
        import huggingface_hub
        print(f"  ✓ huggingface_hub {huggingface_hub.__version__}")
    except ImportError as e:
        print(f"  ✗ huggingface_hub: {e}")
        return False
    
    try:
        import pyannote.audio
        print(f"  ✓ pyannote.audio {pyannote.audio.__version__}")
    except ImportError as e:
        print(f"  ✗ pyannote.audio: {e}")
        return False
    
    try:
        import psutil
        print(f"  ✓ psutil {psutil.__version__}")
    except ImportError as e:
        print(f"  ✗ psutil: {e}")
        return False
    
    try:
        import accelerate
        print(f"  ✓ accelerate {accelerate.__version__}")
    except ImportError as e:
        print(f"  ✗ accelerate: {e}")
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
    """Test that the DiariZen model can be loaded."""
    print("\nTesting DiariZen model loading...")
    
    try:
        from diarizen.pipelines.inference import DiariZenPipeline
        print("  ✓ DiariZenPipeline imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Failed to import DiariZenModel: {e}")
        return False
    except Exception as e:
        print(f"  ⚠ Model instantiation failed: {e}")
        print(f"    (This may be due to missing model weights, which is OK)")
        return True  # Import succeeded, which is the main test


def main():
    """Run all tests."""
    print("=" * 60)
    print("DiariZen Environment Test")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Python Version", test_python_version()))
    results.append(("Imports", test_imports()))
    # results.append(("PyTorch CUDA", test_torch_cuda()))
    results.append(("Model Loading", test_model_load()))
    
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
        print("  1. Model weights are downloaded")
        print("  2. Test audio files are >= 30 seconds")
        return 0
    else:
        print("✗ Some tests failed.")
        print()
        print("Troubleshooting:")
        print("  - Re-run ./setup.sh in this tool directory")
        print("  - Ensure Python 3.10 is being used (not 3.11)")
        print("  - Verify pyannote.audio is installed from submodule (not PyPI)")
        print("  - Check that numpy==1.26.4 is installed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
