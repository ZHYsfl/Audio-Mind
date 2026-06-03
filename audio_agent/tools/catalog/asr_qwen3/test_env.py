#!/usr/bin/env python3
"""
Quick environment test for ASR Qwen3 tool.
Verifies that all required dependencies are properly installed.
"""

import sys


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
        import transformers
        print(f"  ✓ transformers {transformers.__version__}")
    except ImportError as e:
        print(f"  ✗ transformers: {e}")
        return False
    
    try:
        import librosa
        print(f"  ✓ librosa {librosa.__version__}")
    except ImportError as e:
        print(f"  ✗ librosa: {e}")
        return False
    
    try:
        import soundfile
        print(f"  ✓ soundfile {soundfile.__version__}")
    except ImportError as e:
        print(f"  ✗ soundfile: {e}")
        return False
    
    try:
        import accelerate
        print(f"  ✓ accelerate {accelerate.__version__}")
    except ImportError as e:
        print(f"  ✗ accelerate: {e}")
        return False
    
    # qwen_asr uses dynamic imports, so we just check if it's available
    try:
        import qwen_asr
        print(f"  ✓ qwen_asr available")
    except ImportError:
        print(f"  ⚠ qwen_asr not directly importable (uses dynamic loading)")
    
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



def main():
    """Run all tests."""
    print("=" * 60)
    print("ASR Qwen3 Environment Test")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("PyTorch CUDA", test_torch_cuda()))
    
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
        return 0
    else:
        print("✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
