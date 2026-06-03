#!/usr/bin/env python3
"""
Quick environment test for Omni Captioner tool.
Verifies that all required dependencies are properly installed.
"""

import sys


def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import openai
        print(f"  ✓ openai {openai.__version__}")
    except ImportError as e:
        print(f"  ✗ openai: {e}")
        return False
    
    try:
        import soundfile
        print(f"  ✓ soundfile {soundfile.__version__}")
    except ImportError as e:
        print(f"  ✗ soundfile: {e}")
        return False
    
    try:
        import numpy
        print(f"  ✓ numpy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ numpy: {e}")
        return False
    
    return True


def test_model_load():
    """Test that the model can be loaded."""
    print("\nTesting model loading...")
    
    try:
        from model import OmniCaptionerModel
        print("  ✓ OmniCaptionerModel imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import OmniCaptionerModel: {e}")
        return False


def test_api_key():
    """Test if API key is set."""
    print("\nTesting API configuration...")
    
    import os
    api_key = os.environ.get("DASHSCOPE_API_KEY", "")
    
    if api_key:
        print("  ✓ DASHSCOPE_API_KEY is set")
    else:
        print("  ⚠ DASHSCOPE_API_KEY not set (required for actual API calls)")
        print("    Set it with: export DASHSCOPE_API_KEY='your-key'")
    
    return True  # Not a failure - just a warning


def main():
    """Run all tests."""
    print("=" * 60)
    print("Omni Captioner Environment Test")
    print("=" * 60)
    print()
    
    results = [
        ("Imports", test_imports()),
        ("Model Loading", test_model_load()),
        ("API Configuration", test_api_key()),
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
    if all_passed:
        print("✓ All tests passed!")
        print()
        print("Note: This is an API-based tool (no local model).")
        print("Ensure DASHSCOPE_API_KEY is set before use.")
        return 0
    else:
        print("✗ Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
