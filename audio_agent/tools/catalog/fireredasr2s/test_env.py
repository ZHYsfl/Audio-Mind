#!/usr/bin/env python3
"""Test environment for FireRedASR2S tool."""

from __future__ import annotations

import os
import sys


def test_import() -> bool:
    """Test that we can import the FireRedASR2S modules."""
    print("Testing import...")
    try:
        from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
        print("  ✓ FireRedAsr2 imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import FireRedAsr2: {e}")
        return False


def test_model_load() -> bool:
    """Test that we can load the model."""
    print("Testing model load...")
    try:
        from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
        
        # Default resolves under $AUDIO_AGENT_MODELS_DIR (or <repo>/models when unset).
        from pathlib import Path as _P
        _repo_root = _P(__file__).resolve().parents[4]
        _models_dir = os.environ.get("AUDIO_AGENT_MODELS_DIR") or str(_repo_root / "models")
        model_path = os.environ.get(
            "MODEL_PATH",
            os.path.join(_models_dir, "FireRedASR-AED-L"),
        )
        
        if not os.path.exists(model_path):
            print(f"  ⚠ Model path does not exist: {model_path}")
            print("  Skipping load test (model not downloaded yet)")
            return True  # Not a failure, just not ready
        
        config = FireRedAsr2Config(
            use_gpu=False,  # Use CPU for testing
            use_half=False,
            return_timestamp=True,
        )
        
        model = FireRedAsr2.from_pretrained("aed", model_path, config)
        print(f"  ✓ Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_wrapper() -> bool:
    """Test the ModelWrapper class."""
    print("Testing ModelWrapper...")
    try:
        from model import ModelWrapper
        
        wrapper = ModelWrapper(config={"device": "cpu"})
        print("  ✓ ModelWrapper instantiated successfully")
        
        # Test healthcheck before load
        health = wrapper.healthcheck()
        print(f"  ✓ Healthcheck (before load): {health['status']}")
        
        return True
    except Exception as e:
        print(f"  ✗ Failed to test ModelWrapper: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("FireRedASR2S Tool Environment Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Test 1: Import
    results.append(("Import", test_import()))
    print()
    
    # Test 2: Model Wrapper
    results.append(("ModelWrapper", test_model_wrapper()))
    print()
    
    # Test 3: Model Load
    results.append(("Model Load", test_model_load()))
    print()
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("All tests passed!")
        return 0
    else:
        print("Some tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
