#!/usr/bin/env python3
"""Quick environment test for WhisperX tool."""

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
        import torchaudio
        print(f"  ✓ torchaudio {torchaudio.__version__}")
    except ImportError as e:
        print(f"  ✗ torchaudio: {e}")
        return False
    
    try:
        import whisperx
        print(f"  ✓ whisperx available")
    except ImportError as e:
        print(f"  ✗ whisperx: {e}")
        return False
    
    return True


def test_model_wrapper():
    """Test that the model wrapper can be instantiated and healthchecked."""
    print("\nTesting model wrapper...")
    
    try:
        from model import ModelWrapper
        
        wrapper = ModelWrapper()
        health = wrapper.healthcheck()
        
        print(f"  ✓ ModelWrapper instantiated")
        print(f"  ✓ Healthcheck: {health}")
        return True
    except Exception as e:
        print(f"  ✗ ModelWrapper failed: {e}")
        return False


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


def test_inference():
    """Test minimal inference on fixture if available."""
    print("\nTesting inference...")
    
    # Look for test fixtures (override with AUDIO_AGENT_TEST_WAV).
    import os
    fixture_paths = []
    if os.environ.get("AUDIO_AGENT_TEST_WAV"):
        fixture_paths.append(Path(os.environ["AUDIO_AGENT_TEST_WAV"]))
    fixture_paths.append(
        Path(__file__).resolve().parents[4] / "tests" / "fixtures" / "shared" / "asr" / "en_16k.wav"
    )
    
    fixture_path = None
    for fp in fixture_paths:
        if fp.exists():
            fixture_path = fp
            break
    
    if fixture_path is None:
        print(f"  ⚠ No test fixture found, skipping inference test")
        return True
    
    print(f"  Using fixture: {fixture_path}")
    
    try:
        from model import ModelWrapper
        
        wrapper = ModelWrapper()
        print("  Loading model (this may take a while on first run)...")
        wrapper.load()
        print("  Model loaded, running inference...")
        
        result = wrapper.predict(str(fixture_path))
        
        segments = result.get("segments", [])
        language = result.get("language")
        
        print(f"  ✓ Inference passed")
        print(f"  ✓ Detected language: {language}")
        print(f"  ✓ Segments: {len(segments)}")
        
        if segments:
            first_seg = segments[0]
            print(f"  ✓ First segment: {first_seg.get('text', '')[:50]}...")
        
        return True
    except Exception as e:
        print(f"  ✗ Inference failed: {e}")
        return False


def test_diarization_tool():
    """Test that diarization tool is available."""
    print("\nTesting diarization tool availability...")
    
    try:
        from model import ModelWrapper
        
        wrapper = ModelWrapper()
        
        # Check that the method exists
        if hasattr(wrapper, 'predict_with_diarization'):
            print("  ✓ predict_with_diarization method exists")
        else:
            print("  ✗ predict_with_diarization method not found")
            return False
        
        # Verify the method signature accepts the right parameters
        import inspect
        sig = inspect.signature(wrapper.predict_with_diarization)
        params = list(sig.parameters.keys())
        
        expected_params = ['input_data', 'language', 'min_speakers', 'max_speakers']
        for param in expected_params:
            if param in params:
                print(f"  ✓ Parameter '{param}' exists")
            else:
                print(f"  ✗ Parameter '{param}' missing")
                return False
        
        return True
    except Exception as e:
        print(f"  ✗ Diarization tool test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("WhisperX Tool Environment Test")
    print("=" * 60)
    print()
    
    results = [
        ("Imports", test_imports()),
        ("PyTorch CUDA", test_torch_cuda()),
        ("Model Wrapper", test_model_wrapper()),
        ("Diarization Tool", test_diarization_tool()),
    ]
    
    # Only run inference test if basic tests pass
    if all(passed for _, passed in results):
        results.append(("Inference", test_inference()))
    else:
        print("\n⚠ Skipping inference test due to earlier failures")
    
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
