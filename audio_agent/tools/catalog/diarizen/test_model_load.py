#!/usr/bin/env python3
"""
Test script to verify DiariZen model can be loaded from local path.
"""

import os
import sys


def test_model_loading():
    """Test loading the DiariZen pipeline from local model path."""
    model_path = os.environ.get(
        "MODEL_PATH",
        "BUT-FIT/diarizen-wavlm-large-s80-md"
    )
    
    print(f"Testing DiariZen model loading...")
    print(f"Model path: {model_path}")
    print()
    
    # Check if path exists
    if not os.path.exists(model_path):
        print(f"✗ Model path does not exist: {model_path}")
        return False
    
    print(f"✓ Model path exists")
    print(f"  Contents: {os.listdir(model_path)}")
    print()
    
    # Try importing diarizen
    try:
        from diarizen.pipelines.inference import DiariZenPipeline
        print("✓ DiariZenPipeline imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import DiariZenPipeline: {e}")
        return False
    
    # Try loading the pipeline
    print()
    print("Loading pipeline (this may take a minute)...")
    try:
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        
        # Check if local path
        if os.path.isdir(model_path):
            print("  Using local model path...")
            # Download embedding model from HF Hub
            embedding_model = hf_hub_download(
                repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
                filename="pytorch_model.bin"
            )
            # Direct instantiation
            pipeline = DiariZenPipeline(
                diarizen_hub=Path(model_path).expanduser().absolute(),
                embedding_model=embedding_model
            )
        else:
            print("  Using HuggingFace Hub...")
            pipeline = DiariZenPipeline.from_pretrained(model_path)
        
        print("✓ Pipeline loaded successfully!")
        print(f"  Pipeline type: {type(pipeline)}")
        return True
    except Exception as e:
        print(f"✗ Failed to load pipeline: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_diarization():
    """Test diarization with a sample audio file."""
    # Find a sample audio file. The default location is the DiariZen submodule's
    # example clip cloned by setup.sh; override with AUDIO_AGENT_TEST_WAV.
    from pathlib import Path as _P
    _tool_dir = _P(__file__).resolve().parent
    sample_audio = os.environ.get("AUDIO_AGENT_TEST_WAV") or str(
        _tool_dir / "diarizen_src" / "example" / "EN2002a_30s.wav"
    )

    if not os.path.exists(sample_audio):
        print(f"⚠ Sample audio not found: {sample_audio}")
        print("  Skipping diarization test (set AUDIO_AGENT_TEST_WAV to override)")
        return True
    
    print()
    print(f"Testing diarization on: {sample_audio}")
    
    model_path = os.environ.get(
        "MODEL_PATH",
        "BUT-FIT/diarizen-wavlm-large-s80-md"
    )
    
    try:
        from diarizen.pipelines.inference import DiariZenPipeline
        from pathlib import Path
        from huggingface_hub import hf_hub_download
        
        # Load pipeline
        if os.path.isdir(model_path):
            embedding_model = hf_hub_download(
                repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
                filename="pytorch_model.bin"
            )
            pipeline = DiariZenPipeline(
                diarizen_hub=Path(model_path).expanduser().absolute(),
                embedding_model=embedding_model
            )
        else:
            pipeline = DiariZenPipeline.from_pretrained(model_path)
        
        print("Running diarization...")
        diar_results = pipeline(sample_audio)
        
        # Count segments and speakers
        segments = list(diar_results.itertracks(yield_label=True))
        unique_speakers = sorted(set(str(label) for _, _, label in segments))
        
        print(f"✓ Diarization successful!")
        print(f"  Segments: {len(segments)}")
        print(f"  Speakers: {len(unique_speakers)} ({', '.join(unique_speakers)})")
        
        # Show first few segments
        print()
        print("First 3 segments:")
        for turn, _, speaker in segments[:3]:
            print(f"  [{turn.start:.2f}s - {turn.end:.2f}s] {speaker}")
        
        return True
        
    except Exception as e:
        print(f"✗ Diarization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("DiariZen Model Loading Test")
    print("=" * 60)
    print()
    
    # Test 1: Model loading
    load_success = test_model_loading()
    
    # Test 2: Diarization (only if loading succeeded)
    diarize_success = False
    if load_success:
        diarize_success = test_diarization()
    
    print()
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"  Model Loading: {'✓ PASS' if load_success else '✗ FAIL'}")
    print(f"  Diarization: {'✓ PASS' if diarize_success else '✗ FAIL'}")
    
    if load_success and diarize_success:
        print()
        print("✓ All tests passed!")
        return 0
    else:
        print()
        print("✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
