"""WhisperX Wrapper for Audio Agent Framework.

Responsibilities:
- Load WhisperX model from HuggingFace/local path
- Provide ASR inference interface with word-level timestamps
- Manage device placement (CPU/GPU)
- Handle lazy loading

Entry Points:
- ModelWrapper: Main wrapper class
- WhisperXResult: Output type with segments and language

Dependencies:
- whisperx>=3.8.4
- torch>=2.0.0
- torchaudio>=2.0.0

Example:
    model = ModelWrapper(device='cpu')
    result = model.predict("audio.wav")
    print(result.segments)
"""

from __future__ import annotations

import io
import os
import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Any

# Suppress warnings that might pollute stdout (MCP protocol requires clean stdout)
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Redirect all warnings to stderr
warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: print(
    f"{category.__name__}: {message}", file=sys.stderr
)


def _capture_stdout_during_imports(import_func):
    """Capture and discard stdout pollution during imports.
    
    Some libraries (like pyannote) print warnings to stdout which
    corrupts the JSON-RPC protocol. This captures and discards them.
    """
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return import_func()
    finally:
        sys.stdout = old_stdout


class ModelLoadError(RuntimeError):
    """Raised when model loading fails."""
    pass


class InferenceError(RuntimeError):
    """Raised when inference fails."""
    pass


@dataclass
class WhisperXResult:
    """Result of WhisperX transcription.
    
    Attributes:
        segments: List of segment dictionaries with keys:
            - start: Start time in seconds
            - end: End time in seconds
            - text: Transcription text
            - words: List of word-level timestamps (if alignment enabled)
            - speaker: Speaker label (if diarization enabled)
        language: Detected language code
        speakers: List of unique speaker labels (if diarization enabled)
        num_speakers: Number of speakers detected (if diarization enabled)
    """
    segments: list[dict[str, Any]]
    language: str | None = None
    speakers: list[str] | None = None
    num_speakers: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


class ModelWrapper:
    """WhisperX model wrapper for Audio Agent Framework.
    
    This wrapper provides a unified interface for WhisperX ASR with:
    - Lazy model loading
    - Word-level timestamp alignment
    - Configurable device placement
    
    Attributes:
        model_arch: Whisper model size (tiny, base, small, medium, large, etc.)
        device: Compute device (cpu, cuda, auto)
        vad_method: VAD method for segmentation (pyannote, silero, None)
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize wrapper (does not load model yet).
        
        Args:
            config: Optional configuration dictionary with keys:
                - model_arch: Model size (default: from env or 'small')
                - device: Device to use (default: from env or 'cpu')
                - vad_method: VAD method (default: from env or 'pyannote')
                - hf_home: HuggingFace cache directory for Whisper models
        """
        self.config = config or {}
        self.model_arch = self.config.get("model_arch") or os.environ.get("MODEL_ARCH", "small")
        _device = self.config.get("device") or os.environ.get("DEVICE", "cpu")
        if _device == "auto":
            import torch
            _device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = _device
        self.vad_method = self.config.get("vad_method") or os.environ.get("VAD_METHOD", "pyannote")
        self.hf_home = self.config.get("hf_home") or os.environ.get("HF_HOME")
        self._model = None
        self._align_model = None
        self._metadata = None

    def load(self) -> None:
        """Load model weights into memory.
        
        This is called automatically on first predict() if not called explicitly.
        Sets up cache directories and loads both the Whisper model and alignment model.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        if self._model is not None:
            return

        # Set HuggingFace cache directory for model downloads
        if self.hf_home:
            os.environ["HF_HOME"] = self.hf_home

        try:
            # Capture stdout during imports to prevent JSON-RPC pollution
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                import whisperx
                import torch
                
                # Load Whisper model
                compute_type = "float16" if self.device == "cuda" else "int8"
                self._model = whisperx.load_model(
                    self.model_arch,
                    self.device,
                    compute_type=compute_type,
                    vad_method=self.vad_method if self.vad_method != "None" else None,
                )
                
                # Store metadata for alignment
                self._metadata = {"language": None}
            finally:
                sys.stdout = old_stdout
            
        except Exception as exc:
            raise ModelLoadError(f"Failed to load WhisperX model: {exc}") from exc

    def predict(self, input_data: str, language: str | None = None) -> dict[str, Any]:
        """Run inference on audio file.
        
        Args:
            input_data: Path to audio file
            language: Optional language code (e.g., 'en', 'fr'). If not provided
                     or set to 'auto', language will be auto-detected.
        
        Returns:
            Dictionary with segments and detected language
            
        Raises:
            InferenceError: If transcription fails
        """
        if self._model is None:
            self.load()
        
        # Convert 'auto' to None for WhisperX auto-detection
        if language == "auto":
            language = None

        try:
            # Capture stdout during imports to prevent JSON-RPC pollution
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                import whisperx
                import torch
                
                audio_path = input_data
                
                # Load audio
                audio = whisperx.load_audio(audio_path)
                
                # Transcribe
                result = self._model.transcribe(audio, language=language)
                detected_language = result.get("language")
                segments = result.get("segments", [])
                
                # Load alignment model for word-level timestamps
                if detected_language:
                    try:
                        align_model, align_metadata = whisperx.load_align_model(
                            language_code=detected_language,
                            device=self._model.device if hasattr(self._model, 'device') else self.device,
                        )
                        
                        # Align timestamps
                        aligned_result = whisperx.align(
                            segments,
                            align_model,
                            align_metadata,
                            audio,
                            self._model.device if hasattr(self._model, 'device') else self.device,
                            return_char_alignments=False,
                        )
                        segments = aligned_result.get("segments", segments)
                    except (ValueError, Exception) as align_err:
                        # If alignment fails, try falling back to English alignment
                        align_error_msg = str(align_err)
                        if detected_language != "en" and "No default align-model" in align_error_msg:
                            print(f"Warning: No alignment model for '{detected_language}', falling back to English alignment.", file=sys.stderr)
                            try:
                                align_model, align_metadata = whisperx.load_align_model(
                                    language_code="en",
                                    device=self._model.device if hasattr(self._model, 'device') else self.device,
                                )
                                aligned_result = whisperx.align(
                                    segments,
                                    align_model,
                                    align_metadata,
                                    audio,
                                    self._model.device if hasattr(self._model, 'device') else self.device,
                                    return_char_alignments=False,
                                )
                                segments = aligned_result.get("segments", segments)
                                # Keep the original detected language for output, just use English alignment
                            except Exception as fallback_err:
                                print(f"Warning: English alignment fallback also failed: {fallback_err}. Skipping word-level alignment.", file=sys.stderr)
                        else:
                            print(f"Warning: Alignment failed: {align_error_msg}. Skipping word-level alignment.", file=sys.stderr)
                        # Continue with segment-level timestamps only
                
                # Format segments
                formatted_segments = []
                for seg in segments:
                    formatted_seg = {
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text", "").strip(),
                    }
                    # Include word-level timestamps if available
                    if "words" in seg:
                        formatted_seg["words"] = [
                            {
                                "word": w.get("word", "").strip(),
                                "start": w.get("start"),
                                "end": w.get("end"),
                                "score": w.get("score"),
                            }
                            for w in seg["words"]
                            if w.get("word")  # Filter empty words
                        ]
                    formatted_segments.append(formatted_seg)
                
                wrapped = WhisperXResult(
                    segments=formatted_segments,
                    language=detected_language,
                )
                return wrapped.to_dict()
            finally:
                sys.stdout = old_stdout
            
        except Exception as exc:
            raise InferenceError(f"WhisperX inference failed: {exc}") from exc

    def predict_with_diarization(
        self, 
        input_data: str, 
        language: str | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> dict[str, Any]:
        """Run inference with speaker diarization on audio file.
        
        This method performs transcription, alignment, and speaker diarization
        to identify "who spoke when" in the audio.
        
        Args:
            input_data: Path to audio file
            language: Optional language code (e.g., 'en', 'fr'). If not provided
                     or set to 'auto', language will be auto-detected.
            min_speakers: Optional minimum number of speakers
            max_speakers: Optional maximum number of speakers
        
        Returns:
            Dictionary with segments (including speaker labels), detected language,
            speakers list, and num_speakers
            
        Raises:
            InferenceError: If transcription or diarization fails
        """
        if self._model is None:
            self.load()
        
        # Convert 'auto' to None for WhisperX auto-detection
        if language == "auto":
            language = None

        try:
            # Capture stdout during imports to prevent JSON-RPC pollution
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()
            
            try:
                import whisperx
                from whisperx.diarize import DiarizationPipeline
                import torch
                
                audio_path = input_data
                
                # Load audio
                audio = whisperx.load_audio(audio_path)
                
                # Transcribe
                result = self._model.transcribe(audio, language=language)
                detected_language = result.get("language")
                segments = result.get("segments", [])
                
                # Load alignment model for word-level timestamps
                if detected_language:
                    try:
                        align_model, align_metadata = whisperx.load_align_model(
                            language_code=detected_language,
                            device=self._model.device if hasattr(self._model, 'device') else self.device,
                        )
                        
                        # Align timestamps
                        aligned_result = whisperx.align(
                            segments,
                            align_model,
                            align_metadata,
                            audio,
                            self._model.device if hasattr(self._model, 'device') else self.device,
                            return_char_alignments=False,
                        )
                        segments = aligned_result.get("segments", segments)
                    except (ValueError, Exception) as align_err:
                        # If alignment fails, try falling back to English alignment
                        align_error_msg = str(align_err)
                        if detected_language != "en" and "No default align-model" in align_error_msg:
                            print(f"Warning: No alignment model for '{detected_language}', falling back to English alignment.", file=sys.stderr)
                            try:
                                align_model, align_metadata = whisperx.load_align_model(
                                    language_code="en",
                                    device=self._model.device if hasattr(self._model, 'device') else self.device,
                                )
                                aligned_result = whisperx.align(
                                    segments,
                                    align_model,
                                    align_metadata,
                                    audio,
                                    self._model.device if hasattr(self._model, 'device') else self.device,
                                    return_char_alignments=False,
                                )
                                segments = aligned_result.get("segments", segments)
                                # Keep the original detected language for output, just use English alignment
                            except Exception as fallback_err:
                                print(f"Warning: English alignment fallback also failed: {fallback_err}. Skipping word-level alignment.", file=sys.stderr)
                        else:
                            print(f"Warning: Alignment failed: {align_error_msg}. Skipping word-level alignment.", file=sys.stderr)
                        # Continue with segment-level timestamps only
                
                # Perform speaker diarization
                try:
                    # Load diarization model from local path (set via DIARIZATION_MODEL_PATH env var).
                    # Default resolves under $AUDIO_AGENT_MODELS_DIR/pyannote-speaker-diarization-3.1.
                    from pathlib import Path as _P
                    _repo_root = _P(__file__).resolve().parents[4]
                    _models_dir = os.environ.get("AUDIO_AGENT_MODELS_DIR") or str(_repo_root / "models")
                    diarize_model_path = os.environ.get(
                        "DIARIZATION_MODEL_PATH",
                        os.path.join(_models_dir, "pyannote-speaker-diarization-3.1"),
                    )
                    diarize_model = DiarizationPipeline(
                        model_name=diarize_model_path,
                        device=self._model.device if hasattr(self._model, 'device') else self.device,
                    )
                    
                    # Run diarization with optional min/max speakers
                    diarize_kwargs = {}
                    if min_speakers is not None:
                        diarize_kwargs["min_speakers"] = min_speakers
                    if max_speakers is not None:
                        diarize_kwargs["max_speakers"] = max_speakers
                    
                    diarize_segments = diarize_model(audio, **diarize_kwargs)
                    
                    # Assign speakers to words/segments
                    result = whisperx.assign_word_speakers(diarize_segments, {"segments": segments})
                    segments = result.get("segments", segments)
                    
                except Exception as diarize_exc:
                    raise InferenceError(f"Speaker diarization failed: {diarize_exc}") from diarize_exc
                
                # Format segments and extract speakers
                formatted_segments = []
                unique_speakers = set()
                
                for seg in segments:
                    formatted_seg = {
                        "start": seg.get("start"),
                        "end": seg.get("end"),
                        "text": seg.get("text", "").strip(),
                        "speaker": seg.get("speaker"),
                    }
                    # Include word-level timestamps if available
                    if "words" in seg:
                        formatted_seg["words"] = [
                            {
                                "word": w.get("word", "").strip(),
                                "start": w.get("start"),
                                "end": w.get("end"),
                                "score": w.get("score"),
                                "speaker": w.get("speaker"),
                            }
                            for w in seg["words"]
                            if w.get("word")  # Filter empty words
                        ]
                        # Collect speakers from words
                        for w in seg["words"]:
                            if w.get("speaker"):
                                unique_speakers.add(w["speaker"])
                    
                    # Collect speaker from segment
                    if seg.get("speaker"):
                        unique_speakers.add(seg["speaker"])
                    
                    formatted_segments.append(formatted_seg)
                
                speakers_list = sorted(list(unique_speakers))
                
                wrapped = WhisperXResult(
                    segments=formatted_segments,
                    language=detected_language,
                    speakers=speakers_list if speakers_list else None,
                    num_speakers=len(speakers_list) if speakers_list else None,
                )
                return wrapped.to_dict()
            finally:
                sys.stdout = old_stdout
            
        except InferenceError:
            raise
        except Exception as exc:
            raise InferenceError(f"WhisperX diarization failed: {exc}") from exc

    def healthcheck(self) -> dict[str, Any]:
        """Quick check if model is ready.
        
        Returns:
            Dictionary with status information:
            - status: "ready" | "loading" | "error"
            - message: Description of current state
            - model_loaded: Whether model is loaded
            - model_arch: Model architecture being used
            - device: Device being used
            - vad_method: VAD method being used
        """
        return {
            "status": "ready" if self._model is not None else "loading",
            "message": "Model loaded" if self._model is not None else "Model not loaded (will load on first use)",
            "model_loaded": self._model is not None,
            "model_arch": self.model_arch,
            "device": self.device,
            "vad_method": self.vad_method,
        }
