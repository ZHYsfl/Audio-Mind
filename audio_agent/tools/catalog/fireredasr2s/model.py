"""FireRedASR2S Wrapper for Audio Agent Framework.

Responsibilities:
- Load FireRedASR2S AED model from local path
- Provide ASR inference interface with word-level timestamps
- Manage device placement (CPU/GPU)
- Handle lazy loading

Entry Points:
- ModelWrapper: Main wrapper class
- TranscriptionResult: Output type with segments and timestamps

Dependencies:
- fireredasr2s (from GitHub)
- torch>=2.0.0
- torchaudio>=2.0.0

Example:
    model = ModelWrapper(device='cpu')
    result = model.predict("audio.wav")
    print(result.text)
    print(result.segments)
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import asdict, dataclass
from typing import Any

# Suppress warnings that might pollute stdout (MCP protocol requires clean stdout)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Redirect all warnings to stderr
warnings.showwarning = lambda message, category, filename, lineno, file=None, line=None: print(
    f"{category.__name__}: {message}", file=sys.stderr
)


class ModelLoadError(RuntimeError):
    """Raised when model loading fails."""
    pass


class InferenceError(RuntimeError):
    """Raised when inference fails."""
    pass


@dataclass
class TranscriptionResult:
    """Result of FireRedASR2S transcription.
    
    Attributes:
        text: Full transcription text
        segments: List of segment dictionaries with keys:
            - start: Start time in seconds
            - end: End time in seconds
            - text: Transcription text
            - confidence: Confidence score
        language: Detected language code (e.g., 'zh', 'en')
        confidence: Overall confidence score
    """
    text: str
    segments: list[dict[str, Any]]
    language: str | None = None
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


class ModelWrapper:
    """FireRedASR2S AED model wrapper for Audio Agent Framework.
    
    This wrapper provides a unified interface for FireRedASR2S-AED with:
    - Lazy model loading
    - Word-level timestamp support
    - Configurable device placement
    
    Attributes:
        model_path: Path to model weights
        device: Compute device (cpu, cuda, auto)
    """
    
    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize wrapper (does not load model yet).
        
        Args:
            config: Optional configuration dictionary with keys:
                - model_path: Model directory path (default from env or local models)
                - device: Device to use (default from env or 'auto')
        """
        self.config = config or {}
        from pathlib import Path as _P
        _repo_root = _P(__file__).resolve().parents[4]
        _models_dir = os.environ.get("AUDIO_AGENT_MODELS_DIR") or str(_repo_root / "models")
        self.model_path = self.config.get("model_path") or os.environ.get(
            "MODEL_PATH",
            os.path.join(_models_dir, "FireRedASR-AED-L"),
        )
        self.device = self.config.get("device") or os.environ.get("DEVICE", "cpu")
        self._model = None
        self._config = None

    def load(self) -> None:
        """Load model weights into memory.
        
        This is called automatically on first predict() if not called explicitly.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        if self._model is not None:
            return

        try:
            from fireredasr2s.fireredasr2 import FireRedAsr2, FireRedAsr2Config
            import torch
            
            # Determine device
            if self.device == "auto":
                use_gpu = torch.cuda.is_available()
            else:
                use_gpu = self.device in ["cuda", "gpu"]
            
            # Create config for AED model
            self._config = FireRedAsr2Config(
                use_gpu=use_gpu,
                use_half=False,  # Use full precision for stability
                beam_size=3,
                nbest=1,
                decode_max_len=0,
                softmax_smoothing=1.25,
                aed_length_penalty=0.6,
                eos_penalty=1.0,
                return_timestamp=True,  # Enable word-level timestamps
            )
            
            # Load the AED model
            self._model = FireRedAsr2.from_pretrained(
                "aed",
                self.model_path,
                self._config
            )
            
        except Exception as exc:
            raise ModelLoadError(f"Failed to load FireRedASR2S model: {exc}") from exc

    def predict(self, input_data: str | list[str]) -> dict[str, Any]:
        """Run inference on audio file(s).
        
        Args:
            input_data: Path to audio file, or list of paths for batch processing
        
        Returns:
            Dictionary with transcription text, segments, and metadata
            
        Raises:
            InferenceError: If transcription fails
        """
        if self._model is None:
            self.load()

        try:
            # Handle single file or batch
            if isinstance(input_data, str):
                batch_wav_path = [input_data]
                batch_uttid = ["utt_0"]
                single_input = True
            else:
                batch_wav_path = input_data
                batch_uttid = [f"utt_{i}" for i in range(len(input_data))]
                single_input = False
            
            # Run transcription
            results = self._model.transcribe(batch_uttid, batch_wav_path)
            
            # Format results
            if single_input:
                result = results[0] if results else {"text": "", "timestamp": []}
                return self._format_result(result)
            else:
                return {"results": [self._format_result(r) for r in results]}
                
        except Exception as exc:
            raise InferenceError(f"FireRedASR2S inference failed: {exc}") from exc

    def _format_result(self, result: dict[str, Any]) -> dict[str, Any]:
        """Format FireRedASR2S result to standard format.
        
        Args:
            result: Raw result from FireRedAsr2.transcribe()
        
        Returns:
            Formatted TranscriptionResult as dict
        """
        text = result.get("text", "")
        timestamps = result.get("timestamp", [])
        confidence = result.get("confidence")
        
        # Build segments from timestamps
        segments = []
        for word, start_sec, end_sec in timestamps:
            segments.append({
                "text": word,
                "start": start_sec,
                "end": end_sec,
                "confidence": confidence,
            })
        
        # Detect language from text (simple heuristic)
        language = self._detect_language(text)
        
        wrapped = TranscriptionResult(
            text=text,
            segments=segments,
            language=language,
            confidence=confidence,
        )
        return wrapped.to_dict()

    def _detect_language(self, text: str) -> str | None:
        """Detect language from text.
        
        This is a simple heuristic. FireRedASR2S primarily supports:
        - Chinese (Mandarin and dialects)
        - English
        - Code-switching
        
        Args:
            text: Transcription text
        
        Returns:
            Language code or None
        """
        if not text:
            return None
        
        # Check for Chinese characters
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
        # Check for English characters
        has_english = any(c.isalpha() and c.isascii() for c in text)
        
        if has_chinese and has_english:
            return "zh-en"  # Code-switching
        elif has_chinese:
            return "zh"
        elif has_english:
            return "en"
        else:
            return "unknown"

    def healthcheck(self) -> dict[str, Any]:
        """Quick check if model is ready.
        
        Returns:
            Dictionary with status information:
            - status: "ready" | "loading" | "error"
            - message: Description of current state
            - model_loaded: Whether model is loaded
            - model_path: Path being used
            - device: Device being used
        """
        return {
            "status": "ready" if self._model is not None else "loading",
            "message": "Model loaded" if self._model is not None else "Model not loaded (will load on first use)",
            "model_loaded": self._model is not None,
            "model_path": self.model_path,
            "device": self.device,
        }
