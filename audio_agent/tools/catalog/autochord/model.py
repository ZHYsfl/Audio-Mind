"""
autochord Wrapper for Audio Agent Framework.

Responsibilities:
- Wrap autochord.recognize() with a unified ModelWrapper interface
- Return structured chord recognition results with time-localized segments
- Manage lazy model loading (autochord loads on first use, not import)

Entry Points:
- ModelWrapper: Main wrapper class
- ChordSegment: Single chord segment dataclass
- ChordRecognitionResult: Full recognition result

Dependencies:
- autochord>=0.1.4
- soundfile>=0.12.1

Example:
    model = ModelWrapper()
    result = model.predict("audio.wav")
    for seg in result.segments:
        print(f"{seg.start_time:.3f}-{seg.end_time:.3f}: {seg.chord}")
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any

try:
    import soundfile as sf
except ImportError:
    sf = None


@dataclass
class ChordSegment:
    """A single chord segment with timing and label."""
    start_time: float
    end_time: float
    chord: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChordRecognitionResult:
    """Result of chord recognition on an audio file."""
    segments: list[ChordSegment]
    duration: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "duration": self.duration,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ModelWrapper:
    """
    Wrapper for autochord automatic chord recognition.

    autochord uses a BiLSTM-CRF model trained on major/minor chord vocabulary
    (25 classes: N + 12 major + 12 minor). It downloads the model on first import.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._model = None
        self._model_loaded = False

    def _load_model(self) -> Any:
        """Lazy-load autochord module, suppressing stdout to avoid MCP stream corruption."""
        if self._model is not None:
            return self._model

        # Suppress stdout during autochord import (it prints initialization messages
        # and download progress bars that would corrupt the JSON-RPC MCP stream)
        old_stdout = sys.stdout
        sys.stdout = sys.stderr  # Redirect prints to stderr
        try:
            import autochord
            self._model = autochord
            self._model_loaded = True
        finally:
            sys.stdout = old_stdout

        return self._model

    def load(self) -> None:
        """
        Ensure autochord module and model are loaded.
        autochord initializes on import; this method triggers lazy loading.
        """
        _ = self._load_model()

    def healthcheck(self) -> dict[str, Any]:
        """Quick check if autochord runtime is available."""
        try:
            _ = self._load_model()
            return {
                "status": "ready",
                "message": "autochord available",
                "model_loaded": True,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "model_loaded": False,
            }

    def predict(self, audio_path: str) -> ChordRecognitionResult:
        """
        Run chord recognition on the given audio file.

        Args:
            audio_path: Path to the audio file.

        Returns:
            ChordRecognitionResult with time-localized chord segments.
        """
        return self.recognize_chords(audio_path)

    def recognize_chords(self, audio_path: str) -> ChordRecognitionResult:
        """
        Recognize chords in an audio file using autochord.

        Args:
            audio_path: Path to the audio file.

        Returns:
            ChordRecognitionResult with segments.
        """
        autochord = self._load_model()

        audio_path = str(audio_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Get audio duration
        duration = self._get_duration(audio_path)

        # Run autochord recognition
        # Suppress stdout (Keras/TensorFlow progress bars) to avoid corrupting
        # the MCP JSON-RPC stream, which communicates over stdout.
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            raw_results = autochord.recognize(audio_path)
        finally:
            sys.stdout = old_stdout

        segments = [
            ChordSegment(start_time=float(st), end_time=float(ed), chord=str(ch))
            for st, ed, ch in raw_results
        ]

        return ChordRecognitionResult(segments=segments, duration=duration)

    def _get_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        if sf is not None:
            info = sf.info(audio_path)
            return info.duration
        # Fallback via librosa (autochord already depends on librosa)
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return float(len(y)) / sr
        except Exception:
            # Last resort: use last segment end time
            return 0.0

    def predict_with_options(
        self,
        audio_path: str,
        output_lab_path: str | None = None,
    ) -> ChordRecognitionResult:
        """
        Recognize chords with optional LAB file output.

        Args:
            audio_path: Path to the audio file.
            output_lab_path: Optional path to write MIREX-format .lab file.

        Returns:
            ChordRecognitionResult with segments.
        """
        autochord = self._load_model()

        duration = self._get_duration(audio_path)

        # Suppress stdout (Keras/TensorFlow progress bars) to avoid corrupting
        # the MCP JSON-RPC stream, which communicates over stdout.
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            if output_lab_path:
                raw_results = autochord.recognize(audio_path, lab_fn=output_lab_path)
            else:
                raw_results = autochord.recognize(audio_path)
        finally:
            sys.stdout = old_stdout

        segments = [
            ChordSegment(start_time=float(st), end_time=float(ed), chord=str(ch))
            for st, ed, ch in raw_results
        ]

        return ChordRecognitionResult(segments=segments, duration=duration)
