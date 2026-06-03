"""
lv-chordia Wrapper for Audio Agent Framework.

Responsibilities:
- Wrap lv_chordia.chord_recognition with a unified ModelWrapper interface
- Return structured chord recognition results with time-localized segments
- Manage lazy model loading (lv-chordia loads on first use, not import)
- Support multiple chord dictionaries: submission, ismir2017, full

Entry Points:
- ModelWrapper: Main wrapper class
- ChordSegment: Single chord segment dataclass
- ChordRecognitionResult: Full recognition result

Dependencies:
- lv-chordia>=1.0.0
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
from dataclasses import asdict, dataclass
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
    Wrapper for lv-chordia large-vocabulary chord recognition.

    lv-chordia implements the ISMIR 2019 paper "Large-Vocabulary Chord
    Transcription via Chord Structure Decomposition". It uses an ensemble
    of 5 deep CNNs with CQT features and HMM decoding.

    Chord dictionaries:
    - "submission" (~170 chords, default): Best balance for general use
    - "ismir2017" (~25 chords): MIREX/ISMIR2017 standard vocabulary
    - "full" (~600+ chords): Complete MARL dataset vocabulary for jazz
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._model = None
        self._model_loaded = False

    def _load_model(self) -> Any:
        """Lazy-load lv_chordia module, suppressing stdout to avoid MCP stream corruption."""
        if self._model is not None:
            return self._model

        # Force CPU mode: lv-chordia sets use_gpu = torch.cuda.device_count() > 0.
        # When the NVIDIA driver is incompatible with PyTorch, this still returns > 0
        # but .cuda() fails. We monkey-patch device_count to 0 so the model runs on CPU.
        import torch
        torch.cuda.device_count = lambda: 0  # type: ignore[method-assign]

        # Suppress stdout during lv_chordia import (it may print initialization
        # messages that would corrupt the JSON-RPC MCP stream)
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            from lv_chordia.chord_recognition import chord_recognition
            self._model = chord_recognition
            self._model_loaded = True
        finally:
            sys.stdout = old_stdout

        return self._model

    def load(self) -> None:
        """
        Ensure lv_chordia module and model are loaded.
        lv_chordia initializes on import; this method triggers lazy loading.
        """
        _ = self._load_model()

    def healthcheck(self) -> dict[str, Any]:
        """Quick check if lv-chordia runtime is available."""
        try:
            _ = self._load_model()
            return {
                "status": "ready",
                "message": "lv-chordia available",
                "model_loaded": True,
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "model_loaded": False,
            }

    def predict(self, audio_path: str, chord_dict_name: str = "submission") -> ChordRecognitionResult:
        """
        Run chord recognition on the given audio file.

        Args:
            audio_path: Path to the audio file.
            chord_dict_name: Chord dictionary to use ("submission", "ismir2017", "full").

        Returns:
            ChordRecognitionResult with time-localized chord segments.
        """
        return self.recognize_chords(audio_path, chord_dict_name=chord_dict_name)

    def recognize_chords(
        self,
        audio_path: str,
        chord_dict_name: str = "submission",
    ) -> ChordRecognitionResult:
        """
        Recognize chords in an audio file using lv-chordia.

        Args:
            audio_path: Path to the audio file.
            chord_dict_name: Chord dictionary ("submission", "ismir2017", "full").

        Returns:
            ChordRecognitionResult with segments.
        """
        chord_recognition = self._load_model()

        audio_path = str(audio_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Validate chord dictionary
        valid_dicts = {"submission", "ismir2017", "full"}
        if chord_dict_name not in valid_dicts:
            raise ValueError(
                f"Invalid chord_dict_name: {chord_dict_name!r}. "
                f"Must be one of: {', '.join(sorted(valid_dicts))}"
            )

        # Get audio duration
        duration = self._get_duration(audio_path)

        # Run lv-chordia recognition
        # Suppress stdout to avoid corrupting the MCP JSON-RPC stream
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            raw_results = chord_recognition(audio_path, chord_dict_name=chord_dict_name)
        finally:
            sys.stdout = old_stdout

        # raw_results is a list of dicts: [{"start_time": ..., "end_time": ..., "chord": ...}, ...]
        segments = [
            ChordSegment(
                start_time=float(seg.get("start_time", 0.0)),
                end_time=float(seg.get("end_time", 0.0)),
                chord=str(seg.get("chord", "N")),
            )
            for seg in raw_results
        ]

        return ChordRecognitionResult(segments=segments, duration=duration)

    def _get_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        if sf is not None:
            try:
                info = sf.info(audio_path)
                return info.duration
            except Exception:
                pass
        # Fallback via librosa
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            return float(len(y)) / sr
        except Exception:
            # Last resort
            return 0.0
