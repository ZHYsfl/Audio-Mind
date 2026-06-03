"""
Tempo-CNN Wrapper for Audio Agent Framework.

Responsibilities:
- Wrap tempocnn.classifier.TempoClassifier with a unified ModelWrapper interface
- Estimate global musical tempo (BPM) from audio files
- Manage lazy model loading and suppress TF stdout to avoid MCP stream corruption
- Redirect model cache to shared models directory

Entry Points:
- ModelWrapper: Main wrapper class
- TempoResult: Output type with tempo, model_name, interpolate, audio_path

Dependencies:
- tempocnn==0.0.8
- soundfile>=0.12.1

Example:
    model = ModelWrapper()
    result = model.estimate_tempo("audio.wav")
    print(f"Tempo: {result.tempo} BPM")
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Shared models directory for tempocnn checkpoint caching.
# Resolved from $AUDIO_AGENT_MODELS_DIR (or repo-root/models) plus a `tempocnn`
# subdirectory. Override the whole path with $TEMPOCNN_MODELS_DIR.
def _resolve_shared_models_dir() -> Path:
    explicit = os.environ.get("TEMPOCNN_MODELS_DIR")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("AUDIO_AGENT_MODELS_DIR")
    if not base:
        # repo root = five parents up: model.py -> tempo_cnn/ -> catalog/ -> tools/ -> audio_agent/ -> <repo>
        base = str(Path(__file__).resolve().parents[4] / "models")
    return Path(base) / "tempocnn"


SHARED_MODELS_DIR = _resolve_shared_models_dir()


def _patch_tempocnn_cache() -> None:
    """
    Monkey-patch tempocnn's _extract_from_package to first look in the
    shared models directory before falling back to the default ~/.tempocnn cache.
    """
    try:
        import tempocnn.classifier as _classifier_module
    except ImportError:
        return

    # Only patch once
    if getattr(_classifier_module, "_audio_agent_patched", False):
        return

    _orig_extract = _classifier_module._extract_from_package
    _package_version = _classifier_module.package_version

    def _patched_extract(resource: str) -> str:
        # 1. Try shared models directory first
        shared_path = SHARED_MODELS_DIR / resource
        if shared_path.exists():
            return str(shared_path)

        # 2. Try to extract from the installed package (original behavior)
        try:
            data = _classifier_module.pkgutil.get_data("tempocnn", resource)
            if data is not None:
                # Write to shared models dir for future use
                shared_path.parent.mkdir(parents=True, exist_ok=True)
                with open(shared_path, "wb") as f:
                    f.write(data)
                return str(shared_path)
        except (FileNotFoundError, OSError):
            pass

        # 3. Fall back to original behavior (downloads from GitHub if needed)
        return _orig_extract(resource)

    _classifier_module._extract_from_package = _patched_extract
    _classifier_module._audio_agent_patched = True


@dataclass
class TempoResult:
    """Result of tempo estimation on an audio file."""

    tempo: float
    model_name: str
    interpolate: bool
    audio_path: str

    def __post_init__(self) -> None:
        if not isinstance(self.tempo, (int, float)):
            raise ValueError("tempo must be numeric")
        if self.tempo < 0:
            raise ValueError("tempo must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class DualTempoResult:
    """Result of dual-tempo (MIREX-style) estimation on an audio file."""

    tempo1: float
    tempo2: float
    salience: float
    model_name: str
    interpolate: bool
    audio_path: str
    note: str

    def __post_init__(self) -> None:
        for field_name in ("tempo1", "tempo2", "salience"):
            val = getattr(self, field_name)
            if not isinstance(val, (int, float)):
                raise ValueError(f"{field_name} must be numeric")
            if val < 0:
                raise ValueError(f"{field_name} must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class ModelWrapper:
    """Wrapper for Tempo-CNN musical tempo estimation."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.model_name = str(self.config.get("model_name", "fcn"))
        self._classifier = None
        self._model_loaded = False

    def _load_model(self) -> Any:
        """Lazy-load the TempoClassifier, suppressing stdout to avoid MCP stream corruption."""
        if self._classifier is not None:
            return self._classifier

        # Patch cache path before first use
        _patch_tempocnn_cache()

        # Suppress stdout during model load (TF/Keras prints progress bars)
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            from tempocnn.classifier import TempoClassifier

            self._classifier = TempoClassifier(self.model_name)
            self._model_loaded = True
        finally:
            sys.stdout = old_stdout

        return self._classifier

    def load(self) -> None:
        """Explicitly trigger model loading."""
        _ = self._load_model()

    def healthcheck(self) -> dict[str, Any]:
        """Quick check if tempo-cnn runtime is available."""
        try:
            _patch_tempocnn_cache()
            from tempocnn.classifier import TempoClassifier
            from tempocnn.feature import read_features

            return {
                "status": "ready",
                "message": f"tempocnn available (model: {self.model_name})",
                "model_loaded": self._model_loaded,
                "shared_models_dir": str(SHARED_MODELS_DIR),
                "shared_models_dir_exists": SHARED_MODELS_DIR.exists(),
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "model_loaded": False,
                "shared_models_dir": str(SHARED_MODELS_DIR),
                "shared_models_dir_exists": SHARED_MODELS_DIR.exists(),
            }

    def estimate_tempo(
        self,
        audio_path: str,
        model_name: str | None = None,
        interpolate: bool = False,
    ) -> TempoResult:
        """
        Estimate the global tempo of an audio file.

        Args:
            audio_path: Path to the audio file.
            model_name: Optional override for the model name (default: fcn).
            interpolate: If True, use quadratic interpolation for sub-integer BPM.

        Returns:
            TempoResult with tempo in BPM.
        """
        classifier = self._load_model()

        audio_path = str(audio_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        from tempocnn.feature import read_features

        # Suppress stdout during feature extraction and inference
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            features = read_features(audio_path)
            tempo = classifier.estimate_tempo(features, interpolate=interpolate)
        finally:
            sys.stdout = old_stdout

        return TempoResult(
            tempo=float(tempo),
            model_name=model_name or self.model_name,
            interpolate=interpolate,
            audio_path=audio_path,
        )

    def estimate_mirex(
        self,
        audio_path: str,
        model_name: str | None = None,
        interpolate: bool = False,
    ) -> DualTempoResult:
        """
        Estimate the two dominant tempi and salience (MIREX-style).

        This is useful for detecting tempo octave ambiguity. If salience is close
        to 0.5, both tempi are nearly equally likely (e.g., 52 vs 104 BPM).

        Args:
            audio_path: Path to the audio file.
            model_name: Optional override for the model name (default: fcn).
            interpolate: If True, use quadratic interpolation.

        Returns:
            DualTempoResult with tempo1, tempo2, and salience of tempo1.
        """
        classifier = self._load_model()

        audio_path = str(audio_path)
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        from tempocnn.feature import read_features

        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            features = read_features(audio_path)
            t1, t2, s1 = classifier.estimate_mirex(features, interpolate=interpolate)
        finally:
            sys.stdout = old_stdout

        # Generate a helpful note based on salience
        if s1 >= 0.8:
            note = f"Strong single tempo preference: {t1:.1f} BPM is clearly dominant."
        elif s1 >= 0.6:
            note = f"Moderate preference for {t1:.1f} BPM, but {t2:.1f} BPM is a plausible alternative."
        elif s1 >= 0.4:
            note = f"Ambiguous: both {t1:.1f} and {t2:.1f} BPM are roughly equally likely. Check octave relationship (e.g., {min(t1,t2):.0f} x 2 = {min(t1,t2)*2:.0f})."
        else:
            note = f"Weak preference for {t1:.1f} BPM; {t2:.1f} BPM may actually be the perceived tempo."

        return DualTempoResult(
            tempo1=float(t1),
            tempo2=float(t2),
            salience=float(s1),
            model_name=model_name or self.model_name,
            interpolate=interpolate,
            audio_path=audio_path,
            note=note,
        )
