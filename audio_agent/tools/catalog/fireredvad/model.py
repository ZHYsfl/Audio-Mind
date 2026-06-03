from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class VADResult:
    timestamps: list[list[float]]
    dur: float | None = None
    wav_path: str | None = None

    def __post_init__(self) -> None:
        # An empty timestamps list is a legitimate "no speech detected" result
        # (silence, music-only audio, etc.). Reject only structurally invalid
        # entries (e.g., non-list rows).
        for span in self.timestamps:
            if not isinstance(span, (list, tuple)) or len(span) != 2:
                raise ValueError(
                    f"VAD timestamps must be a list of [start, end] pairs; got {span!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


@dataclass
class AEDResult:
    """Result from Audio Event Detection (AED)."""

    event2timestamps: dict[str, list[list[float]]]
    event2ratio: dict[str, float]
    dur: float | None = None
    wav_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class ModelWrapper:
    """Thin wrapper around the validated FireRedVAD non-streaming VAD/AED paths."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.model_root = Path(__file__).resolve().parent
        # Default model paths resolve under $AUDIO_AGENT_MODELS_DIR (or
        # <repo>/models when unset). Override per-call via config or by setting
        # MODEL_PATH for the wrapper directory containing VAD/ and AED/.
        _models_root = os.environ.get("AUDIO_AGENT_MODELS_DIR")
        if not _models_root:
            _models_root = str(Path(__file__).resolve().parents[4] / "models")
        _fireredvad_root = os.environ.get("MODEL_PATH") or str(Path(_models_root) / "FireRedVAD")
        self.vad_model_dir = Path(
            self.config.get("vad_model_dir", str(Path(_fireredvad_root) / "VAD"))
        )
        self.aed_model_dir = Path(
            self.config.get("aed_model_dir", str(Path(_fireredvad_root) / "AED"))
        )
        _env_device = os.environ.get("MODEL_DEVICE", "cpu").lower()
        self.use_gpu = bool(self.config.get("use_gpu", _env_device in ["cuda", "gpu", "auto"]))
        self._vad_model = None
        self._aed_model = None

    def _load_vad(self) -> None:
        """Lazy load the VAD model."""
        if self._vad_model is not None:
            return
        from fireredvad import FireRedVad, FireRedVadConfig

        vad_config = FireRedVadConfig(
            use_gpu=self.use_gpu,
            smooth_window_size=int(self.config.get("smooth_window_size", 5)),
            speech_threshold=float(self.config.get("speech_threshold", 0.4)),
            min_speech_frame=int(self.config.get("min_speech_frame", 20)),
            max_speech_frame=int(self.config.get("max_speech_frame", 2000)),
            min_silence_frame=int(self.config.get("min_silence_frame", 20)),
            merge_silence_frame=int(self.config.get("merge_silence_frame", 0)),
            extend_speech_frame=int(self.config.get("extend_speech_frame", 0)),
            chunk_max_frame=int(self.config.get("chunk_max_frame", 30000)),
        )
        self._vad_model = FireRedVad.from_pretrained(
            str(self.vad_model_dir), vad_config
        )

    def _load_aed(self) -> None:
        """Lazy load the AED model."""
        if self._aed_model is not None:
            return
        from fireredvad import FireRedAed, FireRedAedConfig

        aed_config = FireRedAedConfig(
            use_gpu=self.use_gpu,
            smooth_window_size=int(self.config.get("smooth_window_size", 5)),
            speech_threshold=float(self.config.get("speech_threshold", 0.4)),
            singing_threshold=float(self.config.get("singing_threshold", 0.5)),
            music_threshold=float(self.config.get("music_threshold", 0.5)),
            min_event_frame=int(self.config.get("min_event_frame", 20)),
            max_event_frame=int(self.config.get("max_event_frame", 2000)),
            min_silence_frame=int(self.config.get("min_silence_frame", 20)),
            merge_silence_frame=int(self.config.get("merge_silence_frame", 0)),
            extend_speech_frame=int(self.config.get("extend_speech_frame", 0)),
            chunk_max_frame=int(self.config.get("chunk_max_frame", 30000)),
        )
        self._aed_model = FireRedAed.from_pretrained(
            str(self.aed_model_dir), aed_config
        )

    def _resample_if_needed(self, audio_path: Path, target_sr: int = 16000) -> Path:
        """Check sample rate and resample if needed.
        
        Args:
            audio_path: Path to the input audio file
            target_sr: Target sample rate (default 16000 for AED)
            
        Returns:
            Path to audio file (original if already correct, or resampled temp file)
        """
        import soundfile as sf
        
        # Get current sample rate
        info = sf.info(str(audio_path))
        current_sr = info.samplerate
        
        if current_sr == target_sr:
            return audio_path
        
        # Need to resample
        import librosa
        
        y, sr = librosa.load(str(audio_path), sr=target_sr, mono=True)
        
        # Create temp file
        temp_fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(temp_fd)
        
        sf.write(temp_path, y, target_sr)
        
        return Path(temp_path)

    def healthcheck(self) -> dict[str, Any]:
        return {
            "status": (
                "ready"
                if self._vad_model is not None or self._aed_model is not None
                else "loading"
            ),
            "message": f"vad_model_dir={self.vad_model_dir}, aed_model_dir={self.aed_model_dir}",
            "vad_loaded": self._vad_model is not None,
            "aed_loaded": self._aed_model is not None,
        }

    def predict(self, input_data: str | Path) -> VADResult:
        """Run Voice Activity Detection (VAD) on the input audio."""
        self._load_vad()
        audio_path = Path(input_data).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Input audio not found: {audio_path}")
        raw_result, _ = self._vad_model.detect(str(audio_path))
        timestamps = [list(pair) for pair in raw_result.get("timestamps", [])]
        return VADResult(
            timestamps=timestamps,
            dur=raw_result.get("dur"),
            wav_path=raw_result.get("wav_path"),
        )

    def predict_aed(self, input_data: str | Path) -> AEDResult:
        """Run Audio Event Detection (AED) on the input audio.

        Detects speech, singing, and music events in the audio file.
        Returns timestamps and ratios for each event type.
        
        Note: Automatically resamples audio to 16kHz if needed.
        """
        self._load_aed()
        audio_path = Path(input_data).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Input audio not found: {audio_path}")
        
        # Auto-resample to 16kHz if needed
        temp_file = None
        try:
            processed_path = self._resample_if_needed(audio_path, target_sr=16000)
            
            # If we created a temp file, track it for cleanup
            if processed_path != audio_path:
                temp_file = processed_path
            
            raw_result, _ = self._aed_model.detect(str(processed_path))

            # Convert tuples to lists for JSON serialization
            event2timestamps = {}
            for event_type, timestamps in raw_result.get("event2timestamps", {}).items():
                event2timestamps[event_type] = [list(pair) for pair in timestamps]

            return AEDResult(
                event2timestamps=event2timestamps,
                event2ratio=raw_result.get("event2ratio", {}),
                dur=raw_result.get("dur"),
                wav_path=raw_result.get("wav_path"),
            )
        finally:
            # Cleanup temp file if created
            if temp_file is not None and temp_file.exists():
                try:
                    os.remove(temp_file)
                except OSError:
                    pass


def contract_result_to_json(result: VADResult | AEDResult) -> str:
    return result.to_json()
