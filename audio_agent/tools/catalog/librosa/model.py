from __future__ import annotations

import json
import os
import subprocess
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _coerce_tempo(raw_tempo: Any) -> float:
    if hasattr(raw_tempo, "item"):
        try:
            return float(raw_tempo.item())
        except ValueError:
            pass
    if isinstance(raw_tempo, (list, tuple)) and raw_tempo:
        return float(raw_tempo[0])
    return float(raw_tempo)


@dataclass
class AnalysisResult:
    tempo: float
    beats: list[int]
    beat_times_sec: list[float]

    def __post_init__(self) -> None:
        if not self.beats:
            raise ValueError("beats must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BeatResult:
    tempo_bpm: float
    beat_times: list[float]
    beat_count: int
    units: str = "seconds"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OnsetResult:
    onset_times: list[float]
    onset_count: int
    onset_strength_mean: float
    onset_strength_std: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MFCCResult:
    n_mfcc: int
    mfcc_mean: list[float]
    mfcc_std: list[float]
    sample_rate: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SpectralResult:
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_bandwidth_mean: float
    spectral_bandwidth_std: float
    spectral_rolloff_mean: float
    spectral_rolloff_std: float
    spectral_contrast_mean: list[float]
    spectral_flatness_mean: float
    spectral_flatness_std: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChromaResult:
    chroma_mean: list[float]
    chroma_std: list[float]
    pitch_classes: list[str]
    dominant_pitch_class: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class KeyResult:
    key: str
    confidence: float
    alternative_keys: list[dict[str, Any]]
    method: str = "chroma"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TuningResult:
    deviation_cents: float
    reference_pitch: str
    is_well_tuned: bool
    suggested_correction_cents: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PitchResult:
    mean_pitch_hz: float
    pitch_range_hz: tuple[float, float]
    voiced_frames: int
    unvoiced_frames: int
    voiced_ratio: float
    confidence_mean: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RMSResult:
    rms_mean: float
    rms_std: float
    peak_amplitude: float
    dynamic_range_db: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ZCRResult:
    zcr_mean: float
    zcr_std: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Segment:
    start: float
    end: float
    duration: float


@dataclass
class SegmentResult:
    segments: list[dict[str, float]]
    segment_count: int
    total_speech_duration: float
    total_silence_duration: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": self.segments,
            "segment_count": self.segment_count,
            "total_speech_duration": self.total_speech_duration,
            "total_silence_duration": self.total_silence_duration,
        }


@dataclass
class AudioInfoResult:
    duration_seconds: float
    sample_rate: int
    channels: int
    format: str
    bit_depth: int | None
    total_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TransformResult:
    output_path: str
    original_duration: float
    new_duration: float | None
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SeparationResult:
    """Result of harmonic/percussive source separation.
    
    Uses standard output_path convention with additional_outputs for secondary file.
    """
    output_path: str  # Primary: harmonic file (for backward compatibility)
    percussive_path: str
    harmonic_ratio: float
    percussive_ratio: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_path": self.output_path,
            "percussive_path": self.percussive_path,
            "harmonic_ratio": self.harmonic_ratio,
            "percussive_ratio": self.percussive_ratio,
            "description": "Audio separated into harmonic (melodic) and percussive (rhythmic) components",
        }


class ModelWrapper:
    """Wrapper for comprehensive librosa audio analysis tools."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.model_dir = Path(__file__).resolve().parent
        self.python_bin = Path(
            self.config.get("python_bin", self.model_dir / ".venv" / "bin" / "python")
        )
        self.numba_cache_dir = Path(
            self.config.get(
                "numba_cache_dir",
                Path(tempfile.gettempdir()) / "audio_agent_numba_cache",
            )
        )
        self._loaded = False

    def _build_subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        self.numba_cache_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("NUMBA_CACHE_DIR", str(self.numba_cache_dir))
        return env

    def load(self) -> None:
        if not self.python_bin.exists():
            raise RuntimeError(f"Runtime python not found: {self.python_bin}")
        command = [
            str(self.python_bin),
            "-c",
            "import json, librosa; print(json.dumps({'version': librosa.__version__}))",
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=self.model_dir.parents[4],
            env=self._build_subprocess_env(),
        )
        payload = json.loads(completed.stdout.strip())
        if not payload.get("version"):
            raise RuntimeError("librosa version check returned an empty payload")
        self._loaded = True

    def healthcheck(self) -> dict[str, Any]:
        return {
            "status": "ready" if self.python_bin.exists() else "error",
            "message": f"python_bin={self.python_bin}",
            "model_loaded": self._loaded,
        }

    def _run_librosa_code(self, code: str, audio_path: Path, *extra_args: str) -> dict[str, Any]:
        """Execute librosa code in subprocess and return JSON result."""
        if not self._loaded:
            self.load()
        if not audio_path.exists():
            raise FileNotFoundError(f"Input audio not found: {audio_path}")
        
        command = [str(self.python_bin), "-c", code, str(audio_path), *extra_args]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            cwd=self.model_dir.parents[4],
            env=self._build_subprocess_env(),
        )
        return json.loads(completed.stdout.strip())

    def predict(self, input_data: str | Path) -> AnalysisResult:
        """Original analyze_rhythm method - kept for compatibility."""
        input_path = Path(input_data).resolve()
        code = (
            "import json, librosa, sys, numpy as np; "
            "audio_path = sys.argv[1]; "
            "y, sr = librosa.load(audio_path, sr=None); "
            "onset_env = librosa.onset.onset_strength(y=y, sr=sr); "
            "tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr); "
            "tempo_value = float(tempo.item() if hasattr(tempo, 'item') else tempo[0] if hasattr(tempo, '__len__') else tempo); "
            "beat_list = [int(x) for x in beats.tolist()]; "
            "beat_times = [round(float(x), 6) for x in librosa.frames_to_time(beats, sr=sr).tolist()]; "
            "print(json.dumps({'tempo': tempo_value, 'beats': beat_list, 'beat_times_sec': beat_times}))"
        )
        payload = self._run_librosa_code(code, input_path)
        return AnalysisResult(
            tempo=float(payload["tempo"]),
            beats=[int(x) for x in payload["beats"]],
            beat_times_sec=[float(x) for x in payload["beat_times_sec"]],
        )

    # =========================================================================
    # PHASE 1: TEXT ANALYSIS TOOLS
    # =========================================================================

    def analyze_beats(self, audio_path: str | Path, units: str = "time") -> BeatResult:
        """Detect tempo and beat positions."""
        input_path = Path(audio_path).resolve()
        code = (
            "import json, librosa, sys, numpy as np; "
            "audio_path, units = sys.argv[1], sys.argv[2]; "
            "y, sr = librosa.load(audio_path, sr=None); "
            "onset_env = librosa.onset.onset_strength(y=y, sr=sr); "
            "tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr); "
            "tempo_value = float(tempo.item() if hasattr(tempo, 'item') else tempo[0] if hasattr(tempo, '__len__') else tempo); "
            "beat_times = librosa.frames_to_time(beats, sr=sr).tolist() if units == 'time' else beats.tolist(); "
            "beat_times = [round(float(x), 6) for x in beat_times]; "
            "print(json.dumps({'tempo_bpm': tempo_value, 'beat_times': beat_times, 'beat_count': len(beat_times), 'units': units}))"
        )
        payload = self._run_librosa_code(code, input_path, units)
        return BeatResult(**payload)

    def analyze_onsets(self, audio_path: str | Path) -> OnsetResult:
        """Detect note onset events."""
        input_path = Path(audio_path).resolve()
        code = (
            "import json, librosa, sys, numpy as np; "
            "audio_path = sys.argv[1]; "
            "y, sr = librosa.load(audio_path, sr=None); "
            "onset_env = librosa.onset.onset_strength(y=y, sr=sr); "
            "onset_times = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time'); "
            "onset_times = [round(float(x), 6) for x in onset_times.tolist()]; "
            "strength_mean = float(np.mean(onset_env)); "
            "strength_std = float(np.std(onset_env)); "
            "print(json.dumps({'onset_times': onset_times, 'onset_count': len(onset_times), "
            "'onset_strength_mean': round(strength_mean, 6), 'onset_strength_std': round(strength_std, 6)}))"
        )
        payload = self._run_librosa_code(code, input_path)
        return OnsetResult(**payload)

    def extract_mfcc(self, audio_path: str | Path, n_mfcc: int = 13) -> MFCCResult:
        """Extract MFCC features (timbre)."""
        input_path = Path(audio_path).resolve()
        code = (
            "import json, librosa, sys, numpy as np; "
            "audio_path, n_mfcc = sys.argv[1], int(sys.argv[2]); "
            "y, sr = librosa.load(audio_path, sr=None); "
            "mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc); "
            "mfcc_mean = [round(float(x), 6) for x in np.mean(mfcc, axis=1).tolist()]; "
            "mfcc_std = [round(float(x), 6) for x in np.std(mfcc, axis=1).tolist()]; "
            "print(json.dumps({'n_mfcc': n_mfcc, 'mfcc_mean': mfcc_mean, 'mfcc_std': mfcc_std, 'sample_rate': sr}))"
        )
        payload = self._run_librosa_code(code, input_path, str(n_mfcc))
        return MFCCResult(**payload)

    def analyze_spectral_features(self, audio_path: str | Path) -> SpectralResult:
        """Extract comprehensive spectral features."""
        input_path = Path(audio_path).resolve()
        code = (
            "import json, librosa, sys, numpy as np; "
            "audio_path = sys.argv[1]; "
            "y, sr = librosa.load(audio_path, sr=None); "
            "S = np.abs(librosa.stft(y)); "
            "cent = librosa.feature.spectral_centroid(S=S, sr=sr); "
            "band = librosa.feature.spectral_bandwidth(S=S, sr=sr); "
            "rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr); "
            "contrast = librosa.feature.spectral_contrast(S=S, sr=sr); "
            "flatness = librosa.feature.spectral_flatness(S=S); "
            "result = {"
            "'spectral_centroid_mean': round(float(np.mean(cent)), 6), "
            "'spectral_centroid_std': round(float(np.std(cent)), 6), "
            "'spectral_bandwidth_mean': round(float(np.mean(band)), 6), "
            "'spectral_bandwidth_std': round(float(np.std(band)), 6), "
            "'spectral_rolloff_mean': round(float(np.mean(rolloff)), 6), "
            "'spectral_rolloff_std': round(float(np.std(rolloff)), 6), "
            "'spectral_contrast_mean': [round(float(x), 6) for x in np.mean(contrast, axis=1).tolist()], "
            "'spectral_flatness_mean': round(float(np.mean(flatness)), 6), "
            "'spectral_flatness_std': round(float(np.std(flatness)), 6)}; "
            "print(json.dumps(result))"
        )
        payload = self._run_librosa_code(code, input_path)
        return SpectralResult(**payload)

    def extract_chroma(self, audio_path: str | Path) -> ChromaResult:
        """Extract chroma features (pitch class profile)."""
        input_path = Path(audio_path).resolve()
        code = (
            "import json, librosa, sys, numpy as np; "
            "audio_path = sys.argv[1]; "
            "y, sr = librosa.load(audio_path, sr=None); "
            "chroma = librosa.feature.chroma_stft(y=y, sr=sr); "
            "chroma_mean = [round(float(x), 6) for x in np.mean(chroma, axis=1).tolist()]; "
            "chroma_std = [round(float(x), 6) for x in np.std(chroma, axis=1).tolist()]; "
            "pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']; "
            "dominant_idx = int(np.argmax(chroma_mean)); "
            "print(json.dumps({'chroma_mean': chroma_mean, 'chroma_std': chroma_std, "
            "'pitch_classes': pitch_classes, 'dominant_pitch_class': pitch_classes[dominant_idx]}))"
        )
        payload = self._run_librosa_code(code, input_path)
        return ChromaResult(**payload)

    def detect_key(self, audio_path: str | Path) -> KeyResult:
        """Estimate musical key from chroma features."""
        input_path = Path(audio_path).resolve()
        code = (
            "import json, librosa, sys, numpy as np; "
            "audio_path = sys.argv[1]; "
            "y, sr = librosa.load(audio_path, sr=None); "
            "chroma = librosa.feature.chroma_cqt(y=y, sr=sr); "
            "chroma_mean = np.mean(chroma, axis=1); "
            "key_templates = {"
            "'C major': [1,0,1,0,1,1,0,1,0,1,0,1], 'G major': [1,0,1,0,1,0,1,1,0,1,0,1], "
            "'D major': [0,1,1,0,1,0,1,0,1,1,0,1], 'A major': [0,1,0,1,1,0,1,0,1,0,1,1], "
            "'E major': [1,0,1,0,1,0,1,1,0,1,0,1], 'B major': [0,1,1,0,1,0,1,0,1,1,0,1], "
            "'F# major': [0,1,0,1,1,0,1,0,1,0,1,1], 'C# major': [1,0,1,0,1,0,1,1,0,1,0,1], "
            "'F major': [1,0,1,1,0,1,0,1,0,1,0,1], 'Bb major': [1,1,0,1,0,1,1,0,1,0,1,0], "
            "'Eb major': [0,1,1,0,1,0,1,1,0,1,0,1], 'Ab major': [1,0,1,0,1,0,1,1,0,1,1,0], "
            "'A minor': [1,0,1,1,0,1,0,1,1,0,1,0], 'E minor': [0,1,0,1,1,0,1,0,1,1,0,1], "
            "'B minor': [1,0,1,0,1,1,0,1,0,1,1,0], 'F# minor': [0,1,0,1,0,1,1,0,1,0,1,1], "
            "'C# minor': [1,0,1,0,1,0,1,1,0,1,0,1], 'G# minor': [0,1,0,1,0,1,0,1,1,0,1,1], "
            "'D# minor': [1,0,1,0,1,0,1,0,1,1,0,1], 'Bb minor': [0,1,0,1,0,1,0,1,0,1,1,1], "
            "'F minor': [1,0,1,0,1,0,1,0,1,0,1,1], 'C minor': [1,1,0,1,0,1,0,1,0,1,0,1], "
            "'G minor': [0,1,1,0,1,0,1,0,1,0,1,0], 'D minor': [0,0,1,1,0,1,0,1,0,1,1,0]}; "
            "correlations = {k: np.corrcoef(chroma_mean, v)[0,1] for k, v in key_templates.items()}; "
            "best_key = max(correlations, key=correlations.get); "
            "best_corr = correlations[best_key]; "
            "alternatives = sorted([(k, round(float(v), 4)) for k, v in correlations.items() if k != best_key], "
            "key=lambda x: x[1], reverse=True)[:3]; "
            "print(json.dumps({'key': best_key, 'confidence': round(float(best_corr), 4), "
            "'alternative_keys': [{'key': k, 'confidence': v} for k, v in alternatives], 'method': 'chroma'}))"
        )
        payload = self._run_librosa_code(code, input_path)
        return KeyResult(**payload)

    def estimate_tuning(self, audio_path: str | Path) -> TuningResult:
        """Estimate tuning deviation from A440."""
        input_path = Path(audio_path).resolve()
        code = (
            "import json, librosa, sys, numpy as np; "
            "audio_path = sys.argv[1]; "
            "y, sr = librosa.load(audio_path, sr=None); "
            "tuning = librosa.estimate_tuning(y=y, sr=sr); "
            "deviation_cents = round(float(tuning) * 100, 2); "
            "reference_a4 = round(440.0 * (2 ** (tuning / 12)), 2); "
            "is_well_tuned = abs(deviation_cents) < 10; "
            "print(json.dumps({'deviation_cents': deviation_cents, "
            "'reference_pitch': f'A4={reference_a4}Hz', "
            "'is_well_tuned': is_well_tuned, "
            "'suggested_correction_cents': round(-deviation_cents, 2)}))"
        )
        payload = self._run_librosa_code(code, input_path)
        return TuningResult(**payload)

    def analyze_pitch(self, audio_path: str | Path) -> PitchResult:
        """Extract pitch/F0 contour using pYIN."""
        input_path = Path(audio_path).resolve()
        code = (
            "import json, librosa, sys, numpy as np; "
            "audio_path = sys.argv[1]; "
            "y, sr = librosa.load(audio_path, sr=None); "
            "f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), "
            "fmax=librosa.note_to_hz('C7'), sr=sr); "
            "voiced_frames = int(np.sum(voiced_flag)); "
            "unvoiced_frames = int(np.sum(~voiced_flag)); "
            "voiced_ratio = round(voiced_frames / len(f0), 4) if len(f0) > 0 else 0; "
            "f0_voiced = f0[voiced_flag]; "
            "mean_pitch = round(float(np.mean(f0_voiced)), 2) if len(f0_voiced) > 0 else 0; "
            "pitch_min = round(float(np.min(f0_voiced)), 2) if len(f0_voiced) > 0 else 0; "
            "pitch_max = round(float(np.max(f0_voiced)), 2) if len(f0_voiced) > 0 else 0; "
            "confidence_mean = round(float(np.mean(voiced_probs)), 4); "
            "print(json.dumps({'mean_pitch_hz': mean_pitch, 'pitch_range_hz': [pitch_min, pitch_max], "
            "'voiced_frames': voiced_frames, 'unvoiced_frames': unvoiced_frames, "
            "'voiced_ratio': voiced_ratio, 'confidence_mean': confidence_mean}))"
        )
        payload = self._run_librosa_code(code, input_path)
        # Convert pitch_range list to tuple
        payload['pitch_range_hz'] = tuple(payload['pitch_range_hz'])
        return PitchResult(**payload)

    def extract_rms_energy(self, audio_path: str | Path) -> RMSResult:
        """Extract RMS energy (loudness) features."""
        input_path = Path(audio_path).resolve()
        code = (
            "import json, librosa, sys, numpy as np; "
            "audio_path = sys.argv[1]; "
            "y, sr = librosa.load(audio_path, sr=None); "
            "rms = librosa.feature.rms(y=y)[0]; "
            "rms_mean = round(float(np.mean(rms)), 6); "
            "rms_std = round(float(np.std(rms)), 6); "
            "peak = round(float(np.max(np.abs(y))), 6); "
            "dynamic_range = round(20 * np.log10(peak / (rms_mean + 1e-10)), 2); "
            "print(json.dumps({'rms_mean': rms_mean, 'rms_std': rms_std, "
            "'peak_amplitude': peak, 'dynamic_range_db': dynamic_range}))"
        )
        payload = self._run_librosa_code(code, input_path)
        return RMSResult(**payload)

    def extract_zcr(self, audio_path: str | Path) -> ZCRResult:
        """Extract zero-crossing rate."""
        input_path = Path(audio_path).resolve()
        code = (
            "import json, librosa, sys, numpy as np; "
            "audio_path = sys.argv[1]; "
            "y, sr = librosa.load(audio_path, sr=None); "
            "zcr = librosa.feature.zero_crossing_rate(y)[0]; "
            "zcr_mean = round(float(np.mean(zcr)), 6); "
            "zcr_std = round(float(np.std(zcr)), 6); "
            "print(json.dumps({'zcr_mean': zcr_mean, 'zcr_std': zcr_std}))"
        )
        payload = self._run_librosa_code(code, input_path)
        return ZCRResult(**payload)

    def segment_audio(self, audio_path: str | Path, top_db: int = 40) -> SegmentResult:
        """Segment audio into non-silent regions."""
        input_path = Path(audio_path).resolve()
        code = (
            "import json, librosa, sys, numpy as np; "
            "audio_path, top_db = sys.argv[1], int(sys.argv[2]); "
            "y, sr = librosa.load(audio_path, sr=None); "
            "intervals = librosa.effects.split(y, top_db=top_db); "
            "segments = [{'start': round(float(start) / sr, 3), 'end': round(float(end) / sr, 3), "
            "'duration': round((float(end) - float(start)) / sr, 3)} for start, end in intervals]; "
            "total_speech = sum(segment['duration'] for segment in segments); "
            "total_dur = len(y) / sr; "
            "silence_duration = max(0.0, total_dur - total_speech); "
            "print(json.dumps({'segments': segments, 'segment_count': len(segments), "
            "'total_speech_duration': round(total_speech, 3), "
            "'total_silence_duration': round(silence_duration, 3)}))"
        )
        payload = self._run_librosa_code(code, input_path, str(top_db))
        return SegmentResult(
            segments=payload['segments'],
            segment_count=payload['segment_count'],
            total_speech_duration=payload['total_speech_duration'],
            total_silence_duration=payload['total_silence_duration']
        )

    def get_audio_info(self, audio_path: str | Path) -> AudioInfoResult:
        """Get basic audio file information."""
        input_path = Path(audio_path).resolve()
        code = (
            "import json, librosa, sys, soundfile as sf, os; "
            "audio_path = sys.argv[1]; "
            "info = sf.info(audio_path); "
            "duration = info.duration; "
            "sr = info.samplerate; "
            "channels = info.channels; "
            "subtype = info.subtype; "
            "bit_depth = int(subtype.replace('PCM_', '').replace('FLOAT', '').replace('DOUBLE', '')) "
            "if subtype and any(c.isdigit() for c in subtype) else None; "
            "total_samples = int(info.frames); "
            "ext = os.path.splitext(audio_path)[1].lower().replace('.', ''); "
            "print(json.dumps({'duration_seconds': round(duration, 3), 'sample_rate': sr, "
            "'channels': channels, 'format': ext, 'bit_depth': bit_depth, 'total_samples': total_samples}))"
        )
        payload = self._run_librosa_code(code, input_path)
        return AudioInfoResult(**payload)

    # =========================================================================
    # PHASE 2: AUDIO TRANSFORMATION TOOLS
    # =========================================================================

    def _generate_output_path(self, input_path: Path, suffix: str) -> Path:
        """Generate unique output path in temp directory."""
        stem = input_path.stem
        uid = str(uuid.uuid4())[:8]
        output_dir = Path(tempfile.gettempdir())
        return output_dir / f"{stem}_{suffix}_{uid}.wav"

    def apply_pitch_shift(self, audio_path: str | Path, n_steps: float, 
                          output_path: str | None = None) -> TransformResult:
        """Shift pitch by n_steps semitones."""
        input_path = Path(audio_path).resolve()
        if output_path is None:
            out_path = self._generate_output_path(input_path, f"pitched_{n_steps}")
        else:
            out_path = Path(output_path)
        
        code = (
            "import json, librosa, sys, soundfile as sf; "
            "audio_path, n_steps, output_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]; "
            "y, sr = librosa.load(audio_path, sr=None); "
            "orig_dur = len(y) / sr; "
            "y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps); "
            "sf.write(output_path, y_shifted, sr); "
            "print(json.dumps({'output_path': output_path, 'original_duration': round(orig_dur, 3), "
            "'new_duration': round(len(y_shifted)/sr, 3), 'parameters': {'n_steps': n_steps}}))"
        )
        payload = self._run_librosa_code(code, input_path, str(n_steps), str(out_path))
        return TransformResult(**payload)

    def apply_time_stretch(self, audio_path: str | Path, rate: float,
                           output_path: str | None = None) -> TransformResult:
        """Time-stretch audio by rate (1.0 = original)."""
        input_path = Path(audio_path).resolve()
        if output_path is None:
            out_path = self._generate_output_path(input_path, f"stretched_{rate}")
        else:
            out_path = Path(output_path)
        
        code = (
            "import json, librosa, sys, soundfile as sf, numpy as np; "
            "audio_path, rate, output_path = sys.argv[1], float(sys.argv[2]), sys.argv[3]; "
            "y, sr = librosa.load(audio_path, sr=None); "
            "orig_dur = len(y) / sr; "
            "y_stretched = librosa.effects.time_stretch(y, rate=rate); "
            "sf.write(output_path, y_stretched, sr); "
            "print(json.dumps({'output_path': output_path, 'original_duration': round(orig_dur, 3), "
            "'new_duration': round(len(y_stretched)/sr, 3), 'parameters': {'rate': rate}}))"
        )
        payload = self._run_librosa_code(code, input_path, str(rate), str(out_path))
        return TransformResult(**payload)

    def remove_silence(self, audio_path: str | Path, top_db: int = 60,
                       output_path: str | None = None) -> TransformResult:
        """Remove leading/trailing silence."""
        input_path = Path(audio_path).resolve()
        if output_path is None:
            out_path = self._generate_output_path(input_path, "trimmed")
        else:
            out_path = Path(output_path)
        
        code = (
            "import json, librosa, sys, soundfile as sf; "
            "audio_path, top_db, output_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]; "
            "y, sr = librosa.load(audio_path, sr=None); "
            "orig_dur = len(y) / sr; "
            "y_trimmed, _ = librosa.effects.trim(y, top_db=top_db); "
            "sf.write(output_path, y_trimmed, sr); "
            "silence_removed = orig_dur - (len(y_trimmed) / sr); "
            "print(json.dumps({'output_path': output_path, 'original_duration': round(orig_dur, 3), "
            "'new_duration': round(len(y_trimmed)/sr, 3), "
            "'parameters': {'top_db': top_db, 'silence_removed_sec': round(silence_removed, 3)}}))"
        )
        payload = self._run_librosa_code(code, input_path, str(top_db), str(out_path))
        return TransformResult(**payload)

    def separate_harmonic_percussive(self, audio_path: str | Path,
                                     output_dir: str | None = None) -> SeparationResult:
        """Separate audio into harmonic and percussive components."""
        input_path = Path(audio_path).resolve()
        if output_dir is None:
            output_dir_path = Path(tempfile.gettempdir())
        else:
            output_dir_path = Path(output_dir)
        
        uid = str(uuid.uuid4())[:8]
        harmonic_path = output_dir_path / f"{input_path.stem}_harmonic_{uid}.wav"
        percussive_path = output_dir_path / f"{input_path.stem}_percussive_{uid}.wav"
        
        code = (
            "import json, librosa, sys, soundfile as sf, numpy as np; "
            "audio_path, harmonic_path, percussive_path = sys.argv[1], sys.argv[2], sys.argv[3]; "
            "y, sr = librosa.load(audio_path, sr=None); "
            "y_harmonic, y_percussive = librosa.effects.hpss(y); "
            "sf.write(harmonic_path, y_harmonic, sr); "
            "sf.write(percussive_path, y_percussive, sr); "
            "h_ratio = round(float(np.sum(y_harmonic**2)) / (float(np.sum(y**2)) + 1e-10), 4); "
            "p_ratio = round(float(np.sum(y_percussive**2)) / (float(np.sum(y**2)) + 1e-10), 4); "
            "print(json.dumps({'output_path': harmonic_path, 'percussive_path': percussive_path, "
            "'harmonic_ratio': h_ratio, 'percussive_ratio': p_ratio}))"
        )
        payload = self._run_librosa_code(code, input_path, str(harmonic_path), str(percussive_path))
        return SeparationResult(**payload)


def contract_result_to_json(result: AnalysisResult) -> str:
    return json.dumps(result.to_dict())
