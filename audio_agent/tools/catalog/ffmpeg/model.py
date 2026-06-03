"""FFmpeg Wrapper for Audio Agent Framework - Comprehensive Audio Processing.

Responsibilities:
- Provide 56+ audio processing tools via FFmpeg system calls
- Handle format conversion, effects, analysis, and transformations
- Manage output file paths and temporary files

Entry Points:
- FFmpegWrapper: Main wrapper class with 56+ methods

Dependencies:
- ffmpeg (system binary)
- ffprobe (system binary)

Example:
    wrapper = FFmpegWrapper()
    result = wrapper.loudnorm(input.wav, output.wav, target_lufs=-16)
    stats = wrapper.audio_stats(input.wav)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import Any


class ModelLoadError(RuntimeError):
    """Raised when FFmpeg tools cannot be found."""
    pass


class InferenceError(RuntimeError):
    """Raised when audio processing fails."""
    pass


@dataclass
class AudioProcessResult:
    """Result of audio processing operation."""
    output_path: str
    duration: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    additional_outputs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert isinstance(self.output_path, str)
        assert len(self.output_path) > 0, "Empty output_path"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "output_path": self.output_path,
            "duration": self.duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
        }
        result.update(self.additional_outputs)
        return result


@dataclass
class AnalysisResult:
    """Result of audio analysis operation."""
    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return self.data


class FFmpegWrapper:
    """Comprehensive FFmpeg wrapper for audio processing."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._ffmpeg_path = self.config.get("ffmpeg_path", "ffmpeg")
        self._ffprobe_path = self.config.get("ffprobe_path", "ffprobe")
        self._model_loaded = False

    def load(self) -> None:
        """Verify FFmpeg tools are available."""
        try:
            subprocess.run([self._ffmpeg_path, "-version"], capture_output=True, check=True)
            subprocess.run([self._ffprobe_path, "-version"], capture_output=True, check=True)
            self._model_loaded = True
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise ModelLoadError(f"FFmpeg tools not available: {e}")

    def healthcheck(self) -> dict[str, Any]:
        """Check if FFmpeg tools are ready."""
        try:
            result = subprocess.run([self._ffmpeg_path, "-version"], capture_output=True, check=True)
            version_line = result.stdout.decode().split("\n")[0]
            return {"status": "ready", "message": f"FFmpeg available: {version_line}", "model_loaded": self._model_loaded}
        except Exception as e:
            return {"status": "error", "message": str(e), "model_loaded": False}

    def _ensure_loaded(self):
        if not self._model_loaded:
            self.load()

    def _generate_output_path(self, suffix: str = "processed", ext: str = ".wav") -> str:
        """Generate unique output path in temp directory."""
        uid = uuid.uuid4().hex[:8]
        return os.path.join(tempfile.gettempdir(), f"ffmpeg_{suffix}_{uid}{ext}")

    def _run_ffmpeg(self, cmd: list[str], parse_output: bool = True) -> dict[str, Any]:
        """Execute FFmpeg command and return result."""
        try:
            result = subprocess.run(cmd, capture_output=True, check=True)
            if parse_output:
                return {"success": True, "stdout": result.stdout.decode(), "stderr": result.stderr.decode()}
            return {"success": True}
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode() if e.stderr else "Unknown error"
            if "No such filter" in stderr:
                raise InferenceError(f"FFmpeg unsupported feature: {stderr}")
            raise InferenceError(f"FFmpeg failed: {stderr}")

    def _probe_audio(self, audio_path: str) -> dict[str, Any]:
        """Probe audio file for metadata."""
        try:
            cmd = [
                self._ffprobe_path, "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams", audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, check=True, text=True)
            return json.loads(result.stdout)
        except Exception:
            return {}

    def _extract_metadata(self, audio_path: str) -> dict[str, Any]:
        """Extract duration, sample_rate, channels from audio file."""
        probe_data = self._probe_audio(audio_path)
        metadata = {"duration": None, "sample_rate": None, "channels": None}
        
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "audio":
                metadata["sample_rate"] = int(stream.get("sample_rate", 0)) or None
                metadata["channels"] = stream.get("channels")
                break
        
        format_info = probe_data.get("format", {})
        if duration_str := format_info.get("duration"):
            metadata["duration"] = float(duration_str)
        
        return metadata

    # =============================================================================
    # PHASE 1: FORMAT & CODEC CONVERSION (5 tools)
    # =============================================================================

    def convert_format(self, input_path: str, output_path: str | None = None, 
                       format_ext: str | None = None, codec: str | None = None) -> dict[str, Any]:
        """Convert audio to different format (MP3, WAV, FLAC, AAC, OGG, etc.)."""
        self._ensure_loaded()
        if not output_path:
            ext = format_ext or "wav"
            output_path = self._generate_output_path("converted", f".{ext}")
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path]
        if codec:
            cmd.extend(["-c:a", codec])
        cmd.append(output_path)
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def resample_audio(self, input_path: str, output_path: str | None = None,
                       sample_rate: int = 22050) -> dict[str, Any]:
        """Resample audio to target sample rate."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path(f"resampled_{sample_rate}")
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-ar", str(sample_rate), output_path]
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def change_bit_depth(self, input_path: str, output_path: str | None = None,
                         bit_depth: int = 16) -> dict[str, Any]:
        """Change audio bit depth (8, 16, 24, 32, or 64 for float)."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path(f"bd{bit_depth}")

        ext = os.path.splitext(output_path)[1].lower()
        pcm_codec = {
            8: "pcm_u8",
            16: "pcm_s16le",
            24: "pcm_s24le",
            32: "pcm_s32le",
            64: "pcm_f64le",
        }.get(bit_depth)
        if ext == ".wav" and pcm_codec is not None:
            cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-c:a", pcm_codec, output_path]
        else:
            sample_fmt = {8: "u8", 16: "s16", 24: "s32", 32: "s32", 64: "dbl"}.get(bit_depth, "s16")
            cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-sample_fmt", sample_fmt, output_path]

        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def convert_channels(self, input_path: str, output_path: str | None = None,
                         channels: int = 1) -> dict[str, Any]:
        """Convert channel layout (1=mono, 2=stereo, etc.)."""
        self._ensure_loaded()
        if not output_path:
            ch_name = "mono" if channels == 1 else "stereo" if channels == 2 else f"{channels}ch"
            output_path = self._generate_output_path(ch_name)
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-ac", str(channels), output_path]
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def extract_audio_from_video(self, video_path: str, output_path: str | None = None,
                                  codec: str = "copy") -> dict[str, Any]:
        """Extract audio stream from video file."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("extracted", ".wav")
        
        cmd = [self._ffmpeg_path, "-y", "-i", video_path, "-vn"]
        if codec != "copy":
            cmd.extend(["-c:a", codec])
        else:
            cmd.append("-c:a")
            cmd.append("copy")
        cmd.append(output_path)
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    # =============================================================================
    # PHASE 2: VOLUME & DYNAMICS (7 tools)
    # =============================================================================

    def adjust_volume(self, input_path: str, output_path: str | None = None,
                      volume_db: float | None = None, volume_factor: float | None = None) -> dict[str, Any]:
        """Adjust volume by dB (e.g., 6.0) or factor (e.g., 2.0)."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("volume")
        
        if volume_db is not None:
            volume_str = f"{volume_db}dB"
        elif volume_factor is not None:
            volume_str = str(volume_factor)
        else:
            volume_str = "1.0"
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", f"volume={volume_str}", output_path]
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def loudnorm(self, input_path: str, output_path: str | None = None,
                 target_lufs: float = -16.0, true_peak: float = -1.5,
                 loudness_range: float = 11.0) -> dict[str, Any]:
        """Loudness normalization to LUFS standard (broadcast/podcast)."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("loudnorm")
        
        filter_str = f"loudnorm=I={target_lufs}:TP={true_peak}:LRA={loudness_range}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def dynaudnorm(self, input_path: str, output_path: str | None = None,
                   frame_len: int = 500, filter_size: int = 31) -> dict[str, Any]:
        """Dynamic loudness normalization (gentler than loudnorm)."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("dynaudnorm")
        
        filter_str = f"dynaudnorm=f={frame_len}:g={filter_size}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def acompressor(self, input_path: str, output_path: str | None = None,
                    threshold: float = 0.05, ratio: float = 4.0,
                    attack: int = 200, release: int = 1000) -> dict[str, Any]:
        """Dynamic range compression."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("compressed")
        
        filter_str = f"acompressor=threshold={threshold}:ratio={ratio}:attack={attack}:release={release}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def agate(self, input_path: str, output_path: str | None = None,
              threshold: float = 0.1, attack: int = 50, release: int = 50) -> dict[str, Any]:
        """Noise gate - remove audio below threshold."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("gated")
        
        filter_str = f"agate=threshold={threshold}:attack={attack}:release={release}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def alimiter(self, input_path: str, output_path: str | None = None,
                 limit: float = 0.95, attack: int = 5, release: int = 50) -> dict[str, Any]:
        """Brick-wall limiter - prevent clipping."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("limited")

        filter_str = f"alimiter=limit={limit}:attack={attack}:release={release}:level=false"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def compand(self, input_path: str, output_path: str | None = None,
                attack: float = 0.3, decay: float = 0.8) -> dict[str, Any]:
        """Compand - compress and expand dynamic range."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("companded")
        
        filter_str = f"compand=attacks={attack}:decays={decay}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    # =============================================================================
    # PHASE 3: EQUALIZATION & FILTERING (8 tools)
    # =============================================================================

    def highpass_filter(self, input_path: str, output_path: str | None = None,
                        frequency: float = 80.0, poles: int = 2) -> dict[str, Any]:
        """High-pass filter - remove low frequencies."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path(f"hp{int(frequency)}")
        
        filter_str = f"highpass=f={frequency}:poles={poles}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def lowpass_filter(self, input_path: str, output_path: str | None = None,
                       frequency: float = 8000.0, poles: int = 2) -> dict[str, Any]:
        """Low-pass filter - remove high frequencies."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path(f"lp{int(frequency)}")
        
        filter_str = f"lowpass=f={frequency}:poles={poles}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def bandpass_filter(self, input_path: str, output_path: str | None = None,
                        frequency: float = 1000.0, width: float = 200.0) -> dict[str, Any]:
        """Band-pass filter - keep only frequency range."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path(f"bp{int(frequency)}")
        
        filter_str = f"bandpass=f={frequency}:width_type=h:width={width}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def bandreject_filter(self, input_path: str, output_path: str | None = None,
                          frequency: float = 50.0, width: float = 20.0) -> dict[str, Any]:
        """Band-reject (notch) filter - remove frequency band."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path(f"br{int(frequency)}")
        
        filter_str = f"bandreject=f={frequency}:width_type=h:width={width}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def equalizer(self, input_path: str, output_path: str | None = None,
                  frequency: float = 1000.0, gain: float = 5.0,
                  width: float = 2.0, width_type: str = "o") -> dict[str, Any]:
        """Parametric equalizer - boost/cut specific frequency."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path(f"eq{int(frequency)}")
        
        filter_str = f"equalizer=f={frequency}:width_type={width_type}:width={width}:g={gain}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def anequalizer(self, input_path: str, output_path: str | None = None,
                    bands: list[dict] | None = None) -> dict[str, Any]:
        """Graphic equalizer - multi-band EQ."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("aneq")
        
        # Default 10-band EQ if no bands specified
        if not bands:
            bands = [
                {"f": 31, "w": 50, "g": 0}, {"f": 63, "w": 100, "g": 0},
                {"f": 125, "w": 200, "g": 0}, {"f": 250, "w": 400, "g": 0},
                {"f": 500, "w": 800, "g": 0}, {"f": 1000, "w": 1600, "g": 0},
                {"f": 2000, "w": 3200, "g": 0}, {"f": 4000, "w": 6400, "g": 0},
                {"f": 8000, "w": 12800, "g": 0}, {"f": 16000, "w": 25600, "g": 0},
            ]
        
        eq_params = "|".join([f"c0 f={b['f']} w={b['w']} g={b['g']}" for b in bands])
        filter_str = f"anequalizer={eq_params}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def bass_boost(self, input_path: str, output_path: str | None = None,
                   gain: float = 10.0, frequency: float = 100.0, width: float = 0.5) -> dict[str, Any]:
        """Bass boost/cut (low shelf filter)."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path(f"bass{int(gain)}")
        
        filter_str = f"bass=g={gain}:f={frequency}:w={width}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def treble_boost(self, input_path: str, output_path: str | None = None,
                     gain: float = 5.0, frequency: float = 3000.0, width: float = 0.5) -> dict[str, Any]:
        """Treble boost/cut (high shelf filter)."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path(f"treble{int(gain)}")
        
        filter_str = f"treble=g={gain}:f={frequency}:w={width}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    # =============================================================================
    # PHASE 4: NOISE REDUCTION & RESTORATION (4 tools)
    # =============================================================================

    def afftdn_denoise(self, input_path: str, output_path: str | None = None,
                       noise_reduction: float = 12.0, noise_floor: float = -25.0) -> dict[str, Any]:
        """FFT-based noise reduction."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("denoised")
        
        filter_str = f"afftdn=nr={noise_reduction}:nf={noise_floor}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def afwtdn_denoise(self, input_path: str, output_path: str | None = None,
                       sigma: float = 0.1, levels: int = 10) -> dict[str, Any]:
        """Wavelet denoising."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("wtdn")
        
        filter_str = f"afwtdn=sigma={sigma}:levels={levels}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def adeclick(self, input_path: str, output_path: str | None = None) -> dict[str, Any]:
        """Remove clicks and pops from audio."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("declicked")
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", "adeclick", output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def frequency_filter_combo(self, input_path: str, output_path: str | None = None,
                               highpass_freq: float = 80.0, lowpass_freq: float = 8000.0) -> dict[str, Any]:
        """Apply highpass + lowpass combo for voice isolation."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("bandlimited")
        
        filter_str = f"highpass=f={highpass_freq},lowpass=f={lowpass_freq}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    # =============================================================================
    # PHASE 5: TIME & PITCH MANIPULATION (7 tools)
    # =============================================================================

    def change_tempo(self, input_path: str, output_path: str | None = None,
                     tempo_ratio: float = 1.5) -> dict[str, Any]:
        """Change tempo (speed) without changing pitch."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path(f"tempo{int(tempo_ratio*100)}")
        
        filter_str = f"atempo={tempo_ratio}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def trim_audio(self, input_path: str, output_path: str | None = None,
                   start_time: float = 0.0, duration: float | None = None,
                   end_time: float | None = None) -> dict[str, Any]:
        """Trim audio by start/duration or start/end times."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("trimmed")
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-ss", str(start_time)]
        if duration is not None:
            cmd.extend(["-t", str(duration)])
        elif end_time is not None:
            cmd.extend(["-to", str(end_time)])
        cmd.append(output_path)
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def silenceremove(self, input_path: str, output_path: str | None = None,
                      noise_db: float = -50.0, min_silence_duration: float = 0.5) -> dict[str, Any]:
        """Remove silence sections from audio."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("nosilence")

        filter_str = (
            f"silenceremove=start_periods=1:start_duration={min_silence_duration}:"
            f"start_threshold={noise_db}dB:start_silence=0:"
            f"stop_periods=-1:stop_duration={min_silence_duration}:"
            f"stop_threshold={noise_db}dB:stop_silence=0"
        )
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def pad_silence(self, input_path: str, output_path: str | None = None,
                    pad_start: float = 0.0, pad_end: float = 2.0) -> dict[str, Any]:
        """Add silence padding at start and/or end."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("padded")
        
        filter_parts = []
        if pad_start > 0:
            filter_parts.append(f"adelay=delays={int(pad_start*1000)}:all=1")
        if pad_end > 0:
            filter_parts.append(f"apad=pad_dur={pad_end}")
        
        filter_str = ",".join(filter_parts) if filter_parts else "apad"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def reverse_audio(self, input_path: str, output_path: str | None = None) -> dict[str, Any]:
        """Reverse audio playback."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("reversed")
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", "areverse", output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def add_delay(self, input_path: str, output_path: str | None = None,
                  delay_ms: float = 1000.0, decay: float = 0.5) -> dict[str, Any]:
        """Add echo/delay effect."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("delayed")
        
        delay_sec = delay_ms / 1000.0
        filter_str = f"adelay=delays={int(delay_ms)}:all=1"
        if decay > 0:
            filter_str = f"aecho=0.8:{decay}:{int(delay_ms)}|{int(delay_ms*1.5)}:0.3|0.25"
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def pitch_shift_rubberband(self, input_path: str, output_path: str | None = None,
                               pitch_ratio: float = 1.2, tempo_ratio: float = 1.0) -> dict[str, Any]:
        """High-quality pitch/time shift using rubberband."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path(f"pitch{int(pitch_ratio*100)}")
        
        filter_str = f"rubberband=pitch={pitch_ratio}:tempo={tempo_ratio}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    # =============================================================================
    # PHASE 6: SPATIAL & STEREO PROCESSING (6 tools)
    # =============================================================================

    def pan_channels(self, input_path: str, output_path: str | None = None,
                     channel_layout: str = "mono", pan_expr: str | None = None) -> dict[str, Any]:
        """Pan/remix channels (mono, stereo, custom)."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path(f"pan_{channel_layout}")
        
        if pan_expr:
            filter_str = f"pan={channel_layout}|{pan_expr}"
        elif channel_layout == "mono":
            filter_str = "pan=mono|c0=0.5*c0+0.5*c1"
        else:
            filter_str = f"pan={channel_layout}"
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def split_channels(self, input_path: str, output_dir: str | None = None) -> dict[str, Any]:
        """Split stereo into separate mono files."""
        self._ensure_loaded()
        if not output_dir:
            output_dir = tempfile.gettempdir()
        
        uid = uuid.uuid4().hex[:8]
        left_path = os.path.join(output_dir, f"channel_L_{uid}.wav")
        right_path = os.path.join(output_dir, f"channel_R_{uid}.wav")
        
        # Extract left channel
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", "pan=mono|c0=c0", left_path]
        self._run_ffmpeg(cmd, parse_output=False)
        
        # Extract right channel
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", "pan=mono|c0=c1", right_path]
        self._run_ffmpeg(cmd, parse_output=False)
        
        # Return using standard AudioProcessResult format
        # Primary output is left channel, right channel in additional_outputs
        metadata = self._extract_metadata(left_path)
        return AudioProcessResult(
            output_path=left_path,
            **metadata,
            additional_outputs={
                "right_channel_path": right_path,
                "description": "Stereo channels split into separate mono files",
            }
        ).to_dict()

    def mix_audio(self, input_paths: list[str], output_path: str | None = None,
                  weights: list[float] | None = None) -> dict[str, Any]:
        """Mix multiple audio files together."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("mixed")
        
        # Build amix filter
        n_inputs = len(input_paths)
        inputs_str = "".join([f"[{i}:a]" for i in range(n_inputs)])
        weights_str = f":weights={' '.join([str(w) for w in weights])}" if weights else ""
        filter_str = f"{inputs_str}amix=inputs={n_inputs}:duration=longest:normalize=0{weights_str}"
        
        cmd = [self._ffmpeg_path, "-y"]
        for inp in input_paths:
            cmd.extend(["-i", inp])
        cmd.extend(["-filter_complex", filter_str, output_path])
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def stereotools(self, input_path: str, output_path: str | None = None,
                    mode: str = "ms") -> dict[str, Any]:
        """Stereo tools - balance, width, MS matrix."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("stereotools")

        mode_map = {
            "ms": "lr>ms",
            "lr": "lr>lr",
        }
        filter_str = f"stereotools=mode={mode_map.get(mode, mode)}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def stereowiden(self, input_path: str, output_path: str | None = None,
                    delay: int = 20, feedback: float = 0.3, crossfeed: float = 0.3) -> dict[str, Any]:
        """Stereo widening effect."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("widened")
        
        filter_str = f"stereowiden=delay={delay}:feedback={feedback}:crossfeed={crossfeed}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def crossfeed(self, input_path: str, output_path: str | None = None,
                  strength: float = 0.2, range_hz: float = 0.5) -> dict[str, Any]:
        """Headphone crossfeed filter."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("crossfeed")
        
        filter_str = f"crossfeed=strength={strength}:range={range_hz}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    # =============================================================================
    # PHASE 7: EFFECTS & ENHANCEMENT (8 tools)
    # =============================================================================

    def add_echo(self, input_path: str, output_path: str | None = None,
                 delays: list[float] | None = None, decays: list[float] | None = None) -> dict[str, Any]:
        """Add echo/reverb effect."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("echo")
        
        if not delays:
            delays = [1000, 1800]
        if not decays:
            decays = [0.3, 0.25]
        
        delays_str = "|".join([str(int(d)) for d in delays])
        decays_str = "|".join([str(d) for d in decays])
        filter_str = f"aecho=0.8:0.9:{delays_str}:{decays_str}"
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def chorus_effect(self, input_path: str, output_path: str | None = None) -> dict[str, Any]:
        """Add chorus effect."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("chorus")

        filter_str = "chorus=0.7:0.9:55|60|65:0.4|0.32|0.3:0.25|0.4|0.3:2|2.3|1.3"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def flanger_effect(self, input_path: str, output_path: str | None = None,
                       delay: float = 0.0, depth: float = 2.0, regen: float = 0.5) -> dict[str, Any]:
        """Add flanger effect."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("flanger")

        filter_str = f"flanger=delay={delay}:depth={depth}:regen={regen}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def phaser_effect(self, input_path: str, output_path: str | None = None,
                      speed: float = 0.5, decay: float = 0.5) -> dict[str, Any]:
        """Add phaser effect."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("phaser")
        
        filter_str = f"aphaser=type=t:speed={speed}:decay={decay}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def tremolo_effect(self, input_path: str, output_path: str | None = None,
                       freq: float = 5.0, depth: float = 0.5) -> dict[str, Any]:
        """Add tremolo effect (amplitude modulation)."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("tremolo")
        
        filter_str = f"tremolo=f={freq}:d={depth}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def vibrato_effect(self, input_path: str, output_path: str | None = None,
                       freq: float = 5.0, depth: float = 0.5) -> dict[str, Any]:
        """Add vibrato effect (pitch modulation)."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("vibrato")
        
        filter_str = f"vibrato=f={freq}:d={depth}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def deesser(self, input_path: str, output_path: str | None = None,
                intensity: float = 1.0, frequency: float = 5500.0) -> dict[str, Any]:
        """De-essing - reduce sibilance (S sounds)."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("deessed")

        sample_rate = self._extract_metadata(input_path).get("sample_rate") or 44100
        normalized_frequency = min(max(frequency / (sample_rate / 2), 0.0), 1.0)
        filter_str = f"deesser=i={intensity}:m=0:f={normalized_frequency}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def crystalizer(self, input_path: str, output_path: str | None = None,
                    intensity: float = 2.0) -> dict[str, Any]:
        """Audio sharpening/exciter effect."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("crystalized")
        
        filter_str = f"crystalizer=i={intensity}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    # =============================================================================
    # PHASE 8: ANALYSIS & MEASUREMENT (7 tools)
    # =============================================================================

    def audio_stats(self, input_path: str) -> dict[str, Any]:
        """Get comprehensive audio statistics."""
        self._ensure_loaded()
        probe_data = self._probe_audio(input_path)
        
        result = {
            "format": probe_data.get("format", {}),
            "streams": [],
        }
        
        for stream in probe_data.get("streams", []):
            if stream.get("codec_type") == "audio":
                result["streams"].append({
                    "codec": stream.get("codec_name"),
                    "sample_rate": stream.get("sample_rate"),
                    "channels": stream.get("channels"),
                    "channel_layout": stream.get("channel_layout"),
                    "duration": stream.get("duration"),
                    "bit_rate": stream.get("bit_rate"),
                })
        
        return AnalysisResult(data=result).to_dict()

    def silencedetect(self, input_path: str, noise_db: float = -50.0,
                      min_duration: float = 0.5) -> dict[str, Any]:
        """Detect silence periods in audio."""
        self._ensure_loaded()
        
        filter_str = f"silencedetect=noise={noise_db}dB:d={min_duration}"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, "-f", "null", "-"]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = result.stderr
        
        # Parse silence detection output
        silence_starts = re.findall(r"silence_start: ([\d.]+)", stderr)
        silence_ends = re.findall(r"silence_end: ([\d.]+)", stderr)
        
        silences = []
        for i, start in enumerate(silence_starts):
            end = silence_ends[i] if i < len(silence_ends) else None
            silences.append({
                "start": float(start),
                "end": float(end) if end else None,
                "duration": float(end) - float(start) if end else None,
            })
        
        return AnalysisResult(data={
            "silence_count": len(silences),
            "silences": silences,
            "noise_threshold_db": noise_db,
            "min_silence_duration": min_duration,
        }).to_dict()

    def volumedetect(self, input_path: str) -> dict[str, Any]:
        """Detect max and mean volume levels."""
        self._ensure_loaded()
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", "volumedetect", "-f", "null", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = result.stderr
        
        mean_volume = re.search(r"mean_volume: ([-\d.]+) dB", stderr)
        max_volume = re.search(r"max_volume: ([-\d.]+) dB", stderr)
        
        return AnalysisResult(data={
            "mean_volume_db": float(mean_volume.group(1)) if mean_volume else None,
            "max_volume_db": float(max_volume.group(1)) if max_volume else None,
        }).to_dict()

    def ebur128(self, input_path: str) -> dict[str, Any]:
        """EBU R128 loudness measurement (LUFS)."""
        self._ensure_loaded()
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", "ebur128=metadata=1", "-f", "null", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = result.stderr
        
        # Parse EBU R128 output
        integrated = re.search(r"I:\s+([-\d.]+) LUFS", stderr)
        loudness_range = re.search(r"LRA:\s+([\d.]+) LU", stderr)
        true_peak = re.search(r"Peak:\s+([-\d.]+) dBFS", stderr)
        
        return AnalysisResult(data={
            "integrated_lufs": float(integrated.group(1)) if integrated else None,
            "loudness_range_lu": float(loudness_range.group(1)) if loudness_range else None,
            "true_peak_dbfs": float(true_peak.group(1)) if true_peak else None,
        }).to_dict()

    def replaygain(self, input_path: str) -> dict[str, Any]:
        """Calculate ReplayGain."""
        self._ensure_loaded()
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", "replaygain", "-f", "null", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = result.stderr
        
        track_gain = re.search(r"track_gain = ([-\d.]+) dB", stderr)
        track_peak = re.search(r"track_peak = ([\d.]+)", stderr)
        
        return AnalysisResult(data={
            "track_gain_db": float(track_gain.group(1)) if track_gain else None,
            "track_peak": float(track_peak.group(1)) if track_peak else None,
        }).to_dict()

    def astats(self, input_path: str) -> dict[str, Any]:
        """Detailed audio statistics per frame."""
        self._ensure_loaded()
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", "astats=metadata=1:reset=1", "-f", "null", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        stderr = result.stderr
        
        # Extract overall stats from the output
        stats = {}
        for line in stderr.split("\n"):
            if "Parsed_astats" in line and ":" in line:
                match = re.search(r"\]\s+([^:]+):\s+(.+)$", line)
                if not match:
                    continue
                key = match.group(1).strip().lower().replace(" ", "_")
                val = match.group(2).strip()
                try:
                    stats[key] = float(val)
                except ValueError:
                    stats[key] = val
        
        return AnalysisResult(data={
            "overall_statistics": stats,
        }).to_dict()

    def spectral_stats(self, input_path: str) -> dict[str, Any]:
        """Spectral analysis statistics."""
        self._ensure_loaded()
        
        # Use showfreqs for spectral data
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-lavfi", "showfreqs=s=800x400:mode=line", "-f", "null", "-"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # For now, return basic frequency analysis
        # Full spectral analysis would require parsing the filter output
        return AnalysisResult(data={
            "note": "Spectral analysis available via afft spectrums",
            "recommendation": "Use audio_stats for detailed codec/stream info",
        }).to_dict()

    # =============================================================================
    # PHASE 9: ADVANCED PROCESSING (4 tools)
    # =============================================================================

    def sidechain_compress(self, input_path: str, sidechain_path: str,
                           output_path: str | None = None,
                           threshold: float = 0.1, ratio: float = 4.0) -> dict[str, Any]:
        """Sidechain compression (ducking) - reduce input when sidechain is loud."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("sidechain")
        
        filter_str = f"sidechaincompress=threshold={threshold}:ratio={ratio}"
        cmd = [
            self._ffmpeg_path, "-y",
            "-i", input_path,
            "-i", sidechain_path,
            "-filter_complex", f"[0:a][1:a]{filter_str}",
            output_path
        ]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def crossfade_audio(self, input1_path: str, input2_path: str,
                        output_path: str | None = None,
                        duration: float = 10.0) -> dict[str, Any]:
        """Crossfade between two audio files."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("crossfade")
        
        # Use acrossfade filter
        cmd = [
            self._ffmpeg_path, "-y",
            "-i", input1_path,
            "-i", input2_path,
            "-filter_complex", f"acrossfade=d={duration}",
            output_path
        ]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def concat_audio(self, input_paths: list[str], output_path: str | None = None) -> dict[str, Any]:
        """Concatenate multiple audio files end-to-end."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("concat")
        
        # Create concat file list
        concat_file = os.path.join(tempfile.gettempdir(), f"concat_list_{uuid.uuid4().hex[:8]}.txt")
        with open(concat_file, "w") as f:
            for inp in input_paths:
                f.write(f"file '{inp}'\n")

        try:
            cmd = [
                self._ffmpeg_path, "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_file,
                "-c", "copy",
                output_path
            ]

            self._run_ffmpeg(cmd, parse_output=False)
        finally:
            if os.path.exists(concat_file):
                os.remove(concat_file)
        
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    def dynamic_equalizer(self, input_path: str, output_path: str | None = None) -> dict[str, Any]:
        """Dynamic equalizer - EQ that responds to audio content."""
        self._ensure_loaded()
        if not output_path:
            output_path = self._generate_output_path("dyn_eq")
        
        # Basic dynamic equalizer
        filter_str = "adynamicequalizer"
        cmd = [self._ffmpeg_path, "-y", "-i", input_path, "-af", filter_str, output_path]
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata).to_dict()

    # =============================================================================
    # LEGACY COMPATIBILITY
    # =============================================================================

    def predict(self, input_path: str, output_path: str,
                start_time: float | None = None, duration: float | None = None,
                sample_rate: int | None = None, channels: int | None = None) -> AudioProcessResult:
        """Legacy method - Process audio using FFmpeg."""
        self._ensure_loaded()
        
        cmd = [self._ffmpeg_path, "-y", "-i", input_path]
        if start_time is not None:
            cmd.extend(["-ss", str(start_time)])
        if duration is not None:
            cmd.extend(["-t", str(duration)])
        if sample_rate is not None:
            cmd.extend(["-ar", str(sample_rate)])
        if channels is not None:
            cmd.extend(["-ac", str(channels)])
        cmd.append(output_path)
        
        self._run_ffmpeg(cmd, parse_output=False)
        metadata = self._extract_metadata(output_path)
        return AudioProcessResult(output_path=output_path, **metadata)

    def process_audio(self, input_path: str, output_path: str, **kwargs: Any) -> dict[str, Any]:
        """Legacy convenience method that returns dict instead of dataclass."""
        result = self.predict(input_path, output_path, **kwargs)
        return result.to_dict()
