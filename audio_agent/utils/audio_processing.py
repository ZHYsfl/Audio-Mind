"""Audio preprocessing utilities for API upload."""

from __future__ import annotations

import io
import os
from pathlib import Path

from audio_agent.core.logging import get_logger

logger = get_logger()

# Default threshold: ~7 MB raw file → ~9.3 MB base64, safely under 10 MB API limits.
DEFAULT_MAX_RAW_BYTES: int = 7_000_000
DEFAULT_TARGET_SR: int = 16000


def preprocess_audio_for_api(
    audio_path: str,
    max_raw_bytes: int = DEFAULT_MAX_RAW_BYTES,
    target_sr: int = DEFAULT_TARGET_SR,
) -> bytes:
    """
    Preprocess audio for API upload if it exceeds a size threshold.

    If the file is smaller than *max_raw_bytes* the original bytes are returned
    unchanged.  Otherwise the audio is loaded with *librosa* (which resamples to
    *target_sr* and converts to mono), written back to an in-memory WAV buffer,
    and those bytes are returned.

    On any preprocessing failure a warning is logged and the original file bytes
    are returned so the caller can decide whether to proceed or abort.

    Args:
        audio_path: Path to the audio file.
        max_raw_bytes: Size threshold in bytes.  Files larger than this are
            preprocessed.
        target_sr: Target sample rate in Hz.

    Returns:
        Raw audio bytes (WAV format) ready for base-64 encoding.
    """
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    try:
        file_size = path.stat().st_size
    except OSError as exc:
        raise OSError(f"Cannot stat audio file: {audio_path}") from exc

    if file_size <= max_raw_bytes:
        with open(path, "rb") as f:
            return f.read()

    # File is too large – try to downsample / down-mix in memory.
    try:
        import librosa  # type: ignore[import-untyped]
        import soundfile as sf  # type: ignore[import-untyped]
    except ImportError as exc:
        logger.warning(
            "Audio preprocessing skipped: librosa/soundfile not available (%s). "
            "Returning original %d-byte file.",
            exc,
            file_size,
        )
        with open(path, "rb") as f:
            return f.read()

    try:
        y, sr = librosa.load(str(path), sr=target_sr, mono=True)
    except Exception as exc:
        logger.warning(
            "librosa.load failed for %s (%s). Returning original file.",
            audio_path,
            exc,
        )
        with open(path, "rb") as f:
            return f.read()

    buffer = io.BytesIO()
    try:
        sf.write(buffer, y, target_sr, format="WAV")
    except Exception as exc:
        logger.warning(
            "soundfile.write failed for %s (%s). Returning original file.",
            audio_path,
            exc,
        )
        with open(path, "rb") as f:
            return f.read()

    processed_bytes = buffer.getvalue()
    logger.info(
        "Audio preprocessed: %s | original=%.2fMB → processed=%.2fKB (%dHz mono)",
        path.name,
        file_size / (1024 * 1024),
        len(processed_bytes) / 1024,
        target_sr,
    )
    return processed_bytes
