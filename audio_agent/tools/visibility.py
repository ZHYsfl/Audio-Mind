"""Planner-facing tool visibility helpers."""

from __future__ import annotations

from typing import Literal

from audio_agent.core.schemas import ToolSpec

PlannerToolScope = Literal["core", "all"]

CORE_PLANNER_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "afftdn_denoise",
        "afwtdn_denoise",
        "analyze_onsets",
        "analyze_pitch",
        "analyze_spectral_features",
        "astats",
        "audio_stats",
        "convert_channels",
        "detect_key",
        "diarize",
        "diarize_sortformer",
        "estimate_tempo_cnn_mirex",
        "extract_chroma",
        "extract_mfcc",
        "extract_rms_energy",
        "fireredvad_aed",
        "fireredvad_predict",
        "get_audio_info",
        "highpass_filter",
        "inspect_audio_plots",
        "lowpass_filter",
        "recognize_chords_large_vocab",
        "resample_audio",
        "segment_audio",
        "separate_harmonic_percussive",
        "silencedetect",
        "speaker_verify",
        "spectral_stats",
        "transcribe_fireredasr",
        "transcribe_fireredasr_with_timestamps",
        "transcribe_qwenasr",
        "transcribe_qwenasr_with_timestamps",
        "transcribe_whisperx",
        "transcribe_whisperx_with_diarization",
        "trim_audio",
        "vad_predict",
        "verify_audio_quality",
        "volumedetect",
    }
)


def filter_tool_specs(
    tool_specs: list[ToolSpec],
    scope: PlannerToolScope = "core",
) -> list[ToolSpec]:
    """Filter registered tools to the planner-visible scope."""
    if scope == "all":
        return list(tool_specs)
    if scope == "core":
        return [spec for spec in tool_specs if spec.name in CORE_PLANNER_TOOL_NAMES]

    raise ValueError(f"Invalid planner tool scope: {scope!r}. Expected 'core' or 'all'.")
