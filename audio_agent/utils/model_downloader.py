"""
Model download utility for the audio agent framework.

This module provides functionality to pre-download HuggingFace models
to a local directory to avoid re-downloading on every login.

Usage:
    # As a module
    from audio_agent.utils.model_downloader import download_model, MODELS
    download_model(MODELS["qwen2-audio"])

    # As a CLI command (after pip install)
    audio-agent-download-models --all
    audio-agent-download-models --models qwen2-audio qwen2.5
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def repo_root() -> Path:
    """Resolve the AUDIO_AGENT repo root from this file's location."""
    return Path(__file__).resolve().parents[2]


def resolve_models_dir() -> Path:
    """Resolve models directory: $AUDIO_AGENT_MODELS_DIR, else <repo>/models."""
    env = os.environ.get("AUDIO_AGENT_MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "models"


# Base directory for storing downloaded models.
# Override by setting AUDIO_AGENT_MODELS_DIR; default is <repo>/models.
DEFAULT_MODELS_DIR = resolve_models_dir()

# Model registry - maps friendly names to HuggingFace model IDs
MODELS: dict[str, dict[str, Any]] = {
    "qwen2-audio": {
        "repo_id": "Qwen/Qwen2-Audio-7B-Instruct",
        "description": "Qwen2-Audio 7B Instruct model for audio understanding frontend",
        "subdir": "Qwen2-Audio-7B-Instruct",
    },
    "qwen2.5-omni": {
        "repo_id": "Qwen/Qwen2.5-Omni-7B",
        "description": "Qwen2.5-Omni 7B unified multimodal model for audio understanding frontend",
        "subdir": "Qwen2.5-Omni-7B",
    },
    "qwen3-omni": {
        "repo_id": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "description": "Qwen3-Omni 30B A3B Instruct model for multimodal audio understanding",
        "subdir": "Qwen3-Omni-30B-A3B-Instruct",
    },
    "qwen2.5": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "description": "Qwen2.5 7B Instruct model for planning",
        "subdir": "Qwen2.5-7B-Instruct",
    },
    "qwen3-asr": {
        "repo_id": "Qwen/Qwen3-ASR-1.7B",
        "description": "Qwen3-ASR 1.7B model for speech recognition",
        "subdir": "Qwen3-ASR-1.7B",
    },
    "qwen3-aligner": {
        "repo_id": "Qwen/Qwen3-ForcedAligner-0.6B",
        "description": "Qwen3-ForcedAligner 0.6B model for timestamp generation",
        "subdir": "Qwen3-ForcedAligner-0.6B",
    },
    "diarizen": {
        "repo_id": "BUT-FIT/diarizen-wavlm-large-s80-md",
        "description": "DiariZen speaker diarization model (wavlm-large-s80-md)",
        "subdir": "diarizen-wavlm-large-s80-md",
    },
    "omni-captioner": {
        "repo_id": "Qwen/Qwen3-Omni-30B-A3B-Captioner",
        "description": "Qwen3-Omni captioner for detailed audio description",
        "subdir": "Qwen3-Omni-30B-A3B-Captioner",
    },
    "sortformer-diar": {
        "repo_id": "nvidia/diar_streaming_sortformer_4spk-v2",
        "description": "NVIDIA SortFormer streaming diarization model (4 speakers max)",
        "subdir": "sortformer-diar-streaming-4spk-v2",
    },
    "fireredasr": {
        "repo_id": "FireRedTeam/FireRedASR-AED-L",
        "description": "FireRedASR AED-Large ASR model (Mandarin/English/code-switching)",
        "subdir": "FireRedASR-AED-L",
    },
    "fireredvad": {
        "repo_id": "FireRedTeam/FireRedVAD",
        "description": "FireRedVAD voice activity detection / coarse AED model",
        "subdir": "FireRedVAD",
    },
    # Note: WeSpeaker auto-downloads its 'english' model into $WESPEAKER_HOME on
    # first use. The HF-side mirror has a different file layout than what the
    # library expects, so we skip pre-staging here and let wespeaker fetch.
    # whisperx 3.8.4 requires pyannote-audio>=4.0, whose SpeakerDiarization
    # pipeline loads its components by repo id from the HF *hub cache*:
    #   - pyannote/speaker-diarization-community-1 provides the 4.x default
    #     segmentation/embedding/PLDA components (the pipeline's defaults).
    #   - pyannote/segmentation-3.0 + pyannote/wespeaker-voxceleb-resnet34-LM
    #     are the components declared by the 3.1 config.yaml that whisperx
    #     loads from $AUDIO_AGENT_MODELS_DIR/pyannote-speaker-diarization-3.1.
    # The registry marks those three "hub_cache": True so they land in the HF
    # hub cache (which the runtime consults), not in a plain local_dir snapshot
    # (which the runtime never consults). All four repos are gated.
    "pyannote-diarization": {
        "repo_id": "pyannote/speaker-diarization-3.1",
        "description": "Pyannote speaker-diarization-3.1 pipeline config (whisperx loads it from a local path; components come from pyannote-community/segmentation/embedding). Requires HuggingFace token with accepted user agreement.",
        "subdir": "pyannote-speaker-diarization-3.1",
        "requires_hf_token": True,
    },
    "pyannote-community": {
        "repo_id": "pyannote/speaker-diarization-community-1",
        "description": "Pyannote speaker-diarization-community-1 (pyannote-audio 4.x default components: segmentation + embedding + PLDA). Requires accepting the model agreement at hf.co/pyannote/speaker-diarization-community-1.",
        "subdir": "pyannote-speaker-diarization-community-1",
        "requires_hf_token": True,
        "hub_cache": True,
    },
    "pyannote-segmentation": {
        "repo_id": "pyannote/segmentation-3.0",
        "description": "Pyannote segmentation-3.0 model (whisperx diarization component; loaded from the HF hub cache at runtime). Requires HuggingFace token.",
        "subdir": "pyannote-segmentation-3.0",
        "requires_hf_token": True,
        "hub_cache": True,
    },
    "pyannote-embedding": {
        "repo_id": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "description": "Pyannote wespeaker-voxceleb-resnet34-LM embedding model (whisperx diarization component). Requires accepting the model agreement.",
        "subdir": "pyannote-wespeaker-voxceleb-resnet34-LM",
        "requires_hf_token": True,
        "hub_cache": True,
    },
}

# Convenience constants for local model paths
DEFAULT_QWEN2_AUDIO_PATH = str(DEFAULT_MODELS_DIR / MODELS["qwen2-audio"]["subdir"])
DEFAULT_QWEN25_OMNI_PATH = str(DEFAULT_MODELS_DIR / MODELS["qwen2.5-omni"]["subdir"])
DEFAULT_QWEN3_OMNI_PATH = str(DEFAULT_MODELS_DIR / MODELS["qwen3-omni"]["subdir"])
DEFAULT_QWEN25_PATH = str(DEFAULT_MODELS_DIR / MODELS["qwen2.5"]["subdir"])
DEFAULT_QWEN3_ASR_PATH = str(DEFAULT_MODELS_DIR / MODELS["qwen3-asr"]["subdir"])
DEFAULT_QWEN3_ALIGNER_PATH = str(DEFAULT_MODELS_DIR / MODELS["qwen3-aligner"]["subdir"])
DEFAULT_DIARIZEN_PATH = str(DEFAULT_MODELS_DIR / MODELS["diarizen"]["subdir"])
DEFAULT_OMNI_CAPTIONER_PATH = str(DEFAULT_MODELS_DIR / MODELS["omni-captioner"]["subdir"])
DEFAULT_SORTFORMER_DIAR_PATH = str(DEFAULT_MODELS_DIR / MODELS["sortformer-diar"]["subdir"])
DEFAULT_FIREREDASR_PATH = str(DEFAULT_MODELS_DIR / MODELS["fireredasr"]["subdir"])
DEFAULT_FIREREDVAD_PATH = str(DEFAULT_MODELS_DIR / MODELS["fireredvad"]["subdir"])
DEFAULT_PYANNOTE_DIAR_PATH = str(DEFAULT_MODELS_DIR / MODELS["pyannote-diarization"]["subdir"])
DEFAULT_PYANNOTE_SEG_PATH = str(DEFAULT_MODELS_DIR / MODELS["pyannote-segmentation"]["subdir"])


def get_local_model_path(model_name: str) -> str:
    """
    Get the local path for a model by its friendly name.

    Args:
        model_name: Friendly name of the model (e.g., "qwen2-audio", "qwen2.5")

    Returns:
        Local path where the model should be stored

    Raises:
        KeyError: If model_name is not recognized
    """
    if model_name not in MODELS:
        raise KeyError(
            f"Unknown model: {model_name}. "
            f"Available models: {', '.join(MODELS.keys())}"
        )
    return str(DEFAULT_MODELS_DIR / MODELS[model_name]["subdir"])


def download_model(
    model_name: str,
    models_dir: Path | None = None,
    cache_dir: Path | None = None,
    force_download: bool = False,
) -> Path:
    """
    Download a model from HuggingFace Hub to the local models directory.

    Two download modes, chosen by the "hub_cache" flag in the MODELS registry:
    - default: snapshot into <models_dir>/<subdir>. Used by models that tools
      load straight from disk (Qwen3-ASR, FireRedASR, ...).
    - hub_cache: snapshot into the HF hub cache (<models_dir>/hub by default).
      Required for models that the runtime resolves by repo id: pyannote
      pipelines call ``Model.from_pretrained("pyannote/...")``, which only
      consults the hub cache, never plain local dirs.

    For entries with "requires_hf_token" the ``HF_TOKEN`` environment variable
    is needed, and the repo's user agreement must have been accepted first.

    Args:
        model_name: Friendly name of the model (from MODELS registry)
        models_dir: Directory to store models (defaults to DEFAULT_MODELS_DIR)
        cache_dir: HuggingFace cache directory for hub_cache mode (defaults to
            <models_dir>/hub, which matches the runtime layout where
            config.yaml sets HF_HOME=${AUDIO_AGENT_MODELS_DIR})
        force_download: Whether to re-download even if model exists

    Returns:
        Path to the downloaded model directory (or hub cache root)

    Raises:
        KeyError: If model_name is not recognized
        RuntimeError: If huggingface_hub is not installed, or if a gated
            model is requested without HF_TOKEN
        Exception: If download fails
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required for downloading models. "
            "Install with: pip install huggingface_hub"
        ) from e

    if model_name not in MODELS:
        raise KeyError(
            f"Unknown model: {model_name}. "
            f"Available models: {', '.join(MODELS.keys())}"
        )

    model_info = MODELS[model_name]
    repo_id = model_info["repo_id"]

    # Gated repos need a token; read it from the environment (never store it).
    token = os.environ.get("HF_TOKEN") if model_info.get("requires_hf_token") else None
    if model_info.get("requires_hf_token") and not token:
        raise RuntimeError(
            f"{model_name} ({repo_id}) is gated. Set HF_TOKEN (e.g. "
            f"'export HF_TOKEN=hf_xxx') and accept the user agreement at "
            f"https://hf.co/{repo_id} first."
        )

    if model_info.get("hub_cache"):
        hub = Path(cache_dir) if cache_dir else (models_dir or DEFAULT_MODELS_DIR) / "hub"
        print(f"Downloading {model_name} ({repo_id}) into hub cache...")
        print(f"  Target directory: {hub}")
        hub.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(hub),
            token=token,
            force_download=force_download,
        )
        print(f"  ✓ {model_name} cached at {hub}")
        return hub

    target_dir = (models_dir or DEFAULT_MODELS_DIR) / model_info["subdir"]

    print(f"Downloading {model_name} ({repo_id})...")
    print(f"  Target directory: {target_dir}")

    # Create target directory
    target_dir.mkdir(parents=True, exist_ok=True)

    # Download options
    download_kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "local_dir": str(target_dir),
        "local_dir_use_symlinks": False,
    }
    if cache_dir:
        download_kwargs["cache_dir"] = str(cache_dir)
    if token:
        download_kwargs["token"] = token
    if force_download:
        download_kwargs["force_download"] = True

    try:
        snapshot_download(**download_kwargs)
        print(f"  ✓ {model_name} downloaded successfully to {target_dir}")
        return target_dir
    except Exception as e:
        print(f"  ✗ Failed to download {model_name}: {e}", file=sys.stderr)
        raise


def download_all_models(
    models_dir: Path | None = None,
    cache_dir: Path | None = None,
    force_download: bool = False,
) -> dict[str, Path]:
    """
    Download all registered models.

    Args:
        models_dir: Directory to store models (defaults to DEFAULT_MODELS_DIR)
        cache_dir: HuggingFace cache directory (optional)
        force_download: Whether to re-download even if models exist

    Returns:
        Dictionary mapping model names to their local paths
    """
    results: dict[str, Path] = {}
    print(f"Downloading all models to {models_dir or DEFAULT_MODELS_DIR}...\n")

    for model_name in MODELS:
        try:
            path = download_model(
                model_name,
                models_dir=models_dir,
                cache_dir=cache_dir,
                force_download=force_download,
            )
            results[model_name] = path
            print()
        except Exception as e:
            print(f"  Error downloading {model_name}: {e}\n", file=sys.stderr)

    print(f"Downloaded {len(results)}/{len(MODELS)} models successfully.")
    return results


def list_models() -> None:
    """Print a list of available models."""
    print("Available models:")
    print("-" * 60)
    for name, info in MODELS.items():
        if info.get("hub_cache"):
            # Hub-cache entries resolve at runtime by repo id; look the
            # snapshot up under DEFAULT_MODELS_DIR/hub.
            local_path = DEFAULT_MODELS_DIR / "hub" / (
                "models--" + info["repo_id"].replace("/", "--")
            )
        else:
            local_path = DEFAULT_MODELS_DIR / info["subdir"]
        exists = "✓ Downloaded" if local_path.exists() else "✗ Not downloaded"
        print(f"  {name}")
        print(f"    Repository: {info['repo_id']}")
        print(f"    Description: {info['description']}")
        print(f"    Local path: {local_path}")
        print(f"    Status: {exists}")
        print()


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(
        description="Download HuggingFace models for audio agent framework.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models
  audio-agent-download-models --all

  # Download specific models
  audio-agent-download-models --models qwen2-audio qwen2.5

  # Download to custom directory
  audio-agent-download-models --all --models-dir /path/to/models

  # List available models
  audio-agent-download-models --list
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()),
        metavar="MODEL",
        help=f"Models to download. Choices: {', '.join(MODELS.keys())}",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all registered models",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and their status",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help=f"Directory to store models (default: {DEFAULT_MODELS_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HuggingFace cache directory (optional)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if models exist",
    )

    return parser


def main() -> int:
    """Main entry point for CLI."""
    parser = build_parser()
    args = parser.parse_args()

    # Show list and exit
    if args.list:
        list_models()
        return 0

    # Validate arguments
    if not args.all and not args.models:
        parser.error("Please specify --models or --all")

    # Use default models dir if not specified
    models_dir = args.models_dir or DEFAULT_MODELS_DIR

    # Create models directory
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        if args.all:
            download_all_models(
                models_dir=models_dir,
                cache_dir=args.cache_dir,
                force_download=args.force,
            )
        else:
            for model_name in args.models:
                download_model(
                    model_name,
                    models_dir=models_dir,
                    cache_dir=args.cache_dir,
                    force_download=args.force,
                )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
