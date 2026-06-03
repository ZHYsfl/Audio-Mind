# Environment Setup

Canonical bootstrap sequence that takes a fresh `git clone` all the way to a
working `demo_run.py`.

## 0. Prerequisites

- **Python 3.10+** (3.11 recommended).
- **uv** on `$PATH` — install once via `curl -LsSf https://astral.sh/uv/install.sh | sh`.
- **conda** on `$PATH` *only if you plan to set up the `diarizen` tool*. Any
  miniconda/anaconda install works. On systems where `conda` is not on `$PATH`,
  export `CONDA_SH=/path/to/conda/etc/profile.d/conda.sh` before running
  `diarizen/setup.sh` or `setup_all_tools.sh`.
- **A DashScope API key** (or any OpenAI-compatible key) for the API-based demo:
  ```bash
  export DASHSCOPE_API_KEY="sk-..."
  ```
- **HuggingFace token** (optional, only for the `whisperx` diarization path):
  `huggingface-cli login` after accepting the user agreements on
  `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0`.

## 1. Main framework env

```bash
# Pick one — uv venv:
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e '.[api,dev,download]'

# ...or conda:
conda create -n audio_agent python=3.11 -y
conda activate audio_agent
pip install -e '.[api,dev,download]'
```

Smoke check:
```bash
python -c "from audio_agent.utils.model_downloader import MODELS; print(len(MODELS), 'models registered')"
# expect: 13 models registered
```

## 2. Choose where models live

```bash
# Default: <repo>/models/. Override if you want them elsewhere.
export AUDIO_AGENT_MODELS_DIR="$PWD/models"
```

Every tool's `config.yaml` references `${AUDIO_AGENT_MODELS_DIR}/...`, expanded
at config-load time by `audio_agent/tools/catalog/loader.py`.

## 3. Build every MCP tool env

```bash
./setup_all_tools.sh           # all 14 tools
# or a subset:
./setup_all_tools.sh ffmpeg librosa
```

Output goes to `.artifacts/setup_logs/`. Each tool builds an isolated `.venv`
inside its own catalog directory. `diarizen` uses conda (Python 3.10); every
other tool uses uv (Python 3.11).

## 4. Download model weights

```bash
audio-agent-download-models --all
```

The default registry covers:

| Tool | Model | Size |
|---|---|---|
| asr_qwen3 | Qwen/Qwen3-ASR-1.7B + Qwen/Qwen3-ForcedAligner-0.6B | ~5GB |
| diarizen | BUT-FIT/diarizen-wavlm-large-s80-md | ~1GB |
| sortformer_diarization | nvidia/diar_streaming_sortformer_4spk-v2 | ~450MB |
| fireredasr2s | FireRedTeam/FireRedASR-AED-L | ~4.5GB |
| fireredvad | FireRedTeam/FireRedVAD | ~200MB |
| whisperx (diarization) | pyannote/speaker-diarization-3.1 + segmentation-3.0 | ~150MB (HF token) |
| (local frontend, optional) | Qwen/Qwen2.5-Omni-7B | ~21GB |

`wespeaker` and `tempo_cnn` auto-download their weights on first use; nothing to
pre-stage.

## 5. Verify

Static + import + load test on every tool:
```bash
./verify_all_tools.sh
```

Single targeted demo (any tool, run on a GPU allocation if the tool loads a
local model):
```bash
export DASHSCOPE_API_KEY="sk-..."
python -m audio_agent.examples.demo_run \
  --audio path/to/clip.wav \
  --question "Use the transcribe_qwenasr tool to transcribe this audio." \
  --max-steps 5
```

For larger end-to-end sweeps (run one targeted question per catalog tool, log
each demo to `.artifacts/verify_runs/`), wrap the loop in your own helper
script — the pattern is straightforward and intentionally not in-tree because
runtime details (job-scheduler flags, cache locations, env activation) are
deployment-specific.

## 6. (Optional) Local-model frontend

For `Qwen/Qwen2.5-Omni-7B` (single-GPU friendly) instead of API frontend:

```bash
# Build a separate conda env so torch+CUDA doesn't pollute the API env.
conda create -n audio_agent_omni python=3.10 -y
conda activate audio_agent_omni
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -e '.[api,download]'
pip install transformers qwen_omni_utils accelerate librosa soundfile
audio-agent-download-models --models qwen2.5-omni
```

The adapter is `audio_agent/frontend/qwen25_omni_frontend.py`. Use it by passing
`Qwen25OmniFrontend` to `AudioAgent(frontend=...)`.

## Troubleshooting

- **`uv: command not found`** → install uv (see Prerequisites) or set
  `$REPO_ROOT/.uv/bin/uv` to point at a static binary.
- **`conda: command not found` when running `diarizen/setup.sh`** → either
  install conda or `export CONDA_SH=/path/to/conda/etc/profile.d/conda.sh`.
- **`Illegal instruction` from asr_qwen3 on CPU** → flash-attn or torch wheels
  require AVX-512. Either run on a host with a CUDA GPU (recommended) or
  rebuild torch from a CPU-feature-compatible wheel.
- **`libcudart.so.<N>: cannot open shared object file`** → torch was installed
  for a CUDA major version that doesn't match the host's libcudart. Re-install
  torch with the matching CUDA variant (e.g. `cu121`, `cu124`); for the
  `sortformer_diarization` setup, override `TORCH_CUDA_VARIANT` before
  running `setup.sh`.
- **`FireRedVad.from_pretrained` errors** → model not downloaded. Run
  `audio-agent-download-models --models fireredvad`.
- **WhisperX diarization missing pyannote weights** → accept the model card
  agreements on HuggingFace, `huggingface-cli login`, then
  `audio-agent-download-models --models pyannote-diarization pyannote-segmentation`.
