# Audio Agent Tool Catalog

Each subdirectory is an MCP (Model Context Protocol) tool that runs in its own
isolated environment. The agent talks to them via `config.yaml` →
`audio_agent/tools/catalog/loader.py` → MCP server subprocess.

## Setup

Build every tool's environment with the root-level helper:

```bash
./setup_all_tools.sh             # all tools, auto-discovered
./setup_all_tools.sh ffmpeg lv_chordia   # subset
./verify_all_tools.sh            # run each tool's test_env.sh
```

Full bootstrap (prerequisites, model downloads, troubleshooting) is in
**[../../../ENVIRONMENT_SETUP.md](../../../ENVIRONMENT_SETUP.md)**.

## Tool inventory

| Tool | What it does | Models needed |
|---|---|---|
| `asr_qwen3` | ASR via Qwen3-ASR-1.7B (+ optional forced alignment) | qwen3-asr, qwen3-aligner |
| `autochord` | Coarse major/minor triad chord recognition | bundled (auto-downloaded) |
| `diarizen` | Speaker diarization (WavLM-large, CC BY-NC) | diarizen |
| `ffmpeg` | Audio metadata, filtering, denoise, trim, resample, channel ops | — |
| `fireredasr2s` | Mandarin/English/code-switching ASR | fireredasr |
| `fireredvad` | Voice activity + audio event detection | fireredvad |
| `librosa` | Audio metadata, segmentation, spectral / rhythm / pitch features | — |
| `lv_chordia` | Large-vocabulary chord recognition (7ths, jazz, etc.) | bundled (auto-downloaded) |
| `omni_captioner` | LALM caption + VLM plot inspection (DashScope API) | — (needs `DASHSCOPE_API_KEY`) |
| `snakers4_silero-vad` | Silero VAD | bundled (auto-downloaded) |
| `sortformer_diarization` | NVIDIA SortFormer streaming diarization (≤4 speakers) | sortformer-diar |
| `tempo_cnn` | CNN tempo estimation with octave-ambiguity salience | bundled (auto-downloaded) |
| `wespeaker` | Speaker verification (ResNet embeddings) | auto-downloaded on first use |
| `whisperx` | WhisperX ASR + optional pyannote diarization | pyannote-diarization, pyannote-segmentation (HF token) |

## Per-tool structure

Every tool directory should contain:

```
<tool>/
├── __init__.py
├── config.yaml        # MCP server registration + ${AUDIO_AGENT_MODELS_DIR}/... env vars
├── pyproject.toml     # tool deps for its own .venv
├── server.py          # MCP server entrypoint (top-of-file docstring documents any tool-specific quirks)
├── model.py           # (optional) thin wrapper around the underlying library
├── setup.sh           # creates .venv (uv) or .venv via conda for diarizen
├── test_env.sh        # smoke check; called by verify_all_tools.sh
└── test_env.py        # Python-level smoke check invoked by test_env.sh
```

`.venv/`, `<tool>_tool.egg-info/`, `__pycache__/`, and tool-specific cache
directories are gitignored — built on the host, never committed.

## Adding a new tool

Copy `_template/` and adapt:

```bash
cp -r _template my_new_tool
cd my_new_tool
# edit pyproject.toml, config.yaml, server.py
./setup.sh && ./test_env.sh
```

For the full step-by-step (including the harness-first agent workflow), see
[tool_preparation/README.md](../../../tool_preparation/README.md).
