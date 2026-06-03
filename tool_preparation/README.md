# AUDIO_AGENT Tool Preparation Harness

This directory contains the **Harness-First Agent Workflow** for automated tool environment configuration and onboarding in the Audio Agent Framework.

---

## 🤖 Agent Workflow for Tool Onboarding

The Audio Agent Framework provides a complete agent workflow that automates the entire process of configuring and validating audio processing tools.

### How It Works

1. **Initial Prompt** → Defines the agent's role and workflow
2. **TOOL_INPUT** → Specifies the tool/model to onboard
3. **Automated Execution** → Agent runs the complete pipeline
4. **Artifact Generation** → All results saved as structured files

### Usage

#### Step 1: Initial Prompt

Use this as the system prompt for your AI agent:

```text
cd ${REPO_ROOT}
You are now acting as the tool-onboarding execution agent for the Audio Agent Framework. Your task is not open-ended exploration; instead, strictly follow the harness-first workflow defined in the repository to complete the phase-1 onboarding of a single tool.

You must comply with the following documents:
1. AUDIO_AGENT/tool_preparation/AGENTS.md
2. AUDIO_AGENT/tool_preparation/policies/constitution.md
3. AUDIO_AGENT/tool_preparation/policies/evidence_priority.md
4. AUDIO_AGENT/tool_preparation/policies/retry_and_escalation.md
5. AUDIO_AGENT/tool_preparation/playbooks/env_uv.md
6. AUDIO_AGENT/tool_preparation/contracts/spec_validation.md
7. AUDIO_AGENT/tool_preparation/contracts/fixture_policy.md

Your goals:
- For the given tool, complete the phase-1 end-to-end onboarding
- For now, only evaluate end-to-end success or failure; node-level evaluation analysis is not required
- You must explicitly produce all required artifacts, as well as the conditional artifacts when their conditions are met
- If the workflow fails, you must enter DIAGNOSE / REPLAN, and decide whether to retry or escalate according to policy
- Blind retries are not allowed
- Unrecorded patches are not allowed
- Skipping VALIDATE_SPEC is not allowed

Execute according to the current workflow:
DISCOVER → CLASSIFY → PLAN → VALIDATE_SPEC → BUILD_ENV → FETCH_WEIGHTS → VALIDATE_IMPORT → VALIDATE_LOAD → VALIDATE_INFER → VALIDATE_CONTRACT → GENERATE_WRAPPER → SAVE_ARTIFACTS

Notes on the runtime validation target:
- Phase-1 runtime validation validates the repo-native entrypoint / minimal callable path
- The wrapper is generated after contract validation passes, and is used to integrate with the Audio Agent Framework

Your work requirements:
- All key decisions must be based on evidence and recorded into structured artifacts
- All failures must be classified
- All artifacts must be written to disk
- Finally output verdict.json, and briefly report: success / failure, which step it stopped at, and whether escalation was triggered
- Additionally output a "phase-1 target understanding" section, using 3-8 lines to explain:
  1. The minimal repo-native path to validate for the current tool
  2. Whether the current fixture is task-specific
  3. Whether the current backend choice is a hard constraint or an initial suggestion
  4. When the current run fails, whether to first check integration, dependency, or fixture mismatch

Below is the tool input for this run:

TOOL_INPUT
```

#### Step 2: TOOL_INPUT Format

```yaml
tool_id: owner/model-name
tool_name: ToolName
task_type: asr|s2tt|sd|ser|speech_enhancement|vad|omni|...
deployment_type: local|api

repo:
  url: https://github.com/owner/repo
  commit: null  # or specific commit hash

weights:
  source: huggingface|pip|release_or_pypi
  local_path: null
  required: true

environment_hint:
  preferred_backend: uv|conda|docker
  python_version: "3.11"
  requires_gpu: true|false
  system_packages: [ffmpeg, libsndfile1]

phase1_runtime_target:
  Validate the minimal callable path only:
  - confirm package is importable
  - load model with minimal config
  - run inference on fixture
  This phase does NOT require accuracy evaluation or production validation.

entrypoints:
  import_test: "import package"
  load_test: "model = package.load_model('tiny', 'cpu')"
  infer_test: "model.transcribe('tests/fixtures/shared/asr/en_16k.wav')"

fixture:
  audio: tests/fixtures/shared/asr/en_16k.wav
  task_specific: true|false
  fallback_allowed: true|false

io_contract:
  input_type: audio_path|text|json
  output_type: json|text
  primary_field: text|segments|labels
  required_fields: [field1, field2]
  nonempty_fields: [field1]
  json_serializable: true
```

**Example — local model (WhisperX):**

```yaml
tool_id: m-bain/whisperX
tool_name: WhisperX
task_type: asr
deployment_type: local

repo:
  url: https://github.com/m-bain/whisperX
  commit: null

weights:
  source: huggingface
  local_path: null
  required: true

environment_hint:
  preferred_backend: uv
  python_version: "3.11"
  requires_gpu: true
  system_packages: [ffmpeg]

entrypoints:
  import_test: "import whisperx"
  load_test: "import whisperx; model = whisperx.load_model('tiny', 'cpu')"
  infer_test: "result = model.transcribe('tests/fixtures/shared/asr/en_16k.wav')"

fixture:
  audio: tests/fixtures/shared/asr/en_16k.wav
  task_specific: false
  fallback_allowed: true

io_contract:
  input_type: audio_path
  output_type: json
  primary_field: segments
  required_fields: [segments]
  nonempty_fields: [segments]
  json_serializable: true
```

**Example — API-only model (Qwen3-Omni):**

```yaml
tool_id: qwen3-omni-flash
tool_name: Qwen3OmniAPI
task_type: omni
deployment_type: api

weights:
  source: api
  required: false

environment_hint:
  preferred_backend: api
  requires_gpu: false

entrypoints:
  import_test: "import openai"
  load_test: "client = openai.OpenAI(api_key=os.environ['DASHSCOPE_API_KEY'])"
  infer_test: "client.chat.completions.create(model='qwen3-omni-flash', messages=[...])"

io_contract:
  input_type: audio_path
  output_type: text
  primary_field: text
  required_fields: [text]
  nonempty_fields: [text]
  json_serializable: true
```

> 💡 **Recommended Agents**: Claude Code (Opus) for complex cases, Codex GPT-5.4 for repo analysis. Avoid agents with 60s timeout limits for large installations.

#### Step 3: Run

Send Initial Prompt + TOOL_INPUT to your AI agent. The agent will automatically:
- Discover repository structure
- Select appropriate backend
- Build isolated environment
- Validate import/load/infer/contract
- Generate wrapper files
- Save all artifacts

---

## Configured Tools

Tools successfully onboarded in the Audio Agent Framework:

| Tool | Task | Backend | Status | Notes |
|------|------|---------|--------|-------|
| [asr_qwen3](../audio_agent/tools/catalog/asr_qwen3/) | ASR | uv | ✅ Ready | Qwen3-ASR-1.7B with forced aligner support |
| [diarizen](../audio_agent/tools/catalog/diarizen/) | SD | conda | ✅ Ready | Speaker diarization (DiariZen) |
| [omni_captioner](../audio_agent/tools/catalog/omni_captioner/) | OMNI | uv | ✅ Ready | Qwen3-Omni captioner |

### Reusable Setup Scripts from SURE-EVAL

The SURE-EVAL repository (colleague's repo at `${REPO_ROOT}/sure/`) has already onboarded several additional tools with compatible MCP server implementations. The following tools have high-reusability setup scripts that can be adapted for the Audio Agent Framework:

| Tool | Task | Backend | SURE Path | Reusability |
|------|------|---------|-----------|-------------|
| whisperx | ASR | uv | `sure/src/sure_eval/models/whisperx/` | **High** — Whisper + alignment + diarization |
| deepfilternet | Speech Enhancement | uv | `sure/src/sure_eval/models/deepfilternet/` | **High** — DeepFilterNet2 noise suppression |
| librosa | Music IR | uv | `sure/src/sure_eval/models/librosa/` | **High** — Music feature extraction |
| snakers4_silero-vad | VAD | uv | `sure/src/sure_eval/models/snakers4_silero-vad/` | **High** — Voice activity detection |
| ffmpeg | Utility | uv | `sure/src/sure_eval/models/ffmpeg/` | **High** — Audio processing utility |
| asr_whisper | ASR | uv | `sure/src/sure_eval/models/asr_whisper/` | **High** — OpenAI Whisper base |
| asr_parakeet | ASR | uv | `sure/src/sure_eval/models/asr_parakeet/` | **High** — NVIDIA Parakeet CTC |
| whisper_large_v3_turbo | ASR | uv | `sure/src/sure_eval/models/whisper_large_v3_turbo/` | **High** — OpenAI Whisper Large V3 Turbo |
| parakeet_rnnt_1_1b | ASR | uv | `sure/src/sure_eval/models/parakeet_rnnt_1_1b/` | **High** — NVIDIA Parakeet RNNT 1.1B |
| qwen3_omni | OMNI | API | `sure/src/sure_eval/models/qwen3_omni/` | Partial — API-only omni model |

**How to reuse**: Copy the SURE tool's `model.py`, `server.py`, `pyproject.toml`, and `setup.sh` into `audio_agent/tools/catalog/{tool_name}/`, then adapt paths and `config.yaml` to match the Audio Agent Framework's conventions.

---

## Tool Directory Structure

Each tool directory should contain:

```
audio_agent/tools/catalog/<tool_name>/
├── model.spec.yaml         # Tool specification (optional but recommended)
├── model.py                # Wrapper implementation (recommended)
├── server.py               # MCP server
├── config.yaml             # MCP configuration
├── pyproject.toml          # Dependencies
├── __init__.py             # Package exports
├── setup.sh                # Environment setup
├── test_env.py             # Environment test (Python)
├── test_env.sh             # Environment test (shell wrapper)
├── README.md               # Tool documentation
└── artifacts/              # Generated artifacts
    ├── backend_choice.json
    ├── build.log
    ├── validation.log
    ├── verdict.json
    └── ...
```

### Differences from SURE-EVAL

The Audio Agent Framework has a few conventions that differ from SURE-EVAL:

1. **`config.yaml` is primary**: Unlike SURE, where `model.spec.yaml` is the primary spec, the Audio Agent Framework's catalog loader reads `config.yaml` for MCP tool registration. `model.spec.yaml` is **optional but strongly recommended** as an agent onboarding artifact.

2. **`model.py` is optional**: Many existing tools in the Audio Agent Framework (e.g., `asr_qwen3`) implement model logic directly in `server.py`. The `model.py` wrapper pattern is **recommended for complex models** but not mandatory for simple tools.

3. **Persistent UV**: The framework uses a persistent uv installation at `uv`. Setup scripts should prefer this path.

4. **Model Downloader**: HuggingFace models should be registered in `audio_agent/utils/model_downloader.py` and downloaded via `audio-agent-download-models --models <model-key>`.

5. **Bulk Verification**: Use `./verify_all_tools.sh` from the project root to verify all catalog tools at once.

---

## Manual Tool Development

If you prefer manual development over the agent workflow:

### Required Files

1. **README.md** - Tool documentation
2. **config.yaml** - MCP configuration
3. **server.py** - MCP server
4. **pyproject.toml** - Python dependencies
5. **setup.sh** - Environment setup
6. **test_env.py** / **test_env.sh** - Environment verification
7. **__init__.py** - Package exports

### Example config.yaml

```yaml
name: my_tool
task: ASR
description: "My ASR model"

server:
  command: [".venv/bin/python", "server.py"]
  working_dir: "."
  env:
    MODEL_PATH: "org/model-name"
  timeout: 300
```

### Environment Setup

See [`playbooks/env_uv.md`](./playbooks/env_uv.md) for complete setup.sh and test_env templates.

### Test Your Tool

```python
import asyncio
from audio_agent.tools.catalog import load_mcp_server_config
from audio_agent.tools.mcp import MCPClient

async def test():
    config = load_mcp_server_config("my_tool")
    client = MCPClient(config.command, config.working_dir)
    await client.start()
    tools = await client.list_tools()
    print(f"Available: {[t.name for t in tools]}")
    await client.stop()

asyncio.run(test())
```

---

## See Also

- [AGENTS.md](./AGENTS.md) - Master onboarding workflow
- [Policies](./policies/) - Constitution, evidence priority, retry & escalation
- [Playbooks](./playbooks/) - Environment strategy, failure taxonomy (+ API tools), preflight checklist
- [Contracts](./contracts/) - Spec + runtime validation, fixture policy
- [Templates](./templates/) - model.spec.yaml, verdict.json, artifact_manifest.json
