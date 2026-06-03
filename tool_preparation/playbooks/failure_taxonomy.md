# Failure Taxonomy

The first stage classifies failures into the following categories. Each category includes typical symptoms, common evidence, and a recommended repair direction.

## 1. python_dependency_missing

**Typical symptoms**:
```
ModuleNotFoundError: No module named 'nemo'
ImportError: cannot import name 'ASRModel'
```

**Common evidence**:
- `pip list` is missing the expected package
- `requirements.txt` was not installed correctly
- Package version mismatch

**Recommended repair direction**:
```bash
# Reinstall dependencies
uv pip install --python .venv/bin/python -r requirements.txt
# or
uv pip install --python .venv/bin/python nemo-toolkit[asr]
```

## 2. system_dependency_missing

**Typical symptoms**:
```
OSError: libsndfile.so.1: cannot open shared object file
RuntimeError: ffmpeg not found
```

**Common evidence**:
- System library file is missing
- Not installed via apt/yum

**Recommended repair direction**:
```bash
# Ubuntu/Debian
sudo apt-get install -y libsndfile1 ffmpeg

# Conda environment
conda install -c conda-forge libsndfile
```

## 3. cuda_version_mismatch

**Typical symptoms**:
```
RuntimeError: CUDA error: no kernel image is available
UserWarning: CUDA initialization: The NVIDIA driver is too old
```

**Common evidence**:
- `torch.version.cuda` does not match `nvidia-smi`
- PyTorch CUDA version is higher than what the driver supports

**Recommended repair direction**:
```bash
# Downgrade PyTorch to a version matching the driver
uv pip install --python .venv/bin/python torch==2.0.0 --index-url https://download.pytorch.org/whl/cu118

# or use the CPU version
uv pip install --python .venv/bin/python torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

## 4. wrong_python_version

**Typical symptoms**:
```
SyntaxError: invalid syntax (Python 3.8 encountering 3.10 syntax)
TypeError: unsupported operand type (type annotation issue)
```

**Common evidence**:
- Python version is below the requirement
- Type annotation syntax is incompatible

**Recommended repair direction**:
```bash
# Recreate the environment with the specified version
uv venv --python=python3.11
conda create -n env python=3.10
```

## 5. missing_weights

**Typical symptoms**:
```
FileNotFoundError: model/pytorch_model.bin not found
OSError: nvidia/parakeet-tdt-0.6b-v2 does not exist
```

**Common evidence**:
- Model cache directory is empty
- HuggingFace download was interrupted
- Path is misconfigured

**Recommended repair direction**:
```bash
# Re-download
huggingface-cli download nvidia/parakeet-tdt-0.6b-v2

# or use the Audio Agent model downloader
audio-agent-download-models --models <model-key>

# or set a local path
export MODEL_PATH=/path/to/local/model
```

## 6. wrong_entrypoint

**Typical symptoms**:
```
AttributeError: 'ASRModel' object has no attribute 'transcribe'
TypeError: transcribe() got an unexpected keyword argument 'language'
```

**Common evidence**:
- The API call convention does not match the model
- Incorrect method signature

**Recommended repair direction**:
- Consult the official documentation to confirm the entrypoint
- Adjust the wrapper method signature

## 7. config_not_set

**Typical symptoms**:
```
RuntimeError: DASHSCOPE_API_KEY not set
KeyError: 'MODEL_PATH'
```

**Common evidence**:
- Environment variable was not exported
- config.yaml is missing a required field

**Recommended repair direction**:
```bash
export DASHSCOPE_API_KEY="your-key"
# or write it into a .env file
```

## 8. runtime_backend_incompatible

**Typical symptoms**:
```
RuntimeError: operator torchvision::nms does not exist
AttributeError: np.sctypes was removed
RuntimeError: mentioning torchcodec during audio loading
```

**Common evidence**:
- Incompatibility between library versions
- NumPy 2.0 breaking changes
- PyTorch/Torchvision version mismatch
- torchaudio triggers the torchcodec/CUDA dependency chain

**Recommended repair direction**:
```bash
# Downgrade to a compatible version
uv pip install --python .venv/bin/python numpy==1.26.4
uv pip install --python .venv/bin/python torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
```

### Repair Hint: CPU-friendly torch audio models (e.g. Silero VAD)

When all of the following hold:
- `requires_gpu == false`
- the model is lightweight and CPU-friendly
- `torchaudio` is used mainly for audio I/O
- runtime error mentions `torchcodec`, `torchaudio load/save/info`, or CUDA shared libraries

Prefer the following repair order:
1. Switch to CPU-only PyTorch wheel index (https://download.pytorch.org/whl/cpu)
2. Use aligned torch / torchaudio versions instead of latest CUDA-enabled stack (e.g. 2.2.2+cpu)
3. Pin `numpy<2` if older PyTorch stack requires NumPy 1.x
4. Consider alternative audio I/O backend if the model logic itself does not depend on torchaudio internals

**Do NOT** default to a CUDA stack when the tool spec explicitly says `requires_gpu: false`.

## Failure Classification Flowchart

```
Failure occurs
    ↓
Inspect the error type
    ↓
├── ImportError / ModuleNotFoundError
│   └── python_dependency_missing
│
├── OSError (shared library)
│   └── system_dependency_missing
│
├── RuntimeError (CUDA)
│   └── cuda_version_mismatch
│
├── SyntaxError / TypeError (syntax)
│   └── wrong_python_version
│
├── FileNotFoundError (model file)
│   └── missing_weights
│
├── AttributeError / TypeError (API)
│   └── wrong_entrypoint
│
├── KeyError / RuntimeError (config)
│   └── config_not_set
│
└── RuntimeError / AttributeError (runtime)
    └── runtime_backend_incompatible
```

---

## API Tools

API-backed tools (DashScope / OpenAI / Gemini) skip local weights and GPU; they need a key, a base URL, and a healthcheck instead. Configure in `config.yaml`:

```yaml
api:
  base_url: "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
  api_key_env: "DASHSCOPE_API_KEY"
  timeout: 120
  retry: 3
```

Common API failures and their classes:

| Symptom | Class | Fix |
|---------|-------|-----|
| `401 Unauthorized` | `config_not_set` | API key missing/invalid — check the `api_key_env` variable is set |
| `ReadTimeout` | `runtime_backend_incompatible` | Raise `timeout`, or shorten the request audio |
| `429 Too Many Requests` | (retryable) | Exponential backoff (`sleep(2 ** attempt)`), bounded by the retry-and-escalation policy |
