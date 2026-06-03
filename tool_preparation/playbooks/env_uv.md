# UV Environment Strategy Playbook

## When to Use UV

| Condition | Use UV |
|------|---------|
| Pure Python dependencies | ✅ Recommended |
| Dependencies are mainly PyPI packages | ✅ Recommended |
| No complex C++ extensions | ✅ Recommended |
| No system library dependencies | ✅ Recommended |
| No CUDA compilation requirements | ✅ Recommended |

## Environment Creation

```bash
cd audio_agent/tools/catalog/{tool_name}

# Create the environment
uv venv --python=python3.11

# Install dependencies
uv pip install --python .venv/bin/python -e .
```

> **Critical**: Always use `--python .venv/bin/python` with uv pip install to ensure packages go into the venv, not the base environment.

## Lock/Sync Conventions

```bash
# Export exact dependencies
uv pip freeze --python .venv/bin/python > requirements.lock

# Restore from lock
uv pip install --python .venv/bin/python -r requirements.lock
```

## Common Failures and Fixes

### 1. System PyTorch conflicts with the virtual environment

**Symptom**: `ModuleNotFoundError: No module named 'torch._utils'`

**Cause**: UV environment isolation prevents access to the system-installed PyTorch

**Fix**:
```bash
# Reinstall PyTorch inside the virtual environment
uv pip install --python .venv/bin/python torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2. NumPy version conflict

**Symptom**: `AttributeError: np.sctypes was removed in the NumPy 2.0`

**Cause**: Libraries such as NeMo do not support NumPy 2.0

**Fix**:
```bash
uv pip install --python .venv/bin/python numpy==1.26.4
```

### 3. Torchvision version mismatch

**Symptom**: `RuntimeError: operator torchvision::nms does not exist`

**Fix**:
```bash
# Install a torchvision version that matches PyTorch
uv pip install --python .venv/bin/python torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu
```

### 4. "Multiple top-level modules discovered"

**Symptom**:
```
error: Multiple top-level modules discovered in a flat-layout: ['server', 'test_env'].
```

**Cause**: Missing `[tool.setuptools]` section in `pyproject.toml`.

**Fix**:
Add to `pyproject.toml`:
```toml
[tool.setuptools]
py-modules = ["server", "model"]  # Exclude test_env.py
```

## Verification Commands

```bash
# Verify the Python version
.venv/bin/python --version

# Verify key packages
.venv/bin/python -c "import torch; print(torch.__version__)"
.venv/bin/python -c "import numpy; print(numpy.__version__)"
```

## References

- For complete setup.sh templates, see Section 3 below.
- For persistent uv configuration, see project root `setup_tools_uv_persistent.sh`.
