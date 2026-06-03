# Template tool

Copy this directory to start a new MCP tool:

```bash
cp -r _template my_tool
cd my_tool
# Edit pyproject.toml, config.yaml, server.py (and model.py if you wrap a library)
./setup.sh && ./test_env.sh
```

## Files in a clean tool

| File | Purpose |
|---|---|
| `__init__.py` | Marks the directory as a package |
| `config.yaml` | MCP server registration (command, working_dir, env, resources, tool schemas) |
| `pyproject.toml` | This tool's pip deps for its own `.venv` |
| `server.py` | MCP server entrypoint. Put a top-of-file docstring describing any tool-specific quirks (license, GPU needs, env vars). |
| `model.py` | Optional thin wrapper around the underlying library |
| `setup.sh` | Creates `.venv` (uv) — runs once on host |
| `test_env.sh` | Smoke check; invoked by `./verify_all_tools.sh` |
| `test_env.py` | Python-level checks called from `test_env.sh` |

`.venv/`, `*_tool.egg-info/`, `__pycache__/` are gitignored.

For the full onboarding workflow (harness-first agent-assisted), see
[tool_preparation/README.md](../../../../tool_preparation/README.md).
