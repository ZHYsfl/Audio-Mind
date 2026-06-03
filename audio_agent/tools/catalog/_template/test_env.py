#!/usr/bin/env python3
"""Environment smoke check template — copy and adapt for a new tool.

A passing test_env.py should:
  1. Confirm the Python interpreter is the venv's (sys.executable check).
  2. Import the tool's runtime dependencies.
  3. Optionally instantiate the model wrapper / load a tiny fixture.
"""
from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    print(f"sys.executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    # Confirm we are running inside this tool's .venv.
    expected_venv = Path(__file__).resolve().parent / ".venv"
    if expected_venv not in Path(sys.executable).resolve().parents:
        print(f"warning: sys.executable is not under {expected_venv}", file=sys.stderr)

    # Import the tool's server module to catch import-time errors early.
    try:
        import server  # type: ignore  # noqa: F401
        print("ok: server module imports cleanly")
    except Exception as exc:  # pragma: no cover - smoke test
        print(f"failed: cannot import server.py: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
