#!/usr/bin/env python3
"""Environment smoke check for the WeSpeaker MCP tool.

Verifies that the local .venv has wespeaker + torch + soundfile installed and
that this tool's wrapper / server modules import cleanly. The wespeaker library
auto-downloads its `english` model into $WESPEAKER_HOME on first real use; this
script does not exercise the model, just the imports.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    print(f"Python: {sys.executable} ({sys.version.split()[0]})")

    print("Testing imports...")
    for name in ("wespeaker", "torch", "soundfile"):
        try:
            __import__(name)
            print(f"  ✓ {name}")
        except Exception as exc:
            print(f"  ✗ {name} import failed: {exc}")
            return 1

    print("Testing wrapper...")
    try:
        from model import ModelWrapper, SpeakerVerificationResult  # noqa: F401
        print("  ✓ ModelWrapper / SpeakerVerificationResult imported")
    except Exception as exc:
        print(f"  ✗ wrapper import failed: {exc}")
        return 1

    print("Testing server module...")
    try:
        from server import WeSpeakerServer  # noqa: F401
        print("  ✓ WeSpeakerServer imported")
    except Exception as exc:
        print(f"  ✗ server import failed: {exc}")
        return 1

    print()
    print("All environment tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
