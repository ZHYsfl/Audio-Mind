#!/usr/bin/env python3
"""Environment smoke check for the bilibili tool.

Performs a real MCP handshake against server.py (initialize + tools/list)
and asserts the expected upstream tools are advertised. No network calls
and no credentials required.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

EXPECTED_TOOLS = {
    "search_bilibili_videos",
    "get_video_transcript",
    "get_video_info",
    "get_video_comments",
    "check_bilibili_credentials",
}


def main() -> int:
    print(f"sys.executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")

    expected_venv = Path(__file__).resolve().parent / ".venv"
    if expected_venv not in Path(sys.executable).resolve().parents:
        print(f"warning: sys.executable is not under {expected_venv}", file=sys.stderr)

    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2024-11-05", "capabilities": {},
                    "clientInfo": {"name": "test_env", "version": "0.1"}}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
    ]
    payload = "\n".join(json.dumps(r) for r in requests) + "\n"

    try:
        proc = subprocess.run(
            [sys.executable, "server.py"],
            input=payload, capture_output=True, text=True, timeout=90,
            cwd=Path(__file__).resolve().parent,
        )
    except subprocess.TimeoutExpired:
        print("failed: server.py did not respond within 90s", file=sys.stderr)
        return 1

    tools: list[str] = []
    for line in proc.stdout.splitlines():
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("id") == 1 and "error" in msg:
            print(f"failed: initialize error: {msg['error']}", file=sys.stderr)
            return 1
        if msg.get("id") == 2 and "result" in msg:
            tools = [t["name"] for t in msg["result"]["tools"]]

    if not tools:
        print(f"failed: no tools/list response. stderr: {proc.stderr[-500:]}", file=sys.stderr)
        return 1

    print(f"ok: {len(tools)} tools advertised: {', '.join(tools)}")
    missing = EXPECTED_TOOLS - set(tools)
    if missing:
        print(f"failed: expected tools missing: {sorted(missing)}", file=sys.stderr)
        return 1

    print("ok: all expected tools present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
