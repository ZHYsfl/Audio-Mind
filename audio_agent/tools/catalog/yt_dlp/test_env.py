#!/usr/bin/env python3
"""Environment smoke check for the yt_dlp tool.

Performs a real MCP handshake against server.py (initialize + tools/list)
and asserts the expected upstream tools are advertised. No network calls.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

EXPECTED_TOOLS = {
    "ytdlp_search_videos",
    "ytdlp_download_audio",
    "ytdlp_download_video",
    "ytdlp_get_video_metadata",
    "ytdlp_download_transcript",
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
        # Policy check: download without download_dir must be rejected by the
        # proxy itself (no network involved).
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "ytdlp_download_audio",
                    "arguments": {"url": "https://example.com/x"}}},
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
    download_schema_ok = False
    policy_rejection_ok = False
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
            for t in msg["result"]["tools"]:
                if t["name"] == "ytdlp_download_audio":
                    schema = t.get("inputSchema", {})
                    download_schema_ok = "download_dir" in schema.get("required", [])
        if msg.get("id") == 3 and "result" in msg:
            r = msg["result"]
            text = (r.get("content") or [{}])[0].get("text", "")
            policy_rejection_ok = bool(r.get("isError")) and "download_dir" in text

    if not tools:
        print(f"failed: no tools/list response. stderr: {proc.stderr[-500:]}", file=sys.stderr)
        return 1

    print(f"ok: {len(tools)} tools advertised: {', '.join(tools)}")
    missing = EXPECTED_TOOLS - set(tools)
    if missing:
        print(f"failed: expected tools missing: {sorted(missing)}", file=sys.stderr)
        return 1

    print("ok: all expected tools present")

    if not download_schema_ok:
        print("failed: ytdlp_download_audio schema does not require download_dir", file=sys.stderr)
        return 1
    print("ok: download_dir injected as required argument")

    if not policy_rejection_ok:
        print("failed: download without download_dir was not rejected by the proxy", file=sys.stderr)
        return 1
    print("ok: download without download_dir rejected (policy enforced)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
