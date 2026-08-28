#!/usr/bin/env python3
"""
Bilibili MCP proxy server.

Wraps the vendored Node.js server `@xzxzzx/bilibili-mcp` (installed into
this directory's node_modules by setup.sh) and exposes it through the
catalog's uniform Python entrypoint. MCP traffic (newline-delimited
JSON-RPC on stdio) is relayed verbatim in both directions.

Quirks:
- The upstream CLI guards its entrypoint with
  `process.argv[1] === fileURLToPath(import.meta.url)`, which fails through
  npx/npm symlinked bins on Linux (the process exits silently). This proxy
  therefore invokes dist/cli.js by its real path - do not "simplify" this
  to an npx call.
- Credentials: most tools need Bilibili cookies. The user must run
  `node node_modules/@xzxzzx/bilibili-mcp/dist/cli.js setup` themselves in
  a local terminal (hidden prompt); cookie values must never pass through
  the agent or this repo.
- Optional local ASR (faster-whisper) for videos without subtitles can be
  installed during that same setup flow.

System requirements (checked by setup.sh, not installed by it): node.

Upstream tools provided (pass-through): get_credential_setup_instructions,
check_bilibili_credentials, check_mcp_update, get_video_info,
get_video_comments, get_video_transcript, get_video_metadata,
get_video_chapters, search_bilibili_videos, search_bilibili_creators,
list_bilibili_favorite_videos, get_bilibili_creator_content.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
NODE_ENTRY = (
    TOOL_DIR / "node_modules" / "@xzxzzx" / "bilibili-mcp" / "dist" / "cli.js"
)


def main() -> None:
    if not NODE_ENTRY.exists():
        print(
            f"error: vendored server not found at {NODE_ENTRY}; run ./setup.sh first",
            file=sys.stderr,
        )
        sys.exit(1)

    child = subprocess.Popen(
        ["node", str(NODE_ENTRY)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=dict(os.environ),
    )

    def pump_client_to_child() -> None:
        try:
            for line in sys.stdin.buffer:
                child.stdin.write(line)
                child.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        finally:
            try:
                child.stdin.close()
            except Exception:
                pass

    threading.Thread(target=pump_client_to_child, daemon=True).start()

    try:
        for line in child.stdout:
            sys.stdout.buffer.write(line)
            sys.stdout.buffer.flush()
    finally:
        child.wait()
        sys.exit(child.returncode or 0)


if __name__ == "__main__":
    main()
