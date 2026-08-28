#!/usr/bin/env python3
"""
yt-dlp MCP proxy server.

Wraps the vendored Node.js server `@kevinwatt/yt-dlp-mcp` (installed into
this directory's node_modules by setup.sh) and exposes it through the
catalog's uniform Python entrypoint.

Plain MCP traffic (newline-delimited JSON-RPC on stdio) is relayed verbatim,
except for one enforced policy:

- ytdlp_download_audio / ytdlp_download_video REQUIRE an extra argument
  `download_dir` (absolute path). The upstream tools have no destination
  parameter and would silently drop files into a static default directory;
  this proxy injects `download_dir` into their tools/list schemas as a
  required argument, rejects calls missing it without ever forwarding them,
  and after a successful download moves the file from the internal staging
  dir into `download_dir`, rewriting the path in the response text.

Other shim responsibilities:

- invokes the vendored Node entrypoint by real path (npx/npm bin shims are
  symlinks and fragile under some packages' entrypoint guards),
- prepends ~/.deno/bin and ~/.local/bin to PATH (yt-dlp 2026+ needs Deno as
  its JS runtime for YouTube; yt-dlp itself is expected at ~/.local/bin),
- sets YTDLP_DOWNLOADS_DIR to the internal staging dir
  (<repo>/downloads/.staging or $YTDLP_STAGING_DIR) purely as a transient
  landing zone; callers must always supply `download_dir`.

System requirements (checked by setup.sh, not installed by it):
  node, yt-dlp (`uv tool install yt-dlp`), deno, ffmpeg.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
NODE_ENTRY = (
    TOOL_DIR / "node_modules" / "@kevinwatt" / "yt-dlp-mcp" / "lib" / "index.mjs"
)
# catalog/yt_dlp -> catalog -> tools -> audio_agent -> repo root
REPO_ROOT = TOOL_DIR.parents[3]
STAGING_DIR = Path(
    os.environ.get("YTDLP_STAGING_DIR", REPO_ROOT / "downloads" / ".staging")
)
EXTRA_PATH_DIRS = [Path.home() / ".deno" / "bin", Path.home() / ".local" / "bin"]

DOWNLOAD_TOOLS = {"ytdlp_download_audio", "ytdlp_download_video"}
DOWNLOAD_DIR_SCHEMA = {
    "type": "string",
    "description": "REQUIRED. Absolute path of the directory the downloaded "
    "file must end up in. Created if missing. The download is refused "
    "without it.",
}
# Upstream success message: `... downloaded as "<basename>" to <dirname>`
SUCCESS_RE = re.compile(r'^(?P<head>\w+ successfully downloaded as "(?P<name>.+?)" to )(?P<dir>.+)$')


def _build_child_env() -> dict[str, str]:
    env = dict(os.environ)
    path = env.get("PATH", "")
    prepend = [str(p) for p in EXTRA_PATH_DIRS if p.is_dir() and str(p) not in path]
    if prepend:
        env["PATH"] = os.pathsep.join(prepend + [path])
    env["YTDLP_DOWNLOADS_DIR"] = str(STAGING_DIR)
    return env


def _error_result(request_id, message: str) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [{"type": "text", "text": f"Error: {message}"}],
            "isError": True,
        },
    }


class Proxy:
    """Stateful line-by-line MCP relay enforcing the download_dir policy."""

    def __init__(self, child: subprocess.Popen) -> None:
        self.child = child
        self.pending_downloads: dict = {}  # request id -> target dir (Path)

    # ---- client -> child ----

    def transform_request(self, msg: dict) -> dict | None:
        """Rewrite/validate an outbound request. Returns None to suppress it
        (the proxy answers directly), or the (possibly modified) message."""
        if msg.get("method") != "tools/call":
            return msg
        params = msg.get("params") or {}
        if params.get("name") not in DOWNLOAD_TOOLS:
            return msg

        arguments = params.get("arguments") or {}
        raw_dir = arguments.pop("download_dir", None)
        if not raw_dir or not str(raw_dir).strip():
            return self._reject(msg["id"], "download_dir is required (absolute path); "
                                          "this tool never writes to a default location.")
        target = Path(str(raw_dir)).expanduser()
        if not target.is_absolute():
            return self._reject(msg["id"], f"download_dir must be absolute, got: {raw_dir}")
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return self._reject(msg["id"], f"cannot create download_dir {target}: {exc}")

        self.pending_downloads[msg["id"]] = target
        params["arguments"] = arguments
        msg["params"] = params
        return msg

    def _reject(self, request_id, message: str) -> None:
        self._emit(_error_result(request_id, message))
        return None

    # ---- child -> client ----

    def transform_response(self, msg: dict) -> dict:
        result = msg.get("result")
        if isinstance(result, dict) and isinstance(result.get("tools"), list):
            self._augment_tool_schemas(result["tools"])
            return msg

        request_id = msg.get("id")
        if request_id not in self.pending_downloads:
            return msg
        target = self.pending_downloads.pop(request_id)
        if not isinstance(result, dict) or result.get("isError"):
            return msg

        content = result.get("content") or []
        if not content or content[0].get("type") != "text":
            return msg
        m = SUCCESS_RE.match(content[0]["text"].strip())
        if not m:
            return msg

        src = Path(m.group("dir")) / m.group("name")
        dst = target / m.group("name")
        try:
            shutil.move(str(src), str(dst))
        except OSError as exc:
            return _error_result(
                request_id,
                f"download succeeded but moving {src} to {dst} failed: {exc}",
            )
        content[0]["text"] = m.group("head") + str(target)
        return msg

    @staticmethod
    def _augment_tool_schemas(tools: list) -> None:
        for tool in tools:
            if tool.get("name") not in DOWNLOAD_TOOLS:
                continue
            schema = tool.setdefault("inputSchema", {})
            schema.setdefault("properties", {})["download_dir"] = dict(DOWNLOAD_DIR_SCHEMA)
            required = schema.setdefault("required", [])
            if "download_dir" not in required:
                required.append("download_dir")
            # The upstream description still advertises a default ~/Downloads
            # location, which this proxy overrides; rewrite it so agents are
            # not misled about where files end up.
            desc = tool.get("description") or ""
            # Drop any line referencing the default Downloads folder.
            desc = re.sub(
                r"^.*(?:Downloads folder|~/Downloads).*$\n?", "",
                desc, flags=re.MULTILINE,
            ).strip()
            tool["description"] = desc + (
                "\n\nREQUIRED argument: 'download_dir' (absolute path). "
                "The file is downloaded to an internal staging area and then "
                "moved into 'download_dir'; the call is refused without it. "
                "There is no default destination."
            )

    # ---- pump loops ----

    def _emit(self, msg: dict) -> None:
        sys.stdout.buffer.write(json.dumps(msg, ensure_ascii=False).encode() + b"\n")
        sys.stdout.buffer.flush()

    def pump_client_to_child(self) -> None:
        try:
            for line in sys.stdin.buffer:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    self.child.stdin.write(line)
                    self.child.stdin.flush()
                    continue
                out = self.transform_request(msg)
                if out is None:
                    continue
                self.child.stdin.write(
                    json.dumps(out, ensure_ascii=False).encode() + b"\n"
                )
                self.child.stdin.flush()
        except (BrokenPipeError, ValueError):
            pass
        finally:
            try:
                self.child.stdin.close()
            except Exception:
                pass

    def pump_child_to_client(self) -> None:
        for line in self.child.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                sys.stdout.buffer.write(line)
                sys.stdout.buffer.flush()
                continue
            self._emit(self.transform_response(msg))


def main() -> None:
    if not NODE_ENTRY.exists():
        print(
            f"error: vendored server not found at {NODE_ENTRY}; run ./setup.sh first",
            file=sys.stderr,
        )
        sys.exit(1)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    child = subprocess.Popen(
        ["node", str(NODE_ENTRY)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=_build_child_env(),
    )
    proxy = Proxy(child)

    threading.Thread(target=proxy.pump_client_to_child, daemon=True).start()

    try:
        proxy.pump_child_to_client()
    finally:
        child.wait()
        sys.exit(child.returncode or 0)


if __name__ == "__main__":
    main()
