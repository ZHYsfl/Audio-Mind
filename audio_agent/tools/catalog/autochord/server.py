#!/usr/bin/env python3
"""
MCP Server for autochord automatic chord recognition tool.

Communicates via stdin/stdout using JSON-RPC messages.
"""

from __future__ import annotations

import json
import sys
from typing import Any

try:
    from .model import ModelWrapper
except ImportError:
    from model import ModelWrapper


class AutochordMCPServer:
    def __init__(self) -> None:
        self._initialized = False
        self._wrapper: ModelWrapper | None = None

    def _get_wrapper(self) -> ModelWrapper:
        if self._wrapper is None:
            self._wrapper = ModelWrapper()
        return self._wrapper

    def _handle_initialize(self, request: dict[str, Any]) -> dict[str, Any]:
        self._initialized = True
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "autochord-mcp-server",
                    "version": "1.0.0",
                },
            },
        }

    def _handle_tools_list(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            return self._error_response(request.get("id"), -32001, "Server not initialized")

        tools = [
            {
                "name": "recognize_chords",
                "description": "Recognize time-localized chord labels in clear polyphonic harmonic audio using a 25-class vocabulary: N plus 12 major and 12 minor triads. Useful only as coarse triad-level chord evidence. It does not detect sevenths, diminished or augmented chords, inversions, slash chords, Roman numerals, modulation, Neo-Riemannian relations, or implied harmony in monophonic audio. If the output is N, low coverage, or conflicts with strong frontend perception, treat it as unsupported/uncertain rather than evidence for a different chord option.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to analyze."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "healthcheck",
                "description": "Check whether the autochord runtime is available.",
                "inputSchema": {"type": "object", "properties": {}}
            }
        ]

        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {"tools": tools}
        }

    def _handle_tools_call(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            return self._error_response(request.get("id"), -32001, "Server not initialized")

        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        request_id = request.get("id")

        try:
            if tool_name == "recognize_chords":
                audio_path = arguments.get("audio_path")
                if not audio_path:
                    raise ValueError("audio_path is required")
                result = self._get_wrapper().recognize_chords(audio_path).to_dict()

            elif tool_name == "healthcheck":
                result = self._get_wrapper().healthcheck()

            else:
                raise ValueError(f"Unknown tool: {tool_name}")

            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                    "isError": False,
                },
            }

        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": str(exc)}],
                    "isError": True,
                    "error": str(exc),
                },
            }

    def _error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        if method == "initialize":
            return self._handle_initialize(request)
        if method == "notifications/initialized":
            return None
        if method == "tools/list":
            return self._handle_tools_list(request)
        if method == "tools/call":
            return self._handle_tools_call(request)
        if method == "shutdown":
            return {"jsonrpc": "2.0", "id": request.get("id"), "result": {}}
        return self._error_response(request.get("id"), -32601, f"Method not found: {method}")

    def run(self) -> None:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                request = json.loads(line)
                response = self._handle_request(request)
                if response is not None:
                    sys.stdout.write(json.dumps(response) + "\n")
                    sys.stdout.flush()
            except json.JSONDecodeError as e:
                error_resp = self._error_response(None, -32700, f"Parse error: {e}")
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()
            except Exception as e:
                request_id = request.get("id") if isinstance(request, dict) else None
                error_resp = self._error_response(request_id, -32603, f"Internal error: {e}")
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()


def main() -> None:
    server = AutochordMCPServer()
    server.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
