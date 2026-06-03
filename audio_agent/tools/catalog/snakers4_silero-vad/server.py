#!/usr/bin/env python3
"""MCP Server for Silero VAD Model."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

try:
    from .model import VADModel, VADResult
except ImportError:
    from model import VADModel, VADResult


class SileroVADMCPServer:
    def __init__(self):
        self._initialized = False
        self.model: VADModel | None = None
        import torch
        _env_device = os.environ.get("MODEL_DEVICE", "cpu").lower()
        if _env_device == "auto":
            self.model_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.model_device = _env_device

    def _get_model(self) -> VADModel:
        if self.model is None:
            self.model = VADModel(device=self.model_device)
        return self.model

    def _handle_initialize(self, request: dict[str, Any]) -> dict[str, Any]:
        self._initialized = True
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "silero-vad-mcp-server",
                    "version": "1.0.0"
                }
            }
        }

    def _handle_tools_list(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            return self._error_response(request.get("id"), -32001, "Server not initialized")
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": [
                    {
                        "name": "vad_predict",
                        "description": "Lightweight voice activity detection for quick speech/non-speech segmentation. Useful as a faster, lighter fallback to stronger VAD tools when speed matters more than precision. Empty or boundary-shifted results are not decisive; if speech presence or boundaries matter, cross-check with stronger VAD, ASR, or a focused clip.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "audio_path": {
                                    "type": "string",
                                    "description": "Path to the audio file"
                                },
                                "sampling_rate": {
                                    "type": "integer",
                                    "description": "Target sampling rate (default: 16000)",
                                    "default": 16000
                                }
                            },
                            "required": ["audio_path"]
                        }
                    },
                    {
                        "name": "healthcheck",
                        "description": "Check if the VAD model is ready",
                        "inputSchema": {
                            "type": "object",
                            "properties": {}
                        }
                    }
                ]
            }
        }

    def _handle_tools_call(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self._initialized:
            return self._error_response(request.get("id"), -32001, "Server not initialized")
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        request_id = request.get("id")

        try:
            if tool_name == "vad_predict":
                return self._handle_vad_predict(request_id, arguments)
            elif tool_name == "healthcheck":
                return self._handle_healthcheck(request_id)
            else:
                return self._error_response(
                    request_id,
                    -32601,
                    f"Unknown tool: {tool_name}"
                )
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": str(e)}],
                    "isError": True,
                    "error": str(e)
                }
            }

    def _handle_vad_predict(self, request_id: Any, arguments: dict[str, Any]) -> dict[str, Any]:
        model = self._get_model()
        audio_path = arguments.get("audio_path")
        sampling_rate = arguments.get("sampling_rate", 16000)

        if not audio_path:
            return self._error_response(
                request_id,
                -32602,
                "Missing required parameter: audio_path"
            )

        result = model.predict(audio_path, sampling_rate=sampling_rate)

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": result.to_json()
                    }
                ],
                "isError": False
            }
        }

    def _handle_healthcheck(self, request_id: Any) -> dict[str, Any]:
        model = self._get_model()
        status = model.healthcheck()

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(status)
                    }
                ],
                "isError": False
            }
        }

    def _error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }

    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")

        if method == "initialize":
            return self._handle_initialize(request)
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            return self._handle_tools_list(request)
        elif method == "tools/call":
            return self._handle_tools_call(request)
        elif method == "shutdown":
            return {"jsonrpc": "2.0", "id": request.get("id"), "result": {}}
        else:
            return self._error_response(
                request.get("id"),
                -32601,
                f"Method not found: {method}"
            )

    def run(self):
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
                error_response = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {e}"
                    }
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()
            except Exception as e:
                request_id = request.get("id") if isinstance(request, dict) else None
                error_response = {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {e}"
                    }
                }
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()


def main():
    server = SileroVADMCPServer()
    server.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
