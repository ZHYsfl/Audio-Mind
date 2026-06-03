from __future__ import annotations

import json
import logging
import sys
from typing import Any


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


class MCPServer:
    def __init__(self) -> None:
        self._wrapper = None

    def _wrapper_instance(self):
        if self._wrapper is None:
            try:
                from .model import ModelWrapper
            except ImportError:
                from model import ModelWrapper

            self._wrapper = ModelWrapper()
        return self._wrapper

    def handle_initialize(self, request: dict[str, Any]) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fireredvad-mcp", "version": "1.0.0"},
            },
        }

    def handle_tools_list(self, request: dict[str, Any]) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": [
                    {
                        "name": "fireredvad_predict",
                        "description": "Voice activity detection tool for speech/non-speech segmentation. Returns timestamps of likely speech regions. Useful for routing speech-focused downstream tools, but it does not transcribe, identify speakers, or determine semantic content. Empty or boundary-shifted results can reflect model sensitivity, noise, or preprocessing; if speech presence/boundaries are central, cross-check with ASR, another VAD, or a more focused clip.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "audio_path": {
                                    "type": "string",
                                    "description": "Absolute or relative path to a WAV file.",
                                }
                            },
                            "required": ["audio_path"],
                        },
                    },
                    {
                        "name": "fireredvad_aed",
                        "description": "Coarse audio event detector for speech, singing, and music regions. Returns timestamps and aggregate ratios for those broad classes only; it is not a general sound-event classifier or source-identity tool. Absence of a class is model-dependent evidence, not proof; cross-check when speech/music/singing presence is decisive.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "audio_path": {
                                    "type": "string",
                                    "description": "Absolute or relative path to a WAV file.",
                                }
                            },
                            "required": ["audio_path"],
                        },
                    },
                    {
                        "name": "healthcheck",
                        "description": "Check whether the wrapper has loaded the model.",
                        "inputSchema": {"type": "object", "properties": {}},
                    },
                ]
            },
        }

    def handle_tools_call(self, request: dict[str, Any]) -> dict[str, Any]:
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        try:
            if tool_name == "fireredvad_predict":
                result = self._wrapper_instance().predict(arguments["audio_path"])
                return self._result_response(request.get("id"), result.to_json())
            if tool_name == "fireredvad_aed":
                result = self._wrapper_instance().predict_aed(arguments["audio_path"])
                return self._result_response(request.get("id"), result.to_json())
            if tool_name == "healthcheck":
                status = self._wrapper_instance().healthcheck()
                return self._result_response(request.get("id"), json.dumps(status))
            return self._error_response(
                request.get("id"), -32601, f"Unknown tool: {tool_name}"
            )
        except Exception as exc:  # noqa: BLE001
            import traceback
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.exception(f"Tool execution failed: {error_msg}")
            # Include traceback in the error data for better debugging
            return {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {
                    "code": -32000,
                    "message": error_msg,
                    "data": {
                        "traceback": traceback.format_exc(),
                        "tool": tool_name,
                        "arguments": arguments,
                    }
                }
            }

    def _result_response(self, request_id: Any, text: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"content": [{"type": "text", "text": text}]},
        }

    def _error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        method = request.get("method")
        if method == "initialize":
            return self.handle_initialize(request)
        if method == "tools/list":
            return self.handle_tools_list(request)
        if method == "tools/call":
            return self.handle_tools_call(request)
        if method == "notifications/initialized":
            return None
        return self._error_response(
            request.get("id"), -32601, f"Method not found: {method}"
        )

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            request = json.loads(line)
            response = self.handle_request(request)
            if response is not None:
                print(json.dumps(response), flush=True)


if __name__ == "__main__":
    MCPServer().run()
