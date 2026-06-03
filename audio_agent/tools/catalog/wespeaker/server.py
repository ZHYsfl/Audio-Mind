#!/usr/bin/env python3
"""
WeSpeaker MCP Server.

MCP server for speaker verification using the WeSpeaker English model.
Provides tools for computing similarity scores between audio files.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

try:
    from model import ModelWrapper, contract_result_to_json
except ImportError:
    from .model import ModelWrapper, contract_result_to_json


class WeSpeakerServer:
    """MCP Server for WeSpeaker speaker verification."""
    
    def __init__(self):
        self._initialized = False
        self._wrapper: ModelWrapper | None = None
        
        # Default configuration. WESPEAKER_HOME is set by config.yaml from
        # ${AUDIO_AGENT_MODELS_DIR}/wespeaker; fall back to the same default.
        from pathlib import Path as _P
        _repo_root = _P(__file__).resolve().parents[4]
        _models_dir = os.environ.get("AUDIO_AGENT_MODELS_DIR") or str(_repo_root / "models")
        self._cache_dir = os.environ.get(
            "WESPEAKER_HOME",
            os.path.join(_models_dir, "wespeaker"),
        )
        self._model_name = os.environ.get("WESPEAKER_MODEL", "english")
        self._device = os.environ.get("WESPEAKER_DEVICE", "cpu")
        
        # Tool definitions
        self._tools = [
            {
                "name": "speaker_verify",
                "description": "Verify whether two audio files are likely from the same speaker. Returns a similarity score that is useful for speaker-consistency checks, not for open-set speaker identification by name.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "enrollment_audio": {
                            "type": "string",
                            "description": "Path to enrollment/reference audio file"
                        },
                        "trial_audio": {
                            "type": "string",
                            "description": "Path to trial/test audio file"
                        }
                    },
                    "required": ["enrollment_audio", "trial_audio"]
                }
            },
            {
                "name": "healthcheck",
                "description": "Check if the WeSpeaker model is loaded and ready",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    def _get_wrapper(self) -> ModelWrapper:
        """Get or create the model wrapper instance."""
        if self._wrapper is None:
            self._wrapper = ModelWrapper({
                "cache_dir": self._cache_dir,
                "model_name": self._model_name,
                "device": self._device,
            })
        return self._wrapper
    
    def run(self) -> None:
        """Run the server."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                request = json.loads(line)
                response = self._handle_request(request)
                if response:
                    self._send_response(response)
            except json.JSONDecodeError as e:
                self._send_error(None, -32700, f"Parse error: {e}")
            except Exception as e:
                request_id = request.get("id") if isinstance(request, dict) else None
                self._send_error(request_id, -32603, f"Internal error: {e}")
    
    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle JSON-RPC request."""
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})
        
        if method == "initialize":
            return self._handle_initialize(request_id, params)
        elif method == "notifications/initialized":
            return None
        elif method == "tools/list":
            return self._handle_tools_list(request_id)
        elif method == "tools/call":
            return self._handle_tools_call(request_id, params)
        elif method == "shutdown":
            return self._handle_shutdown(request_id)
        else:
            return self._error_response(request_id, -32601, f"Method not found: {method}")
    
    def _handle_initialize(self, request_id: Any, params: dict) -> dict[str, Any]:
        """Handle initialize request."""
        self._initialized = True
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "wespeaker-server",
                    "version": "1.0.0"
                }
            }
        }
    
    def _handle_tools_list(self, request_id: Any) -> dict[str, Any]:
        """Handle tools/list request."""
        if not self._initialized:
            return self._error_response(request_id, -32001, "Server not initialized")
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": self._tools}
        }
    
    def _handle_tools_call(self, request_id: Any, params: dict) -> dict[str, Any]:
        """Handle tools/call request."""
        if not self._initialized:
            return self._error_response(request_id, -32001, "Server not initialized")
        
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            result = self._execute_tool(tool_name, arguments)
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            print(f"Tool execution error: {error_msg}\n{tb}", file=sys.stderr)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {error_msg}"}],
                    "isError": True,
                    "error": error_msg,
                    "traceback": tb
                }
            }
    
    def _execute_tool(self, tool_name: str, arguments: dict) -> dict[str, Any]:
        """Execute a tool."""
        if tool_name == "speaker_verify":
            return self._speaker_verify(arguments)
        elif tool_name == "healthcheck":
            return self._healthcheck(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _speaker_verify(self, arguments: dict) -> dict[str, Any]:
        """Run speaker verification."""
        enrollment_audio = arguments.get("enrollment_audio", "")
        trial_audio = arguments.get("trial_audio", "")
        
        if not enrollment_audio:
            raise ValueError("enrollment_audio is required")
        if not trial_audio:
            raise ValueError("trial_audio is required")
        
        if not Path(enrollment_audio).exists():
            raise FileNotFoundError(f"Enrollment audio not found: {enrollment_audio}")
        if not Path(trial_audio).exists():
            raise FileNotFoundError(f"Trial audio not found: {trial_audio}")
        
        print(f"Verifying speaker: enrollment={enrollment_audio}, trial={trial_audio}", file=sys.stderr)
        
        try:
            wrapper = self._get_wrapper()
            result = wrapper.predict({
                "enrollment_audio": enrollment_audio,
                "trial_audio": trial_audio
            })
            
            # Format response
            similarity = result.similarity_score
            # Use a conservative decision band so borderline scores remain reviewable.
            verdict = "SAME SPEAKER" if similarity > 0.75 else "DIFFERENT SPEAKERS"
            if 0.55 <= similarity <= 0.75:
                verdict = "UNCERTAIN"
            
            result_text = f"""Speaker Verification Result

**Similarity Score:** {similarity:.4f}
**Verdict:** {verdict}

**Details:**
- Enrollment Audio: {result.enrollment_audio}
- Trial Audio: {result.trial_audio}
- Model: {result.model_name}
- Device: {result.device}

**Interpretation:**
- Score > 0.75: Likely same speaker
- Score < 0.55: Likely different speakers  
- 0.55-0.75: Uncertain, may need manual review
"""
            
            return {
                "content": [{"type": "text", "text": result_text}],
                "isError": False,
                "result_json": result.to_dict()
            }
            
        except Exception as e:
            print(f"Speaker verification failed: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"Speaker verification failed: {e}") from e
    
    def _healthcheck(self, arguments: dict) -> dict[str, Any]:
        """Check model health status."""
        wrapper = self._get_wrapper()
        status = wrapper.healthcheck()
        
        status_text = f"""WeSpeaker Health Status

**Status:** {status['status']}
**Model Loaded:** {status['model_loaded']}
**Model Name:** {status['message']}
**Cache Directory:** {status['cache_dir']}
**Cache Present:** {status['cache_present']}
"""
        
        return {
            "content": [{"type": "text", "text": status_text}],
            "isError": False,
            "health": status
        }
    
    def _handle_shutdown(self, request_id: Any) -> dict[str, Any]:
        """Handle shutdown request."""
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    
    def _error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        """Create error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message}
        }
    
    def _send_response(self, response: dict[str, Any]) -> None:
        """Send response to stdout."""
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    
    def _send_error(self, request_id: Any, code: int, message: str) -> None:
        """Send error response."""
        self._send_response(self._error_response(request_id, code, message))


def main():
    """Main entry point."""
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    server = WeSpeakerServer()
    server.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
