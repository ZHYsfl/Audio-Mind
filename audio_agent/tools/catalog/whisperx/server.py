#!/usr/bin/env python3
"""WhisperX MCP Server for Audio Agent Framework.

MCP server implementation for WhisperX ASR with word-level timestamps.
Uses JSON-RPC protocol over stdin/stdout.
"""

from __future__ import annotations

import io
import json
import os
import sys
import warnings
from typing import Any

# Suppress warnings that might pollute stdout (MCP protocol requires clean stdout)
warnings.filterwarnings("ignore", message="torchcodec is not installed correctly", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Capture any stdout pollution during imports
_old_stdout = sys.stdout
sys.stdout = io.StringIO()

try:
    try:
        from .model import InferenceError, ModelLoadError, ModelWrapper
    except ImportError:
        from model import InferenceError, ModelLoadError, ModelWrapper
finally:
    # Restore stdout and discard any captured output
    _captured = sys.stdout.getvalue()
    sys.stdout = _old_stdout
    if _captured:
        print(f"[DEBUG] Captured during import: {_captured[:200]}", file=sys.stderr)


class WhisperXMCPServer:
    """MCP Server for WhisperX ASR."""
    
    def __init__(self):
        self._initialized = False
        self._model: ModelWrapper | None = None
    
    def _get_model(self) -> ModelWrapper:
        """Get or create model wrapper."""
        if self._model is None:
            self._model = ModelWrapper()
        return self._model
    
    def _handle_initialize(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle initialize request."""
        self._initialized = True
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "whisperx-mcp-server",
                    "version": "1.0.0",
                },
            },
        }
    
    def _handle_tools_list(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/list request."""
        if not self._initialized:
            return self._error_response(request.get("id"), -32001, "Server not initialized")
        
        return {
            "jsonrpc": "2.0",
            "id": request.get("id"),
            "result": {
                "tools": [
                    {
                        "name": "transcribe_whisperx",
                        "description": "Transcribe spoken content to text with timestamps using WhisperX. Useful as a fallback or cross-check against the primary ASR path, especially when an alternative alignment behavior is helpful for clean spoken dialogue. Treat output as recognized text rather than guaranteed ground truth; timestamp and word accuracy can degrade in singing, rap, overlapping speech, loud music/noise, child/cartoon/processed voices, strong accents or dialects, emotional shouting, reverberant audio, or very short clips. Do not use it to override strong frontend perception outside its domain.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "audio_path": {
                                    "type": "string",
                                    "description": "Path to the audio file to transcribe",
                                },
                                "language": {
                                    "type": "string",
                                    "description": "ISO 639-1 language code (e.g., 'en', 'zh', 'fr', 'de', 'ja', 'ko'). Use 2-letter codes, NOT full language names. If not provided, auto-detected.",
                                },
                            },
                            "required": ["audio_path"],
                        },
                    },
                    {
                        "name": "transcribe_whisperx_with_diarization",
                        "description": "Produce timestamped transcription with speaker labels in one integrated pass. Useful when you need a quick combined ASR-plus-diarization result for clean spoken dialogue, but less decomposed and controllable than running separate ASR and diarization tools. Treat speaker labels and timestamps as bounded evidence, not ground truth; trust less for songs, crowd scenes, overlapping speakers, noisy/music-backed audio, child/cartoon/processed voices, emotional speech, or short clips.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "audio_path": {
                                    "type": "string",
                                    "description": "Path to the audio file to transcribe",
                                },
                                "language": {
                                    "type": "string",
                                    "description": "ISO 639-1 language code (e.g., 'en', 'zh', 'fr', 'de', 'ja', 'ko'). Use 2-letter codes, NOT full language names. If not provided, auto-detected.",
                                },
                                "min_speakers": {
                                    "type": "integer",
                                    "description": "Minimum number of speakers (optional)",
                                },
                                "max_speakers": {
                                    "type": "integer",
                                    "description": "Maximum number of speakers (optional)",
                                },
                            },
                            "required": ["audio_path"],
                        },
                    },
                    {
                        "name": "healthcheck",
                        "description": "Check if WhisperX is properly configured and ready",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                ]
            },
        }
    
    def _handle_tools_call(self, request: dict[str, Any]) -> dict[str, Any]:
        """Handle tools/call request."""
        if not self._initialized:
            return self._error_response(request.get("id"), -32001, "Server not initialized")
        
        params = request.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        request_id = request.get("id")
        
        try:
            model = self._get_model()
            
            if tool_name == "transcribe_whisperx":
                audio_path = arguments.get("audio_path")
                language = arguments.get("language")
                
                if not audio_path:
                    return self._error_response(request_id, -32602, "Missing required argument: audio_path")
                
                # Log to stderr (not stdout - that's for JSON-RPC)
                print(f"Transcribing: {audio_path}", file=sys.stderr)
                
                result = model.predict(audio_path, language=language)
                return self._result_response(request_id, result)
            
            elif tool_name == "transcribe_whisperx_with_diarization":
                audio_path = arguments.get("audio_path")
                language = arguments.get("language")
                min_speakers = arguments.get("min_speakers")
                max_speakers = arguments.get("max_speakers")
                
                if not audio_path:
                    return self._error_response(request_id, -32602, "Missing required argument: audio_path")
                
                # Log to stderr (not stdout - that's for JSON-RPC)
                print(f"Transcribing with diarization: {audio_path}", file=sys.stderr)
                
                result = model.predict_with_diarization(
                    audio_path,
                    language=language,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                return self._result_response(request_id, result)
            
            elif tool_name == "healthcheck":
                result = model.healthcheck()
                return self._result_response(request_id, result)
            
            else:
                return self._error_response(request_id, -32601, f"Unknown tool: {tool_name}")
        
        except (ModelLoadError, InferenceError) as exc:
            return self._error_response(request_id, -32000, str(exc))
        except Exception as exc:
            return self._error_response(request_id, -32603, f"Internal error: {exc}")
    
    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Handle JSON-RPC request."""
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
            return self._error_response(request.get("id"), -32601, f"Method not found: {method}")
    
    def run(self) -> None:
        """Run the server."""
        # Ensure unbuffered output for proper JSON-RPC communication
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
            except json.JSONDecodeError as exc:
                error_resp = self._error_response(None, -32700, f"Parse error: {exc}")
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()
            except Exception as exc:
                request_id = request.get("id") if isinstance(request, dict) else None
                error_resp = self._error_response(request_id, -32603, f"Internal error: {exc}")
                sys.stdout.write(json.dumps(error_resp) + "\n")
                sys.stdout.flush()
    
    def _result_response(self, request_id: Any, payload: dict[str, Any]) -> dict[str, Any]:
        """Create successful result response in MCP format."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [
                    {"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=2)}
                ],
                "isError": False,
            },
        }
    
    def _error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        """Create error response."""
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def main() -> None:
    """Main entry point."""
    server = WhisperXMCPServer()
    server.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Log to stderr only - never to stdout (breaks JSON-RPC)
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
