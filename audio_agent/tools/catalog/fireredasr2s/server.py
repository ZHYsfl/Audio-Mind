#!/usr/bin/env python3
"""FireRedASR2S MCP Server for Audio Agent Framework.

MCP server implementation for FireRedASR2S AED ASR.
Uses JSON-RPC protocol over stdin/stdout.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


class FireRedAsr2sMCPServer:
    """MCP Server for FireRedASR2S ASR."""
    
    def __init__(self):
        self._initialized = False
        self._model = None
        # MODEL_PATH is set by config.yaml from ${AUDIO_AGENT_MODELS_DIR}/FireRedASR-AED-L;
        # fall back to the same default when called outside the MCP harness.
        from pathlib import Path as _P
        _repo_root = _P(__file__).resolve().parents[4]
        _models_dir = os.environ.get("AUDIO_AGENT_MODELS_DIR") or str(_repo_root / "models")
        self._model_path = os.environ.get(
            "MODEL_PATH",
            os.path.join(_models_dir, "FireRedASR-AED-L"),
        )
        self._device = os.environ.get("DEVICE", "cpu")
        
        # Tool definitions
        self._tools = [
            {
                "name": "transcribe_fireredasr",
                "description": "Transcribe spoken content in an audio file to text using FireRedASR2S-AED. Useful for Chinese, dialectal Chinese, English, and code-switching speech in clean spoken-dialogue settings. Treat the output as recognized text rather than guaranteed ground truth, especially for singing, rap, overlapping speech, loud music/noise, child/cartoon/processed voices, emotional shouting, reverberant audio, or very short clips. Do not use it to override strong frontend perception outside its domain.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to transcribe (16kHz 16-bit mono PCM WAV recommended)"
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "transcribe_fireredasr_with_timestamps",
                "description": "Transcribe spoken content to text with word-level timestamps. Useful when both lexical content and temporal grounding are needed in clean spoken-dialogue settings. Timestamp quality depends on recognition and alignment and should not be over-read as perfect segmentation; boundary times can be wrong, missing, or shifted in singing, overlapping speech, noisy/music-backed audio, child/cartoon/processed voices, emotional shouting, reverberant audio, or very short clips.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file (16kHz 16-bit mono PCM WAV recommended)"
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "lyric_asr",
                "description": "Transcribe sung lyrics from music audio to text. Useful for lyric-content recovery in singing-heavy audio, especially Chinese and English songs. The output can support lyric reasoning, but should not be treated as guaranteed ground truth or over-read as a full music-structure, melody, emotion, singer-identity, or timing analysis tool. If lyrics conflict with strong frontend perception, treat the disagreement as uncertainty.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file containing singing/music (16kHz 16-bit mono PCM WAV recommended)"
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "healthcheck",
                "description": "Check if FireRedASR2S is properly configured and ready",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    def _load_model(self) -> None:
        """Lazy load the model."""
        if self._model is not None:
            return
        
        try:
            from model import ModelWrapper, ModelLoadError
        except ImportError as e:
            raise RuntimeError(
                f"Failed to import ModelWrapper. Ensure environment is set up: {e}"
            ) from e
        
        print(f"Loading FireRedASR2S model from: {self._model_path}", file=sys.stderr)
        
        try:
            self._model = ModelWrapper(
                config={
                    "model_path": self._model_path,
                    "device": self._device,
                }
            )
            # Don't load yet - let lazy loading happen on first predict
            print("Model wrapper created (will load on first use)", file=sys.stderr)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize model: {e}") from e
    
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
                    "name": "fireredasr2s-server",
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
            print(f"Tool execution error: {error_msg}", file=sys.stderr)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": f"Error: {error_msg}"}],
                    "isError": True,
                    "error": error_msg
                }
            }
    
    def _execute_tool(self, tool_name: str, arguments: dict) -> dict[str, Any]:
        """Execute a tool."""
        self._load_model()
        
        if tool_name == "transcribe_fireredasr":
            return self._transcribe(arguments, include_timestamps=False)
        elif tool_name == "transcribe_fireredasr_with_timestamps":
            return self._transcribe(arguments, include_timestamps=True)
        elif tool_name == "lyric_asr":
            return self._transcribe(arguments, include_timestamps=False)
        elif tool_name == "healthcheck":
            return self._healthcheck()
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _transcribe(
        self, 
        arguments: dict, 
        include_timestamps: bool = False
    ) -> dict[str, Any]:
        """Transcribe audio to text using FireRedASR2S."""
        from model import InferenceError
        
        audio_path = arguments.get("audio_path")
        
        if not audio_path:
            raise ValueError("audio_path is required")
        
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        print(f"Transcribing: {audio_path} (timestamps={include_timestamps})", file=sys.stderr)
        
        try:
            # Run prediction
            result = self._model.predict(audio_path)
            
            # Extract text
            text = result.get("text", "")
            segments = result.get("segments", [])
            language = result.get("language")
            confidence = result.get("confidence")
            
            # Format output
            if include_timestamps and segments:
                # Format with timestamps
                timestamp_lines = []
                for seg in segments:
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)
                    word = seg.get("text", "")
                    timestamp_lines.append(f"[{start:.2f}s - {end:.2f}s]: {word}")
                
                output_text = f"{text}\n\nWord-level timestamps:\n" + "\n".join(timestamp_lines)
            else:
                output_text = text
            
            # Build detailed response
            response_data = {
                "text": text,
                "language": language,
                "confidence": confidence,
            }
            
            if include_timestamps:
                response_data["segments"] = segments
            
            print(f"Transcription result: {text[:100]}...", file=sys.stderr)
            
            return {
                "content": [
                    {"type": "text", "text": output_text},
                    {"type": "text", "text": json.dumps(response_data, ensure_ascii=False, indent=2)}
                ],
                "isError": False
            }
            
        except InferenceError as e:
            print(f"Transcription failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"Transcription failed: {e}") from e
        except Exception as e:
            print(f"Unexpected error during transcription: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"Transcription failed: {e}") from e
    
    def _healthcheck(self) -> dict[str, Any]:
        """Run healthcheck."""
        try:
            self._load_model()
            health = self._model.healthcheck()
            
            return {
                "content": [
                    {"type": "text", "text": json.dumps(health, indent=2)}
                ],
                "isError": False
            }
        except Exception as e:
            return {
                "content": [
                    {"type": "text", "text": f"Healthcheck failed: {e}"}
                ],
                "isError": True,
                "error": str(e)
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
    # Ensure unbuffered output for proper JSON-RPC communication
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    server = FireRedAsr2sMCPServer()
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
