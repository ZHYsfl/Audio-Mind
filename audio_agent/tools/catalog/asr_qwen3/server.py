#!/usr/bin/env python3
"""
ASR Qwen3 MCP Server

MCP server implementation for speech recognition using Qwen3-ASR-1.7B.
Uses the official qwen-asr package.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


class ASRQwen3Server:
    """MCP Server for Qwen3-ASR-1.7B ASR."""
    
    def __init__(self):
        self._initialized = False
        self._model = None
        self._model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen3-ASR-1.7B")
        self._device = os.environ.get("DEVICE", "auto")
        self._default_language = os.environ.get("LANGUAGE", "auto")
        self._aligner_path = os.environ.get("ALIGNER_PATH", "Qwen/Qwen3-ForcedAligner-0.6B")
        self._use_aligner = os.environ.get("USE_ALIGNER", "true").lower() == "true"
        
        # Tool definitions
        self._tools = [
            {
                "name": "transcribe_qwenasr",
                "description": "Transcribe spoken content in an audio file to text. Most useful for clean spoken dialogue, narration, interviews, meetings, and phone-like speech. Treat the output as recognized text rather than guaranteed ground truth, especially for singing, rap, overlapping speech, loud music/noise, child/cartoon/processed voices, strong accents or dialects, emotional shouting, reverberant audio, or very short clips. Do not use it to override strong frontend perception outside its domain.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to transcribe"
                        },
                        "language": {
                            "type": "string",
                            "description": "Language name (e.g., 'auto', 'Chinese', 'English', 'Japanese', 'Korean', 'Cantonese'). Use full language names, NOT ISO codes. Supported: Chinese, English, Cantonese, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Korean, Russian, Thai, Vietnamese, Japanese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian",
                            "default": "auto"
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "transcribe_qwenasr_with_timestamps",
                "description": "Transcribe spoken content to text with word-level timestamps using ASR plus forced alignment. Most useful for clean spoken dialogue when both lexical content and temporal grounding are needed. Timestamp quality depends on the transcript and alignment; boundary times can be wrong, missing, or shifted, especially for singing, overlapping speech, noisy/music-backed audio, child/cartoon/processed voices, strong accents or dialects, emotional shouting, reverberant audio, or very short clips. Do not use it to override strong frontend perception outside its domain.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file"
                        },
                        "language": {
                            "type": "string",
                            "description": "Language name (e.g., 'Chinese', 'English', 'Japanese'). Use full language names, NOT ISO codes.",
                            "default": "auto"
                        }
                    },
                    "required": ["audio_path"]
                }
            }
        ]
    
    def _load_model(self) -> None:
        """Lazy load the model and optional forced aligner."""
        if self._model is not None:
            return
        
        try:
            from qwen_asr import Qwen3ASRModel
            import torch
        except ImportError as e:
            raise RuntimeError(
                f"Missing qwen-asr package. Ensure environment is set up correctly: {e}"
            ) from e
        
        print(f"Loading Qwen3-ASR model: {self._model_path}", file=sys.stderr)
        
        # Determine device and dtype
        if self._device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self._device
        
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        
        try:
            load_kwargs = {
                "dtype": dtype,
                "device_map": device if device != "cpu" else None,
            }
            
            # Load forced aligner if enabled
            if self._use_aligner:
                print(f"Loading forced aligner: {self._aligner_path}", file=sys.stderr)
                load_kwargs["forced_aligner"] = self._aligner_path
                load_kwargs["forced_aligner_kwargs"] = {
                    "dtype": dtype,
                    "device_map": device if device != "cpu" else None,
                }
            
            self._model = Qwen3ASRModel.from_pretrained(
                self._model_path,
                **load_kwargs
            )
            print(f"Model loaded successfully on {self._model.device}", file=sys.stderr)
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e
    
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
                    "name": "asr-qwen3-server",
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
        
        if tool_name == "transcribe_qwenasr":
            return self._transcribe(arguments)
        elif tool_name == "transcribe_qwenasr_with_timestamps":
            # For now, same as transcribe (timestamps require forced aligner)
            return self._transcribe(arguments, include_timestamps=True)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _transcribe(
        self, 
        arguments: dict, 
        include_timestamps: bool = False
    ) -> dict[str, Any]:
        """Transcribe audio to text using Qwen3ASRModel."""
        audio_path = arguments.get("audio_path")
        language = arguments.get("language", self._default_language)
        
        if not audio_path:
            raise ValueError("audio_path is required")
        
        # Use audio path as-is (client should provide valid path)
        print(f"Transcribing: {audio_path} (language={language}, timestamps={include_timestamps})", file=sys.stderr)
        
        try:
            # Use the model's transcribe method
            # Qwen3ASRModel.transcribe(audio, context='', language=None, return_time_stamps=False)
            result = self._model.transcribe(
                audio_path,
                language=language if language != "auto" else None,
                return_time_stamps=include_timestamps,
            )
            
            # The result is a list of ASRTranscription objects
            # Each has .text and optionally .time_stamps
            if result and len(result) > 0:
                transcription = result[0]
                text = transcription.text
                
                # If timestamps requested, include them in the output
                if include_timestamps and hasattr(transcription, 'time_stamps') and transcription.time_stamps:
                    timestamp_text = "\n".join([
                        f"[{ts.start_time:.2f}s - {ts.end_time:.2f}s]: {ts.text}"
                        for ts in transcription.time_stamps
                    ])
                    text = f"{text}\n\nTimestamps:\n{timestamp_text}"
            else:
                text = ""
            
            print(f"Transcription result: {text[:100]}...", file=sys.stderr)
            
            return {
                "content": [{"type": "text", "text": text}],
                "isError": False
            }
            
        except Exception as e:
            print(f"Transcription failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"Transcription failed: {e}") from e
    
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
    
    server = ASRQwen3Server()
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
