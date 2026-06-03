#!/usr/bin/env python3
"""
DiariZen Speaker Diarization MCP Server.

MCP server implementation for speaker diarization using DiariZen.
Uses the BUT-FIT/diarizen-wavlm-large-s80-md model.

Tool-specific notes:
- This is the **only** tool in the catalog that uses **conda** (Python 3.10)
  instead of uv. setup.sh creates `.venv/` via `conda create --prefix`.
- The model weights are licensed **CC BY-NC 4.0 (Non-Commercial)**. Verify
  your downstream use is allowed before deploying.
- Memory: pipeline loads ~8 GB RAM; set `DEVICE=cpu` if no GPU.
- Env vars: `MODEL_PATH` (resolved from $AUDIO_AGENT_MODELS_DIR by
  config.yaml), `DEVICE` (auto/cpu/cuda).
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


class DiariZenServer:
    """MCP Server for DiariZen Speaker Diarization."""
    
    def __init__(self):
        self._initialized = False
        self._pipeline = None
        self._model_path = os.environ.get(
            "MODEL_PATH", 
            "BUT-FIT/diarizen-wavlm-large-s80-md"
        )
        self._device = os.environ.get("DEVICE", "auto")
        
        # Tool definitions
        self._tools = [
            {
                "name": "diarize",
                "description": "Estimate who speaks when in an audio file by returning speaker-labeled time segments. Most useful for meeting, interview, narration, or clean multi-speaker dialogue with reasonably separated turns. The output identifies anonymous speaker clusters rather than real identities or guaranteed-perfect boundaries. Trust it less for songs, crowd scenes, TV/movie audio, laughter/noise-heavy clips, overlapping speakers, short clips, child/cartoon/processed voices, or cases where speaker role depends on semantics rather than voice clustering. Do not use it to override strong frontend perception outside its domain.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to diarize"
                        },
                    },
                    "required": ["audio_path"]
                }
            }
        ]
    
    def _load_pipeline(self) -> None:
        """Lazy load the DiariZen pipeline."""
        if self._pipeline is not None:
            return
        
        try:
            from diarizen.pipelines.inference import DiariZenPipeline
            from pathlib import Path
            import torch
        except ImportError as e:
            raise RuntimeError(
                f"Missing diarizen package. Ensure environment is set up correctly: {e}"
            ) from e
        
        print(f"Loading DiariZen pipeline: {self._model_path}", file=sys.stderr)
        
        try:
            # DiariZen prints to stdout which breaks JSON-RPC
            # Redirect stdout to stderr during loading
            old_stdout = sys.stdout
            sys.stdout = sys.stderr
            
            # Check if model_path is a local directory
            if os.path.isdir(self._model_path):
                # Load from local path using direct instantiation
                print(f"Loading from local directory: {self._model_path}", file=sys.stderr)
                
                # Download embedding model from HF Hub (required component)
                from huggingface_hub import hf_hub_download
                embedding_model = hf_hub_download(
                    repo_id="pyannote/wespeaker-voxceleb-resnet34-LM",
                    filename="pytorch_model.bin"
                )
                
                # Directly instantiate the pipeline
                self._pipeline = DiariZenPipeline(
                    diarizen_hub=Path(self._model_path).expanduser().absolute(),
                    embedding_model=embedding_model
                )
            else:
                # Load from HuggingFace Hub using from_pretrained
                self._pipeline = DiariZenPipeline.from_pretrained(self._model_path)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            print(f"Pipeline loaded successfully", file=sys.stderr)
        except Exception as e:
            # Restore stdout on error
            sys.stdout = old_stdout
            raise RuntimeError(f"Failed to load pipeline: {e}") from e
    
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
                    "name": "diarizen-server",
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
        self._load_pipeline()
        
        if tool_name == "diarize":
            return self._diarize(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _diarize(self, arguments: dict) -> dict[str, Any]:
        """Perform speaker diarization on an audio file."""
        audio_path = arguments.get("audio_path")
        
        if not audio_path:
            raise ValueError("audio_path is required")
        
        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")
        
        print(f"Diarizing: {audio_path}", file=sys.stderr)
        
        # Redirect stdout to stderr during diarization (DiariZen prints progress)
        old_stdout = sys.stdout
        sys.stdout = sys.stderr
        
        try:
            # Run diarization
            # DiariZenPipeline.__call__ only accepts in_wav and sess_name
            sess_name = os.path.splitext(os.path.basename(audio_path))[0]
            diar_results = self._pipeline(audio_path, sess_name=sess_name)
            
            # Format results
            segments = []
            for turn, _, speaker in diar_results.itertracks(yield_label=True):
                segments.append({
                    "start": round(turn.start, 2),
                    "end": round(turn.end, 2),
                    "speaker": str(speaker)
                })
            
            # Build output text
            output_lines = [
                f"Speaker diarization results for: {os.path.basename(audio_path)}",
                f"Total segments: {len(segments)}",
                "",
                "Segments:"
            ]
            
            for seg in segments:
                output_lines.append(
                    f"  [{seg['start']:6.2f}s - {seg['end']:6.2f}s] {seg['speaker']}"
                )
            
            # Add summary
            unique_speakers = sorted(set(seg["speaker"] for seg in segments))
            output_lines.extend([
                "",
                f"Detected speakers: {', '.join(unique_speakers)}"
            ])
            
            result_text = "\n".join(output_lines)
            
            print(f"Diarization complete: {len(segments)} segments, {len(unique_speakers)} speakers", 
                  file=sys.stderr)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            return {
                "content": [{"type": "text", "text": result_text}],
                "isError": False
            }
            
        except Exception as e:
            # Restore stdout on error
            sys.stdout = old_stdout
            print(f"Diarization failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"Diarization failed: {e}") from e
    
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
    
    server = DiariZenServer()
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
