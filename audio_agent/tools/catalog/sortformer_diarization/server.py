#!/usr/bin/env python3
"""
SortFormer Streaming Diarization MCP Server

MCP server implementation for speaker diarization using NVIDIA SortFormer.
Uses the nv-community/diar_streaming_sortformer_4spk-v2 model.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Redirect NeMo logs to stderr before any nemo imports.
# NeMo's Logger.add_stream_handlers() sends INFO-level logs to stdout by
# default. Setting this env var forces all logs to stderr, keeping stdout
# clean for JSON-RPC responses.
os.environ.setdefault("NEMO_REDIRECT_LOGS_TO_STDERR", "1")
os.environ.setdefault("NEMO_LOGGING_LEVEL", "WARNING")
logging.getLogger("nemo").setLevel(logging.WARNING)
logging.getLogger("nemo_logger").setLevel(logging.WARNING)


class SortformerDiarizationServer:
    """MCP Server for SortFormer Speaker Diarization."""

    def __init__(self):
        self._initialized = False
        self._model = None
        # Default resolves under $AUDIO_AGENT_MODELS_DIR (set by config.yaml env-var
        # expansion). Falls back to <repo>/models/... when neither env var is set.
        _repo_root = Path(__file__).resolve().parents[4]
        _models_dir = os.environ.get("AUDIO_AGENT_MODELS_DIR") or str(_repo_root / "models")
        self._model_path = os.environ.get(
            "MODEL_PATH",
            os.path.join(
                _models_dir,
                "sortformer-diar-streaming-4spk-v2",
                "diar_streaming_sortformer_4spk-v2.nemo",
            ),
        )
        self._device = os.environ.get("DEVICE", "auto")

        # Tool definitions
        self._tools = [
            {
                "name": "diarize_sortformer",
                "description": "Estimate who speaks when in an audio file by returning speaker-labeled time segments using NVIDIA SortFormer. Most useful for meeting, interview, narration, or clean multi-speaker dialogue with reasonably separated turns. Supports up to 4 speakers. The output identifies anonymous speaker clusters rather than real identities or guaranteed-perfect boundaries. Trust it less for songs, crowd scenes, TV/movie audio, laughter/noise-heavy clips, overlapping speakers, short clips, child/cartoon/processed voices, or cases where speaker role depends on semantics rather than voice clustering. Do not use it to override strong frontend perception outside its domain.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to diarize (16kHz mono WAV recommended)",
                        },
                    },
                    "required": ["audio_path"],
                },
            }
        ]

    def _resolve_device(self) -> str:
        """Resolve device string to actual device."""
        if self._device == "auto":
            try:
                import torch

                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self._device

    def _load_model(self) -> None:
        """Lazy load the SortFormer model."""
        if self._model is not None:
            return

        try:
            from nemo.collections.asr.models import SortformerEncLabelModel
        except ImportError as e:
            raise RuntimeError(
                f"Missing NeMo ASR package. Ensure environment is set up correctly: {e}"
            ) from e

        print(f"Loading SortFormer model: {self._model_path}", file=sys.stderr)

        # Redirect stdout to stderr during loading (NeMo prints progress/logs)
        old_stdout = sys.stdout
        sys.stdout = sys.stderr

        try:
            device = self._resolve_device()

            # Load model with fallback
            try:
                self._model = SortformerEncLabelModel.restore_from(
                    self._model_path, map_location=device, strict=False
                )
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and device == "cuda":
                    print(
                        "CUDA OOM, falling back to CPU...", file=sys.stderr
                    )
                    self._model = SortformerEncLabelModel.restore_from(
                        self._model_path, map_location="cpu", strict=False
                    )
                else:
                    raise

            self._model.eval()

            # Configure for high-latency offline mode (best accuracy)
            self._model.sortformer_modules.chunk_len = 340
            self._model.sortformer_modules.chunk_right_context = 40
            self._model.sortformer_modules.fifo_len = 40
            self._model.sortformer_modules.spkcache_update_period = 300
            self._model.sortformer_modules.spkcache_len = 188

            # Restore stdout
            sys.stdout = old_stdout

            print(f"Model loaded successfully on {self._model.device}", file=sys.stderr)
        except Exception as e:
            # Restore stdout on error
            sys.stdout = old_stdout
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
                    "name": "sortformer-diarization-server",
                    "version": "1.0.0",
                },
            },
        }

    def _handle_tools_list(self, request_id: Any) -> dict[str, Any]:
        """Handle tools/list request."""
        if not self._initialized:
            return self._error_response(request_id, -32001, "Server not initialized")

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": self._tools},
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
                    "error": error_msg,
                },
            }

    def _execute_tool(self, tool_name: str, arguments: dict) -> dict[str, Any]:
        """Execute a tool."""
        self._load_model()

        if tool_name == "diarize_sortformer":
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

        # Redirect stdout to stderr during diarization (NeMo prints progress)
        old_stdout = sys.stdout
        sys.stdout = sys.stderr

        try:
            # Run diarization
            predicted_segments = self._model.diarize(
                audio=[audio_path], batch_size=1
            )

            # Parse segments
            segments = []
            for seg_str in predicted_segments[0]:
                parts = seg_str.strip().split()
                if len(parts) >= 3:
                    segments.append(
                        {
                            "start": round(float(parts[0]), 2),
                            "end": round(float(parts[1]), 2),
                            "speaker": str(parts[2]),
                        }
                    )

            # Restore stdout
            sys.stdout = old_stdout

            # Build output text matching diarizen format
            output_lines = [
                f"Speaker diarization results for: {os.path.basename(audio_path)}",
                f"Total segments: {len(segments)}",
                "",
                "Segments:",
            ]

            for seg in segments:
                output_lines.append(
                    f"  [{seg['start']:6.2f}s - {seg['end']:6.2f}s] {seg['speaker']}"
                )

            # Add summary
            unique_speakers = sorted(set(seg["speaker"] for seg in segments))
            output_lines.extend(
                [
                    "",
                    f"Detected speakers: {', '.join(unique_speakers)}",
                ]
            )

            result_text = "\n".join(output_lines)

            print(
                f"Diarization complete: {len(segments)} segments, {len(unique_speakers)} speakers",
                file=sys.stderr,
            )

            return {
                "content": [{"type": "text", "text": result_text}],
                "isError": False,
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
            "error": {"code": code, "message": message},
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

    server = SortformerDiarizationServer()
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
