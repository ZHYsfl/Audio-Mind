#!/usr/bin/env python3
"""
Omni Captioner MCP Server.

MCP server for Qwen3-Omni API via DashScope.
Supports audio captioning with optional audio response.

Tool-specific notes:
- **API-based**: no local model download required, but needs a DashScope key:
    export DASHSCOPE_API_KEY="sk-..."
  Get one at https://dashscope.console.aliyun.com/.
- Env vars (set via config.yaml or before launch):
    DASHSCOPE_API_KEY  - required
    DASHSCOPE_BASE_URL - default: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
    DEFAULT_MODEL      - default: qwen3-omni-flash
    DEFAULT_VOICE      - default: Cherry (only used by omni_caption_with_audio)
- Three tools exposed: omni_caption, omni_caption_with_audio,
  inspect_audio_plots, verify_audio_quality. See config.yaml for input
  schemas. inspect_audio_plots is the only one in the default planner
  scope; the others are LALM-style perception that's normally served
  by the framework's CALL_FRONTEND action.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


class OmniCaptionerServer:
    """MCP Server for Omni Captioner API."""
    
    def __init__(self):
        self._initialized = False
        self._client = None
        
        # API configuration from environment
        self._api_key = os.environ.get("DASHSCOPE_API_KEY", "")
        self._base_url = os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self._default_model = os.environ.get("DEFAULT_MODEL", "qwen3.5-omni-plus")
        self._default_vlm_model = os.environ.get("DEFAULT_VLM_MODEL", "qwen3-vl-plus")
        self._default_voice = os.environ.get("DEFAULT_VOICE", "Cherry")
        self._default_audio_format = os.environ.get("DEFAULT_AUDIO_FORMAT", "wav")
        self._default_sample_rate = int(os.environ.get("DEFAULT_SAMPLE_RATE", "24000"))
        
        # Tool definitions
        self._tools = [
            {
                "name": "omni_caption",
                "description": "Generate a model-based perceptual caption for an audio file. Useful for broad summaries of what the model hears and for open-ended description prompts. The output is a model judgment rather than an independent measurement and should not be treated as direct verification of other model outputs.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to caption"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Prompt for captioning task",
                            "default": "Describe this audio in detail."
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "omni_caption_with_audio",
                "description": "Generate a model-based caption plus spoken audio response for an audio file. Useful for presentation-style or interactive output. It does not provide independent analysis beyond the captioning model's own judgment.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to caption"
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Prompt for captioning task",
                            "default": "Describe this audio in detail."
                        },
                        "voice": {
                            "type": "string",
                            "description": "Voice for audio response (e.g., 'Cherry')",
                            "default": "Cherry"
                        },
                        "output_audio_path": {
                            "type": "string",
                            "description": "Path to save the generated audio response",
                            "default": ""
                        }
                    },
                    "required": ["audio_path"]
                }
            },
            {
                "name": "verify_audio_quality",
                "description": "Inspect processed audio for obvious artifacts, distortion, or quality regressions using a model over spectrogram-like representations. Useful for coarse post-processing checks. It should not be treated as a direct judge of semantic correctness or content preservation.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the processed/enhanced audio file to verify (REQUIRED)"
                        },
                        "reference_audio_path": {
                            "type": "string",
                            "description": "Optional: Path to original audio for comparison"
                        },
                        "verification_prompt": {
                            "type": "string",
                            "description": "Specific instructions on what to check (e.g., 'Check for denoising artifacts', 'Compare with reference for content loss')"
                        }
                    },
                    "required": ["audio_path", "verification_prompt"]
                }
            },
            {
                "name": "inspect_audio_plots",
                "description": "Generate one combined audio-plot image and ask a VLM for bounded visual-acoustic evidence. Useful for visible structure only: activity, silence/gaps, rough loudness changes, transients, rough event boundaries, spectral brightness/muffling, and coarse rhythm. It does not listen to audio and is not valid for cowbell detection, exact pitch naming, chord recognition, instrument taxonomy, semantic music-theory labels, source identity, or final answering.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "audio_path": {
                            "type": "string",
                            "description": "Path to the audio file to inspect (REQUIRED)"
                        },
                        "question": {
                            "type": "string",
                            "description": "Original user question for context only; the tool will not answer it directly"
                        },
                        "analysis_focus": {
                            "type": "string",
                            "description": "Specific visual-acoustic uncertainty to inspect, e.g. repeated transients, silent gaps, loudness change, or brightness change"
                        },
                        "plot_types": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "waveform",
                                    "mel_spectrogram",
                                    "rms_energy",
                                    "onset_envelope",
                                    "spectral_rolloff",
                                    "cqt_chroma",
                                    "bpm_curve"
                                ]
                            },
                            "description": "Optional selected plot rows. Defaults to waveform, mel_spectrogram, rms_energy, onset_envelope, spectral_rolloff."
                        },
                        "time_range": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "number"},
                                "end": {"type": "number"}
                            },
                            "description": "Optional time range in seconds to crop before plotting"
                        }
                    },
                    "required": ["audio_path", "question", "analysis_focus"]
                }
            }
        ]
    
    def _get_client(self):
        """Lazy initialize OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise RuntimeError(f"Missing openai package: {e}") from e
            
            if not self._api_key:
                raise RuntimeError(
                    "DASHSCOPE_API_KEY not set. Please provide API key."
                )
            
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
            )
        return self._client
    
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
                    "name": "omni-captioner-server",
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
        if tool_name == "omni_caption":
            return self._omni_caption(arguments, generate_audio=False)
        elif tool_name == "omni_caption_with_audio":
            return self._omni_caption(arguments, generate_audio=True)
        elif tool_name == "verify_audio_quality":
            return self._verify_audio_quality(arguments)
        elif tool_name == "inspect_audio_plots":
            return self._inspect_audio_plots(arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _omni_caption(self, arguments: dict, generate_audio: bool) -> dict[str, Any]:
        """Call Qwen3-Omni API for audio captioning."""
        from model import OmniCaptionerModel
        
        audio_path = arguments.get("audio_path", "")
        prompt = arguments.get("prompt", "Describe this audio in detail.")
        voice = arguments.get("voice", self._default_voice)
        output_audio_path = arguments.get("output_audio_path", "")
        
        if not audio_path:
            raise ValueError("audio_path is required")
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Captioning audio: {audio_path}, audio={generate_audio}", file=sys.stderr)
        
        try:
            model = OmniCaptionerModel(
                api_key=self._api_key,
                base_url=self._base_url,
                model=self._default_model,
                voice=voice,
            )
            
            result = model.caption_audio(
                audio_path=audio_path,
                prompt=prompt,
                generate_audio=generate_audio,
                output_audio_path=output_audio_path if output_audio_path else None,
            )
            
            # Build response
            result_text = result.text
            if result.audio_path:
                result_text += f"\n\n[Audio response saved to: {result.audio_path}]"
            elif generate_audio and not output_audio_path:
                result_text += "\n\n[Audio response generated but not saved (no output path provided)]"
            
            return {
                "content": [{"type": "text", "text": result_text}],
                "isError": False
            }
            
        except Exception as e:
            print(f"API call failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"API call failed: {e}") from e
    
    def _verify_audio_quality(self, arguments: dict) -> dict[str, Any]:
        """Verify audio quality using VLM analysis of spectrograms."""
        from model import OmniCaptionerModel
        
        audio_path = arguments.get("audio_path", "")
        verification_prompt = arguments.get("verification_prompt", "")
        reference_audio_path = arguments.get("reference_audio_path", "")
        
        if not audio_path:
            raise ValueError("audio_path is required")
        if not verification_prompt:
            raise ValueError("verification_prompt is required")
        
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Verifying audio quality: {audio_path}", file=sys.stderr)
        
        try:
            model = OmniCaptionerModel(
                api_key=self._api_key,
                base_url=self._base_url,
                model=self._default_model,
                vlm_model=self._default_vlm_model,
            )
            
            result = model.verify_audio_quality(
                audio_path=audio_path,
                verification_prompt=verification_prompt,
                reference_audio_path=reference_audio_path if reference_audio_path else None,
            )
            
            # Build response
            status_emoji = "✅" if result.verification_passed else "❌"
            quality_emoji = {
                "Good": "🟢",
                "Acceptable": "🟡",
                "Poor": "🔴"
            }.get(result.quality_assessment, "⚪")
            
            result_text = f"""{status_emoji} Audio Quality Verification Result

**Verification Passed:** {result.verification_passed}
**Quality Assessment:** {quality_emoji} {result.quality_assessment}
**VLM Model:** {result.vlm_model}

**Issues Found:**
"""
            if result.issues_found:
                for issue in result.issues_found:
                    result_text += f"- {issue}\n"
            else:
                result_text += "- None\n"
            
            result_text += "\n**Recommendations:**\n"
            if result.recommendations:
                for rec in result.recommendations:
                    result_text += f"- {rec}\n"
            else:
                result_text += "- None\n"
            
            result_text += f"\n**Analysis:**\n{result.analysis}\n"
            result_text += f"\n[Spectrogram saved to: {result.spectrogram_path}]"
            
            return {
                "content": [{"type": "text", "text": result_text}],
                "isError": False
            }
            
        except Exception as e:
            print(f"Verification failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"Verification failed: {e}") from e

    def _inspect_audio_plots(self, arguments: dict) -> dict[str, Any]:
        """Inspect combined audio plots using a bounded VLM prompt."""
        from model import OmniCaptionerModel

        audio_path = arguments.get("audio_path", "")
        question = arguments.get("question", "")
        analysis_focus = arguments.get("analysis_focus", "")
        plot_types = arguments.get("plot_types")
        time_range = arguments.get("time_range")

        if not audio_path:
            raise ValueError("audio_path is required")
        if not question:
            raise ValueError("question is required")
        if not analysis_focus:
            raise ValueError("analysis_focus is required")
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        if plot_types is not None and not isinstance(plot_types, list):
            raise ValueError("plot_types must be an array when provided")
        if time_range is not None and not isinstance(time_range, dict):
            raise ValueError("time_range must be an object when provided")

        print(f"Inspecting audio plots: {audio_path}", file=sys.stderr)

        try:
            model = OmniCaptionerModel(
                api_key=self._api_key,
                base_url=self._base_url,
                model=self._default_model,
                vlm_model=self._default_vlm_model,
            )
            result = model.inspect_audio_plots(
                audio_path=audio_path,
                question=question,
                analysis_focus=analysis_focus,
                plot_types=plot_types,
                time_range=time_range,
            )

            output = dict(result.structured_result)
            output["plot_path"] = result.plot_path
            output["vlm_model"] = result.vlm_model
            if result.parsing_warning:
                output["parsing_warning"] = result.parsing_warning
                output["raw_response"] = result.raw_response

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(output, ensure_ascii=False, indent=2),
                    }
                ],
                "isError": False,
            }
        except Exception as e:
            print(f"Audio plot inspection failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)
            raise RuntimeError(f"Audio plot inspection failed: {e}") from e

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
    
    server = OmniCaptionerServer()
    server.run()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
