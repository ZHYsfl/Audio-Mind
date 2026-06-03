"""
Omni Captioner Model Wrapper.

API client for Qwen3-Omni multi-modal model via DashScope.
Supports audio captioning and text/audio generation.
"""

from __future__ import annotations

import os
import base64
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CaptionResult:
    """Result of caption generation."""
    text: str
    audio_path: str | None = None
    audio_data: bytes | None = None


@dataclass
class VerificationResult:
    """Result of audio quality verification."""
    verification_passed: bool
    issues_found: list[str]
    quality_assessment: str  # "Good", "Acceptable", "Poor"
    recommendations: list[str]
    spectrogram_path: str
    analysis: str
    vlm_model: str


@dataclass
class PlotInspectionResult:
    """Result of VLM inspection over a combined audio-plot image."""
    plot_path: str
    structured_result: dict[str, Any]
    raw_response: str
    vlm_model: str
    parsing_warning: str | None = None


class OmniCaptionerModel:
    """Wrapper for Qwen3-Omni API for audio captioning."""

    DEFAULT_AUDIO_PLOT_TYPES = (
        "waveform",
        "mel_spectrogram",
        "rms_energy",
        "onset_envelope",
        "spectral_rolloff",
    )
    OPTIONAL_AUDIO_PLOT_TYPES = ("cqt_chroma", "bpm_curve")
    SUPPORTED_AUDIO_PLOT_TYPES = DEFAULT_AUDIO_PLOT_TYPES + OPTIONAL_AUDIO_PLOT_TYPES
    
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "qwen3.5-omni-plus",
        vlm_model: str | None = None,
        voice: str = "Cherry",
        audio_format: str = "wav",
        sample_rate: int = 24000,
    ):
        """
        Initialize Omni Captioner model.
        
        Args:
            api_key: DashScope API key (or set DASHSCOPE_API_KEY env var)
            base_url: API base URL
            model: Audio-capable model ID for omni captioning
            vlm_model: Vision-language model ID for image-based analysis tools
            voice: Voice for audio generation
            audio_format: Audio format (wav, mp3, etc.)
            sample_rate: Audio sample rate
        """
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.base_url = base_url or os.environ.get(
            "DASHSCOPE_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.model = model
        self.vlm_model = vlm_model or os.environ.get("DEFAULT_VLM_MODEL") or model
        self.voice = voice
        self.audio_format = audio_format
        self.sample_rate = sample_rate
        self.vlm_enable_thinking = self._env_bool("DEFAULT_VLM_ENABLE_THINKING", False)
        self.vlm_thinking_budget = self._env_int("DEFAULT_VLM_THINKING_BUDGET")
        
        self._client = None

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        """Read a boolean environment variable using common truthy values."""
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_int(name: str) -> int | None:
        """Read an integer environment variable when set."""
        value = os.environ.get(name)
        if value is None or not value.strip():
            return None
        return int(value)
    
    def _get_client(self):
        """Lazy initialize OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise RuntimeError(
                    "openai not installed. Run: pip install openai"
                )
            
            if not self.api_key:
                raise RuntimeError(
                    "DASHSCOPE_API_KEY not set. Please provide API key."
                )
            
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client
    
    def caption_audio(
        self,
        audio_path: str | Path,
        prompt: str = "Describe this audio in detail.",
        generate_audio: bool = False,
        output_audio_path: str | Path | None = None,
    ) -> CaptionResult:
        """
        Generate caption for an audio file.
        
        Args:
            audio_path: Path to the audio file to caption
            prompt: Prompt for the captioning task
            generate_audio: Whether to generate audio response
            output_audio_path: Path to save the generated audio file
            
        Returns:
            CaptionResult with text and optional audio
        """
        import numpy as np
        
        client = self._get_client()
        
        # Read and encode audio
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        
        # Determine audio format from file extension
        audio_format = audio_path.suffix.lstrip(".").lower()
        if audio_format not in ["wav", "mp3", "ogg", "m4a", "flac"]:
            audio_format = "wav"  # Default fallback
        
        # Build request with audio input
        # Note: data:;base64, prefix is required for base64 audio data
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": f"data:;base64,{audio_base64}",
                            "format": audio_format,
                        }
                    }
                ]
            }
        ]
        
        request_params = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        
        # qwen3.5-omni-plus requires modalities parameter
        if generate_audio:
            request_params["modalities"] = ["text", "audio"]
        else:
            request_params["modalities"] = ["text"]  # Text-only output
        
        if generate_audio:
            request_params["modalities"] = ["text", "audio"]
            request_params["audio"] = {
                "voice": self.voice,
                "format": self.audio_format
            }
        
        # Call API
        completion = client.chat.completions.create(**request_params)
        
        # Process response
        text_response = ""
        audio_response_base64 = ""
        
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                text_response += chunk.choices[0].delta.content
            
            if (chunk.choices and 
                hasattr(chunk.choices[0].delta, "audio") and 
                chunk.choices[0].delta.audio):
                audio_response_base64 += chunk.choices[0].delta.audio.get("data", "")
        
        # Save audio if generated
        audio_data = None
        saved_path = None
        
        if generate_audio and audio_response_base64 and output_audio_path:
            wav_bytes = base64.b64decode(audio_response_base64)
            audio_data = wav_bytes
            
            # Save to file
            output_path = Path(output_audio_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import soundfile as sf
            audio_np = np.frombuffer(wav_bytes, dtype=np.int16)
            sf.write(str(output_path), audio_np, samplerate=self.sample_rate)
            saved_path = str(output_path)
        
        return CaptionResult(
            text=text_response,
            audio_path=saved_path,
            audio_data=audio_data,
        )
    
    def caption_text_only(
        self,
        audio_path: str | Path,
        prompt: str = "Describe this audio in detail.",
    ) -> str:
        """
        Generate text caption for an audio file (no audio output).
        
        Args:
            audio_path: Path to the audio file to caption
            prompt: Prompt for the captioning task
            
        Returns:
            Text caption
        """
        result = self.caption_audio(audio_path, prompt, generate_audio=False)
        return result.text

    def _generate_spectrogram(
        self,
        audio_path: str | Path,
        output_path: str | Path | None = None,
    ) -> str:
        """
        Generate magnitude spectrogram using librosa and matplotlib.
        
        Args:
            audio_path: Path to the audio file
            output_path: Optional path to save spectrogram. If None, creates temp file.
            
        Returns:
            Path to the generated spectrogram image
        """
        try:
            import librosa
            import librosa.display
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as e:
            raise RuntimeError(f"Required packages not installed: {e}")
        
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Create temp file if output path not provided
        if output_path is None:
            output_fd, output_path = tempfile.mkstemp(suffix=".png")
            os.close(output_fd)
        else:
            output_path = Path(output_path)
        
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=None)
        
        # Generate spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        
        # Plot
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Spectrogram: {audio_path.name}')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(output_path)

    def _load_audio_for_plots(
        self,
        audio_path: str | Path,
        time_range: dict[str, Any] | None = None,
    ) -> tuple[Any, int, float, float]:
        """Load mono audio and optionally crop it to a validated time range."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            import librosa
            import numpy as np
        except ImportError as e:
            raise RuntimeError(f"Required packages not installed: {e}") from e

        y, sr = librosa.load(str(path), sr=None, mono=True)
        if sr <= 0:
            raise ValueError(f"Invalid sample rate for audio file: {sr}")
        if y.size == 0:
            raise ValueError("Audio file is empty")

        duration = float(len(y) / sr)
        start = 0.0
        end = duration

        if time_range:
            if not isinstance(time_range, dict):
                raise ValueError("time_range must be an object with optional start/end")
            raw_start = float(time_range.get("start", 0.0))
            raw_end = float(time_range.get("end", duration))
            # Planners sometimes request a window past the clip end (or otherwise out of
            # bounds). Clamp to [0, duration] and analyze the valid overlap instead of
            # failing the tool; fall back to the full clip if nothing valid remains.
            start = min(max(raw_start, 0.0), duration)
            end = min(max(raw_end, 0.0), duration)
            if end <= start:
                start, end = 0.0, duration
            start_sample = int(round(start * sr))
            end_sample = int(round(end * sr))
            y = y[start_sample:end_sample]
            if y.size == 0:
                raise ValueError("Selected time_range produced empty audio")

        if not np.isfinite(y).all():
            raise ValueError("Audio contains non-finite samples")

        return y, sr, start, end

    def _validate_plot_types(self, plot_types: list[str] | None) -> list[str]:
        """Validate plot type selection and preserve caller order."""
        selected = list(plot_types) if plot_types else list(self.DEFAULT_AUDIO_PLOT_TYPES)
        if not selected:
            raise ValueError("plot_types cannot be empty")

        supported = set(self.SUPPORTED_AUDIO_PLOT_TYPES)
        invalid = [plot_type for plot_type in selected if plot_type not in supported]
        if invalid:
            raise ValueError(
                "Unsupported plot_types: "
                + ", ".join(invalid)
                + ". Supported: "
                + ", ".join(self.SUPPORTED_AUDIO_PLOT_TYPES)
            )

        deduped: list[str] = []
        for plot_type in selected:
            if plot_type not in deduped:
                deduped.append(plot_type)
        return deduped

    def _generate_combined_audio_plots(
        self,
        audio_path: str | Path,
        plot_types: list[str] | None = None,
        time_range: dict[str, Any] | None = None,
        output_path: str | Path | None = None,
    ) -> tuple[str, list[str], float, float]:
        """Generate a single stacked PNG containing selected audio visualizations."""
        selected_plot_types = self._validate_plot_types(plot_types)
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        try:
            import librosa
            import librosa.display
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as e:
            raise RuntimeError(f"Required packages not installed: {e}") from e

        y, sr, start_time, end_time = self._load_audio_for_plots(path, time_range)

        if output_path is None:
            output_fd, output_path = tempfile.mkstemp(suffix=".png")
            os.close(output_fd)
        else:
            output_path = Path(output_path)

        duration = float(len(y) / sr)
        time_offset = start_time
        rows = len(selected_plot_types)
        fig, axes = plt.subplots(rows, 1, figsize=(14, max(3.0 * rows, 4.0)), squeeze=False)
        axes_list = axes[:, 0]
        time_axis = np.linspace(time_offset, time_offset + duration, num=len(y), endpoint=False)

        for ax, plot_type in zip(axes_list, selected_plot_types):
            if plot_type == "waveform":
                ax.plot(time_axis, y, linewidth=0.7)
                ax.set_ylabel("Amplitude")
                ax.set_title("Waveform")
                ax.set_xlim(time_offset, time_offset + duration)

            elif plot_type == "mel_spectrogram":
                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=96)
                mel_db = librosa.power_to_db(mel, ref=np.max)
                img = librosa.display.specshow(
                    mel_db,
                    sr=sr,
                    x_axis="time",
                    y_axis="mel",
                    ax=ax,
                    x_coords=None,
                )
                ax.set_title("Mel Spectrogram (dB)")
                fig.colorbar(img, ax=ax, format="%+2.0f dB")

            elif plot_type == "rms_energy":
                rms = librosa.feature.rms(y=y)[0]
                times = librosa.frames_to_time(range(len(rms)), sr=sr) + time_offset
                ax.plot(times, rms, linewidth=1.0)
                ax.set_ylabel("RMS")
                ax.set_title("RMS Energy")
                ax.set_xlim(time_offset, time_offset + duration)

            elif plot_type == "onset_envelope":
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                times = librosa.frames_to_time(range(len(onset_env)), sr=sr) + time_offset
                ax.plot(times, onset_env, linewidth=1.0)
                ax.set_ylabel("Strength")
                ax.set_title("Onset Envelope")
                ax.set_xlim(time_offset, time_offset + duration)

            elif plot_type == "spectral_rolloff":
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
                times = librosa.frames_to_time(range(len(rolloff)), sr=sr) + time_offset
                ax.plot(times, rolloff, linewidth=1.0)
                ax.set_ylabel("Hz")
                ax.set_title("Spectral Rolloff")
                ax.set_xlim(time_offset, time_offset + duration)

            elif plot_type == "cqt_chroma":
                chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
                img = librosa.display.specshow(
                    chroma,
                    sr=sr,
                    x_axis="time",
                    y_axis="chroma",
                    ax=ax,
                )
                ax.set_title("CQT Chroma")
                fig.colorbar(img, ax=ax)

            elif plot_type == "bpm_curve":
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)
                bpm_values = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)
                dominant_bpm = bpm_values[np.argmax(tempogram, axis=0)]
                times = librosa.frames_to_time(range(tempogram.shape[1]), sr=sr) + time_offset
                ax.plot(times, dominant_bpm, linewidth=1.0)
                ax.set_ylabel("BPM")
                ax.set_title("Dominant Local Tempo Estimate")
                ax.set_xlim(time_offset, time_offset + duration)

            ax.set_xlabel("Time (s)")
            ax.grid(alpha=0.2)

        title = f"Audio Plot Inspection: {path.name}"
        if time_range:
            title += f" ({start_time:.2f}s-{end_time:.2f}s)"
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
        plt.close(fig)

        return str(output_path), selected_plot_types, start_time, end_time

    def _build_audio_plot_inspection_prompt(
        self,
        question: str,
        analysis_focus: str,
        plot_types: list[str],
        start_time: float,
        end_time: float,
    ) -> str:
        """Build the fixed bounded prompt for VLM plot inspection."""
        plot_list = ", ".join(plot_types)
        return f"""You are inspecting a single combined image of audio-derived plots. You are not listening to the audio.

Your job is to extract visual-acoustic evidence only. Do not directly answer the user question from plots alone.
Do not infer speech content, accent, speaker identity, relationship, intent, emotion, animal understanding, or object identity from these plots.
If a claim cannot be determined from the plots, explicitly say it cannot be determined.

User question for context:
{question}

Planner-requested analysis focus:
{analysis_focus}

Plot types shown:
{plot_list}

Audio time span shown:
{start_time:.3f}s to {end_time:.3f}s

Return only valid JSON with this exact schema:
{{
  "visual_acoustic_clues": ["short visual-acoustic observations"],
  "timeline": [
    {{"start": 0.0, "end": 1.5, "observation": "visible acoustic pattern"}}
  ],
  "focus_relevant_evidence": "what the plots support about the requested focus",
  "uncertain_or_not_determinable": ["claims that cannot be determined from these plots"],
  "recommended_next_tools": ["specific next evidence tools if needed"],
  "reliability": "high | medium | low"
}}
"""

    def _call_vlm_with_single_image(self, prompt: str, image_path: str | Path) -> str:
        """Call the configured VLM with exactly one image input."""
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                ],
            }
        ]

        return self._call_vlm_messages(messages)

    def _call_vlm_messages(self, messages: list[dict[str, Any]]) -> str:
        """Call the configured image-capable VLM and return streamed text content."""
        client = self._get_client()
        request_params: dict[str, Any] = {
            "model": self.vlm_model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.vlm_enable_thinking:
            extra_body: dict[str, Any] = {"enable_thinking": True}
            if self.vlm_thinking_budget is not None:
                extra_body["thinking_budget"] = self.vlm_thinking_budget
            request_params["extra_body"] = extra_body

        completion = client.chat.completions.create(**request_params)
        text_response = ""
        for chunk in completion:
            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                text_response += content
        return text_response

    def _parse_plot_inspection_response(
        self,
        response: str,
        plot_path: str,
    ) -> tuple[dict[str, Any], str | None]:
        """Parse VLM JSON response, preserving raw output on format failure."""
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            return (
                {
                    "visual_acoustic_clues": [],
                    "timeline": [],
                    "focus_relevant_evidence": "",
                    "uncertain_or_not_determinable": [
                        "The VLM response did not follow the required JSON format."
                    ],
                    "recommended_next_tools": [],
                    "reliability": "low",
                    "plot_path": plot_path,
                    "raw_response": response,
                },
                f"Failed to parse VLM response as JSON: {exc}",
            )

        if not isinstance(parsed, dict):
            return (
                {
                    "visual_acoustic_clues": [],
                    "timeline": [],
                    "focus_relevant_evidence": "",
                    "uncertain_or_not_determinable": [
                        "The VLM response was valid JSON but not an object."
                    ],
                    "recommended_next_tools": [],
                    "reliability": "low",
                    "plot_path": plot_path,
                    "raw_response": response,
                },
                "VLM response JSON was not an object.",
            )

        parsed["plot_path"] = plot_path
        return parsed, None

    def inspect_audio_plots(
        self,
        audio_path: str | Path,
        question: str,
        analysis_focus: str,
        plot_types: list[str] | None = None,
        time_range: dict[str, Any] | None = None,
    ) -> PlotInspectionResult:
        """Inspect a single combined audio-plot image with a bounded VLM prompt."""
        if not question or not question.strip():
            raise ValueError("question is required")
        if not analysis_focus or not analysis_focus.strip():
            raise ValueError("analysis_focus is required")

        plot_path, selected_plot_types, start_time, end_time = self._generate_combined_audio_plots(
            audio_path=audio_path,
            plot_types=plot_types,
            time_range=time_range,
        )
        prompt = self._build_audio_plot_inspection_prompt(
            question=question,
            analysis_focus=analysis_focus,
            plot_types=selected_plot_types,
            start_time=start_time,
            end_time=end_time,
        )

        try:
            raw_response = self._call_vlm_with_single_image(prompt, plot_path)
            structured_result, parsing_warning = self._parse_plot_inspection_response(
                raw_response,
                plot_path,
            )
            return PlotInspectionResult(
                plot_path=plot_path,
                structured_result=structured_result,
                raw_response=raw_response,
                vlm_model=self.vlm_model,
                parsing_warning=parsing_warning,
            )
        except Exception as e:
            try:
                Path(plot_path).unlink(missing_ok=True)
            except OSError:
                pass
            raise RuntimeError(f"Audio plot inspection failed: {e}") from e

    def verify_audio_quality(
        self,
        audio_path: str | Path,
        verification_prompt: str,
        reference_audio_path: str | Path | None = None,
    ) -> VerificationResult:
        """
        Verify audio quality by generating spectrogram and analyzing with VLM.
        
        Args:
            audio_path: Path to the processed/enhanced audio file to verify
            verification_prompt: Specific instructions on what to check
            reference_audio_path: Optional path to original audio for comparison
            
        Returns:
            VerificationResult with pass/fail status and analysis
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Generate spectrogram for the audio to verify
        spectrogram_path = self._generate_spectrogram(audio_path)
        
        try:
            # Build the verification prompt
            base_prompt = """You are an expert audio quality analyst. Examine the provided spectrogram(s) and assess audio quality.

Look for these common issues in the spectrogram:
- Horizontal bands indicating constant noise/hum
- Vertical spikes suggesting clicks, pops, or transient artifacts
- Irregular patterns indicating distortion or processing artifacts
- Missing frequency content suggesting over-filtering
- Unnatural gaps indicating dropped audio or severe denoising
- Excessive high-frequency noise or aliasing artifacts

{verification_prompt}

Respond in this exact format:
VERIFICATION_PASSED: [true/false]
QUALITY_ASSESSMENT: [Good/Acceptable/Poor]
ISSUES_FOUND:
- [issue 1]
- [issue 2]
...
RECOMMENDATIONS:
- [recommendation 1]
- [recommendation 2]
...
ANALYSIS: [Your detailed analysis of what you see in the spectrogram]
"""
            
            full_prompt = base_prompt.format(verification_prompt=verification_prompt)
            
            # Build message content with image(s)
            content = [{"type": "text", "text": full_prompt}]
            
            # Add spectrogram image
            with open(spectrogram_path, "rb") as f:
                img_base64 = base64.b64encode(f.read()).decode("utf-8")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            })
            
            # If reference audio provided, generate and add its spectrogram
            ref_spectrogram_path = None
            if reference_audio_path:
                ref_path = Path(reference_audio_path)
                if ref_path.exists():
                    ref_spectrogram_path = self._generate_spectrogram(ref_path)
                    content.append({
                        "type": "text",
                        "text": "\nReference (original) audio spectrogram for comparison:"
                    })
                    with open(ref_spectrogram_path, "rb") as f:
                        ref_img_base64 = base64.b64encode(f.read()).decode("utf-8")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{ref_img_base64}"}
                    })
            
            messages = [{"role": "user", "content": content}]
            text_response = self._call_vlm_messages(messages)
            
            # Parse response
            return self._parse_verification_response(
                text_response, spectrogram_path, ref_spectrogram_path
            )
            
        except Exception as e:
            # Clean up spectrogram files on error
            try:
                Path(spectrogram_path).unlink(missing_ok=True)
                if ref_spectrogram_path:
                    Path(ref_spectrogram_path).unlink(missing_ok=True)
            except:
                pass
            raise RuntimeError(f"Verification failed: {e}") from e

    def _parse_verification_response(
        self,
        response: str,
        spectrogram_path: str,
        ref_spectrogram_path: str | None,
    ) -> VerificationResult:
        """
        Parse VLM response into VerificationResult.
        
        Args:
            response: Raw text response from VLM
            spectrogram_path: Path to the generated spectrogram
            ref_spectrogram_path: Path to reference spectrogram (if any)
            
        Returns:
            Parsed VerificationResult
        """
        import re
        
        # Default values
        verification_passed = False
        quality_assessment = "Unknown"
        issues_found = []
        recommendations = []
        analysis = response  # Default to full response
        
        # Parse VERIFICATION_PASSED
        passed_match = re.search(r'VERIFICATION_PASSED:\s*(true|false)', response, re.IGNORECASE)
        if passed_match:
            verification_passed = passed_match.group(1).lower() == 'true'
        
        # Parse QUALITY_ASSESSMENT
        quality_match = re.search(r'QUALITY_ASSESSMENT:\s*(\w+)', response, re.IGNORECASE)
        if quality_match:
            quality_assessment = quality_match.group(1).capitalize()
        
        # Parse ISSUES_FOUND
        issues_section = re.search(r'ISSUES_FOUND:(.*?)(?=RECOMMENDATIONS:|ANALYSIS:|$)', 
                                   response, re.DOTALL | re.IGNORECASE)
        if issues_section:
            issues_text = issues_section.group(1)
            issues_found = [line.strip('- ').strip() 
                          for line in issues_text.split('\n') 
                          if line.strip().startswith('-')]
        
        # Parse RECOMMENDATIONS
        recs_section = re.search(r'RECOMMENDATIONS:(.*?)(?=ANALYSIS:|$)', 
                                 response, re.DOTALL | re.IGNORECASE)
        if recs_section:
            recs_text = recs_section.group(1)
            recommendations = [line.strip('- ').strip() 
                             for line in recs_text.split('\n') 
                             if line.strip().startswith('-')]
        
        # Parse ANALYSIS
        analysis_match = re.search(r'ANALYSIS:\s*(.+)', response, re.DOTALL | re.IGNORECASE)
        if analysis_match:
            analysis = analysis_match.group(1).strip()
        
        # Clean up reference spectrogram if it exists (keep main one for evidence)
        if ref_spectrogram_path:
            try:
                Path(ref_spectrogram_path).unlink(missing_ok=True)
            except:
                pass
        
        return VerificationResult(
            verification_passed=verification_passed,
            issues_found=issues_found,
            quality_assessment=quality_assessment,
            recommendations=recommendations,
            spectrogram_path=spectrogram_path,
            analysis=analysis,
            vlm_model=self.vlm_model,
        )
