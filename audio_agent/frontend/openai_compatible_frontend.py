"""
OpenAI-compatible API frontend implementation.

Uses any OpenAI-compatible API that supports audio input.
Tested with qwen3-omni-flash via DashScope.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any

from audio_agent.core.errors import FrontendError
from audio_agent.frontend.model_frontend import (
    BaseModelFrontend,
    FrontendInputFormat,
    UnifiedFrontendInput,
)
from audio_agent.utils.audio_processing import preprocess_audio_for_api
from audio_agent.utils.prompt_io import load_prompt


# System prompt for direct-answer mode (AgentConfig.frontend_direct_answer): the frontend answers
# the question directly from the audio instead of producing a question-oriented-prompt caption.
_DIRECT_SYSTEM_PROMPT = (
    "You are an expert audio reasoning assistant. "
    "Listen to the provided audio carefully. The audio may contain sound events, music, speech, "
    "or any mixture of them. "
    "Analyze what you hear deeply and reason step by step. "
    "After your reasoning, conclude your response with a line starting with 'Final Answer:' "
    "followed by your chosen option. "
    "You may write either the full option text or the letter label (A, B, C, D)."
)


class OpenAICompatibleFrontend(BaseModelFrontend):
    """
    Generic frontend for OpenAI-compatible APIs with audio support.

    This frontend works with any API that follows the OpenAI chat completions
    format and supports audio input, including qwen3-omni-flash via DashScope.

    The frontend sends audio as base64-encoded data and receives
    a text caption relevant to the question.

    Args:
        model: Model name (e.g., "qwen3-omni-flash")
        api_key: API key. If None, reads from api_key_env environment variable.
        base_url: API base URL. Defaults to DashScope.
        api_key_env: Environment variable name to read API key from.
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        timeout: API request timeout in seconds
        direct_answer: If True, answer the question directly from the audio (no QoP caption)
    """

    def __init__(
        self,
        model: str = "qwen3-omni-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        temperature: float = 0.05,
        max_tokens: int = 4096,
        timeout: float = 120.0,
        max_retries: int = 3,
        direct_answer: bool = True,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        self._api_key_env = api_key_env
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._direct_answer = direct_answer

        super().__init__(max_retries=max_retries)

    @property
    def name(self) -> str:
        return f"openai_compatible_frontend_{self._model}"

    @property
    def direct_answer(self) -> bool:
        """Whether this frontend answers the question directly (no QoP-guided caption)."""
        return self._direct_answer

    @direct_answer.setter
    def direct_answer(self, value: bool) -> None:
        """Set the mode. AudioAgent uses this to apply AgentConfig.frontend_direct_answer."""
        self._direct_answer = bool(value)

    @property
    def input_format(self) -> FrontendInputFormat:
        return FrontendInputFormat.API_MODEL

    def initialize_model(self) -> Any:
        """Initialize and return OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError as e:
            raise FrontendError(
                "Missing openai package. Install with: pip install openai",
                details={"hint": "pip install openai"},
            ) from e

        # Get API key from constructor or environment
        api_key = self._api_key or os.environ.get(self._api_key_env)

        # Also try common alternative env vars
        if not api_key:
            for alt_env in ["DASHSCOPE_API_KEY", "OPENAI_API_KEY"]:
                api_key = os.environ.get(alt_env)
                if api_key:
                    break

        if not api_key:
            raise FrontendError(
                f"API key required for {self._model}",
                details={
                    "hint": f"Provide api_key parameter or set {self._api_key_env} environment variable",
                    "alternative_env_vars": ["DASHSCOPE_API_KEY", "OPENAI_API_KEY"],
                },
            )

        try:
            return OpenAI(
                api_key=api_key,
                base_url=self._base_url,
                timeout=self._timeout,
            )
        except Exception as e:
            raise FrontendError(
                f"Failed to initialize OpenAI client: {e}",
                details={"base_url": self._base_url, "model": self._model},
            ) from e

    def _encode_audio(self, audio_path: str) -> tuple[str, str]:
        """
        Read and encode audio file to base64.

        Automatically preprocesses oversized audio (e.g. multi-channel high-
        sample-rate WAV) by downsampling to 16 kHz mono so it stays within API
        base-64 data-URI limits.

        Args:
            audio_path: Path to the audio file

        Returns:
            Tuple of (base64_data_url, audio_format)
            The data URL format is: data:;base64,{encoded_data}

        Raises:
            FrontendError: If audio file not found or cannot be read
        """
        path = Path(audio_path)
        if not path.exists():
            raise FrontendError(
                f"Audio file not found: {audio_path}",
                details={"audio_path": audio_path},
            )

        # Determine format from extension
        audio_format = path.suffix.lstrip(".").lower()
        if audio_format not in ["wav", "mp3", "ogg", "m4a", "flac"]:
            audio_format = "wav"  # Default fallback

        try:
            audio_bytes = preprocess_audio_for_api(str(path))
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            return f"data:;base64,{audio_base64}", audio_format
        except Exception as e:
            raise FrontendError(
                f"Failed to read/encode audio file: {e}",
                details={"audio_path": audio_path},
            ) from e

    def build_api_model_input(
        self,
        question: str,
        audio_paths: list[str],
        question_oriented_prompt: str | None = None,
    ) -> UnifiedFrontendInput:
        """
        Build API input with audio as base64.

        Overrides base to include audio in the messages using the input_audio
        content type supported by OpenAI-compatible APIs.

        Note: audio_paths will always have exactly one audio when this is called,
        since the base run() method iterates through multiple audios separately.
        """
        # Encode audio to base64 (single audio)
        audio_path = audio_paths[0]
        audio_data_url, audio_format = self._encode_audio(audio_path)

        # Direct-answer mode (AgentConfig.frontend_direct_answer): answer the question directly;
        # the answer becomes the downstream evidence. When False, the QoP-caption path below runs.
        if self._direct_answer:
            direct_messages = [
                {"role": "system", "content": _DIRECT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},  # already includes the Options block
                        {
                            "type": "input_audio",
                            "input_audio": {"data": audio_data_url, "format": audio_format},
                        },
                    ],
                },
            ]
            return UnifiedFrontendInput(
                system_prompt=_DIRECT_SYSTEM_PROMPT,
                question=question,
                audio_paths=audio_paths,
                user_payload={
                    "question": question,
                    "audio": {"kind": "path", "value": audio_path},
                    "task": "direct_answer",
                    "output_format": "plain_text",
                },
                messages=direct_messages,
                metadata={
                    "frontend_name": self.name,
                    "input_format": FrontendInputFormat.API_MODEL.value,
                    "model": self._model,
                    "direct_caption_mode": True,
                },
            )

        # Load prompts from markdown files
        system_prompt = load_prompt("frontend_system")
        audio_list_text = f"- Audio 0: {audio_path}"
        prompt_text = question_oriented_prompt or "No customized prompt available."
        user_text = load_prompt("frontend_user").format(
            question=question,
            audio_list=audio_list_text,
            question_oriented_prompt=prompt_text,
        )

        # Build messages with single audio
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_data_url,
                            "format": audio_format,
                        },
                    },
                ],
            },
        ]

        return UnifiedFrontendInput(
            system_prompt=system_prompt,
            question=question,
            audio_paths=audio_paths,
            user_payload={
                "question": question,
                "audio": {"kind": "path", "value": audio_path},
                "task": "question_guided_audio_captioning",
                "output_format": "plain_text_caption",
                "question_oriented_prompt": question_oriented_prompt,
            },
            messages=messages,
            metadata={
                "frontend_name": self.name,
                "input_format": FrontendInputFormat.API_MODEL.value,
                "model": self._model,
                "question_oriented_prompt": question_oriented_prompt,
            },
        )

    def build_followup_api_model_input(
        self,
        question: str,
        audio_paths: list[str],
        followup_prompt: str,
    ) -> UnifiedFrontendInput:
        """Build API input for follow-up; only audio packaging is provider-specific."""
        stripped_paths = [p.strip() for p in audio_paths]
        common = self.build_followup_common_fields(
            question, stripped_paths, followup_prompt, FrontendInputFormat.API_MODEL
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": common["user_text"]}]
        for audio_path in stripped_paths:
            audio_data_url, audio_format = self._encode_audio(audio_path)
            content.append(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_data_url,
                        "format": audio_format,
                    },
                }
            )

        return UnifiedFrontendInput(
            system_prompt=common["system_prompt"],
            question=question,
            audio_paths=stripped_paths,
            user_payload=common["user_payload"],
            messages=[
                {"role": "system", "content": common["system_prompt"]},
                {"role": "user", "content": content},
            ],
            metadata={
                **common["metadata"],
                "model": self._model,
            },
        )

    def call_model(self, model_input: UnifiedFrontendInput) -> str:
        """
        Call the API model and return the caption text.

        Uses streaming API to handle potentially long responses.
        """
        client = self.model_handle

        # Prepare API call parameters
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": model_input.messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
            "modalities": ["text"],  # Text-only output (disable audio generation)
        }

        # Make API call
        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise FrontendError(
                f"API call failed for model {self._model}: {e}",
                details={"model": self._model, "error_type": type(e).__name__},
            ) from e

        # Process streaming response
        text_response = ""

        try:
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    text_response += chunk.choices[0].delta.content
        except Exception as e:
            raise FrontendError(
                f"Error processing API response: {e}",
                details={"model": self._model},
            ) from e

        if not text_response.strip():
            raise FrontendError(
                "API returned empty response",
                details={"model": self._model},
            )

        return text_response.strip()

    def generate_final_answer(
        self,
        question: str,
        audio_paths: list[str],
        context: dict[str, Any],
    ) -> str:
        """
        Generate final answer using the API frontend model with all audio and context.

        Overrides base to send multiple audios as base64 input_audio blocks.
        """
        self.validate_inputs(question, audio_paths)
        stripped_paths = [p.strip() for p in audio_paths]

        # Build unified input using base method
        model_input = self.build_final_answer_model_input(question.strip(), stripped_paths, context)
        if not isinstance(model_input, UnifiedFrontendInput):
            raise FrontendError(
                "Malformed model input: builder must return UnifiedFrontendInput",
                details={"returned_type": type(model_input).__name__},
            )

        # Replace local audio references with base64-encoded input_audio blocks
        original_messages = model_input.messages
        messages: list[dict[str, Any]] = []
        for msg in original_messages:
            if msg.get("role") == "user":
                content = msg.get("content", [])
                new_content: list[dict[str, Any]] = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "audio":
                        audio_path = block.get("audio")
                        audio_data_url, audio_format = self._encode_audio(audio_path)
                        new_content.append(
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_data_url,
                                    "format": audio_format,
                                },
                            }
                        )
                    else:
                        new_content.append(block)
                messages.append({"role": "user", "content": new_content})
            else:
                messages.append(msg)

        # Update model_input with API-compatible messages
        model_input.messages = messages
        model_input.metadata["input_format"] = FrontendInputFormat.API_MODEL.value

        # Make API call (non-streaming for final answer reliability)
        client = self.model_handle
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=model_input.messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                stream=False,
                modalities=["text"],
            )
        except Exception as e:
            raise FrontendError(
                f"API call failed for final answer: {e}",
                details={"model": self._model, "error_type": type(e).__name__},
            ) from e

        if not response.choices or len(response.choices) == 0:
            raise FrontendError(
                "Empty response from API",
                details={"model": self._model},
            )

        text_response = response.choices[0].message.content or ""
        if not text_response.strip():
            raise FrontendError(
                "API returned empty final answer",
                details={"model": self._model},
            )

        return text_response.strip()
