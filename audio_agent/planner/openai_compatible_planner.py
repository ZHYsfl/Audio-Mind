"""
Generic OpenAI-compatible API planner implementation.

Works with any OpenAI-compatible API including:
- qwen3.5-plus (with optional thinking mode)
- kimi-k2.5
- OpenAI GPT models
- Any other OpenAI-compatible endpoint

Example:
    planner = OpenAICompatiblePlanner(
        model="qwen3.5-plus",
        api_key="sk-xxx",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        enable_thinking=True,
    )
"""

from __future__ import annotations

import os
import sys
from typing import Any

from audio_agent.core.errors import PlannerError
from audio_agent.planner.model_planner import (
    BaseModelPlanner,
    PlannerInputFormat,
    UnifiedPlannerInput,
)


class OpenAICompatiblePlanner(BaseModelPlanner):
    """
    Generic planner for OpenAI-compatible APIs.
    
    This planner works with any API that follows the OpenAI chat completions
    format, including qwen3.5-plus, kimi-k2.5, and OpenAI's own models.
    
    Args:
        model: Model name (e.g., "qwen3.5-plus", "kimi-k2.5", "gpt-4")
        api_key: API key. If None, reads from api_key_env environment variable.
        base_url: API base URL. Defaults to OpenAI's official API.
        api_key_env: Environment variable name to read API key from.
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        enable_thinking: Enable thinking/reasoning mode (qwen3.5-plus specific)
        timeout: API request timeout in seconds
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        temperature: float = 0.05,
        max_tokens: int = 4096,
        enable_thinking: bool = False,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._api_key_env = api_key_env
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._enable_thinking = enable_thinking
        self._timeout = timeout

        super().__init__(max_retries=max_retries)

    @property
    def name(self) -> str:
        return f"openai_compatible_{self._model}"

    @property
    def input_format(self) -> PlannerInputFormat:
        return PlannerInputFormat.API_MODEL

    def initialize_model(self) -> Any:
        """Initialize and return OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError as e:
            raise PlannerError(
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
            raise PlannerError(
                f"API key required for {self._model}",
                details={
                    "hint": f"Provide api_key parameter or set {self._api_key_env} environment variable",
                    "alternative_env_vars": ["DASHSCOPE_API_KEY", "OPENAI_API_KEY"],
                },
            )

        # Build client kwargs
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self._timeout,
        }
        if self._base_url:
            client_kwargs["base_url"] = self._base_url

        try:
            return OpenAI(**client_kwargs)
        except Exception as e:
            raise PlannerError(
                f"Failed to initialize OpenAI client: {e}",
                details={"base_url": self._base_url, "model": self._model},
            ) from e

    def call_model(self, model_input: UnifiedPlannerInput) -> str:
        """
        Call the API model and return the response text.
        
        Handles different response formats from various API providers.
        """
        client = self.model_handle

        # Prepare API call parameters
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": model_input.messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }

        # Add model-specific extras
        if self._enable_thinking:
            kwargs["extra_body"] = {"enable_thinking": True}

        # Make API call
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise PlannerError(
                f"API call failed for model {self._model}: {e}",
                details={"model": self._model, "error_type": type(e).__name__},
            ) from e

        # Validate response
        if not response.choices or len(response.choices) == 0:
            raise PlannerError(
                "Empty response from API",
                details={"model": self._model},
            )

        message = response.choices[0].message
        return self._extract_content(message)

    def supports_native_tools(self) -> bool:
        """OpenAI-compatible providers (incl. DashScope) all support tools=."""
        return True

    def call_model_with_tools(
        self,
        model_input: UnifiedPlannerInput,
        tools: list[dict[str, Any]],
        tool_choice: Any = "required",
        parallel_tool_calls: bool = True,
    ) -> Any:
        """
        Call the API model with native function-calling enabled and return
        the raw response ``message`` object (not just its text content).

        The caller is responsible for parsing ``message.tool_calls`` and
        ``message.content``. Unlike :meth:`call_model`, this does NOT collapse
        the response down to a string — the structured tool_calls are exactly
        what we want.

        Args:
            model_input: The same unified planner input used by ``call_model``.
            tools: List of OpenAI-shaped tool definitions
                (``[{"type": "function", "function": {...}}, ...]``). Build
                via ``BaseModelPlanner._to_openai_tools(...)``.
            tool_choice: Either ``"auto"``, ``"required"``, ``"none"``, or
                ``{"type": "function", "function": {"name": "<name>"}}`` to
                force a specific tool (e.g. ``emit_final_answer`` on the
                final allowed planner round).
            parallel_tool_calls: Allow the model to emit multiple tool calls
                in one response. Defaults to ``True`` — that's the whole
                point of this path.

        Returns:
            ``response.choices[0].message`` (OpenAI SDK message object with
            ``.content``, ``.tool_calls``, etc).
        """
        client = self.model_handle

        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": model_input.messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "tools": tools,
            "tool_choice": tool_choice,
            "parallel_tool_calls": parallel_tool_calls,
        }
        # DashScope's qwen models default to thinking mode for some variants,
        # which rejects ``tool_choice`` values of ``"required"`` or a forced
        # function dict. We must pass the flag explicitly (in either
        # direction) when using native function calling so the model knows
        # to disable thinking mode. The text-only ``call_model`` path stays
        # with its previous behaviour (only sets the flag when True).
        kwargs["extra_body"] = {"enable_thinking": bool(self._enable_thinking)}

        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            raise PlannerError(
                f"API call (with tools) failed for model {self._model}: {e}",
                details={
                    "model": self._model,
                    "error_type": type(e).__name__,
                    "tool_count": len(tools),
                },
            ) from e

        if not response.choices:
            raise PlannerError(
                "Empty response from API (with tools)",
                details={"model": self._model},
            )

        return response.choices[0].message

    def _extract_content(self, message) -> str:
        """
        Extract text content from API response message.
        
        Handles different response formats:
        - Standard OpenAI: message.content
        - qwen3.5-plus thinking: message.reasoning_content (logged) + message.content (returned)
        - Others: fallback to content
        """
        # Log thinking content for debugging if present (but don't include in output)
        if hasattr(message, "reasoning_content") and message.reasoning_content:
            # Log to stderr for debugging purposes
            print(f"[Thinking] {message.reasoning_content[:500]}...", file=sys.stderr)

        # Return only the actual content (JSON response from model)
        if hasattr(message, "content") and message.content:
            return message.content

        raise PlannerError(
            "Empty content in API response",
            details={"message_attrs": dir(message)},
        )
