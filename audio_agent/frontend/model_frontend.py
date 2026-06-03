"""
Model-backed frontend base classes.

This module mirrors the structure of the model-backed planner:
- explicit unified input schema
- explicit input format dispatch
- template-method hooks for model initialization and invocation
- strict output parsing and normalization
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from audio_agent.core.errors import FrontendError
from audio_agent.core.schemas import FrontendOutput
from audio_agent.core.logging import get_logger
from audio_agent.frontend.base import BaseFrontend
from audio_agent.utils.model_io import validate_message_sequence
from audio_agent.utils.prompt_io import load_prompt


class UnifiedFrontendInput(BaseModel):
    """
    Unified model-ready input for frontend calls.

    This contract keeps prompt and multimodal request shape consistent across providers.
    Provider-specific frontend implementations should consume this structure and only
    customize model initialization plus call mechanics.
    """

    system_prompt: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    audio_paths: list[str] = Field(..., min_length=1)
    user_payload: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FrontendInputFormat(str, Enum):
    """Supported default frontend input composition modes."""

    API_MODEL = "api_model"
    LOCAL_MULTIMODAL = "local_multimodal"


class BaseModelFrontend(BaseFrontend):
    """
    Template-method base class for model-backed frontends.

    Shared behavior lives here:
    - input validation
    - model input composition
    - output parsing and normalization

    Concrete frontends mainly implement:
    - `initialize_model`
    - `call_model`
    """

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model_config = model_config or {}
        self.max_retries = max_retries
        self.model_handle = self.initialize_model()

    @abstractmethod
    def initialize_model(self) -> Any:
        """Initialize and return provider/model handle."""
        raise NotImplementedError

    @abstractmethod
    def call_model(self, model_input: UnifiedFrontendInput) -> Any:
        """Invoke provider/model with unified input and return raw model output."""
        raise NotImplementedError

    @property
    def input_format(self) -> FrontendInputFormat:
        """
        Input format mode used by build_model_input dispatcher.

        Subclasses can override this to switch default composition mode.
        """
        return FrontendInputFormat.API_MODEL

    def build_frontend_task_instruction(
        self,
        question: str,
        audio_paths: list[str],
        question_oriented_prompt: str | None = None,
    ) -> str:
        """Shared instruction text reused across input builders."""
        audio_list_text = "\n".join([f"- Audio {i}: {path}" for i, path in enumerate(audio_paths)])
        prompt_text = question_oriented_prompt or "No customized prompt available."
        return load_prompt("frontend_user").format(
            question=question,
            audio_list=audio_list_text,
            question_oriented_prompt=prompt_text,
        )

    def build_followup_task_instruction(
        self,
        question: str,
        audio_paths: list[str],
        followup_prompt: str,
    ) -> str:
        """Instruction text for planner-requested frontend follow-up calls."""
        audio_list_text = "\n".join([f"- Audio {i}: {path}" for i, path in enumerate(audio_paths)])
        return load_prompt("frontend_followup_user").format(
            question=question,
            audio_list=audio_list_text,
            followup_prompt=followup_prompt,
        )

    def build_followup_common_fields(
        self,
        question: str,
        audio_paths: list[str],
        followup_prompt: str,
        input_format: FrontendInputFormat,
    ) -> dict[str, Any]:
        """
        Build provider-independent fields for frontend follow-up calls.

        Provider subclasses should reuse this and only customize the audio
        transport/package shape when the base message layout is not valid for
        their backend.
        """
        user_payload = self._build_common_user_payload(question, audio_paths)
        user_payload.update(
            {
                "task": "frontend_followup",
                "followup_prompt": followup_prompt,
            }
        )
        return {
            "system_prompt": load_prompt("frontend_followup_system"),
            "user_text": self.build_followup_task_instruction(
                question, audio_paths, followup_prompt
            ),
            "user_payload": user_payload,
            "metadata": {
                "frontend_name": self.name,
                "input_format": input_format.value,
                "task": "frontend_followup",
                "audio_count": len(audio_paths),
            },
        }

    def _build_common_user_payload(self, question: str, audio_paths: list[str]) -> dict[str, Any]:
        """Normalized provider-agnostic payload for adapters/logging."""
        return {
            "question": question,
            "audio": {
                "kind": "paths",
                "value": audio_paths,
                "count": len(audio_paths),
            },
            "task": "question_guided_audio_captioning",
            "output_format": "plain_text_caption",
        }

    def build_api_model_input(
        self,
        question: str,
        audio_paths: list[str],
        question_oriented_prompt: str | None = None,
    ) -> UnifiedFrontendInput:
        """
        Build API-hosted chat style input:
        - one system message
        - one user message with readable task text + audio reference(s)
        """
        user_payload = self._build_common_user_payload(question, audio_paths)
        user_payload["question_oriented_prompt"] = question_oriented_prompt
        system_prompt = load_prompt("frontend_system")
        user_text = self.build_frontend_task_instruction(
            question, audio_paths, question_oriented_prompt
        )

        return UnifiedFrontendInput(
            system_prompt=system_prompt,
            question=question,
            audio_paths=audio_paths,
            user_payload=user_payload,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            metadata={
                "frontend_name": self.name,
                "input_format": FrontendInputFormat.API_MODEL.value,
                "audio_count": len(audio_paths),
                "question_oriented_prompt": question_oriented_prompt,
            },
        )

    def build_local_multimodal_model_input(
        self,
        question: str,
        audio_paths: list[str],
        question_oriented_prompt: str | None = None,
    ) -> UnifiedFrontendInput:
        """
        Build local multimodal style input:
        - system prompt
        - user content list with text instruction + audio reference(s)
        """
        user_payload = self._build_common_user_payload(question, audio_paths)
        user_payload["question_oriented_prompt"] = question_oriented_prompt
        system_prompt = load_prompt("frontend_system")
        user_text = self.build_frontend_task_instruction(
            question, audio_paths, question_oriented_prompt
        )

        # Build content list with text and all audio files
        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for audio_path in audio_paths:
            content.append({"type": "audio", "audio": audio_path})

        return UnifiedFrontendInput(
            system_prompt=system_prompt,
            question=question,
            audio_paths=audio_paths,
            user_payload=user_payload,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": content,
                },
            ],
            metadata={
                "frontend_name": self.name,
                "input_format": FrontendInputFormat.LOCAL_MULTIMODAL.value,
                "audio_count": len(audio_paths),
                "question_oriented_prompt": question_oriented_prompt,
            },
        )

    def _validate_built_model_input(self, model_input: UnifiedFrontendInput) -> None:
        """Fail-fast validation for builder outputs."""
        if not model_input.system_prompt.strip():
            raise FrontendError("Malformed model input: empty system_prompt")
        if not model_input.question.strip():
            raise FrontendError("Malformed model input: empty question")
        if not model_input.audio_paths or len(model_input.audio_paths) == 0:
            raise FrontendError("Malformed model input: empty audio_paths")
        for i, path in enumerate(model_input.audio_paths):
            if not path or not path.strip():
                raise FrontendError(f"Malformed model input: empty audio path at index {i}")
        validate_message_sequence(
            model_input.messages,
            error_cls=FrontendError,
            context="Malformed model input",
        )

    def build_model_input(
        self,
        question: str,
        audio_paths: list[str],
        question_oriented_prompt: str | None = None,
    ) -> UnifiedFrontendInput:
        """Build model input via explicit format-mode dispatch."""
        mode = self.input_format
        if isinstance(mode, str):
            try:
                mode = FrontendInputFormat(mode)
            except ValueError as e:
                raise FrontendError(
                    "Unsupported frontend input format",
                    details={"input_format": mode},
                ) from e
        elif not isinstance(mode, FrontendInputFormat):
            raise FrontendError(
                "Unsupported frontend input format type",
                details={"input_format_type": type(mode).__name__},
            )

        if mode == FrontendInputFormat.API_MODEL:
            model_input = self.build_api_model_input(
                question, audio_paths, question_oriented_prompt
            )
        elif mode == FrontendInputFormat.LOCAL_MULTIMODAL:
            model_input = self.build_local_multimodal_model_input(
                question, audio_paths, question_oriented_prompt
            )
        else:
            raise FrontendError(
                "Unsupported frontend input format",
                details={"input_format": mode.value},
            )

        if not isinstance(model_input, UnifiedFrontendInput):
            raise FrontendError(
                "Malformed model input: builder must return UnifiedFrontendInput",
                details={"returned_type": type(model_input).__name__},
            )

        self._validate_built_model_input(model_input)
        return model_input

    def build_followup_api_model_input(
        self,
        question: str,
        audio_paths: list[str],
        followup_prompt: str,
    ) -> UnifiedFrontendInput:
        """Build API-style input for frontend follow-up re-perception."""
        common = self.build_followup_common_fields(
            question, audio_paths, followup_prompt, FrontendInputFormat.API_MODEL
        )

        return UnifiedFrontendInput(
            system_prompt=common["system_prompt"],
            question=question,
            audio_paths=audio_paths,
            user_payload=common["user_payload"],
            messages=[
                {"role": "system", "content": common["system_prompt"]},
                {"role": "user", "content": common["user_text"]},
            ],
            metadata=common["metadata"],
        )

    def build_followup_local_multimodal_model_input(
        self,
        question: str,
        audio_paths: list[str],
        followup_prompt: str,
    ) -> UnifiedFrontendInput:
        """Build local multimodal input for frontend follow-up re-perception."""
        common = self.build_followup_common_fields(
            question, audio_paths, followup_prompt, FrontendInputFormat.LOCAL_MULTIMODAL
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": common["user_text"]}]
        for audio_path in audio_paths:
            content.append({"type": "audio", "audio": audio_path})

        return UnifiedFrontendInput(
            system_prompt=common["system_prompt"],
            question=question,
            audio_paths=audio_paths,
            user_payload=common["user_payload"],
            messages=[
                {"role": "system", "content": common["system_prompt"]},
                {"role": "user", "content": content},
            ],
            metadata=common["metadata"],
        )

    def build_followup_model_input(
        self,
        question: str,
        audio_paths: list[str],
        followup_prompt: str,
    ) -> UnifiedFrontendInput:
        """Build follow-up model input via explicit format-mode dispatch."""
        if not followup_prompt or not followup_prompt.strip():
            raise FrontendError("Frontend follow-up requires a non-empty followup_prompt")

        mode = self.input_format
        if isinstance(mode, str):
            try:
                mode = FrontendInputFormat(mode)
            except ValueError as e:
                raise FrontendError(
                    "Unsupported frontend input format",
                    details={"input_format": mode},
                ) from e
        elif not isinstance(mode, FrontendInputFormat):
            raise FrontendError(
                "Unsupported frontend input format type",
                details={"input_format_type": type(mode).__name__},
            )

        if mode == FrontendInputFormat.API_MODEL:
            model_input = self.build_followup_api_model_input(
                question, audio_paths, followup_prompt
            )
        elif mode == FrontendInputFormat.LOCAL_MULTIMODAL:
            model_input = self.build_followup_local_multimodal_model_input(
                question, audio_paths, followup_prompt
            )
        else:
            raise FrontendError(
                "Unsupported frontend input format",
                details={"input_format": mode.value},
            )

        if not isinstance(model_input, UnifiedFrontendInput):
            raise FrontendError(
                "Malformed follow-up model input: builder must return UnifiedFrontendInput",
                details={"returned_type": type(model_input).__name__},
            )
        self._validate_built_model_input(model_input)
        return model_input

    def normalize_model_output(
        self,
        raw_output: Any,
        model_input: UnifiedFrontendInput,
    ) -> FrontendOutput:
        """
        Normalize model output into FrontendOutput.

        Supports:
        - FrontendOutput (returned directly)
        - Plain text string (treated as caption directly)
        - Dict (for backward compatibility with JSON outputs)

        Fail-fast requirement:
        - Empty outputs are rejected
        """
        if isinstance(raw_output, FrontendOutput):
            return raw_output

        # Treat string output as plain text caption (new default behavior)
        if isinstance(raw_output, str):
            caption = raw_output.strip()
            if not caption:
                raise FrontendError(
                    "Frontend returned empty caption",
                    details={"frontend": self.name},
                )
            return FrontendOutput(question_guided_caption=caption)

        # Dict output is not supported - model should return plain text
        if isinstance(raw_output, dict):
            raise FrontendError(
                "Frontend returned dict instead of plain text. "
                "The model should return plain text caption only.",
                details={"output_keys": list(raw_output.keys())},
            )

        raise FrontendError(
            "Malformed frontend output: expected str or FrontendOutput",
            details={"output_type": type(raw_output).__name__, "question": model_input.question},
        )

    def _call_with_retries(self, callable, context: str):
        """Call model and normalize output with retries on FrontendError."""
        import random

        logger = get_logger()
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return callable()
            except FrontendError as e:
                last_error = e
                if attempt < self.max_retries:
                    sleep_time = 0.5
                    logger.warning(
                        f"{context} failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {sleep_time:.1f}s: {e}"
                    )
                    import time

                    time.sleep(sleep_time)
                else:
                    logger.error(f"{context} exhausted all retries: {e}")
        raise FrontendError(
            f"{context} failed after {self.max_retries + 1} attempts",
            details={"last_error": str(last_error), "retries": self.max_retries},
        ) from last_error

    def run(
        self, question: str, audio_paths: list[str], question_oriented_prompt: str | None = None
    ) -> FrontendOutput:
        """
        Standardized frontend execution path:
        validate -> build unified input -> call model -> normalize output

        For multiple audios, makes separate API calls and combines results
        into a single FrontendOutput (since API models don't support multiple
        audios in one call).
        """
        self.validate_inputs(question, audio_paths)
        # Strip all paths
        stripped_paths = [p.strip() for p in audio_paths]

        if len(stripped_paths) == 1:
            # Single audio - normal processing path
            model_input = self.build_model_input(
                question.strip(), stripped_paths, question_oriented_prompt
            )

            def _call_single():
                try:
                    raw_output = self.call_model(model_input)
                except FrontendError:
                    raise
                except Exception as e:
                    raise FrontendError(
                        f"Model call failed: {type(e).__name__}: {e}",
                        details={"frontend": self.name},
                    ) from e
                return self.normalize_model_output(raw_output, model_input)

            return self._call_with_retries(_call_single, "run()")
        else:
            # Multiple audios - separate calls, combine results
            # API models like qwen3-omni-flash don't support multiple audios in one call
            captions = []
            for i, path in enumerate(stripped_paths):
                single_input = self.build_model_input(
                    question.strip(), [path], question_oriented_prompt
                )

                def _call_multi():
                    try:
                        raw_output = self.call_model(single_input)
                    except FrontendError:
                        raise
                    except Exception as e:
                        raise FrontendError(
                            f"Model call failed for audio {i}: {type(e).__name__}: {e}",
                            details={"frontend": self.name, "audio_index": i},
                        ) from e
                    return self.normalize_model_output(raw_output, single_input)

                output = self._call_with_retries(_call_multi, f"run() audio {i}")
                captions.append(f"Audio {i}: {output.question_guided_caption}")

            # Combine into single FrontendOutput
            combined_caption = "\n\n".join(captions)
            return FrontendOutput(question_guided_caption=combined_caption)

    def run_followup(
        self,
        question: str,
        audio_paths: list[str],
        followup_prompt: str,
    ) -> FrontendOutput:
        """Run a planner-authored follow-up request on selected audio artifact(s)."""
        self.validate_inputs(question, audio_paths)
        if not followup_prompt or not followup_prompt.strip():
            raise FrontendError("Frontend follow-up requires a non-empty followup_prompt")
        stripped_paths = [p.strip() for p in audio_paths]
        model_input = self.build_followup_model_input(
            question.strip(), stripped_paths, followup_prompt.strip()
        )

        def _call_followup():
            try:
                raw_output = self.call_model(model_input)
            except FrontendError:
                raise
            except Exception as e:
                raise FrontendError(
                    f"Frontend follow-up model call failed: {type(e).__name__}: {e}",
                    details={"frontend": self.name},
                ) from e
            return self.normalize_model_output(raw_output, model_input)

        return self._call_with_retries(_call_followup, "run_followup()")

    def build_final_answer_model_input(
        self,
        question: str,
        audio_paths: list[str],
        context: dict[str, Any],
    ) -> UnifiedFrontendInput:
        """Build model input for frontend final answer generation."""
        system_prompt = load_prompt("frontend_final_answer_system")

        evidence_summary = context.get("evidence_summary")
        evidence_log = context.get("evidence_log", [])
        evidence_text = (
            "\n".join(f"[{item.source}] {item.content}" for item in evidence_log)
            if evidence_log
            else "No evidence collected."
        )

        planner_trace = context.get("planner_trace", [])
        planner_trace_text = (
            "\n".join(
                f"Step {i+1}: {d.action.value} - {d.rationale}" for i, d in enumerate(planner_trace)
            )
            if planner_trace
            else "No planner decisions yet."
        )

        tool_history = context.get("tool_call_history", [])
        tool_history_text = (
            "\n".join(
                f"- {record.request.tool_name}: success={record.result.success}"
                for record in tool_history
            )
            if tool_history
            else "No tools called."
        )

        initial_plan = context.get("initial_plan")
        initial_plan_text = initial_plan.approach if initial_plan else "No initial plan."

        initial_frontend_output = context.get("initial_frontend_output")
        frontend_initial_text = (
            initial_frontend_output.question_guided_caption
            if initial_frontend_output
            else "No initial frontend output."
        )

        audio_summary = (
            "\n".join(f"- {a.audio_id}: {a.description}" for a in context.get("audio_list", []))
            if context.get("audio_list")
            else "No audio information."
        )

        expected_output_format = (
            context.get("expected_output_format") or "No specific format required."
        )
        format_critique = context.get("format_critique")
        format_critique_section = (
            f"\n## Format Critique (previous attempt failed)\n{format_critique}\n"
            if format_critique
            else ""
        )

        # Build the combined evidence/history section.
        # If a summary exists, it replaces the raw evidence log, planner trace, and tool history.
        if evidence_summary:
            evidence_and_history_text = (
                f"## Evidence and Reasoning Summary\n" f"{evidence_summary}\n" f"\n"
            )
        else:
            evidence_and_history_text = (
                f"## Evidence Log\n"
                f"{evidence_text}\n"
                f"\n"
                f"## Planner Reasoning Trace\n"
                f"{planner_trace_text}\n"
                f"\n"
                f"## Tool Call History\n"
                f"{tool_history_text}\n"
                f"\n"
            )

        user_text = load_prompt("frontend_final_answer_user").format(
            question=question,
            expected_output_format=expected_output_format,
            initial_plan_text=initial_plan_text,
            frontend_initial_text=frontend_initial_text,
            evidence_and_history_text=evidence_and_history_text,
            audio_summary=audio_summary,
            format_critique_section=format_critique_section,
        )

        content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
        for audio_path in audio_paths:
            content.append({"type": "audio", "audio": audio_path})

        return UnifiedFrontendInput(
            system_prompt=system_prompt,
            question=question,
            audio_paths=audio_paths,
            user_payload={
                "question": question,
                "audio": {"kind": "paths", "value": audio_paths, "count": len(audio_paths)},
                "task": "final_answer_generation",
            },
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
            metadata={
                "frontend_name": self.name,
                "input_format": FrontendInputFormat.LOCAL_MULTIMODAL.value,
                "task": "final_answer_generation",
                "audio_count": len(audio_paths),
            },
        )

    def generate_final_answer(
        self,
        question: str,
        audio_paths: list[str],
        context: dict[str, Any],
    ) -> str:
        """Generate final answer using the frontend model with all audio and context."""
        self.validate_inputs(question, audio_paths)
        stripped_paths = [p.strip() for p in audio_paths]

        model_input = self.build_final_answer_model_input(question.strip(), stripped_paths, context)
        if not isinstance(model_input, UnifiedFrontendInput):
            raise FrontendError(
                "Malformed model input: builder must return UnifiedFrontendInput",
                details={"returned_type": type(model_input).__name__},
            )
        self._validate_built_model_input(model_input)

        def _call():
            try:
                raw_output = self.call_model(model_input)
            except FrontendError:
                raise
            except Exception as e:
                raise FrontendError(
                    f"Final answer generation failed: {type(e).__name__}: {e}",
                    details={"frontend": self.name},
                ) from e
            if isinstance(raw_output, str):
                answer = raw_output.strip()
                if not answer:
                    raise FrontendError(
                        "Frontend returned empty final answer",
                        details={"frontend": self.name},
                    )
                return answer
            raise FrontendError(
                "Frontend returned non-string final answer",
                details={
                    "output_type": type(raw_output).__name__,
                    "frontend": self.name,
                },
            )

        return self._call_with_retries(_call, "generate_final_answer()")
