"""
Model-backed planner base classes.

This module mirrors the structure of the model-backed frontend:
- explicit unified input schema
- explicit input format dispatch
- template-method hooks for model initialization and invocation
- strict output parsing and normalization
"""

from __future__ import annotations

from abc import abstractmethod
from enum import Enum
import json
from typing import Any

from pydantic import BaseModel, Field

from audio_agent.core.errors import PlannerError
from audio_agent.core.logging import get_logger
from audio_agent.core.schemas import (
    FormatCheckResult,
    FrontendOutput,
    InitialPlan,
    PlannerActionType,
    PlannerDecision,
    ToolCallRequest,
    ToolSpec,
)
from audio_agent.core.state import AgentState
from audio_agent.planner.base import BasePlanner
from audio_agent.tools.inventory import (
    PLANNER_TOOL_CATEGORY_ORDER,
    load_planner_tool_category_definitions,
)
from audio_agent.utils.model_io import parse_json_object_text, validate_message_sequence
from audio_agent.utils.prompt_io import load_prompt
from audio_agent.utils.skill_io import render_skills_reference


class PlannerInputFormat(str, Enum):
    """Supported planner backend input modes."""

    API_MODEL = "api_model"
    LOCAL_MODEL = "local_model"


class UnifiedPlannerInput(BaseModel):
    """Backend-agnostic planner input wrapper."""

    system_prompt: str = Field(..., min_length=1)
    task_type: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    messages: list[dict[str, Any]] = Field(default_factory=list)
    user_payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "forbid"}


class BaseModelPlanner(BasePlanner):
    """
    Template-method base for model-backed planners.

    Concrete subclasses mainly implement:
    - initialize_model()
    - call_model()
    """

    def __init__(
        self,
        model_config: dict[str, Any] | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model_config = model_config or {}
        self.max_retries = max_retries
        self.model_handle = self.initialize_model()

    @property
    def input_format(self) -> PlannerInputFormat:
        """Default planner input mode."""
        return PlannerInputFormat.API_MODEL

    @abstractmethod
    def initialize_model(self) -> Any:
        """Initialize and return provider/model handle."""
        raise NotImplementedError

    @abstractmethod
    def call_model(self, model_input: UnifiedPlannerInput) -> Any:
        """Invoke model/backend and return raw output."""
        raise NotImplementedError

    def call_model_with_tools(
        self,
        model_input: UnifiedPlannerInput,
        tools: list[dict[str, Any]],
        tool_choice: Any = "required",
        parallel_tool_calls: bool = True,
    ) -> Any:
        """Invoke model with native function-calling enabled.

        Subclasses that support OpenAI-style native function calling
        (``OpenAICompatiblePlanner`` and friends) override this. The default
        raises ``NotImplementedError`` so backends without native support
        (Gemini, local-model planners, etc.) fall back to the JSON-text
        path inside :meth:`decide` rather than silently misbehaving.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support native function calling. "
            "Override call_model_with_tools or extend OpenAICompatiblePlanner."
        )

    def supports_native_tools(self) -> bool:
        """Whether this backend supports :meth:`call_model_with_tools`.

        Used by :meth:`decide` to route through the native-tools path when
        available, and to fall back to the legacy JSON-text path otherwise.
        Subclasses that implement ``call_model_with_tools`` should return
        ``True``.
        """
        return False

    def build_initial_prompt_system_prompt(self) -> str:
        """Build system prompt for the question-oriented-prompt generation
        phase.

        Renders ``prompts/initial_prompt_system.md`` with the static
        Task-Oriented Caption Skills Reference (from
        ``prompts/task_oriented_caption_skill.md``) inlined into the
        ``{caption_skills_reference}`` placeholder. Both pieces are
        iteration-invariant for a given install, so they live together
        in the system slot — mirroring how ``decide_system.md`` and
        ``plan_system.md`` carry their static references.
        """
        try:
            caption_skills = load_prompt("task_oriented_caption_skill")
        except Exception:
            caption_skills = "(no caption skills reference available)"
        return load_prompt("initial_prompt_system").format(
            caption_skills_reference=caption_skills,
        )

    def build_initial_prompt_user_instruction(self, question: str) -> str:
        """Build user instruction for the question-oriented-prompt
        generation phase.

        Renders ``prompts/initial_prompt_user.md`` with iteration-volatile
        state only: the user question. Static material (the role,
        formatting rules, and the caption-skills reference) lives in
        the system prompt.
        """
        return load_prompt("initial_prompt_user").format(question=question)

    def build_api_model_input_for_initial_prompt(self, question: str) -> UnifiedPlannerInput:
        """Build API-style planner input for question-oriented prompt generation."""
        system_prompt = self.build_initial_prompt_system_prompt()
        user_text = self.build_initial_prompt_user_instruction(question)
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="initial_prompt",
            question=question,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": question, "task": "initial_prompt"},
            metadata={"planner_name": self.name, "input_format": PlannerInputFormat.API_MODEL.value},
        )

    def build_local_model_input_for_initial_prompt(self, question: str) -> UnifiedPlannerInput:
        """Build local-text-model input for question-oriented prompt generation."""
        system_prompt = self.build_initial_prompt_system_prompt()
        user_text = self.build_initial_prompt_user_instruction(question)
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="initial_prompt",
            question=question,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": question, "task": "initial_prompt"},
            metadata={"planner_name": self.name, "input_format": PlannerInputFormat.LOCAL_MODEL.value},
        )

    def build_plan_system_prompt(self) -> str:
        """Build system prompt for the initial planning phase.

        Renders ``prompts/plan_system.md`` with the static Task Skills
        Reference (from ``prompts/task_skills.yaml``) inlined into the
        ``{task_skills_reference}`` placeholder. Both pieces are
        iteration-invariant for a given install, so they live together
        in the system slot — mirroring how ``decide_system.md`` carries
        ``{decision_rules}`` and ``{tool_category_definitions}``.
        """
        skills_ref = render_skills_reference()
        return load_prompt("plan_system").format(
            task_skills_reference=(
                skills_ref or "(no task skills reference configured)"
            ),
        )

    def build_plan_user_instruction(
        self,
        question: str,
        frontend_output: FrontendOutput | None = None,
    ) -> str:
        """Build user instruction for the initial planning phase.

        Renders ``prompts/plan_user.md`` with iteration-volatile state
        only: the user question and the frontend caption. Static
        material (planning rules, schema, detailed-plan patterns, task
        skills reference) lives in the system prompt.
        """
        frontend_caption = (
            frontend_output.question_guided_caption
            if frontend_output else "No frontend caption available."
        )
        return load_prompt("plan_user").format(
            question=question,
            frontend_caption=frontend_caption,
        )

    def build_decision_system_prompt(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> str:
        """Build system prompt for the action decision phase.

        Renders ``prompts/decide_system.md`` with the static sections that
        govern agent reasoning across all iterations of a run:

        - ``{decision_rules}`` — full text of ``prompts/decide_rules.md``.
        - ``{tool_category_definitions}`` — definition + guideline per
          category present in the current ``available_tools`` (filtered via
          ``planner_tool_inventory.yaml``).

        The tool catalog itself is NOT rendered into the system text — it
        is delivered to the model via the ``tools=`` API parameter (see
        :meth:`_to_openai_tools`). ``available_tools`` is still accepted
        here so we can build the category-definitions block from it.
        """
        tool_category_definitions = self._build_tool_category_definitions(
            state, available_tools
        )
        return load_prompt("decide_system").format(
            decision_rules=load_prompt("decide_rules"),
            tool_category_definitions=(
                json.dumps(
                    tool_category_definitions, indent=2, ensure_ascii=False
                )
                if tool_category_definitions
                else "(no category definitions configured)"
            ),
        )

    def build_decision_user_instruction(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> str:
        """Build user instruction for the action decision phase.

        Renders ``prompts/decide_user.md`` with iteration-volatile state
        only: question, initial plan, loop budget, audio list, and a
        unified Evidence Ledger that merges ``evidence_log`` with
        ``tool_call_history`` (so tool-derived entries carry call args and
        success alongside the fuser-formatted content). The
        ``available_tools`` argument is kept for symmetry with the
        system-prompt builder; the catalog itself is rendered into the
        system message, not here.
        """
        initial_plan = state["initial_plan"]
        evidence_log = state.get("evidence_log", [])
        tool_history = state.get("tool_call_history", [])
        audio_list = state.get("audio_list", [])
        planner_trace = state.get("planner_trace", [])

        evidence_ledger = self._build_evidence_ledger(evidence_log, tool_history)
        reasoning_trace = self._format_planner_reasoning_trace(planner_trace)

        audio_summary_lines = [
            f"- {a.audio_id}: {a.description} (source: {a.source})"
            for a in audio_list
        ]
        audio_summary_text = (
            "\n".join(audio_summary_lines)
            if audio_summary_lines
            else "- audio_0: original input audio (source: original)"
        )

        return load_prompt("decide_user").format(
            question=state["question"],
            initial_plan=(
                json.dumps(
                    initial_plan.model_dump(mode="json"),
                    indent=2,
                    ensure_ascii=False,
                )
                if initial_plan is not None
                else "(no initial plan)"
            ),
            planner_reasoning_trace=reasoning_trace,
            evidence_ledger=(
                json.dumps(evidence_ledger, indent=2, ensure_ascii=False)
                if evidence_ledger
                else "(no evidence yet)"
            ),
            audio_list=audio_summary_text,
            step_count=state.get("step_count", 0),
            max_steps=state.get("max_steps", 10),
        )

    @staticmethod
    def _format_planner_reasoning_trace(
        planner_trace: list[PlannerDecision],
    ) -> str:
        """Render past planner rationales as a numbered chronological list.

        Each entry is the model's brief preamble (captured from
        ``message.content`` and stored in ``PlannerDecision.rationale``) for
        that round's tool emission. Empty rationales render as
        ``(no reasoning recorded)`` so the round numbering stays continuous.
        """
        if not planner_trace:
            return "(no prior rounds — this is your first decision)"
        lines: list[str] = []
        for i, decision in enumerate(planner_trace, start=1):
            rationale = (decision.rationale or "").strip()
            if not rationale:
                rationale = "(no reasoning recorded)"
            lines.append(f"Round {i}: {rationale}")
        return "\n".join(lines)

    @staticmethod
    def _build_evidence_ledger(
        evidence_log: list,
        tool_history: list,
    ) -> list[dict]:
        """Merge ``evidence_log`` and ``tool_call_history`` into one
        chronological ledger.

        Each tool call appends exactly one ``EvidenceItem`` (via the fuser,
        with ``evidence_type`` of ``"tool_output"`` for success or
        ``"error"`` for failure) and exactly one ``ToolCallRecord``. The Nth
        tool-derived evidence item corresponds to the Nth tool record, so
        we pair them by FIFO order over the chronologically-ordered
        ``evidence_log``. Frontend captions, frontend follow-ups, and
        format critiques are not paired (no tool record exists for them)
        and pass through unchanged.

        Per-entry shape::

            {
              "source": str,             # e.g. "trim_audio", "frontend:qwen", "format_check:..."
              "type":   str,             # evidence_type
              "content": str,            # fuser-formatted or raw content
              # Only for tool-derived entries:
              "step":        int,
              "args":        dict,
              "success":     bool,
              "output_keys": list[str],
            }
        """
        tool_records = list(tool_history)
        next_tool_idx = 0
        ledger: list[dict] = []
        for item in evidence_log:
            entry: dict = {
                "source": item.source,
                "type": item.evidence_type,
                "content": item.content,
            }
            if (
                item.evidence_type in ("tool_output", "error")
                and next_tool_idx < len(tool_records)
            ):
                rec = tool_records[next_tool_idx]
                next_tool_idx += 1
                entry["step"] = rec.step_number
                entry["args"] = rec.request.args
                entry["success"] = rec.result.success
                entry["output_keys"] = (
                    list(rec.result.output.keys()) if rec.result.output else []
                )
            ledger.append(entry)
        return ledger

    # =========================================================================
    # Native function-calling helpers (DashScope `tools=` parameter)
    # =========================================================================

    # Names of the three synthetic action-tools exposed alongside the real
    # catalog. The planner picks one of these when the round's decision is
    # to answer / hand off to the frontend / give up. See _to_openai_tools
    # below for their JSON-Schema definitions, and _parse_tool_calls_to_decision
    # for how they are converted back into PlannerDecision objects.
    ACTION_TOOL_ANSWER: str = "emit_final_answer"
    ACTION_TOOL_ASK_FRONTEND: str = "ask_frontend"
    ACTION_TOOL_GIVE_UP: str = "give_up"

    @classmethod
    def _exclusive_action_tool_names(cls) -> frozenset[str]:
        """Action tools that must be the only call in their round."""
        return frozenset(
            {
                cls.ACTION_TOOL_ANSWER,
                cls.ACTION_TOOL_ASK_FRONTEND,
                cls.ACTION_TOOL_GIVE_UP,
            }
        )

    @classmethod
    def _to_openai_tools(
        cls,
        available_tools: list[ToolSpec],
    ) -> list[dict[str, Any]]:
        """Convert the planner-visible catalog into OpenAI ``tools=`` shape.

        Wraps each real tool's ``input_schema`` (already JSON Schema) into a
        ``{"type": "function", "function": {...}}`` envelope, then appends
        the three synthetic action-tools (``emit_final_answer``,
        ``ask_frontend``, ``give_up``) so the model picks an action by
        calling a tool, rather than emitting a JSON ``PlannerDecision`` as
        text. Each synthetic tool's description states the exclusivity
        rule — the parser also enforces it server-side.
        """
        tools: list[dict[str, Any]] = []

        # Real tools — pass-through, only the wrapping is added.
        for spec in available_tools:
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": spec.input_schema or {"type": "object", "properties": {}},
                    },
                }
            )

        # Synthetic action-tools.
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": cls.ACTION_TOOL_ANSWER,
                    "description": (
                        "Signal that the accumulated evidence is sufficient and the "
                        "frontend final-answer node should generate the answer. "
                        "EXCLUSIVE: this MUST be the only tool call in the round."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rationale": {
                                "type": "string",
                                "description": (
                                    "Why the current evidence is sufficient to answer. "
                                    "Cite the specific evidence items / tool outputs that ground the answer."
                                ),
                            },
                            "confidence": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "description": "Confidence in answer-readiness (0.0 to 1.0).",
                            },
                        },
                        "required": ["rationale"],
                    },
                },
            }
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": cls.ACTION_TOOL_ASK_FRONTEND,
                    "description": (
                        "Re-perceive selected audios with the LALM frontend to resolve "
                        "uncertainty that cannot be answered by metadata/measurements/derivation. "
                        "EXCLUSIVE: this MUST be the only tool call in the round."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "selected_audio_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Non-empty list of audio_ids from Available Audio Files to send "
                                    "to the frontend for re-perception (e.g. ['audio_1'])."
                                ),
                                "minItems": 1,
                            },
                            "frontend_followup_prompt": {
                                "type": "string",
                                "description": "Exact prompt sent to the frontend model.",
                            },
                            "frontend_followup_goal": {
                                "type": "string",
                                "description": (
                                    "Optional record-only metadata describing what uncertainty "
                                    "this follow-up resolves; not sent to the frontend."
                                ),
                            },
                        },
                        "required": ["selected_audio_ids", "frontend_followup_prompt"],
                    },
                },
            }
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": cls.ACTION_TOOL_GIVE_UP,
                    "description": (
                        "Signal that the question cannot be answered with the available "
                        "tools and evidence. EXCLUSIVE: this MUST be the only tool call in the round."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Why the task cannot be completed (concrete and specific).",
                            },
                        },
                        "required": ["reason"],
                    },
                },
            }
        )

        return tools

    @classmethod
    def _parse_tool_calls_to_decision(
        cls,
        message: Any,
        available_tool_names: set[str] | None = None,
        is_final_step: bool = False,
    ) -> PlannerDecision:
        """Convert a model response message into a ``PlannerDecision``.

        The model is called with ``tool_choice="required"`` (or forced to
        ``emit_final_answer`` on the last allowed round), so every message
        is expected to carry ``message.tool_calls``. The message's free-text
        ``content`` (which may be empty) is captured as the round's
        ``rationale``.

        Action-tool exclusivity is enforced:

        - If any of ``emit_final_answer`` / ``ask_frontend`` / ``give_up``
          appears in the list, that one action wins. Other tool calls in the
          same round are discarded with a warning log.
        - Otherwise, every tool call must name a real tool from the supplied
          ``available_tool_names`` (if provided). Unknown names are NOT
          silently dropped — they are surfaced as invalid-tool-call markers
          (``ToolCallRequest`` with ``context["_invalid_tool_call"]=True``).
          The tool executor recognizes the marker and synthesizes a failing
          ``ToolResult`` so the failure becomes a normal evidence entry the
          next round can read and self-correct against. Same treatment for
          a response with no tool_calls at all, or one whose tool_calls all
          lack a function name.

        This is intentionally not a ``PlannerError`` path: raising would
        trigger the API retry wrapper, and a deterministic model would
        just re-emit the same bad output. Surfacing as evidence lets the
        main agent loop give the model corrective feedback over multiple
        rounds, bounded by ``max_steps``.
        """
        logger = get_logger()
        rationale = (getattr(message, "content", None) or "").strip()

        raw_calls = getattr(message, "tool_calls", None) or []
        if not raw_calls:
            if is_final_step:
                # Final round: emit_final_answer was forced but the response carried no
                # tool_calls (API gateway fallback, or the model ignored tool_choice).
                # Terminate by delegating to the frontend final-answer node instead of
                # emitting an invalid CALL_TOOL marker that would loop until the
                # recursion limit and yield no answer.
                logger.warning(
                    "Planner emitted no tool_calls on the final step; "
                    "synthesizing an ANSWER decision to terminate cleanly."
                )
                return PlannerDecision(
                    action=PlannerActionType.ANSWER,
                    rationale=rationale or "Final step reached; delegating to the frontend.",
                    draft_answer=rationale or None,
                    confidence=0.5,
                )
            # No tool_calls in response. Surface as an invalid marker so
            # the next round sees the failure and can correct.
            logger.warning(
                "Planner emitted no tool_calls (tool_choice='required'); "
                "surfacing as invalid-tool-call for next-round visibility."
            )
            return PlannerDecision(
                action=PlannerActionType.CALL_TOOL,
                rationale=rationale,
                selected_tool_calls=[
                    cls._build_invalid_tool_call(
                        tool_name="__no_tool_call__",
                        args={},
                        reason=(
                            "Model emitted no tool_calls in its response, "
                            "even though tool_choice='required' was set. "
                            "Possible causes: API gateway fallback, or "
                            "model ignored the tool_choice constraint. On "
                            "the next round, call one of the available "
                            "tools — or emit_final_answer / ask_frontend / "
                            "give_up if appropriate."
                        ),
                    )
                ],
                confidence=0.5,
            )

        # Normalize each tool_call into (name, args_dict, raw_args_str).
        parsed: list[tuple[str, dict[str, Any], str]] = []
        for raw in raw_calls:
            func = getattr(raw, "function", None) or {}
            name = getattr(func, "name", None) if not isinstance(func, dict) else func.get("name")
            raw_args = (
                getattr(func, "arguments", None)
                if not isinstance(func, dict)
                else func.get("arguments")
            )
            if not name or not isinstance(name, str):
                logger.warning("Planner emitted tool_call with no name; skipping.")
                continue
            args: dict[str, Any] = {}
            if raw_args is None:
                pass
            elif isinstance(raw_args, dict):
                args = raw_args
            else:
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Planner tool_call '%s' had malformed JSON arguments (%s); using empty args.",
                        name,
                        exc,
                    )
                    args = {}
            parsed.append((name, args, raw_args if isinstance(raw_args, str) else ""))

        if not parsed:
            # Every tool_call in the response lacked a usable function
            # name. Surface as an invalid marker rather than raise — see
            # the docstring rationale at the top of this method.
            logger.warning(
                "All %d emitted tool_calls had missing/invalid names; "
                "surfacing as invalid-tool-call for next-round visibility.",
                len(raw_calls),
            )
            return PlannerDecision(
                action=PlannerActionType.CALL_TOOL,
                rationale=rationale,
                selected_tool_calls=[
                    cls._build_invalid_tool_call(
                        tool_name="__nameless_tool_call__",
                        args={},
                        reason=(
                            f"All {len(raw_calls)} tool_calls in the "
                            "response had missing or non-string function "
                            "names; nothing usable. On the next round, "
                            "ensure each tool call has a valid function "
                            "name from the available catalog."
                        ),
                    )
                ],
                confidence=0.5,
            )

        # Detect exclusive action-tools first. If multiple were emitted in
        # one round (unlikely), the first one wins; the rest are ignored.
        action_tools = cls._exclusive_action_tool_names()
        action_hits = [(name, args) for name, args, _ in parsed if name in action_tools]
        if action_hits:
            picked_name, picked_args = action_hits[0]
            if len(parsed) > 1:
                discarded = [n for n, _, _ in parsed if n != picked_name]
                logger.warning(
                    "Planner emitted action tool '%s' alongside other tool_calls %s; "
                    "discarding the rest (exclusivity rule).",
                    picked_name,
                    discarded,
                )
            return cls._build_action_tool_decision(picked_name, picked_args, rationale)

        # All real-tool calls. Validate names against the catalog. Known
        # names go straight in as normal ToolCallRequests; unknown names
        # are surfaced as invalid-tool-call markers (NOT silently dropped)
        # so the executor records each as a failure and the next round
        # can see + correct against it. Mixed rounds (some valid, some
        # hallucinated) preserve both for full audit-trail symmetry.
        tool_calls: list[ToolCallRequest] = []
        for name, args, _ in parsed:
            if (
                available_tool_names is not None
                and name not in available_tool_names
            ):
                logger.warning(
                    "Planner emitted unknown tool '%s'; surfacing as "
                    "invalid-tool-call for next-round visibility.",
                    name,
                )
                tool_calls.append(
                    cls._build_invalid_tool_call(
                        tool_name=name,
                        args=args,
                        reason=(
                            f"Tool name '{name}' is not in the available "
                            "tool catalog. The available tools were "
                            "provided to you via the API's tools= "
                            "parameter. On the next round, call one of "
                            "those tools — or emit_final_answer / "
                            "ask_frontend / give_up if appropriate."
                        ),
                    )
                )
                continue
            tool_calls.append(
                ToolCallRequest(tool_name=name, args=args, context={})
            )

        # tool_calls is guaranteed non-empty here: ``parsed`` was checked
        # non-empty above, and every parsed entry produces exactly one
        # ToolCallRequest (real or invalid marker).
        primary_audio_id = cls._first_audio_id_from_args(parsed[0][1])

        return PlannerDecision(
            action=PlannerActionType.CALL_TOOL,
            rationale=rationale,
            selected_tool_calls=tool_calls,
            selected_audio_id=primary_audio_id,
            confidence=0.8,
        )

    @classmethod
    def _build_invalid_tool_call(
        cls,
        tool_name: str,
        args: dict[str, Any],
        reason: str,
    ) -> ToolCallRequest:
        """Build a ToolCallRequest that the tool executor will surface as
        a failure WITHOUT dispatching to the registry.

        Used when the model emits something we cannot honor (an unknown
        tool name, no tool_calls at all, etc.) and we want the failure
        to land in the evidence log so the next round can react —
        rather than raising and triggering the API retry wrapper, which
        a deterministic model would just repeat.

        The marker is ``context["_invalid_tool_call"] = True`` and the
        explanatory text lives at ``context["_invalid_reason"]``. The
        tool executor (``audio_agent/graph/nodes.py``'s
        ``tool_executor_node``) checks for the marker before
        ``_prepare_tool_request`` runs and synthesizes a failing
        ``ToolResult`` whose ``error_message`` is the reason text.
        """
        return ToolCallRequest(
            tool_name=tool_name,
            args=args,
            context={
                "_invalid_tool_call": True,
                "_invalid_reason": reason,
            },
        )

    @classmethod
    def _build_action_tool_decision(
        cls,
        name: str,
        args: dict[str, Any],
        rationale: str,
    ) -> PlannerDecision:
        """Synthesize a PlannerDecision for an exclusive action-tool call."""
        if name == cls.ACTION_TOOL_ANSWER:
            return PlannerDecision(
                action=PlannerActionType.ANSWER,
                rationale=rationale or str(args.get("rationale") or "(no rationale)"),
                confidence=float(args.get("confidence") or 0.7),
            )
        if name == cls.ACTION_TOOL_GIVE_UP:
            return PlannerDecision(
                action=PlannerActionType.FAIL,
                rationale=rationale or str(args.get("reason") or "(no reason given)"),
                confidence=0.0,  # the give_up tool schema declares no confidence field
            )
        if name == cls.ACTION_TOOL_ASK_FRONTEND:
            audio_ids = args.get("selected_audio_ids") or []
            if isinstance(audio_ids, str):  # tolerate the model sending a single string
                audio_ids = [audio_ids]
            return PlannerDecision(
                action=PlannerActionType.CALL_FRONTEND,
                rationale=rationale or "(no rationale)",
                selected_audio_ids=[str(aid) for aid in audio_ids],
                frontend_followup_prompt=str(args.get("frontend_followup_prompt") or "").strip() or None,
                frontend_followup_goal=(
                    str(args["frontend_followup_goal"]) if args.get("frontend_followup_goal") else None
                ),
                confidence=0.7,
            )
        raise PlannerError(
            f"Unknown action-tool name '{name}'",
            details={"args": args},
        )

    @staticmethod
    def _first_audio_id_from_args(args: dict[str, Any]) -> str | None:
        """Extract the first audio_id-looking value from a tool's args dict.

        Used only to populate ``PlannerDecision.selected_audio_id`` for
        logging / display continuity with the old singular-tool contract.
        Returns ``None`` if nothing audio-id-shaped is found.
        """
        for value in args.values():
            if isinstance(value, str) and value.startswith("audio_"):
                return value
        # Lists of audio_ids (e.g. some tools accept multiple inputs).
        for value in args.values():
            if isinstance(value, list) and value and isinstance(value[0], str) and value[0].startswith("audio_"):
                return value[0]
        return None

    # =========================================================================

    def _build_tool_category_definitions(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> list[dict[str, str]]:
        """Build category definitions for categories present in available tools."""
        categories_in_tools = {
            category
            for tool in available_tools
            for category in self._extract_tool_categories(tool.description)
        }
        config = state.get("config") or {}
        inventory_path = config.get("planner_tool_inventory_path")
        if not inventory_path or not categories_in_tools:
            return []

        definitions = load_planner_tool_category_definitions(inventory_path)
        return [
            {
                "category": category,
                "definition": definitions[category]["definition"],
                "guideline": definitions[category]["guideline"],
            }
            for category in PLANNER_TOOL_CATEGORY_ORDER
            if category in categories_in_tools
        ]

    def _extract_tool_categories(self, description: str) -> list[str]:
        """Extract planner inventory category labels from a formatted tool description."""
        categories: list[str] = []
        for line in description.splitlines():
            if line.startswith("Category:"):
                category = line.removeprefix("Category:").strip()
                if category:
                    categories.append(category)
        return categories

    def build_api_model_input_for_plan(self, question: str, frontend_output: FrontendOutput | None = None) -> UnifiedPlannerInput:
        """Build API-style planner input for initial planning."""
        system_prompt = self.build_plan_system_prompt()
        user_text = self.build_plan_user_instruction(question, frontend_output)
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="initial_plan",
            question=question,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": question, "task": "initial_plan"},
            metadata={"planner_name": self.name, "input_format": PlannerInputFormat.API_MODEL.value},
        )

    def build_local_model_input_for_plan(self, question: str, frontend_output: FrontendOutput | None = None) -> UnifiedPlannerInput:
        """Build local-text-model input for initial planning."""
        system_prompt = self.build_plan_system_prompt()
        user_text = self.build_plan_user_instruction(question, frontend_output)
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="initial_plan",
            question=question,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": question, "task": "initial_plan"},
            metadata={"planner_name": self.name, "input_format": PlannerInputFormat.LOCAL_MODEL.value},
        )

    def build_api_model_input_for_decision(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> UnifiedPlannerInput:
        """Build API-style planner input for action decision."""
        system_prompt = self.build_decision_system_prompt(state, available_tools)
        user_text = self.build_decision_user_instruction(state, available_tools)
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="decision",
            question=state["question"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": state["question"], "task": "decision"},
            metadata={"planner_name": self.name, "input_format": PlannerInputFormat.API_MODEL.value},
        )

    def build_local_model_input_for_decision(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> UnifiedPlannerInput:
        """Build local-text-model input for action decision."""
        system_prompt = self.build_decision_system_prompt(state, available_tools)
        user_text = self.build_decision_user_instruction(state, available_tools)
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="decision",
            question=state["question"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": state["question"], "task": "decision"},
            metadata={"planner_name": self.name, "input_format": PlannerInputFormat.LOCAL_MODEL.value},
        )

    def _validate_built_model_input(self, model_input: UnifiedPlannerInput) -> None:
        """Fail-fast validation for planner model input."""
        if not model_input.system_prompt.strip():
            raise PlannerError("Malformed planner model input: empty system_prompt")
        if not model_input.task_type.strip():
            raise PlannerError("Malformed planner model input: empty task_type")
        if not model_input.question.strip():
            raise PlannerError("Malformed planner model input: empty question")
        validate_message_sequence(
            model_input.messages,
            error_cls=PlannerError,
            context="Malformed planner model input",
        )

    def build_initial_prompt_model_input(self, question: str) -> UnifiedPlannerInput:
        """Dispatch question-oriented prompt input build by backend mode."""
        question = self.validate_question(question)
        mode = self.input_format
        if isinstance(mode, str):
            try:
                mode = PlannerInputFormat(mode)
            except ValueError as e:
                raise PlannerError(
                    "Unsupported planner input format",
                    details={"input_format": mode},
                ) from e
        elif not isinstance(mode, PlannerInputFormat):
            raise PlannerError(
                "Unsupported planner input format type",
                details={"input_format_type": type(mode).__name__},
            )

        if mode == PlannerInputFormat.API_MODEL:
            model_input = self.build_api_model_input_for_initial_prompt(question)
        elif mode == PlannerInputFormat.LOCAL_MODEL:
            model_input = self.build_local_model_input_for_initial_prompt(question)
        else:
            raise PlannerError(
                "Unsupported planner input format",
                details={"input_format": mode.value},
            )

        self._validate_built_model_input(model_input)
        return model_input

    def build_plan_model_input(self, question: str, frontend_output: FrontendOutput | None = None) -> UnifiedPlannerInput:
        """Dispatch planner initial-plan input build by backend mode."""
        question = self.validate_question(question)
        mode = self.input_format
        if isinstance(mode, str):
            try:
                mode = PlannerInputFormat(mode)
            except ValueError as e:
                raise PlannerError(
                    "Unsupported planner input format",
                    details={"input_format": mode},
                ) from e
        elif not isinstance(mode, PlannerInputFormat):
            raise PlannerError(
                "Unsupported planner input format type",
                details={"input_format_type": type(mode).__name__},
            )

        if mode == PlannerInputFormat.API_MODEL:
            model_input = self.build_api_model_input_for_plan(question, frontend_output)
        elif mode == PlannerInputFormat.LOCAL_MODEL:
            model_input = self.build_local_model_input_for_plan(question, frontend_output)
        else:
            raise PlannerError(
                "Unsupported planner input format",
                details={"input_format": mode.value},
            )

        self._validate_built_model_input(model_input)
        return model_input

    def build_decision_model_input(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> UnifiedPlannerInput:
        """Dispatch planner decision input build by backend mode."""
        self.validate_state(state)
        mode = self.input_format
        if isinstance(mode, str):
            try:
                mode = PlannerInputFormat(mode)
            except ValueError as e:
                raise PlannerError(
                    "Unsupported planner input format",
                    details={"input_format": mode},
                ) from e
        elif not isinstance(mode, PlannerInputFormat):
            raise PlannerError(
                "Unsupported planner input format type",
                details={"input_format_type": type(mode).__name__},
            )

        if mode == PlannerInputFormat.API_MODEL:
            model_input = self.build_api_model_input_for_decision(state, available_tools)
        elif mode == PlannerInputFormat.LOCAL_MODEL:
            model_input = self.build_local_model_input_for_decision(state, available_tools)
        else:
            raise PlannerError(
                "Unsupported planner input format",
                details={"input_format": mode.value},
            )

        self._validate_built_model_input(model_input)
        return model_input

    def normalize_plan_output(self, raw_output: Any) -> InitialPlan:
        """Normalize model output into InitialPlan."""
        if isinstance(raw_output, InitialPlan):
            return raw_output
        if isinstance(raw_output, str):
            raw_output = parse_json_object_text(
                raw_output,
                error_cls=PlannerError,
                subject="Planner",
            )
        if isinstance(raw_output, dict):
            required = {"approach", "focus_points", "possible_tool_types"}
            keys = set(raw_output.keys())
            missing = sorted(required - keys)
            if missing:
                raise PlannerError(
                    "Malformed initial plan output: missing required fields",
                    details={
                        "missing_fields": missing,
                        "output_keys": sorted(keys),
                        "raw_output": raw_output,
                    },
                )
            # Sanitize: remove None values for fields that have defaults
            fields_with_defaults = {"notes", "clarified_intent", "expected_output_format", 
                                    "requires_audio_output", "detailed_plan"}
            sanitized_output = {
                k: v for k, v in raw_output.items() 
                if v is not None or k not in fields_with_defaults
            }
            try:
                return InitialPlan(**sanitized_output)
            except Exception as e:
                raise PlannerError(
                    "Malformed initial plan output: schema validation failed",
                    details={
                        "error": str(e),
                        "output_keys": sorted(keys),
                        "raw_output": raw_output,
                        "sanitized_output": sanitized_output,
                    },
                ) from e
        raise PlannerError(
            "Malformed initial plan output: expected dict, JSON text, or InitialPlan",
            details={"output_type": type(raw_output).__name__, "raw_output": str(raw_output)[:1000]},
        )

    def normalize_decision_output(self, raw_output: Any) -> PlannerDecision:
        """Normalize model output into PlannerDecision."""
        if isinstance(raw_output, PlannerDecision):
            return raw_output
        if isinstance(raw_output, str):
            raw_output = parse_json_object_text(
                raw_output,
                error_cls=PlannerError,
                subject="Planner",
            )
        if isinstance(raw_output, dict):
            required = {"action", "rationale"}
            keys = set(raw_output.keys())
            missing = sorted(required - keys)
            if missing:
                raise PlannerError(
                    "Malformed planner decision output: missing required fields",
                    details={
                        "missing_fields": missing,
                        "output_keys": sorted(keys),
                        "raw_output": raw_output,
                    },
                )

            # Translate the legacy singular-tool fields
            # ({"selected_tool_name", "selected_tool_args"}) into the new
            # ``selected_tool_calls`` list. This keeps backends that still
            # emit a JSON PlannerDecision (dummy / gemini / mimo / qwen25)
            # compatible after the schema migration without rewriting them.
            translated_output = dict(raw_output)
            legacy_name = translated_output.pop("selected_tool_name", None)
            legacy_args = translated_output.pop("selected_tool_args", None)
            if legacy_name:
                translated_output["selected_tool_calls"] = [
                    ToolCallRequest(
                        tool_name=legacy_name,
                        args=legacy_args if isinstance(legacy_args, dict) else {},
                        context={},
                    )
                ]
            elif legacy_args is not None:
                # Args without a name — drop the orphan.
                pass

            # Sanitize: remove None values for fields that have defaults
            # so Pydantic uses defaults instead of failing validation.
            fields_with_defaults = {
                "confidence",
                "selected_tool_calls",
                "selected_audio_id",
                "selected_audio_ids",
                "frontend_followup_prompt",
                "frontend_followup_goal",
                "draft_answer",
            }
            sanitized_output = {
                k: v for k, v in translated_output.items()
                if v is not None or k not in fields_with_defaults
            }
            try:
                return PlannerDecision(**sanitized_output)
            except Exception as e:
                raise PlannerError(
                    "Malformed planner decision output: schema validation failed",
                    details={
                        "error": str(e),
                        "output_keys": sorted(keys),
                        "raw_output": raw_output,
                        "sanitized_output": sanitized_output,
                    },
                ) from e
        raise PlannerError(
            "Malformed planner decision output: expected dict, JSON text, or PlannerDecision",
            details={"output_type": type(raw_output).__name__, "raw_output": str(raw_output)[:1000]},
        )

    def _call_with_retries(self, callable, context: str):
        """Call model and normalize output with retries on PlannerError."""
        import random
        logger = get_logger()
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return callable()
            except PlannerError as e:
                last_error = e
                if attempt < self.max_retries:
                    delay = min(1.0 * (2 ** attempt), 8.0)
                    jitter = delay * random.uniform(0, 0.25)
                    sleep_time = delay + jitter
                    logger.warning(
                        f"{context} failed (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {sleep_time:.1f}s: {e}"
                    )
                    import time
                    time.sleep(sleep_time)
                else:
                    logger.error(f"{context} exhausted all retries: {e}")
        raise PlannerError(
            f"{context} failed after {self.max_retries + 1} attempts",
            details={"last_error": str(last_error), "retries": self.max_retries},
        ) from last_error

    def generate_question_oriented_prompt(self, question: str) -> str:
        """Generate a question-oriented prompt for the frontend model."""
        question = self.validate_question(question)
        model_input = self.build_initial_prompt_model_input(question)

        def _call():
            try:
                raw_output = self.call_model(model_input)
            except PlannerError:
                raise
            except Exception as e:
                raise PlannerError(
                    f"Planner model call failed during question-oriented prompt generation: {type(e).__name__}: {e}",
                    details={"planner": self.name},
                ) from e
            return self.normalize_question_oriented_prompt_output(raw_output)

        return self._call_with_retries(_call, "generate_question_oriented_prompt()")

    def normalize_question_oriented_prompt_output(self, raw_output: Any) -> str:
        """Normalize model output into a plain string prompt."""
        if isinstance(raw_output, str):
            stripped = raw_output.strip()
            if not stripped:
                raise PlannerError(
                    "Question-oriented prompt output is empty",
                    details={"raw_output": raw_output},
                )
            return stripped
        if isinstance(raw_output, dict):
            prompt = raw_output.get("question_oriented_prompt") or raw_output.get("prompt")
            if not prompt or not str(prompt).strip():
                raise PlannerError(
                    "Malformed question-oriented prompt output: missing prompt text",
                    details={"output_keys": sorted(raw_output.keys()), "raw_output": raw_output},
                )
            return str(prompt).strip()
        raise PlannerError(
            "Malformed question-oriented prompt output: expected str or dict",
            details={"output_type": type(raw_output).__name__, "raw_output": str(raw_output)[:1000]},
        )

    def plan(self, question: str, frontend_output: FrontendOutput | None = None) -> InitialPlan:
        """Question-and-caption initial planning phase."""
        question = self.validate_question(question)
        model_input = self.build_plan_model_input(question, frontend_output)

        def _call():
            try:
                raw_output = self.call_model(model_input)
            except PlannerError:
                raise
            except Exception as e:
                raise PlannerError(
                    f"Planner model call failed during initial planning: {type(e).__name__}: {e}",
                    details={"planner": self.name},
                ) from e
            return self.normalize_plan_output(raw_output)

        return self._call_with_retries(_call, "plan()")

    def decide(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> PlannerDecision:
        """Action decision phase using state + available tools.

        When the backend supports native function calling
        (:meth:`supports_native_tools` returns ``True``), the planner
        emits structured ``tool_calls`` directly:

        - Real tool names from ``available_tools`` map to
          ``PlannerDecision(action=CALL_TOOL, selected_tool_calls=[...])``.
        - Synthetic action-tools (``emit_final_answer`` / ``ask_frontend``
          / ``give_up``) map to ANSWER / CALL_FRONTEND / FAIL.

        On the final allowed round (``step_count >= max_steps - 1``),
        ``tool_choice`` is set to force ``emit_final_answer``, replacing the
        old "synthesize ANSWER inside the node" hack.

        Backends without native function calling fall back to the legacy
        JSON-text path via :meth:`normalize_decision_output`.
        """
        self.validate_state(state)
        model_input = self.build_decision_model_input(state, available_tools)

        if self.supports_native_tools():
            return self._decide_with_native_tools(model_input, state, available_tools)

        # Legacy JSON-text path (for backends without function-calling support).
        def _legacy_call():
            try:
                raw_output = self.call_model(model_input)
            except PlannerError:
                raise
            except Exception as e:
                raise PlannerError(
                    f"Planner model call failed during decision phase: {type(e).__name__}: {e}",
                    details={"planner": self.name},
                ) from e
            return self.normalize_decision_output(raw_output)

        return self._call_with_retries(_legacy_call, "decide()")

    def _decide_with_native_tools(
        self,
        model_input: "UnifiedPlannerInput",
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> PlannerDecision:
        """Decide via the native ``tools=`` API path."""
        tools = self._to_openai_tools(available_tools)
        available_tool_names = {spec.name for spec in available_tools} | self._exclusive_action_tool_names()

        step_count = int(state.get("step_count", 0) or 0)
        max_steps = int(state.get("max_steps", 10) or 10)
        is_final_step = step_count >= max_steps - 1
        if is_final_step:
            tool_choice: Any = {
                "type": "function",
                "function": {"name": self.ACTION_TOOL_ANSWER},
            }
        else:
            tool_choice = "required"

        def _call():
            try:
                message = self.call_model_with_tools(
                    model_input,
                    tools=tools,
                    tool_choice=tool_choice,
                    parallel_tool_calls=True,
                )
            except PlannerError:
                raise
            except Exception as e:
                raise PlannerError(
                    f"Planner native-tools call failed during decision phase: {type(e).__name__}: {e}",
                    details={"planner": self.name, "tool_choice": tool_choice},
                ) from e
            return self._parse_tool_calls_to_decision(
                message, available_tool_names, is_final_step=is_final_step
            )

        return self._call_with_retries(_call, "decide() [native tools]")

    # =============================================================================
    # Format Check Methods
    # =============================================================================

    def build_format_check_system_prompt(self) -> str:
        """Build system prompt for format checking phase."""
        return load_prompt("format_check_system")

    def build_format_check_user_instruction(
        self,
        proposed_answer: str,
        expected_format: str | None,
        question: str,
        requires_audio_output: bool = False,
    ) -> str:
        """Build user instruction for format checking phase."""
        return load_prompt("format_check_user").format(
            question=question,
            expected_format=expected_format or "No specific format required",
            proposed_answer=proposed_answer,
            is_audio_output_task="Yes" if requires_audio_output else "No",
        )

    def build_format_check_model_input(
        self,
        proposed_answer: str,
        expected_format: str | None,
        question: str,
        requires_audio_output: bool = False,
    ) -> UnifiedPlannerInput:
        """Build model input for format checking."""
        system_prompt = self.build_format_check_system_prompt()
        user_text = self.build_format_check_user_instruction(
            proposed_answer, expected_format, question, requires_audio_output
        )
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="format_check",
            question=question,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={
                "question": question,
                "task": "format_check",
                "expected_format": expected_format,
            },
            metadata={
                "planner_name": self.name,
                "task_type": "format_check",
            },
        )

    def normalize_format_check_output(self, raw_output: Any) -> FormatCheckResult:
        """Normalize model output into FormatCheckResult."""
        if isinstance(raw_output, FormatCheckResult):
            return raw_output
        if isinstance(raw_output, str):
            raw_output = parse_json_object_text(
                raw_output,
                error_cls=PlannerError,
                subject="Planner",
            )
        if isinstance(raw_output, dict):
            required = {"passed"}
            keys = set(raw_output.keys())
            missing = sorted(required - keys)
            if missing:
                raise PlannerError(
                    "Malformed format check output: missing required fields",
                    details={
                        "missing_fields": missing,
                        "output_keys": sorted(keys),
                        "raw_output": raw_output,
                    },
                )
            # Sanitize: remove None values for fields that have defaults
            fields_with_defaults = {"critique", "confidence"}
            sanitized_output = {
                k: v for k, v in raw_output.items() 
                if v is not None or k not in fields_with_defaults
            }
            try:
                return FormatCheckResult(**sanitized_output)
            except Exception as e:
                raise PlannerError(
                    "Malformed format check output: schema validation failed",
                    details={
                        "error": str(e),
                        "output_keys": sorted(keys),
                        "raw_output": raw_output,
                        "sanitized_output": sanitized_output,
                    },
                ) from e
        raise PlannerError(
            "Malformed format check output: expected dict, JSON text, or FormatCheckResult",
            details={"output_type": type(raw_output).__name__, "raw_output": str(raw_output)[:1000]},
        )

    def check_format(
        self,
        proposed_answer: str,
        expected_format: str | None,
        question: str,
        requires_audio_output: bool = False,
    ) -> FormatCheckResult:
        """
        Check if the proposed answer follows the expected output format.
        
        Validates format compliance only - does NOT check content correctness.
        """
        # If no format specified and answer is non-empty, auto-pass
        if not expected_format and proposed_answer and proposed_answer.strip():
            return FormatCheckResult(
                passed=True,
                critique=None,
                confidence=1.0,
            )
        
        model_input = self.build_format_check_model_input(
            proposed_answer, expected_format, question, requires_audio_output
        )

        def _call():
            try:
                raw_output = self.call_model(model_input)
            except PlannerError:
                raise
            except Exception as e:
                raise PlannerError(
                    f"Planner model call failed during format check: {type(e).__name__}: {e}",
                    details={"planner": self.name},
                ) from e
            return self.normalize_format_check_output(raw_output)

        return self._call_with_retries(_call, "check_format()")

    # =============================================================================
    # Evidence Summary Methods
    # =============================================================================

    def build_evidence_summary_system_prompt(self) -> str:
        """Build system prompt for evidence summarization phase."""
        return load_prompt("evidence_summary_system")

    def build_evidence_summary_user_instruction(self, state: AgentState) -> str:
        """Build user instruction for evidence summarization phase.

        The frontend caption is NOT rendered as its own section here —
        it already appears in ``evidence_log`` (as the first entry with
        ``evidence_type="question_guided_caption"``), so a standalone
        ``## Initial Frontend Output`` block would be a verbatim
        duplicate.
        """
        question = state["question"]
        evidence_log = state.get("evidence_log", [])
        planner_trace = state.get("planner_trace", [])
        tool_history = state.get("tool_call_history", [])
        clarified_intent = state.get("clarified_intent")
        expected_output_format = state.get("expected_output_format")

        evidence_text = "\n".join(
            f"[{item.source}] {item.content}"
            for item in evidence_log
        ) if evidence_log else "No evidence yet."

        planner_trace_text = "\n".join(
            f"Step {i+1}: {d.action.value} - {d.rationale}"
            for i, d in enumerate(planner_trace)
        ) if planner_trace else "No planner decisions yet."

        tool_history_text = "\n".join(
            f"- {record.request.tool_name}: success={record.result.success}"
            for record in tool_history
        ) if tool_history else "No tools called."

        return load_prompt("evidence_summary_user").format(
            question=question,
            evidence_text=evidence_text,
            planner_trace_text=planner_trace_text,
            tool_history_text=tool_history_text,
            clarified_intent=clarified_intent or "Not specified",
            expected_output_format=expected_output_format or "Not yet specified",
        )

    def build_evidence_summary_model_input(self, state: AgentState) -> UnifiedPlannerInput:
        """Build model input for evidence summarization."""
        system_prompt = self.build_evidence_summary_system_prompt()
        user_text = self.build_evidence_summary_user_instruction(state)
        return UnifiedPlannerInput(
            system_prompt=system_prompt,
            task_type="evidence_summary",
            question=state["question"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text},
            ],
            user_payload={"question": state["question"], "task": "evidence_summary"},
            metadata={"planner_name": self.name, "task_type": "evidence_summary"},
        )

    def normalize_evidence_summary_output(self, raw_output: Any) -> str:
        """Normalize model output into a plain string summary."""
        if isinstance(raw_output, str):
            stripped = raw_output.strip()
            if not stripped:
                raise PlannerError(
                    "Evidence summary output is empty",
                    details={"raw_output": raw_output},
                )
            return stripped
        if isinstance(raw_output, dict):
            summary = raw_output.get("summary") or raw_output.get("evidence_summary")
            if not summary or not str(summary).strip():
                raise PlannerError(
                    "Malformed evidence summary output: missing summary text",
                    details={"output_keys": sorted(raw_output.keys()), "raw_output": raw_output},
                )
            return str(summary).strip()
        raise PlannerError(
            "Malformed evidence summary output: expected str or dict",
            details={"output_type": type(raw_output).__name__, "raw_output": str(raw_output)[:1000]},
        )

    def summarize_evidence(self, state: AgentState) -> str:
        """
        Summarize accumulated evidence into a concise narrative.
        
        Uses the planner (text LLM) to compress evidence_log, planner_trace,
        and tool_call_history into a single neutral summary.
        """
        model_input = self.build_evidence_summary_model_input(state)

        def _call():
            try:
                raw_output = self.call_model(model_input)
            except PlannerError:
                raise
            except Exception as e:
                raise PlannerError(
                    f"Planner model call failed during evidence summarization: {type(e).__name__}: {e}",
                    details={"planner": self.name},
                ) from e
            return self.normalize_evidence_summary_output(raw_output)

        return self._call_with_retries(_call, "summarize_evidence()")
