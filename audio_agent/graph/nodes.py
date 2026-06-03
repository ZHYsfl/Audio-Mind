"""
LangGraph node functions for the audio agent.

Each node is a pure function that takes state and returns partial state updates.
Nodes follow fail-fast principles with explicit validation.
"""

import glob
import os
from typing import Any

from audio_agent.core.state import AgentState
from audio_agent.core.schemas import (
    EvidenceItem,
    FrontendOutput,
    InitialPlan,
    PlannerDecision,
    PlannerActionType,
    ToolCallRequest,
    ToolCallRecord,
    ToolResult,
    FinalAnswer,
    AudioItem,
    AudioOutput,
    FormatCheckResult,
)
from audio_agent.utils.model_io import parse_json_object_text
from audio_agent.core.constants import AgentStatus
from audio_agent.core.errors import (
    StateValidationError,
    FrontendError,
    PlannerError,
    ToolExecutionError,
    FusionError,
)
from audio_agent.core.logging import (
    log_node_start,
    log_node_end,
    log_planner_decision,
    log_error,
    log_state_transition,
    log_warning,
    log_info,
)
from audio_agent.utils.validation import validate_state_has_fields
from audio_agent.frontend.base import BaseFrontend
from audio_agent.planner.base import BasePlanner
from audio_agent.tools.executor import ToolExecutor
from audio_agent.tools.registry import ToolRegistry
from audio_agent.tools.inventory import apply_planner_tool_inventory
from audio_agent.tools.visibility import PlannerToolScope, filter_tool_specs
from audio_agent.fusion.base import BaseEvidenceFuser


def _build_audio_description(tool_name: str, args: dict, source_audio_id: str | None) -> str:
    """
    Build a descriptive string for generated audio based on tool and arguments.

    Args:
        tool_name: Name of the tool that generated the audio
        args: Tool arguments used
        source_audio_id: The audio_id of the source audio (if available)

    Returns:
        Human-readable description of the generated audio
    """
    # Extract source audio info - args may contain resolved paths or audio_ids
    # Try to find the source audio from args
    source = source_audio_id or "unknown"

    # Tool-specific description building
    if tool_name == "trim_audio":
        start = args.get("start_time", "?")
        duration = args.get("duration")
        end = args.get("end_time")

        if duration is not None:
            return f"Trimmed segment from {source} starting at {start}s, duration {duration}s"
        elif end is not None:
            return f"Trimmed segment from {source} from {start}s to {end}s"
        else:
            return f"Trimmed segment from {source} starting at {start}s"

    elif tool_name == "resample_audio":
        sample_rate = args.get("sample_rate", "?")
        return f"Resampled {source} to {sample_rate}Hz"

    elif tool_name == "convert_format":
        format_ext = args.get("format_ext", args.get("codec", "unknown"))
        return f"Converted {source} to {format_ext} format"

    elif tool_name == "convert_channels":
        channels = args.get("channels", "?")
        layout = "mono" if channels == 1 else "stereo" if channels == 2 else f"{channels}ch"
        return f"Converted {source} to {layout}"

    elif tool_name == "adjust_volume":
        volume_db = args.get("volume_db")
        volume_factor = args.get("volume_factor")
        if volume_db is not None:
            return f"Volume adjusted {source} by {volume_db}dB"
        elif volume_factor is not None:
            return f"Volume adjusted {source} by factor {volume_factor}"
        return f"Volume adjusted {source}"

    elif tool_name in ("loudnorm", "dynaudnorm"):
        return f"Loudness normalized {source}"

    elif tool_name == "highpass_filter":
        freq = args.get("frequency", "?")
        return f"High-pass filtered {source} at {freq}Hz"

    elif tool_name == "lowpass_filter":
        freq = args.get("frequency", "?")
        return f"Low-pass filtered {source} at {freq}Hz"

    elif tool_name == "afftdn_denoise":
        return f"Denoised {source} (FFT)"

    elif tool_name == "silenceremove":
        return f"Silence removed from {source}"

    elif tool_name == "change_tempo":
        ratio = args.get("tempo_ratio", "?")
        return f"Tempo changed {source} by {ratio}x"

    elif tool_name == "pitch_shift_rubberband":
        ratio = args.get("pitch_ratio", "?")
        return f"Pitch shifted {source} by {ratio}x"

    elif tool_name == "mix_audio":
        return f"Mixed multiple audio files"

    elif tool_name == "concat_audio":
        return f"Concatenated multiple audio files"

    elif tool_name == "add_echo":
        return f"Added echo to {source}"

    elif tool_name == "reverse_audio":
        return f"Reversed {source}"

    # Generic fallback - include key args
    key_args = []
    for k, v in args.items():
        if k in ("output_path", "audio_path"):
            continue
        if isinstance(v, (int, float, str)) and len(str(v)) < 20:
            key_args.append(f"{k}={v}")

    if key_args:
        return f"{tool_name} on {source} ({', '.join(key_args)})"
    return f"Generated by {tool_name} from {source}"


def create_initial_prompt_node(planner: BasePlanner):
    """
    Factory to create an initial prompt node with the given planner.

    Args:
        planner: Planner instance to generate the question-oriented prompt

    Returns:
        Node function compatible with LangGraph
    """

    def initial_prompt_node(state: AgentState) -> dict:
        """
        Generate a question-oriented prompt to guide the frontend model.

        Validates:
        - question exists

        Updates:
        - question_oriented_prompt
        """
        log_node_start(
            "initial_prompt_node",
            {
                "question": state.get("question", "")[:50],
            },
        )

        validate_state_has_fields(
            state,
            ["question"],
            context="initial_prompt_node",
        )

        question = state["question"]

        try:
            prompt = planner.generate_question_oriented_prompt(question)
        except PlannerError:
            raise
        except Exception as e:
            log_error("initial_prompt_node", e)
            raise PlannerError(
                f"Initial prompt generation failed: {e}",
                details={"planner": planner.name},
            ) from e

        if not prompt:
            raise PlannerError(
                "Planner returned empty question-oriented prompt",
                details={"planner": planner.name},
            )

        log_node_end(
            "initial_prompt_node",
            {
                "prompt_length": len(prompt),
            },
        )

        return {
            "question_oriented_prompt": prompt,
        }

    return initial_prompt_node


def create_frontend_evidence_node(frontend: BaseFrontend):
    """
    Factory to create a frontend evidence node with the given frontend.

    Args:
        frontend: Frontend instance to use for audio processing

    Returns:
        Node function compatible with LangGraph
    """

    def frontend_evidence_node(state: AgentState) -> dict:
        """
        Process audio(s) through frontend and generate initial evidence.

        Validates:
        - question and audio_list are present
        - at least one original audio exists in audio_list

        Updates:
        - initial_frontend_output
        - evidence_log (appends initial question-guided caption evidence)
        """
        log_node_start(
            "frontend_evidence_node",
            {
                "question": state.get("question", "")[:50],
                "audio_count": len(state.get("audio_list", [])),
            },
        )

        validate_state_has_fields(
            state,
            ["question", "audio_list"],
            context="frontend_evidence_node",
        )

        question = state["question"]
        audio_list = state["audio_list"]
        question_oriented_prompt = state.get("question_oriented_prompt")

        # Get all original audio paths (source == "original")
        # These are the input audios provided by the user, ordered by audio_id
        original_audios = sorted(
            [a for a in audio_list if a.source == "original"], key=lambda a: a.audio_id
        )
        if not original_audios:
            raise StateValidationError(
                "No original audio found in audio_list",
                details={"audio_ids": [a.audio_id for a in audio_list]},
            )

        audio_paths = [a.path for a in original_audios]

        try:
            output = frontend.run(question, audio_paths, question_oriented_prompt)
        except FrontendError:
            raise
        except Exception as e:
            log_error("frontend_evidence_node", e)
            raise FrontendError(f"Frontend failed: {e}", details={"frontend": frontend.name}) from e

        if output is None:
            raise FrontendError("Frontend returned None", details={"frontend": frontend.name})

        evidence_items = [
            EvidenceItem(
                source=f"frontend:{frontend.name}",
                content=output.question_guided_caption,
                evidence_type="question_guided_caption",
                confidence=0.5,
                metadata={},
            )
        ]

        log_node_end(
            "frontend_evidence_node",
            {
                "caption_length": len(output.question_guided_caption),
            },
        )

        return {
            "initial_frontend_output": output,
            "evidence_log": evidence_items,
        }

    return frontend_evidence_node


def create_initial_plan_node(planner: BasePlanner):
    """
    Factory to create an initial planning node.

    Initial planning must depend only on question (not frontend caption).
    """

    def initial_plan_node(state: AgentState) -> dict:
        """
        Generate initial plan from question only.

        Validates:
        - question exists

        Updates:
        - initial_plan
        - initial_plan_trace (appends plan)
        """
        log_node_start(
            "initial_plan_node",
            {
                "question": state.get("question", "")[:50],
            },
        )

        validate_state_has_fields(
            state,
            ["question", "initial_frontend_output"],
            context="initial_plan_node",
        )

        question = state["question"]
        frontend_output = state["initial_frontend_output"]

        try:
            plan = planner.plan(question, frontend_output)
        except PlannerError:
            raise
        except Exception as e:
            log_error("initial_plan_node", e)
            raise PlannerError(
                f"Initial planning failed: {e}",
                details={"planner": planner.name},
            ) from e

        if plan is None:
            raise PlannerError(
                "Planner returned None for initial plan",
                details={"planner": planner.name},
            )
        if not isinstance(plan, InitialPlan):
            raise PlannerError(
                "Planner returned malformed initial plan type",
                details={"actual_type": type(plan).__name__},
            )

        log_node_end(
            "initial_plan_node",
            {
                "focus_points": len(plan.focus_points),
                "possible_tool_types": len(plan.possible_tool_types),
                "clarified_intent": plan.clarified_intent,
                "expected_output_format": plan.expected_output_format,
            },
        )

        return {
            "initial_plan": plan,
            "initial_plan_trace": [plan],
            "clarified_intent": plan.clarified_intent,
            "expected_output_format": plan.expected_output_format,
        }

    return initial_plan_node


def create_planner_decision_node(
    planner: BasePlanner,
    registry: ToolRegistry,
    planner_tool_scope: PlannerToolScope = "core",
    planner_tool_inventory_path: str | None = None,
):
    """
    Factory to create an action decision planner node.

    Args:
        planner: Planner instance for action decision
        registry: Tool registry for available tools
        planner_tool_scope: Tool scope exposed to the planner ("core" or "all")
        planner_tool_inventory_path: Optional standalone inventory file for planner-facing tool text

    Returns:
        Node function compatible with LangGraph
    """

    def planner_decision_node(state: AgentState) -> dict:
        """
        Make an action decision based on current state.

        Termination strategy depends on the planner backend:

        - **Native function calling** (OpenAI-compatible planners): the
          planner forces ``emit_final_answer`` on the last allowed round
          via ``tool_choice``, so this node does not need a fallback.
        - **Legacy JSON-text backends** (dummy planner, custom planners
          that don't implement ``call_model_with_tools``): we keep the
          original synthesis hack — if ``step_count >= max_steps - 1``,
          synthesize an ANSWER decision and skip the planner LLM call.
          This guarantees termination for backends without per-call
          ``tool_choice`` control.

        Validates:
        - question exists
        - initial_frontend_output exists
        - initial_plan exists

        Updates:
        - current_decision
        - planner_trace (appends decision)
        """
        step_count = state.get("step_count", 0)
        max_steps = state.get("max_steps", 10)
        is_final_step = step_count >= max_steps - 1

        # Synthesis fallback for backends without native function calling:
        # force ANSWER on the final round to guarantee termination.
        if is_final_step and not planner.supports_native_tools():
            log_node_start(
                "planner_decision_node",
                {"step_count": step_count, "mode": "final_answer_fallback"},
            )
            decision = PlannerDecision(
                action=PlannerActionType.ANSWER,
                rationale=(
                    f"Maximum steps ({max_steps}) reached. Delegating final "
                    "answer generation to the frontend model. (Synthesised "
                    "fallback for non-native-tools planner.)"
                ),
                draft_answer=None,
                confidence=0.7,
            )
            log_planner_decision("answer", decision.rationale, None)
            log_node_end(
                "planner_decision_node",
                {"action": "answer", "mode": "final_answer_fallback"},
            )
            return {
                "current_decision": decision,
                "planner_trace": [decision],
            }

        log_node_start(
            "planner_decision_node",
            {
                "step_count": step_count,
                "evidence_count": len(state.get("evidence_log", [])),
            },
        )

        validate_state_has_fields(
            state,
            ["question", "initial_frontend_output", "initial_plan"],
            context="planner_decision_node",
        )

        available_tools = filter_tool_specs(
            registry.list_specs(),
            scope=planner_tool_scope,
        )
        available_tools = apply_planner_tool_inventory(
            available_tools,
            inventory_path=planner_tool_inventory_path,
        )

        try:
            decision = planner.decide(state, available_tools)
        except PlannerError:
            raise
        except Exception as e:
            log_error("planner_decision_node", e)
            raise PlannerError(
                f"Planner decision failed: {e}", details={"planner": planner.name}
            ) from e

        if decision is None:
            raise PlannerError("Planner returned None", details={"planner": planner.name})

        # Summarize the round's tool emission for the structured log.
        tool_call_summary = (
            ", ".join(tc.tool_name for tc in decision.selected_tool_calls)
            if decision.selected_tool_calls
            else None
        )
        log_planner_decision(
            decision.action.value,
            decision.rationale,
            tool_call_summary,
        )

        log_node_end(
            "planner_decision_node",
            {
                "action": decision.action.value,
                "tool_calls": tool_call_summary,
            },
        )

        return {
            "current_decision": decision,
            "planner_trace": [decision],
        }

    return planner_decision_node


def create_tool_executor_node(executor: ToolExecutor):
    """
    Factory to create a tool executor node.

    Args:
        executor: ToolExecutor instance for running tools

    Returns:
        Async node function compatible with LangGraph
    """

    async def tool_executor_node(state: AgentState) -> dict:
        """
        Execute all tool calls emitted by the planner for this round.

        With native function calling enabled, a single planner round can
        emit multiple parallel tool calls (``decision.selected_tool_calls``
        is a list). This node iterates the list sequentially, executing
        each call and accumulating:

        - One ``ToolCallRecord`` per call (appended to ``tool_call_history``).
        - One ``ToolResult`` per call (kept in ``latest_tool_results`` so the
          downstream ``evidence_fusion_node`` can fuse all of them).
        - Zero or one new ``AudioItem`` per audio-producing tool, appended
          to the running ``audio_list`` so a later call in the **same round**
          could reference it (though the planner is instructed not to do this).

        Within-round dependency errors (a tool referencing an audio_id
        that is not yet visible) fail that single call cleanly with an
        error ``ToolResult``, log a warning, and the loop continues. The
        planner sees the error in the next round's evidence and self-corrects.

        Validates:
        - ``current_decision`` exists and is CALL_TOOL
        - ``selected_tool_calls`` is non-empty

        Updates:
        - ``latest_tool_results`` (list)
        - ``tool_call_history`` (appends N records, one per executed call)
        - ``audio_list`` (extended with any new audio produced this round)
        """
        log_node_start("tool_executor_node")

        validate_state_has_fields(
            state,
            ["current_decision", "audio_list"],
            context="tool_executor_node",
        )

        decision: PlannerDecision = state["current_decision"]
        starting_audio_list: list[AudioItem] = list(state["audio_list"])

        if decision.action != PlannerActionType.CALL_TOOL:
            raise StateValidationError(
                f"tool_executor_node called with non-CALL_TOOL action: {decision.action}",
                details={"action": decision.action.value},
            )

        if not decision.selected_tool_calls:
            raise StateValidationError(
                "CALL_TOOL decision has no selected_tool_calls",
                details={"decision": decision.model_dump()},
            )

        temp_dir = state.get("temp_dir", "")
        step_count = state.get("step_count", 0)

        # Running audio_list grows as audio-producing tools complete within
        # this round; later tools in the same round see the new audio_ids
        # appended here. This is intentionally extensible despite the design
        # rule that the planner should not emit within-round dependencies —
        # if it does anyway, the resolver succeeds rather than failing.
        running_audio_list: list[AudioItem] = list(starting_audio_list)
        accumulated_records: list[ToolCallRecord] = []
        accumulated_results: list[ToolResult] = []
        per_call_summaries: list[dict[str, Any]] = []

        for call_index, tool_call in enumerate(decision.selected_tool_calls):
            tool_name = tool_call.tool_name
            ctx = tool_call.context or {}

            # Invalid-tool-call marker (model hallucinated a tool name,
            # emitted no tool_calls at all, etc. — see
            # ``BaseModelPlanner._build_invalid_tool_call``). Bypass
            # _prepare_tool_request + executor entirely and synthesize a
            # failing ToolResult whose error_message is the reason text.
            # This routes the failure into the normal evidence log so the
            # next planner round can see what went wrong and self-correct,
            # rather than dying inside the API-retry wrapper.
            if ctx.get("_invalid_tool_call"):
                reason = ctx.get("_invalid_reason") or (
                    f"Tool name '{tool_name}' is not in the available "
                    "tool catalog."
                )
                failed_result = ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output={},
                    error_message=reason,
                )
                accumulated_records.append(
                    ToolCallRecord(
                        request=tool_call,
                        result=failed_result,
                        step_number=step_count,
                    )
                )
                accumulated_results.append(failed_result)
                per_call_summaries.append(
                    {
                        "tool": tool_name,
                        "success": False,
                        "error": "invalid_tool_call",
                    }
                )
                continue

            try:
                request, auto_gen_id, auto_gen_path = _prepare_tool_request(
                    executor=executor,
                    tool_name=tool_name,
                    raw_args=tool_call.args or {},
                    audio_list=running_audio_list,
                    temp_dir=temp_dir,
                    state_question=state.get("question", ""),
                    step_count=step_count,
                )
            except StateValidationError as prep_error:
                # The most common reason for this is a within-round audio_id
                # dependency: the planner referenced an audio_id that doesn't
                # exist yet because the producer is later in this same round.
                # Surface it as a failing ToolResult and continue.
                error_msg = f"Argument resolution failed for '{tool_name}': {prep_error}"
                log_error("tool_executor_node", prep_error)
                failed_result = ToolResult(
                    tool_name=tool_name,
                    success=False,
                    output={},
                    error_message=error_msg,
                )
                failed_request = ToolCallRequest(
                    tool_name=tool_name,
                    args=tool_call.args or {},
                    context={
                        "question": state.get("question", ""),
                        "step_count": step_count,
                        "call_index": call_index,
                        "resolution_error": str(prep_error),
                    },
                )
                accumulated_records.append(
                    ToolCallRecord(
                        request=failed_request,
                        result=failed_result,
                        step_number=step_count,
                    )
                )
                accumulated_results.append(failed_result)
                per_call_summaries.append(
                    {
                        "tool": tool_name,
                        "success": False,
                        "error": "argument_resolution_failed",
                    }
                )
                continue

            try:
                result = await executor.execute(request)
            except ToolExecutionError:
                raise
            except Exception as e:
                log_error("tool_executor_node", e)
                raise ToolExecutionError(
                    f"Tool execution failed: {e}",
                    details={"tool_name": tool_name},
                ) from e

            accumulated_records.append(
                ToolCallRecord(
                    request=request,
                    result=result,
                    step_number=step_count,
                )
            )
            accumulated_results.append(result)

            # Register any newly-produced audio so the next call in this
            # round (if any) and the next round can see it.
            generated_path = result.output.get("generated_audio_path") if isinstance(result.output, dict) else None
            # Only register produced audio when the tool actually succeeded; a failed
            # audio-producing tool must not register an AudioItem pointing at a path it
            # never wrote (later rounds would hand that nonexistent path downstream).
            if result.success and (generated_path or auto_gen_id):
                if auto_gen_id and auto_gen_path:
                    actual_new_id = auto_gen_id
                    actual_path = generated_path or auto_gen_path
                else:
                    actual_new_id = f"audio_{len(running_audio_list)}"
                    actual_path = generated_path

                # Defensive: if the tool returned a relative path (some
                # tools echo back exactly what the planner gave them),
                # promote it to absolute under temp_dir so audio_id lookups
                # in future rounds get a path the dispatch layer can open.
                if (
                    actual_path
                    and isinstance(actual_path, str)
                    and not os.path.isabs(actual_path)
                    and temp_dir
                ):
                    actual_path = os.path.join(temp_dir, actual_path)

                tool_description = (
                    result.output.get("audio_description")
                    if isinstance(result.output, dict)
                    else None
                )
                if tool_description:
                    description = tool_description
                else:
                    description = _build_audio_description(
                        tool_name,
                        request.args,
                        request.context.get("selected_audio_id") if isinstance(request.context, dict) else None,
                    )
                running_audio_list.append(
                    AudioItem(
                        audio_id=actual_new_id,
                        path=actual_path,
                        source=tool_name,
                        description=description,
                        metadata=result.output.get("audio_metadata", {}) if isinstance(result.output, dict) else {},
                    )
                )
                per_call_summaries.append(
                    {
                        "tool": tool_name,
                        "success": result.success,
                        "new_audio_id": actual_new_id,
                    }
                )
            else:
                per_call_summaries.append(
                    {
                        "tool": tool_name,
                        "success": result.success,
                    }
                )

        updates: dict = {
            "latest_tool_results": accumulated_results,
            "tool_call_history": accumulated_records,
        }
        if len(running_audio_list) != len(starting_audio_list):
            updates["audio_list"] = running_audio_list

        log_node_end(
            "tool_executor_node",
            {
                "call_count": len(decision.selected_tool_calls),
                "calls": per_call_summaries,
            },
        )

        return updates

    return tool_executor_node


def _prepare_tool_request(
    executor: ToolExecutor,
    tool_name: str,
    raw_args: dict,
    audio_list: list[AudioItem],
    temp_dir: str,
    state_question: str,
    step_count: int,
) -> tuple[ToolCallRequest, str | None, str | None]:
    """Resolve audio_ids → paths and auto-fill output_path for one tool call.

    Returns ``(request, auto_generated_audio_id, auto_generated_output_path)``.
    When the tool produces audio but the planner didn't specify
    ``output_path``, we auto-generate one under ``temp_dir`` using the next
    audio_id (``audio_{len(audio_list)}``); the caller registers the new
    ``AudioItem`` only if execution succeeds.

    Raises ``StateValidationError`` when an audio_id referenced in args is
    not yet visible in ``audio_list`` (typically a within-round dependency
    the planner shouldn't have emitted).
    """
    args = dict(raw_args)

    def resolve_audio_id(audio_id: str) -> str:
        item = next((a for a in audio_list if a.audio_id == audio_id), None)
        if item:
            return item.path
        if temp_dir and os.path.isdir(temp_dir):
            matches = glob.glob(os.path.join(temp_dir, f"{audio_id}.*"))
            if matches:
                return matches[0]
        raise StateValidationError(
            f"Audio ID '{audio_id}' not found in audio_list or temp_dir",
            details={
                "audio_id": audio_id,
                "available_ids": [a.audio_id for a in audio_list],
                "temp_dir": temp_dir,
            },
        )

    auto_generated_id: str | None = None
    auto_generated_path: str | None = None
    primary_audio_id: str | None = None
    primary_audio_description: str | None = None

    try:
        tool = executor._registry.get(tool_name)
        input_schema = tool.spec.input_schema or {}
        properties = input_schema.get("properties", {})

        audio_params: list[str] = []
        for param_name, param_spec in properties.items():
            param_desc = param_spec.get("description", "").lower() if isinstance(param_spec, dict) else ""
            if "audio" in param_name or "audio" in param_desc or "path" in param_name:
                audio_params.append(param_name)

        resolved_args: dict[str, Any] = {}
        for param_name in audio_params:
            param_value = args.get(param_name)
            if param_value:
                if isinstance(param_value, str) and param_value.startswith("audio_"):
                    resolved_path = resolve_audio_id(param_value)
                    resolved_args[param_name] = resolved_path
                    if primary_audio_id is None and param_name in ("audio_path", "input_audio", "enrollment_audio"):
                        primary_audio_id = param_value
                        primary_audio_description = next(
                            (a.description for a in audio_list if a.audio_id == param_value),
                            None,
                        )
                else:
                    resolved_args[param_name] = param_value

        args = {**args, **resolved_args}

        # Auto-generate output_path for audio-producing tools when omitted.
        if "output_path" in audio_params and "output_path" not in args:
            if temp_dir and os.path.isdir(temp_dir):
                auto_generated_id = f"audio_{len(audio_list)}"
                auto_generated_path = os.path.join(temp_dir, f"{auto_generated_id}.wav")
                args["output_path"] = auto_generated_path
                log_info(
                    "auto_generated_output_path",
                    {
                        "audio_id": auto_generated_id,
                        "path": auto_generated_path,
                        "tool": tool_name,
                    },
                )
        elif "output_path" in args and isinstance(args["output_path"], str):
            # The system-prompt rule promises that a bare-filename
            # output_path will be placed under temp_dir. Honor it: if the
            # planner gave us a relative filename, resolve it against
            # temp_dir so the audio_list registers an absolute path.
            user_output_path = args["output_path"]
            if not os.path.isabs(user_output_path) and temp_dir:
                args["output_path"] = os.path.join(temp_dir, user_output_path)
                log_info(
                    "resolved_relative_output_path",
                    {
                        "original": user_output_path,
                        "resolved": args["output_path"],
                        "tool": tool_name,
                    },
                )
    except StateValidationError:
        raise
    except Exception as e:
        # Tool not in registry / schema introspection failed, or arg handling hit an
        # unexpected error. Fall through with raw args (the executor will surface a clean
        # error), but log it so a real bug in this block is not silently swallowed.
        log_warning(
            "tool_arg_resolution_failed",
            {"tool": tool_name, "error": f"{type(e).__name__}: {e}"},
        )

    request = ToolCallRequest(
        tool_name=tool_name,
        args=args,
        context={
            "question": state_question,
            "step_count": step_count,
            "selected_audio_id": primary_audio_id,
            "selected_audio_description": primary_audio_description,
        },
    )
    return request, auto_generated_id, auto_generated_path


def create_evidence_fusion_node(fuser: BaseEvidenceFuser):
    """
    Factory to create an evidence fusion node.

    Args:
        fuser: Evidence fuser instance

    Returns:
        Node function compatible with LangGraph
    """

    def evidence_fusion_node(state: AgentState) -> dict:
        """
        Fuse the round's results into evidence items.

        Handles both tool results (``latest_tool_results`` — a list, one
        entry per tool call in the round) and frontend follow-up results
        (``latest_frontend_followup_output``).

        For the tool path, the fuser is called once per ``ToolResult``;
        the accumulated EvidenceItems are appended to ``evidence_log``.
        For the frontend follow-up path, evidence was already added by
        ``frontend_followup_node``; this node just increments the step
        counter.

        ``step_count`` increments by 1 per round, regardless of how many
        tool calls executed in that round (per the "one LLM round = one
        step" budget convention).

        Validates:
        - Either ``latest_tool_results`` is non-empty OR
          ``latest_frontend_followup_output`` is set.

        Updates:
        - ``evidence_log`` (appends fused evidence for the tool path)
        - ``step_count`` (increments by 1)
        - ``latest_tool_results`` (clears to ``[]``)
        - ``latest_frontend_followup_output`` (clears to ``None``)
        """
        log_node_start("evidence_fusion_node")

        tool_results = state.get("latest_tool_results") or []
        followup_output = state.get("latest_frontend_followup_output")

        if not tool_results and followup_output is None:
            raise StateValidationError(
                "evidence_fusion_node requires either latest_tool_results or latest_frontend_followup_output",
                details={"context": "evidence_fusion_node"},
            )

        current_step = state.get("step_count", 0)
        updates: dict = {
            "step_count": current_step + 1,
            "latest_tool_results": [],
            "latest_frontend_followup_output": None,
        }

        if tool_results:
            # Tool path: fuse each result into one or more EvidenceItems.
            fused_items: list = []
            for tr in tool_results:
                try:
                    items = fuser.fuse(state, tr)
                except FusionError:
                    raise
                except Exception as e:
                    log_error("evidence_fusion_node", e)
                    raise FusionError(
                        f"Evidence fusion failed: {e}", details={"fuser": fuser.name}
                    ) from e
                if items is None:
                    raise FusionError(
                        "Fuser returned None", details={"fuser": fuser.name}
                    )
                fused_items.extend(items)

            updates["evidence_log"] = fused_items
            log_node_end(
                "evidence_fusion_node",
                {
                    "new_evidence_count": len(fused_items),
                    "tool_results_fused": len(tool_results),
                    "step_count": current_step + 1,
                },
            )
        else:
            # Frontend follow-up path: evidence was already added by frontend_followup_node
            log_node_end(
                "evidence_fusion_node",
                {
                    "new_evidence_count": 0,
                    "step_count": current_step + 1,
                    "source": "frontend_followup",
                },
            )

        return updates

    return evidence_fusion_node


def create_evidence_summarization_node(planner: BasePlanner):
    """
    Factory to create an evidence summarization node.

    Uses the planner (text LLM) to compress evidence_log, planner_trace,
    and tool_call_history into a single neutral narrative before final answer.

    Args:
        planner: Planner instance for summarization

    Returns:
        Node function compatible with LangGraph
    """

    def evidence_summarization_node(state: AgentState) -> dict:
        """
        Summarize accumulated evidence into a concise narrative.

        Validates:
        - current_decision exists and is ANSWER

        Updates:
        - evidence_summary
        """
        log_node_start("evidence_summarization_node")

        validate_state_has_fields(
            state,
            ["current_decision"],
            context="evidence_summarization_node",
        )

        decision: PlannerDecision = state["current_decision"]

        if decision.action != PlannerActionType.ANSWER:
            raise StateValidationError(
                f"evidence_summarization_node called with non-ANSWER action: {decision.action}",
                details={"action": decision.action.value},
            )

        try:
            summary = planner.summarize_evidence(state)
        except PlannerError:
            raise
        except Exception as e:
            log_error("evidence_summarization_node", e)
            raise PlannerError(
                f"Evidence summarization failed: {e}", details={"planner": planner.name}
            ) from e

        if not summary:
            raise PlannerError(
                "Planner returned empty evidence summary", details={"planner": planner.name}
            )

        log_node_end(
            "evidence_summarization_node",
            {
                "summary_length": len(summary),
            },
        )

        return {
            "evidence_summary": summary,
        }

    return evidence_summarization_node


def create_final_answer_node(frontend: BaseFrontend):
    """
    Factory to create a final answer node.

    Uses the frontend (audio-capable model) to generate the final answer
    from the original audio(s) and all accumulated context.

    Args:
        frontend: Frontend instance with generate_final_answer capability

    Returns:
        Node function compatible with LangGraph
    """

    def final_answer_node(state: AgentState) -> dict:
        """
        Generate the final answer using the frontend model.

        Validates:
        - current_decision exists and is ANSWER
        - audio_list contains at least one original audio

        Updates:
        - current_decision (sets draft_answer to the generated answer)
        """
        log_node_start("final_answer_node")

        validate_state_has_fields(
            state,
            ["current_decision", "audio_list", "question"],
            context="final_answer_node",
        )

        decision: PlannerDecision = state["current_decision"]
        audio_list: list[AudioItem] = state["audio_list"]
        question: str = state["question"]

        if decision.action != PlannerActionType.ANSWER:
            raise StateValidationError(
                f"final_answer_node called with non-ANSWER action: {decision.action}",
                details={"action": decision.action.value},
            )

        # Gather original audio paths
        original_audios = sorted(
            [a for a in audio_list if a.source == "original"], key=lambda a: a.audio_id
        )
        if not original_audios:
            raise StateValidationError(
                "No original audio found in audio_list",
                details={"audio_ids": [a.audio_id for a in audio_list]},
            )

        audio_paths = [a.path for a in original_audios]

        # Build context for the frontend
        format_check_result = state.get("format_check_result")
        format_critique = None
        if format_check_result and not format_check_result.passed:
            format_critique = format_check_result.critique

        context = {
            "evidence_summary": state.get("evidence_summary"),
            "evidence_log": state.get("evidence_log", []),
            "planner_trace": state.get("planner_trace", []),
            "tool_call_history": state.get("tool_call_history", []),
            "initial_plan": state.get("initial_plan"),
            "initial_frontend_output": state.get("initial_frontend_output"),
            "clarified_intent": state.get("clarified_intent"),
            "expected_output_format": state.get("expected_output_format"),
            "audio_list": audio_list,
            "format_critique": format_critique,
        }

        try:
            answer_text = frontend.generate_final_answer(
                question=question,
                audio_paths=audio_paths,
                context=context,
            )
        except FrontendError:
            raise
        except Exception as e:
            log_error("final_answer_node", e)
            raise FrontendError(
                f"Final answer generation failed: {e}", details={"frontend": frontend.name}
            ) from e

        if not answer_text:
            raise FrontendError(
                "Frontend returned empty final answer", details={"frontend": frontend.name}
            )

        # Parse JSON output from frontend to extract final_answer and rationale
        final_answer_text = answer_text
        rationale_text = None
        try:
            parsed = parse_json_object_text(
                answer_text,
                error_cls=FrontendError,
                subject="frontend final answer",
            )
            if "final_answer" in parsed and isinstance(parsed["final_answer"], str):
                final_answer_text = parsed["final_answer"].strip()
            if "rationale" in parsed and isinstance(parsed["rationale"], str):
                rationale_text = parsed["rationale"].strip()
        except FrontendError:
            # Fallback: treat entire response as final_answer, no rationale
            pass

        if not final_answer_text:
            raise FrontendError(
                "Frontend returned empty final answer after JSON parsing",
                details={"frontend": frontend.name},
            )

        # Update the decision with the extracted final answer
        updated_decision = decision.model_copy(update={"draft_answer": final_answer_text})

        log_node_end(
            "final_answer_node",
            {
                "answer_length": len(final_answer_text),
                "has_rationale": rationale_text is not None,
            },
        )

        return {
            "current_decision": updated_decision,
            "final_answer_rationale": rationale_text,
        }

    return final_answer_node


def answer_node(state: AgentState) -> dict:
    """
    Finalize the agent with an answer.

    Validates:
    - current_decision exists and is ANSWER
    - draft_answer is present

    Updates:
    - final_answer (with output_audio if applicable)
    - status (to ANSWERED)
    """
    log_node_start("answer_node")

    validate_state_has_fields(
        state,
        ["current_decision"],
        context="answer_node",
    )

    decision: PlannerDecision = state["current_decision"]

    if decision.action != PlannerActionType.ANSWER:
        raise StateValidationError(
            f"answer_node called with non-ANSWER action: {decision.action}",
            details={"action": decision.action.value},
        )

    if not decision.draft_answer:
        raise StateValidationError(
            "ANSWER decision has no draft_answer", details={"decision": decision.model_dump()}
        )

    # Build final answer
    evidence_log = state.get("evidence_log", [])
    evidence_summary = "\n".join(f"- [{e.source}] {e.content[:100]}..." for e in evidence_log)

    planner_trace = state.get("planner_trace", [])
    reasoning_trace = "\n".join(
        f"Step {i+1}: {d.action.value} - {d.rationale}" for i, d in enumerate(planner_trace)
    )

    # Determine output audio
    output_audio = None
    initial_plan = state.get("initial_plan")
    audio_list = state.get("audio_list", [])

    # Check if audio output is expected and available
    if initial_plan and initial_plan.requires_audio_output and audio_list:
        # Find the last non-original audio (most likely the output)
        generated_audios = [a for a in audio_list if a.source != "original"]
        if generated_audios:
            last_audio = generated_audios[-1]
            output_audio = AudioOutput(
                audio_id=last_audio.audio_id,
                path=last_audio.path,
                description=last_audio.description,
                metadata=last_audio.metadata,
            )

    # Build final answer with output_audio and rationale
    final_answer_rationale = state.get("final_answer_rationale")
    final_answer = FinalAnswer(
        answer=decision.draft_answer,
        confidence=decision.confidence,
        evidence_summary=evidence_summary,
        reasoning_trace=reasoning_trace,
        rationale=final_answer_rationale,
        output_audio=output_audio,
    )

    # Log warning if audio was expected but not found
    if initial_plan and initial_plan.requires_audio_output and not output_audio:
        log_warning(
            "answer_node", {"message": "Audio output was expected but not found in audio_list"}
        )

    log_state_transition(
        state.get("status", AgentStatus.RUNNING).value,
        AgentStatus.ANSWERED.value,
        "Planner provided final answer",
    )

    log_node_end(
        "answer_node",
        {
            "answer_length": len(final_answer.answer),
            "has_output_audio": output_audio is not None,
        },
    )

    return {
        "final_answer": final_answer,
        "status": AgentStatus.ANSWERED,
    }


def failure_node(state: AgentState) -> dict:
    """
    Handle agent failure.

    Updates:
    - error_message
    - status (to FAILED or EXHAUSTED)
    """
    log_node_start("failure_node")

    decision = state.get("current_decision")
    step_count = state.get("step_count", 0)
    max_steps = state.get("max_steps", 10)

    # Determine failure reason
    if step_count >= max_steps:
        error_message = f"Agent exhausted: reached max_steps ({max_steps})"
        new_status = AgentStatus.EXHAUSTED
    elif decision and decision.action == PlannerActionType.FAIL:
        error_message = f"Planner requested failure: {decision.rationale}"
        new_status = AgentStatus.FAILED
    else:
        error_message = "Agent failed for unknown reason"
        new_status = AgentStatus.FAILED

    log_state_transition(
        state.get("status", AgentStatus.RUNNING).value,
        new_status.value,
        error_message,
    )

    log_node_end("failure_node", {"status": new_status.value})

    return {
        "error_message": error_message,
        "status": new_status,
    }


def create_format_check_node(planner: BasePlanner):
    """
    Factory to create a format check node.

    Uses the planner (text LLM) to check if the proposed answer follows
    the expected output format requirements.

    Args:
        planner: Planner instance with check_format capability

    Returns:
        Node function compatible with LangGraph
    """

    def format_check_node(state: AgentState) -> dict:
        """
        Check the format of the proposed answer in current_decision.

        Validates:
        - current_decision exists and is ANSWER
        - draft_answer is present (the answer to check)

        Updates:
        - format_check_result
        - format_check_count (increments)
        - evidence_log (appends format critique as evidence if failed)
        """
        log_node_start("format_check_node")

        validate_state_has_fields(
            state,
            ["current_decision", "question"],
            context="format_check_node",
        )

        decision: PlannerDecision = state["current_decision"]
        question: str = state["question"]

        if decision.action != PlannerActionType.ANSWER:
            raise StateValidationError(
                f"format_check_node called with non-ANSWER action: {decision.action}",
                details={"action": decision.action.value},
            )

        if not decision.draft_answer:
            raise StateValidationError(
                "ANSWER decision has no draft_answer for format check",
                details={"decision": decision.model_dump()},
            )

        proposed_answer = decision.draft_answer
        expected_format = state.get("expected_output_format")

        # Check if this is an audio output task
        initial_plan = state.get("initial_plan")
        requires_audio_output = initial_plan.requires_audio_output if initial_plan else False

        # Call planner to check format
        try:
            format_check_result = planner.check_format(
                proposed_answer=proposed_answer,
                expected_format=expected_format,
                question=question,
                requires_audio_output=requires_audio_output,
            )
        except PlannerError:
            raise
        except Exception as e:
            log_error("format_check_node", e)
            raise PlannerError(
                f"Format check failed: {e}", details={"planner": planner.name}
            ) from e

        if format_check_result is None:
            raise PlannerError(
                "Planner returned None for format check", details={"planner": planner.name}
            )

        # Update format check count
        current_count = state.get("format_check_count", 0)
        new_count = current_count + 1

        # Prepare return updates
        updates: dict = {
            "format_check_result": format_check_result,
            "format_check_count": new_count,
        }

        # If format check failed, add critique as evidence
        if not format_check_result.passed and format_check_result.critique:
            critique_evidence = EvidenceItem(
                source=f"format_check:{planner.name}",
                content=f"Format check failed: {format_check_result.critique}",
                evidence_type="format_critique",
                confidence=format_check_result.confidence,
                metadata={
                    "proposed_answer": proposed_answer[:200],  # Truncate for metadata
                    "format_check_passed": format_check_result.passed,
                    "expected_format": expected_format,
                },
            )
            updates["evidence_log"] = [critique_evidence]
            log_node_end(
                "format_check_node",
                {
                    "passed": format_check_result.passed,
                    "confidence": format_check_result.confidence,
                    "critique": format_check_result.critique[:100],
                },
            )
        else:
            log_node_end(
                "format_check_node",
                {
                    "passed": format_check_result.passed,
                    "confidence": format_check_result.confidence,
                },
            )

        return updates

    return format_check_node


def create_frontend_followup_node(frontend: BaseFrontend):
    """
    Factory for the frontend follow-up node.

    This node allows the planner to explicitly request the frontend model
    to re-perceive selected audio artifacts (trimmed, isolated, denoised, etc.)
    with a custom prompt. The output is added as evidence.
    """

    def frontend_followup_node(state: AgentState) -> dict:
        log_node_start("frontend_followup_node", state)

        # Validate required state fields
        validate_state_has_fields(
            state,
            ["current_decision", "audio_list", "question"],
            context="frontend_followup_node",
        )

        decision = state["current_decision"]
        if decision.action != PlannerActionType.CALL_FRONTEND:
            raise StateValidationError(
                f"Expected action=CALL_FRONTEND, got {decision.action.value}",
                details={"context": "frontend_followup_node"},
            )

        # Validate follow-up fields
        audio_ids = decision.selected_audio_ids or []
        if not audio_ids:
            raise StateValidationError(
                "CALL_FRONTEND requires at least one selected_audio_id",
                details={"context": "frontend_followup_node"},
            )
        prompt = decision.frontend_followup_prompt
        if not prompt or not prompt.strip():
            raise StateValidationError(
                "CALL_FRONTEND requires non-empty frontend_followup_prompt",
                details={"context": "frontend_followup_node"},
            )

        # Resolve audio IDs to paths
        audio_list = state.get("audio_list", [])
        audio_map = {a.audio_id: a for a in audio_list}

        selected_audios = []
        for aid in audio_ids:
            if aid not in audio_map:
                raise StateValidationError(
                    f"Audio ID '{aid}' not found in audio_list",
                    details={
                        "available_ids": list(audio_map.keys()),
                        "context": "frontend_followup_node",
                    },
                )
            selected_audios.append(audio_map[aid])

        selected_paths = [a.path for a in selected_audios]

        # Call frontend on selected audio(s) using the dedicated follow-up prompt path.
        try:
            output = frontend.run_followup(
                question=state["question"],
                audio_paths=selected_paths,
                followup_prompt=prompt.strip(),
            )
        except Exception as e:
            raise FrontendError(
                f"Frontend follow-up failed: {type(e).__name__}: {e}",
                details={"frontend": frontend.name, "audio_ids": audio_ids},
            ) from e

        # Build evidence item from follow-up output
        evidence = EvidenceItem(
            source=f"frontend:{frontend.name}:followup",
            content=output.question_guided_caption,
            evidence_type="frontend_followup",
            confidence=0.7,
            metadata={
                "selected_audio_ids": audio_ids,
                "goal": decision.frontend_followup_goal,
                "prompt": prompt,
                "frontend_name": frontend.name,
            },
        )

        log_node_end("frontend_followup_node", state)

        return {
            "latest_frontend_followup_output": output,
            "evidence_log": [evidence],
        }

    return frontend_followup_node


# Convenience aliases for node creation
frontend_evidence_node = create_frontend_evidence_node
initial_plan_node = create_initial_plan_node
planner_decision_node = create_planner_decision_node
planner_node = create_planner_decision_node  # Backward-compatible alias
tool_executor_node = create_tool_executor_node
evidence_fusion_node = create_evidence_fusion_node
final_answer_node = create_final_answer_node
format_check_node = create_format_check_node
frontend_followup_node = create_frontend_followup_node
