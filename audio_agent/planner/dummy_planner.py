"""
Dummy planner implementation for testing and development.

Uses simple deterministic rules instead of an LLM.
"""

from audio_agent.core.state import AgentState
from audio_agent.core.schemas import (
    InitialPlan,
    PlannerDecision,
    PlannerActionType,
    ToolCallRequest,
    ToolSpec,
    FormatCheckResult,
)
from audio_agent.core.errors import PlannerError
from audio_agent.planner.base import BasePlanner


class DummyPlanner(BasePlanner):
    """
    Dummy planner with deterministic behavior.
    
    Logic:
    - plan(question): produce a simple deterministic initial plan
    - decide(...):
      1. If no tools have been called yet, call "dummy_asr"
      2. After one tool call, ANSWER using accumulated evidence
      3. If step_count >= max_steps, FAIL
    """
    
    @property
    def name(self) -> str:
        return "dummy_planner"

    def generate_question_oriented_prompt(self, question: str) -> str:
        """Generate a deterministic question-oriented prompt."""
        question = self.validate_question(question)
        return (
            f"Clarified Question: Understand the audio content related to: {question}\n"
            f"Focus Points: 1) Identify relevant sounds and speech, 2) Note any unclear or ambiguous elements\n"
            f"Tasks: 1) Listen for content relevant to the question, 2) Form an initial hypothesis, 3) List uncertainties that need verification."
        )

    def plan(self, question: str, frontend_output=None) -> InitialPlan:
        """Produce deterministic initial plan from question and optional frontend caption."""
        question = self.validate_question(question)
        
        # Simple keyword-based detection for audio output requirements
        audio_processing_keywords = [
            "trim", "cut", "crop", "slice", "extract", "split",
            "merge", "mix", "concatenate", "join", "combine",
            "convert", "resample", "change format",
            "normalize", "denoise", "enhance", "filter", "eq",
            "pitch", "speed", "tempo", "stretch",
            "volume", "amplify", "compress", "limit",
            "save", "output", "export", "generate",
        ]
        question_lower = question.lower()
        requires_audio_output = any(kw in question_lower for kw in audio_processing_keywords)
        
        # Adjust approach based on audio output requirement
        if requires_audio_output:
            approach = "Process the audio as requested and produce the transformed audio file."
            focus_points = [
                "Identify the specific audio processing required",
                "Apply the correct transformation to the audio",
                "Ensure output audio is properly generated and saved",
            ]
            possible_tool_types = ["ffmpeg", "audio_processing"]
        else:
            approach = "Start with speech transcription, then synthesize direct evidence for the question."
            focus_points = [
                "Identify question-relevant speech content",
                "Keep answer grounded in observed audio evidence",
            ]
            possible_tool_types = ["asr", "event_detection"]
        
        notes = f"[DummyPlanner] Initial plan generated for question: {question}"
        if frontend_output:
            notes += f" | Frontend caption: {frontend_output.question_guided_caption[:100]}..."
        
        return InitialPlan(
            approach=approach,
            focus_points=focus_points,
            possible_tool_types=possible_tool_types,
            requires_audio_output=requires_audio_output,
            notes=notes,
        )
    
    def decide(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> PlannerDecision:
        """Make a deterministic decision based on simple rules."""
        self.validate_state(state)
        
        step_count = state.get("step_count", 0)
        max_steps = state.get("max_steps", 10)
        tool_history = state.get("tool_call_history", [])
        evidence_log = state.get("evidence_log", [])
        initial_plan: InitialPlan = state["initial_plan"]
        audio_list = state.get("audio_list", [])
        
        # Check step limit first
        if step_count >= max_steps:
            return PlannerDecision(
                action=PlannerActionType.FAIL,
                rationale=f"Exceeded maximum steps ({max_steps})",
                confidence=1.0,
            )
        
        # Build tool name lookup
        available_tool_names = {t.name for t in available_tools}

        # Decision logic based on tool call count
        num_tools_called = len(tool_history)
        
        # Get first audio from audio_list (original audio)
        selected_audio_id = audio_list[0].audio_id if audio_list else "audio_0"
        
        # Check if we should do a frontend follow-up on generated audio
        non_original_audios = [a for a in audio_list if a.source != "original"]
        has_done_followup = any(
            d.action == PlannerActionType.CALL_FRONTEND
            for d in state.get("planner_trace", [])
        )
        
        if non_original_audios and not has_done_followup:
            return PlannerDecision(
                action=PlannerActionType.CALL_FRONTEND,
                rationale="A tool generated a new audio artifact. Re-perceiving it with the frontend may resolve remaining uncertainty.",
                selected_audio_ids=[non_original_audios[0].audio_id],
                frontend_followup_prompt=f"Describe the content of this audio artifact in detail, focusing on aspects relevant to: {state['question']}",
                frontend_followup_goal="Gather detailed perception of the transformed audio",
                confidence=0.7,
            )
        
        if num_tools_called == 0:
            # First iteration: call ASR tool if available
            target_tool = "dummy_asr"
            if target_tool not in available_tool_names:
                return PlannerDecision(
                    action=PlannerActionType.FAIL,
                    rationale=f"Required tool '{target_tool}' not available",
                    confidence=1.0,
                )
            return PlannerDecision(
                action=PlannerActionType.CALL_TOOL,
                rationale=(
                    "Initial plan indicates transcription-first strategy; "
                    "call ASR to gather direct textual evidence."
                ),
                selected_tool_calls=[
                    ToolCallRequest(tool_name=target_tool, args={}, context={}),
                ],
                selected_audio_id=selected_audio_id,
                confidence=0.8,
            )

        else:
            # After first tool call, answer
            return self._build_answer_decision(state, evidence_log, initial_plan)
    
    def _build_answer_decision(
        self,
        state: AgentState,
        evidence_log: list,
        initial_plan: InitialPlan,
    ) -> PlannerDecision:
        """Build an ANSWER decision from accumulated evidence."""
        return PlannerDecision(
            action=PlannerActionType.ANSWER,
            rationale="Sufficient evidence collected from multiple tools. Frontend model will generate the final answer.",
            draft_answer=None,
            confidence=0.75,
        )

    def check_format(
        self,
        proposed_answer: str,
        expected_format: str | None,
        question: str,
        requires_audio_output: bool = False,
    ) -> FormatCheckResult:
        """
        Check format compliance using dummy logic.
        
        Always passes format check for dummy planner.
        """
        # Dummy planner always passes format check
        return FormatCheckResult(
            passed=True,
            critique=None,
            confidence=1.0,
        )

    def summarize_evidence(self, state: AgentState) -> str:
        """
        Produce a deterministic summary of accumulated evidence.
        
        Concatenates evidence sources and planner decisions without LLM call.
        """
        evidence_log = state.get("evidence_log", [])
        planner_trace = state.get("planner_trace", [])
        tool_history = state.get("tool_call_history", [])
        frontend_output = state.get("initial_frontend_output")
        
        parts = []
        if frontend_output:
            parts.append(f"Frontend observation: {frontend_output.question_guided_caption}")
        
        if evidence_log:
            parts.append("Evidence items:")
            for item in evidence_log:
                parts.append(f"- [{item.source}] {item.content}")
        
        if planner_trace:
            parts.append("Planner decisions:")
            for i, decision in enumerate(planner_trace, 1):
                parts.append(f"Step {i}: {decision.action.value} - {decision.rationale}")
        
        if tool_history:
            parts.append("Tool calls:")
            for record in tool_history:
                parts.append(
                    f"- {record.request.tool_name} (success={record.result.success})"
                )
        
        return "\n".join(parts) if parts else "No evidence collected."
