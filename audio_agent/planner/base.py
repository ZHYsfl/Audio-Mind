"""
Abstract base class for planner modules.

The planner is the "brain" of the agent, deciding what action to take
based on accumulated evidence and available tools.
"""

from abc import ABC, abstractmethod

from audio_agent.core.state import AgentState
from audio_agent.core.schemas import FrontendOutput, InitialPlan, PlannerDecision, ToolSpec, FormatCheckResult
from audio_agent.core.errors import PlannerError


class BasePlanner(ABC):
    """
    Abstract base class for planners.

    The planner is the agent's Text-LLM "brain" and plays EVERY Text-LLM role
    in the paper's question lifecycle (Fig 1a) — one instance, several hats.
    Roles are switched by dedicated system prompts (prompts/*.md), not by
    swapping models:

    Phase 1 (Setup)     generate_question_oriented_prompt  -> initial_prompt_node
                        (QoP generation; skipped in direct-answer mode)
    Phase 2 (Setup)     NOT the planner — the frontend LALM (frontend/base.py)
    Phase 3 (Setup)     plan                              -> initial_plan_node
    Phase 4 (Loop)      decide                            -> planner_decision_node
    Phase 5a (Finalize) summarize_evidence                -> evidence_summarization_node
    Phase 5b (Finalize) NOT the planner — the frontend LALM (frontend/base.py)
    Phase 5c (Finalize) check_format                      -> format_check_node

    Concrete implementations might use:
    - Local LLMs
    - Remote API-based LLMs (OpenAI, Anthropic, etc.)
    - Rule-based planners for testing
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this planner for logging and identification."""
        raise NotImplementedError

    # Infra flag for the Phase 4 decide() path: native ``tools=`` function
    # calling vs the legacy JSON-text fallback (affects the synthesis fallback
    # on the final allowed round).
    def supports_native_tools(self) -> bool:
        """Whether this planner uses the native ``tools=`` API path.

        Subclasses that override ``decide`` to emit / parse structured
        ``tool_calls`` (via the OpenAI-compatible function-calling
        interface) should return ``True``. The graph uses this flag to
        decide whether to apply a synthesis fallback on the final allowed
        round (legacy backends without per-call ``tool_choice`` control
        rely on the synthesis to guarantee termination).
        """
        return False

    # ── Phase 3 (Setup): Initial Plan Construction ──
    # Text-LLM role. Reads the frontend caption, returns an InitialPlan
    # (approach / clarified intent / expected output format / focus points /
    # candidate tool types). Uses plan_system.md; called from initial_plan_node.
    @abstractmethod
    def plan(self, question: str, frontend_output: FrontendOutput | None = None) -> InitialPlan:
        """
        Build an initial plan using question and optional frontend caption.

        Args:
            question: User question
            frontend_output: Optional frontend output with question-guided caption

        Returns:
            InitialPlan

        Raises:
            PlannerError: If question is invalid or planning fails
        """
        raise NotImplementedError

    # ── Phase 1 (Setup): Question-Oriented Prompt Generation ──
    # Text-LLM role. Turns the raw question into a self-contained listening
    # prompt (clarified question + decomposed tasks + focus points) for the
    # LALM in Phase 2. Uses initial_prompt_system.md with the caption-skills
    # reference inlined; called from initial_prompt_node. Skipped entirely in
    # direct-answer mode (frontend_direct_answer=True).
    @abstractmethod
    def generate_question_oriented_prompt(self, question: str) -> str:
        """
        Generate a question-oriented prompt to guide the frontend audio model.

        The output should be a single string that contains:
        - clarified question
        - decomposed tasks
        - focus points for the specific task

        Args:
            question: User question

        Returns:
            A question-oriented prompt string

        Raises:
            PlannerError: If generation fails
        """
        raise NotImplementedError

    # ── Phase 4 (Main Loop): Decide → Execute → Fuse (per round) ──
    # Text-LLM role. Emits exactly ONE PlannerAction per round:
    # CALL_TOOL / CALL_FRONTEND / ANSWER / FAIL, via native function calling
    # with the 3 synthetic tools (emit_final_answer / ask_frontend / give_up).
    # Uses decide_system.md + decide_rules.md; called from planner_decision_node.
    @abstractmethod
    def decide(
        self,
        state: AgentState,
        available_tools: list[ToolSpec],
    ) -> PlannerDecision:
        """
        Make a decision based on current state and available tools.

        Args:
            state: Current agent state with evidence and history
            available_tools: List of tool specifications the planner can choose from

        Returns:
            PlannerDecision indicating next action

        Raises:
            PlannerError: If decision cannot be made or state is invalid
        """
        raise NotImplementedError

    # ── Phase 5c (Finalize): Format Check ──
    # Text-LLM role. Validates the proposed answer against the expected output
    # format — structure only, NOT content correctness. Up to max_format_checks
    # retries before the answer is accepted / failing the run.
    # Called from format_check_node.
    @abstractmethod
    def check_format(
        self,
        proposed_answer: str,
        expected_format: str | None,
        question: str,
        requires_audio_output: bool = False,
    ) -> FormatCheckResult:
        """
        Check if the proposed answer follows the expected output format.

        This method validates format compliance only - it does NOT check
        content correctness. The format check ensures the answer adheres to
        any structural or formatting requirements specified in the question or
        initial plan.

        Args:
            proposed_answer: The answer to check for format compliance
            expected_format: The expected output format (may be None if not specified)
            question: The original user question (for context)
            requires_audio_output: Whether this task expects an audio file as output

        Returns:
            FormatCheckResult indicating whether format requirements are met

        Raises:
            PlannerError: If format check fails
        """
        raise NotImplementedError

    # ── Phase 5a (Finalize): Evidence Summary ──
    # Text-LLM role. Compresses the evidence log + planner trace + tool history
    # into ONE neutral narrative ("court stenographer": no credibility judgment,
    # no contradiction resolution) for the LALM in Phase 5b.
    # Called from evidence_summarization_node.
    @abstractmethod
    def summarize_evidence(self, state: AgentState) -> str:
        """
        Summarize accumulated evidence, planner trace, and tool history
        into a single neutral narrative.

        This is called before the final answer node to compress context
        for the frontend model. The summarizer must NOT judge credibility
        or resolve contradictions.

        Args:
            state: Current agent state with accumulated evidence

        Returns:
            A comprehensive, neutral summary string

        Raises:
            PlannerError: If summarization fails
        """
        raise NotImplementedError
    
    def validate_state(self, state: AgentState) -> None:
        """
        Validate that state has required fields for action decision.
        
        Raises:
            PlannerError: If state is invalid for planning
        """
        if not state.get("question"):
            raise PlannerError(
                "Cannot decide without a question",
                details={"state_keys": list(state.keys())}
            )
        if state.get("initial_frontend_output") is None:
            raise PlannerError(
                "Cannot decide without frontend output",
                details={"question": state.get("question")}
            )
        if state.get("initial_plan") is None:
            raise PlannerError(
                "Cannot decide without initial plan",
                details={"question": state.get("question")}
            )

    def validate_question(self, question: str) -> str:
        """
        Validate question input for initial planning phase.

        Returns:
            Stripped question string.
        """
        if question is None or not isinstance(question, str):
            raise PlannerError(
                "Question must be a non-empty string for initial planning",
                details={"question_type": type(question).__name__ if question is not None else "None"},
            )
        stripped = question.strip()
        if not stripped:
            raise PlannerError("Question must be a non-empty string for initial planning")
        return stripped
