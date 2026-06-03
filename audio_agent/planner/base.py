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
    
    Planner has two phases:
    - plan(question): produce an initial approach from question only
    - decide(state, tools): produce concrete action decision
    
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
