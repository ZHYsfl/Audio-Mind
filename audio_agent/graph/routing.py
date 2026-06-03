"""
Routing logic for the LangGraph workflow.

Routers examine state and return the name of the next node.
All routing decisions are explicit and logged.
"""

from audio_agent.core.state import AgentState
from audio_agent.core.schemas import PlannerActionType
from audio_agent.core.errors import GraphRoutingError
from audio_agent.core.logging import get_logger


# Node name constants
NODE_INITIAL_PROMPT = "initial_prompt_node"
NODE_INITIAL_PLAN = "initial_plan_node"
NODE_PLANNER_DECISION = "planner_decision_node"
NODE_ANSWER = "answer_node"
NODE_TOOL_EXECUTOR = "tool_executor_node"
NODE_FAILURE = "failure_node"
NODE_EVIDENCE_FUSION = "evidence_fusion_node"
NODE_FINAL_ANSWER = "final_answer_node"
NODE_FORMAT_CHECK = "format_check_node"
NODE_EVIDENCE_SUMMARIZATION = "evidence_summarization_node"
NODE_FRONTEND_FOLLOWUP = "frontend_followup_node"
END = "__end__"


def route_after_planner_decision(state: AgentState) -> str:
    """
    Route after the planner decision node based on its decision.
    
    Routes:
    - ANSWER -> final_answer_node (frontend generates the answer)
    - CALL_TOOL -> tool_executor_node
    - FAIL -> failure_node
    
    Note: max_steps exhaustion is handled in planner_decision_node by
    forcing answer generation on the final step.
    
    Args:
        state: Current agent state
    
    Returns:
        Name of the next node
    
    Raises:
        GraphRoutingError: If routing cannot be determined
    """
    logger = get_logger()
    
    # Get decision
    decision = state.get("current_decision")
    
    if decision is None:
        raise GraphRoutingError(
            "Cannot route: current_decision is None",
            details={"step_count": state.get("step_count", 0)}
        )
    
    action = decision.action
    
    if action == PlannerActionType.ANSWER:
        # Summarize evidence before frontend generates the final answer
        logger.info(f"ROUTING: action={action.value} -> {NODE_EVIDENCE_SUMMARIZATION}")
        return NODE_EVIDENCE_SUMMARIZATION
    
    elif action == PlannerActionType.CALL_TOOL:
        tool_names = ", ".join(tc.tool_name for tc in decision.selected_tool_calls) or "<none>"
        logger.info(f"ROUTING: action={action.value}, tools=[{tool_names}] -> {NODE_TOOL_EXECUTOR}")
        return NODE_TOOL_EXECUTOR
    
    elif action == PlannerActionType.CALL_FRONTEND:
        logger.info(f"ROUTING: action={action.value} -> {NODE_FRONTEND_FOLLOWUP}")
        return NODE_FRONTEND_FOLLOWUP
    
    elif action == PlannerActionType.FAIL:
        logger.info(f"ROUTING: action={action.value} -> {NODE_FAILURE}")
        return NODE_FAILURE
    
    else:
        raise GraphRoutingError(
            f"Unknown planner action: {action}",
            details={"action": str(action)}
        )


def route_after_planner(state: AgentState) -> str:
    """Backward-compatible alias for planner decision routing."""
    return route_after_planner_decision(state)


def route_after_tool(state: AgentState) -> str:
    """
    Route after tool execution.
    
    Always routes to evidence_fusion_node to process the result.
    
    Args:
        state: Current agent state
    
    Returns:
        Name of the next node
    """
    logger = get_logger()
    
    # Check if any tool result exists
    if not state.get("latest_tool_results"):
        raise GraphRoutingError(
            "Cannot route after tool: latest_tool_results is empty"
        )
    
    logger.info(f"ROUTING: after tool -> {NODE_EVIDENCE_FUSION}")
    return NODE_EVIDENCE_FUSION


def route_after_format_check(state: AgentState) -> str:
    """
    Route after format check based on the result.
    
    Routes:
    - format check passed -> answer_node (finalize the answer)
    - format check failed -> planner_decision_node (adds critique as evidence)
    
    Args:
        state: Current agent state
    
    Returns:
        Name of the next node
    
    Raises:
        GraphRoutingError: If routing cannot be determined
    """
    logger = get_logger()
    
    # Get format check result
    format_check_result = state.get("format_check_result")
    
    if format_check_result is None:
        raise GraphRoutingError(
            "Cannot route after format check: format_check_result is None"
        )
    
    if format_check_result.passed:
        logger.info(f"ROUTING: format check passed -> {NODE_ANSWER}")
        return NODE_ANSWER

    # Cap the format-check retry loop: after max_format_checks failed attempts, accept the
    # current draft (best effort) rather than looping back to the planner indefinitely.
    config = state.get("config") or {}
    max_format_checks = config.get("max_format_checks", 2)
    format_check_count = state.get("format_check_count", 0)
    if format_check_count >= max_format_checks:
        logger.info(
            f"ROUTING: format check failed but reached max_format_checks={max_format_checks} "
            f"(count={format_check_count}) -> {NODE_ANSWER} (best effort)"
        )
        return NODE_ANSWER

    # Format check failed and under the cap - critique was added as evidence.
    # Loop back to planner to regenerate the answer with format feedback.
    logger.info(f"ROUTING: format check failed (critique added) -> {NODE_PLANNER_DECISION}")
    return NODE_PLANNER_DECISION


def route_after_frontend_followup(state: AgentState) -> str:
    """
    Route after frontend follow-up.
    
    Always routes to evidence_fusion_node to incorporate the follow-up output
    into the evidence log, then loops back to planner_decision_node.
    
    Args:
        state: Current agent state
    
    Returns:
        Name of the next node
    
    Raises:
        GraphRoutingError: If routing cannot be determined
    """
    logger = get_logger()
    
    if state.get("latest_frontend_followup_output") is None:
        raise GraphRoutingError(
            "Cannot route after frontend follow-up: latest_frontend_followup_output is None"
        )
    
    logger.info(f"ROUTING: after frontend follow-up -> {NODE_EVIDENCE_FUSION}")
    return NODE_EVIDENCE_FUSION
