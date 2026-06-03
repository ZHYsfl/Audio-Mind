"""
LangGraph graph builder for the audio agent.

Constructs the agent workflow with proper node connections and routing.
"""

from langgraph.graph import StateGraph, START, END

from audio_agent.core.state import AgentState
from audio_agent.graph.nodes import (
    create_initial_prompt_node,
    create_frontend_evidence_node,
    create_initial_plan_node,
    create_planner_decision_node,
    create_tool_executor_node,
    create_evidence_fusion_node,
    create_evidence_summarization_node,
    create_final_answer_node,
    create_format_check_node,
    create_frontend_followup_node,
    answer_node,
    failure_node,
)
from audio_agent.graph.routing import (
    route_after_planner_decision,
    route_after_format_check,
    route_after_frontend_followup,
    NODE_ANSWER,
    NODE_TOOL_EXECUTOR,
    NODE_FAILURE,
    NODE_EVIDENCE_FUSION,
    NODE_INITIAL_PROMPT,
    NODE_INITIAL_PLAN,
    NODE_PLANNER_DECISION,
    NODE_EVIDENCE_SUMMARIZATION,
    NODE_FINAL_ANSWER,
    NODE_FORMAT_CHECK,
    NODE_FRONTEND_FOLLOWUP,
)
from audio_agent.frontend.base import BaseFrontend
from audio_agent.planner.base import BasePlanner
from audio_agent.tools.registry import ToolRegistry
from audio_agent.tools.executor import ToolExecutor
from audio_agent.fusion.base import BaseEvidenceFuser
from audio_agent.config.settings import AgentConfig


def build_graph(
    frontend: BaseFrontend,
    planner: BasePlanner,
    registry: ToolRegistry,
    fuser: BaseEvidenceFuser,
    config: AgentConfig | None = None,
    checkpointer=None,
) -> StateGraph:
    """
    Build the complete audio agent LangGraph workflow.

    Graph structure:

    START
      -> initial_prompt_node            (skipped in direct-answer mode)
      -> frontend_evidence_node
      -> initial_plan_node
      -> planner_decision_node
      -> [conditional routing based on decision]
         - ANSWER -> evidence_summarization_node -> final_answer_node -> format_check_node -> [conditional]
            * Format OK -> answer_node -> END
            * Format Failed (under the cap) -> planner_decision_node (loop with critique)
         - CALL_TOOL -> tool_executor_node -> evidence_fusion_node -> planner_decision_node (loop)
         - CALL_FRONTEND -> frontend_followup_node -> evidence_fusion_node -> planner_decision_node (loop)
         - FAIL -> failure_node -> END

    The format-check node is wired only when config.enable_format_check is True.

    Args:
        frontend: Frontend instance for initial audio processing
        planner: Planner instance for decision making
        registry: Tool registry containing available tools
        fuser: Evidence fuser for converting tool results
        config: Optional agent configuration
        checkpointer: Optional LangGraph checkpointer for state persistence /
            resumable executions / debugging with history

    Returns:
        Compiled LangGraph StateGraph ready for execution
    """
    if frontend is None:
        raise ValueError("frontend cannot be None")
    if planner is None:
        raise ValueError("planner cannot be None")
    if registry is None:
        raise ValueError("registry cannot be None")
    if fuser is None:
        raise ValueError("fuser cannot be None")

    enable_format_check = config.enable_format_check if config else True

    # Create executor from registry
    executor = ToolExecutor(registry)

    # Create node functions with injected dependencies
    initial_prompt_node_fn = create_initial_prompt_node(planner)
    frontend_node = create_frontend_evidence_node(frontend)
    initial_plan_node_fn = create_initial_plan_node(planner)
    planner_tool_scope = config.planner_tool_scope if config else "core"
    planner_tool_inventory_path = config.planner_tool_inventory_path if config else None
    planner_decision_node_fn = create_planner_decision_node(
        planner,
        registry,
        planner_tool_scope=planner_tool_scope,
        planner_tool_inventory_path=planner_tool_inventory_path,
    )
    tool_executor_node_fn = create_tool_executor_node(executor)
    evidence_fusion_node_fn = create_evidence_fusion_node(fuser)
    evidence_summarization_node_fn = create_evidence_summarization_node(planner)
    final_answer_node_fn = create_final_answer_node(frontend)
    format_check_node_fn = create_format_check_node(planner)
    frontend_followup_node_fn = create_frontend_followup_node(frontend)

    # Build the graph
    graph = StateGraph(AgentState)

    # Add nodes
    # Direct-answer mode (frontend answers the question directly) skips the initial_prompt/QoP
    # node, saving one planner call per question. The mode lives on the frontend
    # (frontend.direct_answer); AudioAgent sets it from AgentConfig.frontend_direct_answer, the
    # source of truth. Frontends that don't support direct answering keep the QoP node.
    _direct_answer = getattr(frontend, "direct_answer", False)
    if not _direct_answer:
        graph.add_node(NODE_INITIAL_PROMPT, initial_prompt_node_fn)
    graph.add_node("frontend_evidence_node", frontend_node)
    graph.add_node(NODE_INITIAL_PLAN, initial_plan_node_fn)
    graph.add_node(NODE_PLANNER_DECISION, planner_decision_node_fn)
    graph.add_node(NODE_TOOL_EXECUTOR, tool_executor_node_fn)
    graph.add_node(NODE_EVIDENCE_FUSION, evidence_fusion_node_fn)
    graph.add_node(NODE_EVIDENCE_SUMMARIZATION, evidence_summarization_node_fn)
    graph.add_node(NODE_FINAL_ANSWER, final_answer_node_fn)
    if enable_format_check:
        graph.add_node(NODE_FORMAT_CHECK, format_check_node_fn)
    graph.add_node(NODE_FRONTEND_FOLLOWUP, frontend_followup_node_fn)
    graph.add_node(NODE_ANSWER, answer_node)
    graph.add_node(NODE_FAILURE, failure_node)

    # Add edges
    # START -> initial_prompt -> frontend  (direct-answer mode: START -> frontend, skipping QoP)
    if _direct_answer:
        graph.add_edge(START, "frontend_evidence_node")
    else:
        graph.add_edge(START, NODE_INITIAL_PROMPT)
        graph.add_edge(NODE_INITIAL_PROMPT, "frontend_evidence_node")

    # frontend_evidence_node -> initial_plan_node
    graph.add_edge("frontend_evidence_node", NODE_INITIAL_PLAN)

    # initial_plan_node -> planner_decision_node
    graph.add_edge(NODE_INITIAL_PLAN, NODE_PLANNER_DECISION)

    # planner_decision_node -> conditional routing
    # ANSWER routes through evidence summarization before final answer and format check.
    graph.add_conditional_edges(
        NODE_PLANNER_DECISION,
        route_after_planner_decision,
        {
            NODE_EVIDENCE_SUMMARIZATION: NODE_EVIDENCE_SUMMARIZATION,
            NODE_TOOL_EXECUTOR: NODE_TOOL_EXECUTOR,
            NODE_FRONTEND_FOLLOWUP: NODE_FRONTEND_FOLLOWUP,
            NODE_FAILURE: NODE_FAILURE,
        }
    )

    # tool_executor_node -> evidence_fusion_node
    graph.add_edge(NODE_TOOL_EXECUTOR, NODE_EVIDENCE_FUSION)

    # frontend_followup_node -> evidence_fusion_node
    graph.add_edge(NODE_FRONTEND_FOLLOWUP, NODE_EVIDENCE_FUSION)

    # evidence_fusion_node -> planner_decision_node (loop back)
    graph.add_edge(NODE_EVIDENCE_FUSION, NODE_PLANNER_DECISION)

    # evidence_summarization_node -> final_answer_node
    graph.add_edge(NODE_EVIDENCE_SUMMARIZATION, NODE_FINAL_ANSWER)

    if enable_format_check:
        # final_answer_node -> format_check_node -> conditional routing based on result
        graph.add_edge(NODE_FINAL_ANSWER, NODE_FORMAT_CHECK)
        graph.add_conditional_edges(
            NODE_FORMAT_CHECK,
            route_after_format_check,
            {
                NODE_ANSWER: NODE_ANSWER,
                NODE_PLANNER_DECISION: NODE_PLANNER_DECISION,
            }
        )
    else:
        # Format checking disabled: accept the drafted answer directly.
        graph.add_edge(NODE_FINAL_ANSWER, NODE_ANSWER)

    # Terminal nodes -> END
    graph.add_edge(NODE_ANSWER, END)
    graph.add_edge(NODE_FAILURE, END)

    if checkpointer is not None:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
