"""Graph module: LangGraph workflow definition."""

try:
    from audio_agent.graph.builder import build_graph
except ModuleNotFoundError as e:
    if e.name == "langgraph":
        build_graph = None  # type: ignore[assignment]
    else:
        raise
from audio_agent.graph.nodes import (
    frontend_evidence_node,
    initial_plan_node,
    planner_decision_node,
    planner_node,
    tool_executor_node,
    evidence_fusion_node,
    answer_node,
    failure_node,
)
from audio_agent.graph.routing import route_after_planner_decision, route_after_planner, route_after_tool

__all__ = [
    "build_graph",
    "frontend_evidence_node",
    "initial_plan_node",
    "planner_decision_node",
    "planner_node",
    "tool_executor_node",
    "evidence_fusion_node",
    "answer_node",
    "failure_node",
    "route_after_planner_decision",
    "route_after_planner",
    "route_after_tool",
]
