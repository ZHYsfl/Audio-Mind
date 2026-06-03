"""
Simple structured logging utilities for the audio agent framework.

Provides readable, consistent logging for debugging and monitoring.
"""

import logging
import sys
from datetime import datetime
from typing import Any


def setup_logger(
    name: str = "audio_agent",
    level: int = logging.INFO,
    format_string: str | None = None,
) -> logging.Logger:
    """
    Set up and return a configured logger.
    
    Args:
        name: Logger name
        level: Logging level
        format_string: Optional custom format string
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s [%(name)s] %(message)s"
    
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


# Global logger instance
_logger: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """Get the global audio agent logger."""
    global _logger
    if _logger is None:
        _logger = setup_logger()
    return _logger


def set_debug_mode(enabled: bool = True) -> None:
    """Enable or disable debug logging."""
    logger = get_logger()
    level = logging.DEBUG if enabled else logging.INFO
    logger.setLevel(level)
    # Also update all handlers
    for handler in logger.handlers:
        handler.setLevel(level)


# =============================================================================
# Structured logging helpers
# =============================================================================

def log_node_start(node_name: str, state_summary: dict[str, Any] | None = None) -> None:
    """Log the start of a graph node execution."""
    logger = get_logger()
    msg = f"NODE START: {node_name}"
    if state_summary:
        msg += f" | State: {state_summary}"
    logger.info(msg)


def log_node_end(node_name: str, result_summary: dict[str, Any] | None = None) -> None:
    """Log the end of a graph node execution."""
    logger = get_logger()
    msg = f"NODE END: {node_name}"
    if result_summary:
        msg += f" | Result: {result_summary}"
    logger.info(msg)


def log_tool_call(tool_name: str, args: dict[str, Any]) -> None:
    """Log a tool invocation."""
    logger = get_logger()
    logger.info(f"TOOL CALL: {tool_name} | Args: {args}")


def log_tool_result(tool_name: str, success: bool, summary: str) -> None:
    """Log a tool result."""
    logger = get_logger()
    status = "SUCCESS" if success else "FAILURE"
    logger.info(f"TOOL RESULT: {tool_name} | {status} | {summary}")


def log_planner_decision(action: str, rationale: str, tool_name: str | None = None) -> None:
    """Log a planner decision."""
    logger = get_logger()
    msg = f"PLANNER DECISION: {action} | Rationale: {rationale}"
    if tool_name:
        msg += f" | Tool: {tool_name}"
    logger.info(msg)


def log_error(component: str, error: Exception, context: dict[str, Any] | None = None) -> None:
    """Log an error with context."""
    logger = get_logger()
    msg = f"ERROR in {component}: {type(error).__name__}: {error}"
    if context:
        msg += f" | Context: {context}"
    logger.error(msg)


def log_state_transition(from_status: str, to_status: str, reason: str) -> None:
    """Log an agent status transition."""
    logger = get_logger()
    logger.info(f"STATUS TRANSITION: {from_status} -> {to_status} | Reason: {reason}")


def log_info(event: str, details: dict[str, Any] | None = None) -> None:
    """Log a general informational event."""
    logger = get_logger()
    msg = f"EVENT: {event}"
    if details:
        msg += f" | Details: {details}"
    logger.info(msg)


def log_warning(event: str, details: dict[str, Any] | None = None) -> None:
    """Log a warning event."""
    logger = get_logger()
    msg = f"WARNING: {event}"
    if details:
        msg += f" | Details: {details}"
    logger.warning(msg)
