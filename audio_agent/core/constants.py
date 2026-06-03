"""
Constants and enums used throughout the audio agent framework.
"""

from enum import Enum


class AgentStatus(str, Enum):
    """
    Status of the agent execution.
    
    - RUNNING: Agent is actively processing
    - ANSWERED: Agent has produced a final answer
    - FAILED: Agent encountered an unrecoverable error
    - EXHAUSTED: Agent exceeded max_steps without answering
    """
    RUNNING = "running"
    ANSWERED = "answered"
    FAILED = "failed"
    EXHAUSTED = "exhausted"


# Default configuration values
DEFAULT_MAX_STEPS = 10
DEFAULT_DEBUG = False
