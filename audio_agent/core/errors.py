"""
Custom exceptions for the audio agent framework.

All exceptions inherit from AudioAgentError for easy catching.
Each exception type corresponds to a specific failure domain.
"""


class AudioAgentError(Exception):
    """Base exception for all audio agent errors."""
    
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class StateValidationError(AudioAgentError):
    """Raised when agent state is invalid or missing required fields."""
    pass


class ToolRegistryError(AudioAgentError):
    """Raised for tool registry operations: duplicate names, unknown tools, etc."""
    pass


class ToolExecutionError(AudioAgentError):
    """Raised when a tool fails to execute or returns invalid output."""
    pass


class PlannerError(AudioAgentError):
    """Raised when the planner produces invalid output or fails."""
    pass


class FrontendError(AudioAgentError):
    """Raised when the frontend fails to process input or returns invalid output."""
    pass


class FusionError(AudioAgentError):
    """Raised when evidence fusion fails."""
    pass


class GraphRoutingError(AudioAgentError):
    """Raised when graph routing encounters an invalid state or decision."""
    pass
