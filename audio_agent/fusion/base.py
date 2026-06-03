"""
Abstract base class for evidence fusers.

Fusers convert tool results into structured evidence items
that can be accumulated in the agent state.
"""

from abc import ABC, abstractmethod

from audio_agent.core.state import AgentState
from audio_agent.core.schemas import ToolResult, EvidenceItem
from audio_agent.core.errors import FusionError


class BaseEvidenceFuser(ABC):
    """
    Abstract base class for evidence fusion.
    
    Evidence fusers are responsible for:
    - Converting tool outputs to evidence items
    - Optionally summarizing/combining evidence
    - Maintaining evidence quality metadata
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this fuser for logging."""
        raise NotImplementedError
    
    @abstractmethod
    def fuse(
        self,
        state: AgentState,
        tool_result: ToolResult,
    ) -> list[EvidenceItem]:
        """
        Fuse a tool result into evidence items.
        
        Args:
            state: Current agent state (for context)
            tool_result: Result from a tool execution
        
        Returns:
            List of EvidenceItem objects to append to evidence_log
        
        Raises:
            FusionError: If fusion fails
        """
        raise NotImplementedError
    
    def validate_tool_result(self, tool_result: ToolResult) -> None:
        """
        Validate tool result before fusion.
        
        Raises:
            FusionError: If tool result is invalid
        """
        if tool_result is None:
            raise FusionError("Cannot fuse None tool result")
        if not isinstance(tool_result, ToolResult):
            raise FusionError(
                f"Expected ToolResult, got {type(tool_result).__name__}",
                details={"actual_type": type(tool_result).__name__}
            )
