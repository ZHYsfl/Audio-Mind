"""
Default evidence fuser implementation.

Converts tool results into evidence items with reasonable defaults.
"""

import json

from audio_agent.core.state import AgentState
from audio_agent.core.schemas import ToolResult, EvidenceItem
from audio_agent.fusion.base import BaseEvidenceFuser


class DefaultEvidenceFuser(BaseEvidenceFuser):
    """
    Default evidence fuser that creates evidence items from tool outputs.
    
    Behavior:
    - Creates one evidence item per tool result
    - Extracts key fields from output
    - Preserves confidence from tool result
    """
    
    @property
    def name(self) -> str:
        return "default_fuser"
    
    def fuse(
        self,
        state: AgentState,
        tool_result: ToolResult,
    ) -> list[EvidenceItem]:
        """Convert tool result into evidence items."""
        self.validate_tool_result(tool_result)
        
        # Handle failed tool results
        if not tool_result.success:
            return [
                EvidenceItem(
                    source=tool_result.tool_name,
                    content=f"Tool failed: {tool_result.error_message or 'Unknown error'}",
                    evidence_type="error",
                    confidence=0.0,
                    metadata={"success": False},
                )
            ]
        
        # Extract content from output
        output = tool_result.output
        content = self._format_output(output)
        
        # Determine confidence (exclude bool, which is an int subclass, so a tool
        # returning confidence=True does not silently become 1.0).
        raw_confidence = output.get("confidence")
        confidence = (
            float(raw_confidence)
            if isinstance(raw_confidence, (int, float)) and not isinstance(raw_confidence, bool)
            else 0.5
        )
        
        return [
            EvidenceItem(
                source=tool_result.tool_name,
                content=content,
                evidence_type="tool_output",
                confidence=confidence,
                metadata={
                    "tool_name": tool_result.tool_name,
                    "execution_time_ms": tool_result.execution_time_ms,
                    "output_keys": list(output.keys()) if isinstance(output, dict) else [],
                },
            )
        ]
    
    def _format_output(self, output: dict) -> str:
        """Format tool output as readable content."""
        if not output:
            return "No output from tool"
        
        # Try to extract common fields
        parts = []
        
        if "transcript" in output:
            parts.append(f"Transcript: {output['transcript']}")
        
        if "events" in output:
            events = output["events"]
            if isinstance(events, list):
                event_strs = [
                    f"{e.get('event_type', 'unknown')} ({e.get('confidence', 0):.2f})"
                    for e in events[:5]  # Limit to first 5
                ]
                parts.append(f"Detected events: {', '.join(event_strs)}")
        
        if "caption" in output:
            parts.append(f"Caption: {output['caption']}")
        
        # If no known fields, dump as JSON
        if not parts:
            try:
                parts.append(json.dumps(output, indent=2, default=str))
            except (TypeError, ValueError):
                parts.append(str(output))
        
        return "\n".join(parts)
