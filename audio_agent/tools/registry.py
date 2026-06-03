"""
Tool registry for managing available tools.

The registry provides a central place to:
- Register internal tools (in-process)
- Register MCP tools (external processes)
- Look up tools by name
- List available tool specs for the planner
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from audio_agent.core.schemas import ToolSpec
from audio_agent.core.errors import ToolRegistryError
from audio_agent.core.logging import get_logger
from audio_agent.tools.base import BaseTool


class ToolRegistry:
    """
    Registry for managing both internal and MCP tools.
    
    Supports:
    - Internal tools: In-process, direct implementation
    - MCP tools: External processes via Model Context Protocol
    
    Thread-safety note: This implementation is not thread-safe.
    For concurrent access, wrap with appropriate locks.
    """
    
    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._internal_tools: dict[str, BaseTool] = {}
        self._mcp_tools: dict[str, BaseTool] = {}
        self._logger = get_logger()
    
    def register(self, tool: BaseTool) -> None:
        """
        Register an internal tool in the registry.
        
        This is a convenience alias for register_internal().
        
        Args:
            tool: Tool instance to register
        
        Raises:
            ToolRegistryError: If tool name is empty or already registered
        """
        self.register_internal(tool)
    
    def register_internal(self, tool: BaseTool) -> None:
        """
        Register an internal (in-process) tool.
        
        Args:
            tool: Tool instance to register
        
        Raises:
            ToolRegistryError: If tool name is empty or already registered
        """
        if not isinstance(tool, BaseTool):
            raise ToolRegistryError(
                f"Cannot register non-tool object: {type(tool).__name__}",
                details={"type": type(tool).__name__}
            )
        
        raw_name = tool.spec.name
        name = raw_name.strip() if raw_name else ""
        
        if not name:
            raise ToolRegistryError(
                "Cannot register tool with empty name",
                details={"tool_type": type(tool).__name__, "raw_name": repr(raw_name)}
            )
        
        if name in self._internal_tools or name in self._mcp_tools:
            raise ToolRegistryError(
                f"Tool '{name}' is already registered",
                details={"existing_tool": type(self._get_tool(name)).__name__}
            )
        
        self._internal_tools[name] = tool
        self._logger.info(f"Registered internal tool: {name}")
    
    def register_mcp(self, tool: BaseTool) -> None:
        """
        Register an MCP tool.
        
        Args:
            tool: MCP tool adapter instance
        
        Raises:
            ToolRegistryError: If tool name is empty or already registered
        """
        if not isinstance(tool, BaseTool):
            raise ToolRegistryError(
                f"Cannot register non-tool object: {type(tool).__name__}",
                details={"type": type(tool).__name__}
            )
        
        raw_name = tool.spec.name
        name = raw_name.strip() if raw_name else ""
        
        if not name:
            raise ToolRegistryError(
                "Cannot register tool with empty name",
                details={"tool_type": type(tool).__name__, "raw_name": repr(raw_name)}
            )
        
        if name in self._internal_tools or name in self._mcp_tools:
            raise ToolRegistryError(
                f"Tool '{name}' is already registered",
                details={"existing_tool": type(self._get_tool(name)).__name__}
            )
        
        self._mcp_tools[name] = tool
        self._logger.info(f"Registered MCP tool: {name}")
    
    def _get_tool(self, name: str) -> BaseTool | None:
        """Get tool without raising - internal use."""
        if name in self._internal_tools:
            return self._internal_tools[name]
        if name in self._mcp_tools:
            return self._mcp_tools[name]
        return None
    
    def get(self, tool_name: str) -> BaseTool:
        """
        Get a tool by name (internal or MCP).
        
        Args:
            tool_name: Name of the tool to retrieve
        
        Returns:
            The registered tool
        
        Raises:
            ToolRegistryError: If tool is not found
        """
        if not tool_name:
            raise ToolRegistryError("Cannot look up tool with empty name")
        
        tool = self._get_tool(tool_name)
        if tool is None:
            available = self.list_names()
            raise ToolRegistryError(
                f"Unknown tool: '{tool_name}'",
                details={"available_tools": available}
            )
        
        return tool
    
    def list_specs(self) -> list[ToolSpec]:
        """
        List specifications of all registered tools (internal + MCP).
        
        Returns:
            List of ToolSpec objects for the planner
        """
        internal_specs = [tool.spec for tool in self._internal_tools.values()]
        mcp_specs = [tool.spec for tool in self._mcp_tools.values()]
        return internal_specs + mcp_specs
    
    def list_names(self) -> list[str]:
        """
        List names of all registered tools.
        
        Returns:
            List of tool names
        """
        internal = list(self._internal_tools.keys())
        mcp = list(self._mcp_tools.keys())
        return internal + mcp
    
    def list_internal_names(self) -> list[str]:
        """List names of internal tools only."""
        return list(self._internal_tools.keys())
    
    def list_mcp_names(self) -> list[str]:
        """List names of MCP tools only."""
        return list(self._mcp_tools.keys())
    
    def __len__(self) -> int:
        """Return number of registered tools."""
        return len(self._internal_tools) + len(self._mcp_tools)
    
    def __contains__(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self._internal_tools or tool_name in self._mcp_tools
    
    def is_internal(self, tool_name: str) -> bool:
        """Check if a tool is an internal tool."""
        return tool_name in self._internal_tools
    
    def is_mcp(self, tool_name: str) -> bool:
        """Check if a tool is an MCP tool."""
        return tool_name in self._mcp_tools
