"""
MCP (Model Context Protocol) tool integration.

This package provides infrastructure for running tools in separate processes
using the Model Context Protocol.
"""

from audio_agent.tools.mcp.schemas import (
    MCPRequest,
    MCPResponse,
    MCPToolInfo,
    MCPCallResult,
    MCPServerConfig,
)
from audio_agent.tools.mcp.client import MCPClient
from audio_agent.tools.mcp.server_manager import MCPServerManager
from audio_agent.tools.mcp.tool_adapter import MCPToolAdapter

__all__ = [
    "MCPRequest",
    "MCPResponse",
    "MCPToolInfo",
    "MCPCallResult",
    "MCPServerConfig",
    "MCPClient",
    "MCPServerManager",
    "MCPToolAdapter",
]
