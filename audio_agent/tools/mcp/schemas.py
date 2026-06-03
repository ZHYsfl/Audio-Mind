"""
MCP (Model Context Protocol) schema definitions.

These types define the JSON-RPC protocol used for communication between
the agent and MCP servers.
"""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field


class MCPRequest(BaseModel):
    """JSON-RPC request to MCP server."""
    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None = Field(default=None)
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class MCPErrorDetail(BaseModel):
    """JSON-RPC error detail."""
    code: int
    message: str
    data: dict[str, Any] | None = None


class MCPResponse(BaseModel):
    """JSON-RPC response from MCP server."""
    jsonrpc: Literal["2.0"] = "2.0"
    id: int | str | None = None
    result: dict[str, Any] | None = None
    error: MCPErrorDetail | None = None


class MCPToolInfo(BaseModel):
    """Information about an MCP tool."""
    name: str
    description: str
    inputSchema: dict[str, Any] = Field(default_factory=dict)


class MCPTextContent(BaseModel):
    """Text content in MCP tool result."""
    type: Literal["text"] = "text"
    text: str


class MCPImageContent(BaseModel):
    """Image content in MCP tool result."""
    type: Literal["image"] = "image"
    data: str  # base64 encoded
    mimeType: str


class MCPCallResult(BaseModel):
    """Result of calling an MCP tool."""
    content: list[MCPTextContent | MCPImageContent] = Field(default_factory=list)
    isError: bool = False
    error: str | None = None


class MCPServerResources(BaseModel):
    """Resource requirements for MCP server."""
    memory_gb: int | None = None
    gpu: bool = False
    cpu_cores: int | None = None


class MCPServerConfig(BaseModel):
    """Configuration for an MCP server."""
    command: list[str]
    working_dir: str | None = Field(
        default=None,
        description="Working directory for server execution. Relative paths resolved from tool directory."
    )
    python_path: str | None = Field(
        default=None,
        description="Explicit Python interpreter path. If provided, prepends to command."
    )
    env: dict[str, str] = Field(default_factory=dict)
    resources: MCPServerResources = Field(default_factory=MCPServerResources)
    lifecycle: Literal["per_call", "session", "persistent"] = "session"
    startup_timeout_sec: int = 300
    
    model_config = {"extra": "forbid"}
