"""
Tool catalog for MCP-based tools.

This package contains tool implementations that run as separate processes
using the Model Context Protocol (MCP).

Utilities provided:
- loader: Load tool configurations with path resolution
- register_all_mcp_tools: Auto-register all MCP tools from catalog

For tool onboarding, see tool_preparation/README.md (harness-first workflow).
For manual setup, use ./setup.sh in each tool directory.
"""

from audio_agent.tools.catalog.loader import (
    get_catalog_dir,
    get_tool_dir,
    load_mcp_server_config,
    load_tool_config,
    list_available_tools,
    register_all_mcp_tools,
    resolve_config_paths,
    resolve_path,
)

__all__ = [
    "get_catalog_dir",
    "get_tool_dir",
    "load_mcp_server_config",
    "load_tool_config",
    "list_available_tools",
    "register_all_mcp_tools",
    "resolve_config_paths",
    "resolve_path",
]
