#!/usr/bin/env python3
"""
Tool configuration loader with path resolution.

This module provides utilities for loading MCP tool configurations
from catalog directories with proper path resolution.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from audio_agent.tools.mcp.schemas import MCPServerConfig


def _default_audio_agent_env() -> dict[str, str]:
    """Defaults injected into config-time env expansion when the variable is unset.

    Keeps configs portable: tools can reference ${AUDIO_AGENT_MODELS_DIR} without
    every consumer having to export it first.
    """
    defaults: dict[str, str] = {}
    if "AUDIO_AGENT_MODELS_DIR" not in os.environ:
        # repo root = three parents up from this file: catalog/ -> tools/ -> audio_agent/ -> <repo>
        repo_root = Path(__file__).resolve().parents[3]
        defaults["AUDIO_AGENT_MODELS_DIR"] = str(repo_root / "models")
    return defaults


def _expand_env_vars(value: Any, fallback_env: dict[str, str]) -> Any:
    """Recursively expand ${VAR} / $VAR references in all string values.

    Uses the process environment first; falls back to fallback_env for any
    variable not present in os.environ. Non-string values pass through.
    """
    if isinstance(value, str):
        # os.path.expandvars only consults os.environ; layer fallbacks manually.
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            for k, v in fallback_env.items():
                expanded = expanded.replace(f"${{{k}}}", v).replace(f"${k}", v)
        return expanded
    if isinstance(value, dict):
        return {k: _expand_env_vars(v, fallback_env) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(v, fallback_env) for v in value]
    return value


def expand_config_env_vars(config: dict[str, Any]) -> dict[str, Any]:
    """Expand ${VAR}/$VAR in every string in the config tree.

    Process env wins; falls back to internal defaults (currently
    AUDIO_AGENT_MODELS_DIR) so configs work out of the box on a fresh clone.
    """
    fallback = _default_audio_agent_env()
    return _expand_env_vars(config, fallback)

# Type hints for optional imports
if False:
    from audio_agent.tools.mcp import MCPServerManager, MCPToolAdapter
    from audio_agent.tools.registry import ToolRegistry


def get_catalog_dir() -> Path:
    """Get the path to the tools catalog directory."""
    return Path(__file__).parent


def get_tool_dir(tool_name: str, catalog_dir: Path | None = None) -> Path:
    """Get the path to a specific tool's directory."""
    if catalog_dir is None:
        catalog_dir = get_catalog_dir()
    return catalog_dir / tool_name


def resolve_path(path: str, base_dir: Path) -> Path:
    """
    Resolve a path relative to a base directory.
    
    Args:
        path: Path string (can be relative or absolute)
        base_dir: Base directory to resolve relative paths from
        
    Returns:
        Resolved absolute Path
    """
    p = Path(path)
    if p.is_absolute():
        return p
    return (base_dir / p).resolve()


def load_tool_config(
    tool_name: str,
    catalog_dir: Path | None = None,
    resolve_relative_paths: bool = True,
) -> dict[str, Any]:
    """
    Load a tool's configuration from config.yaml.
    
    Args:
        tool_name: Name of the tool
        catalog_dir: Optional catalog directory path
        resolve_relative_paths: If True, resolve relative paths in config
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If tool directory or config.yaml not found
        ValueError: If yaml is not installed
    """
    if not HAS_YAML:
        raise ValueError(
            "PyYAML is required to load tool configs. "
            "Install with: pip install pyyaml"
        )
    
    if catalog_dir is None:
        catalog_dir = get_catalog_dir()
    
    tool_dir = get_tool_dir(tool_name, catalog_dir)
    
    if not tool_dir.exists():
        raise FileNotFoundError(f"Tool directory not found: {tool_dir}")
    
    config_path = tool_dir / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Expand ${VAR}/$VAR references before path resolution so env-var-based
    # paths (e.g. ${AUDIO_AGENT_MODELS_DIR}/...) resolve correctly.
    config = expand_config_env_vars(config)

    if resolve_relative_paths:
        config = resolve_config_paths(config, tool_dir)

    return config


def resolve_config_paths(config: dict[str, Any], tool_dir: Path) -> dict[str, Any]:
    """
    Resolve relative paths in a tool configuration.
    
    Resolves:
    - server.working_dir: Relative to tool directory
    - server.python_path: Relative to tool directory
    
    Args:
        config: Configuration dictionary
        tool_dir: Path to the tool's directory
        
    Returns:
        Configuration with resolved paths
    """
    config = dict(config)  # Shallow copy
    
    server_config = config.get("server", {})
    if server_config:
        server_config = dict(server_config)  # Shallow copy
        
        # Resolve working_dir
        if "working_dir" in server_config:
            working_dir = server_config["working_dir"]
            if working_dir:
                resolved = resolve_path(working_dir, tool_dir)
                server_config["working_dir"] = str(resolved)
        
        # Resolve python_path
        if "python_path" in server_config:
            python_path = server_config["python_path"]
            if python_path:
                resolved = resolve_path(python_path, tool_dir)
                server_config["python_path"] = str(resolved)
        
        config["server"] = server_config
    
    return config


def load_mcp_server_config(
    tool_name: str,
    catalog_dir: Path | None = None,
) -> MCPServerConfig:
    """
    Load an MCP server configuration for a tool.
    
    This function loads the tool's config.yaml, resolves relative paths,
    and returns an MCPServerConfig instance.
    
    Args:
        tool_name: Name of the tool
        catalog_dir: Optional catalog directory path
        
    Returns:
        MCPServerConfig instance
        
    Raises:
        FileNotFoundError: If tool or config not found
        ValueError: If config is invalid
    """
    config = load_tool_config(tool_name, catalog_dir, resolve_relative_paths=True)
    
    server_config = config.get("server")
    if not server_config:
        raise ValueError(f"No server configuration found for tool: {tool_name}")
    
    return MCPServerConfig(**server_config)


def list_available_tools(catalog_dir: Path | None = None) -> list[str]:
    """
    List all available tools in the catalog.
    
    Args:
        catalog_dir: Optional catalog directory path
        
    Returns:
        List of tool names
    """
    if catalog_dir is None:
        catalog_dir = get_catalog_dir()
    
    tools = []
    for item in catalog_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            if (item / "config.yaml").exists():
                tools.append(item.name)
    
    return sorted(tools)


def get_tool_readme(tool_name: str, catalog_dir: Path | None = None) -> str | None:
    """
    Get the contents of a tool's README.md.
    
    Args:
        tool_name: Name of the tool
        catalog_dir: Optional catalog directory path
        
    Returns:
        README contents or None if not found
    """
    if catalog_dir is None:
        catalog_dir = get_catalog_dir()
    
    tool_dir = get_tool_dir(tool_name, catalog_dir)
    readme_path = tool_dir / "README.md"
    
    if not readme_path.exists():
        return None
    
    with open(readme_path, "r", encoding="utf-8") as f:
        return f.read()


async def register_all_mcp_tools(
    registry: ToolRegistry,
    server_manager: MCPServerManager,
    catalog_dir: Path | None = None,
    tool_names: list[str] | None = None,
    verbose: bool = False,
) -> list[str]:
    """
    Auto-register all MCP tools from the catalog.
    
    This function discovers all available MCP tools in the catalog and
    registers them with the provided registry and server manager.
    
    Args:
        registry: Tool registry to register tools to
        server_manager: MCP server manager for tool execution
        catalog_dir: Optional catalog directory path (defaults to built-in catalog)
        tool_names: Optional list of specific tools to register. If None,
                   all available tools are registered.
        verbose: If True, print registration progress
        
    Returns:
        List of registered tool names
        
    Example:
        >>> from audio_agent.tools.catalog import register_all_mcp_tools
        >>> from audio_agent.tools.mcp import MCPServerManager
        >>> from audio_agent.tools.registry import ToolRegistry
        >>> 
        >>> registry = ToolRegistry()
        >>> server_manager = MCPServerManager()
        >>> registered = await register_all_mcp_tools(
        ...     registry, server_manager, verbose=True
        ... )
        >>> print(f"Registered {len(registered)} tools: {registered}")
    """
    # Import here to avoid circular imports
    from audio_agent.tools.mcp import MCPToolAdapter
    
    if catalog_dir is None:
        catalog_dir = get_catalog_dir()
    
    # Discover tools to register
    if tool_names is None:
        tool_names = list_available_tools(catalog_dir)
    
    registered: list[str] = []
    
    for tool_name in tool_names:
        try:
            if verbose:
                print(f"Registering MCP tool: {tool_name}...")
            
            # Load config and register with server manager
            config = load_mcp_server_config(tool_name, catalog_dir)
            server_manager.register_config(tool_name, config)
            
            # Get client and discover tools
            client = await server_manager.get_client(tool_name)
            tools = await client.list_tools()
            
            # Register each tool from the server
            for tool_info in tools:
                # Skip healthcheck tools - they're for environment verification only
                if tool_info.name == "healthcheck":
                    if verbose:
                        print(f"  ⏭ Skipped: healthcheck (environment verification only)")
                    continue
                
                adapter = MCPToolAdapter(
                    server_name=tool_name,
                    tool_info=tool_info,
                    server_manager=server_manager,
                )
                registry.register_mcp(adapter)
                registered.append(tool_info.name)
                
                if verbose:
                    print(f"  ✓ Registered: {tool_info.name}")
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Failed to register {tool_name}: {e}")
            # Continue with other tools even if one fails
            continue
    
    if verbose:
        print(f"\nTotal registered: {len(registered)} tool(s)")
    
    return registered
