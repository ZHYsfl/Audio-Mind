"""
MCP Server Manager for managing multiple MCP server processes.
"""

from __future__ import annotations

import asyncio
from typing import Any

from audio_agent.core.errors import ToolExecutionError
from audio_agent.core.logging import get_logger
from audio_agent.tools.mcp.client import MCPClient
from audio_agent.tools.mcp.schemas import MCPServerConfig


class MCPServerManager:
    """
    Manages lifecycle of MCP server processes.
    
    Handles:
    - Server configuration registration
    - Lazy server startup
    - Session-based lifecycle management
    - Graceful shutdown
    """
    
    def __init__(self):
        """Initialize empty server manager."""
        self._configs: dict[str, MCPServerConfig] = {}
        self._clients: dict[str, MCPClient] = {}
        self._lock = asyncio.Lock()
        self._logger = get_logger()
    
    def register_config(self, name: str, config: MCPServerConfig) -> None:
        """
        Register an MCP server configuration.
        
        Args:
            name: Unique server name
            config: Server configuration
        """
        self._configs[name] = config
        self._logger.info(f"Registered MCP server config: {name}")
    
    def register_config_dict(self, name: str, config_dict: dict[str, Any]) -> None:
        """
        Register config from dictionary.
        
        Args:
            name: Unique server name
            config_dict: Configuration as dictionary
        """
        config = MCPServerConfig(**config_dict)
        self.register_config(name, config)
    
    async def get_client(self, name: str) -> MCPClient:
        """
        Get or create client for named server.
        
        For "session" lifecycle, keeps server running.
        For "per_call" lifecycle, creates new client each time.
        
        Args:
            name: Server name
            
        Returns:
            MCPClient for the server
            
        Raises:
            ToolExecutionError: If server not configured or fails to start
        """
        if name not in self._configs:
            raise ToolExecutionError(
                f"MCP server '{name}' not configured",
                details={"available": list(self._configs.keys())}
            )
        
        config = self._configs[name]
        
        # For "per_call" lifecycle, always create new client
        if config.lifecycle == "per_call":
            client = MCPClient(
                command=config.command,
                env=config.env,
                startup_timeout_sec=config.startup_timeout_sec,
                working_dir=config.working_dir,
            )
            await client.start()
            return client
        
        # For "session" or "persistent", use cached client
        async with self._lock:
            if name not in self._clients:
                client = MCPClient(
                    command=config.command,
                    env=config.env,
                    startup_timeout_sec=config.startup_timeout_sec,
                    working_dir=config.working_dir,
                )
                await client.start()
                self._clients[name] = client
                self._logger.info(f"Started MCP server: {name}")
            
            return self._clients[name]
    
    async def list_servers(self) -> list[str]:
        """List names of all registered servers."""
        return list(self._configs.keys())
    
    async def list_running_servers(self) -> list[str]:
        """List names of currently running servers."""
        running = []
        for name, client in self._clients.items():
            if client.is_running:
                running.append(name)
        return running
    
    def is_per_call(self, name: str) -> bool:
        """Whether the named server uses the per_call (ephemeral) lifecycle."""
        config = self._configs.get(name)
        return bool(config and config.lifecycle == "per_call")

    async def stop_server(self, name: str) -> None:
        """
        Stop a specific server.
        
        Args:
            name: Server name to stop
        """
        async with self._lock:
            if name in self._clients:
                client = self._clients.pop(name)
                await client.stop()
                self._logger.info(f"Stopped MCP server: {name}")
    
    async def shutdown_all(self) -> None:
        """Stop all managed servers gracefully."""
        async with self._lock:
            for name, client in list(self._clients.items()):
                try:
                    await client.stop()
                    self._logger.info(f"Stopped MCP server: {name}")
                except Exception as e:
                    self._logger.error(f"Error stopping server {name}: {e}")
            
            self._clients.clear()
    
    async def health_check(self, name: str) -> bool:
        """
        Check if a server is healthy.
        
        Args:
            name: Server name
            
        Returns:
            True if server is running and responsive
        """
        if name not in self._clients:
            return False
        
        client = self._clients[name]
        if not client.is_running:
            return False
        
        # Try to list tools as health check
        try:
            await asyncio.wait_for(client.list_tools(), timeout=10.0)
            return True
        except Exception:
            return False
