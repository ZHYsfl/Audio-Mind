"""
MCP Client for communicating with MCP servers via stdio.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from audio_agent.core.errors import ToolExecutionError
from audio_agent.tools.mcp.schemas import (
    MCPRequest,
    MCPResponse,
    MCPToolInfo,
    MCPCallResult,
)


class MCPClient:
    """
    Client for communicating with an MCP server via stdio.
    
    Manages the server process lifecycle and JSON-RPC communication.
    """
    
    def __init__(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        startup_timeout_sec: float = 300.0,
        working_dir: str | None = None,
    ):
        """
        Initialize MCP client.
        
        Args:
            command: Command to spawn server (e.g., ["python", "server.py"])
            env: Additional environment variables for server
            startup_timeout_sec: Timeout for server startup
            working_dir: Working directory for server process
        """
        self._command = command
        self._env = env or {}
        self._startup_timeout = startup_timeout_sec
        self._working_dir = working_dir
        self._process: asyncio.subprocess.Process | None = None
        self._request_id = 0
        self._lock = asyncio.Lock()
        self._stderr_buffer: list[str] = []
        self._stderr_task: asyncio.Task | None = None
        
    async def _read_stderr(self) -> None:
        """Continuously read stderr and store for debugging."""
        if self._process is None or self._process.stderr is None:
            return
        
        try:
            while True:
                line = await self._process.stderr.readline()
                if not line:
                    break
                stderr_line = line.decode().strip()
                self._stderr_buffer.append(stderr_line)
                # Keep only last 100 lines to prevent memory issues
                if len(self._stderr_buffer) > 100:
                    self._stderr_buffer.pop(0)
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
    
    def get_stderr_logs(self) -> str:
        """Get captured stderr logs for debugging."""
        return "\n".join(self._stderr_buffer)
        
    async def start(self) -> None:
        """
        Start the MCP server process and perform handshake.
        
        Raises:
            ToolExecutionError: If server fails to start
        """
        if self._process is not None:
            return  # Already started
            
        # Merge environment variables
        env = {**os.environ, **self._env}
        
        # Prepare cwd parameter
        cwd = self._working_dir if self._working_dir else None
        
        try:
            self._process = await asyncio.create_subprocess_exec(
                *self._command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=cwd,
            )
        except Exception as e:
            raise ToolExecutionError(
                f"Failed to start MCP server: {e}",
                details={"command": self._command}
            ) from e
        
        # Start stderr reader task
        self._stderr_task = asyncio.create_task(self._read_stderr())
        
        # Perform initialize handshake
        try:
            await self._send_initialize()
        except Exception as e:
            await self.stop()
            raise ToolExecutionError(
                f"MCP server initialization failed: {e}",
                details={"command": self._command, "stderr": self.get_stderr_logs()}
            ) from e
    
    async def _send_initialize(self) -> None:
        """Send initialize request to server."""
        init_request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "audio-agent",
                    "version": "0.1.0"
                }
            }
        }
        
        response = await self._send_request_raw(init_request)
        if response.get("error"):
            raise ToolExecutionError(
                f"MCP initialize failed: {response['error']}"
            )
        
        # Send initialized notification
        initialized_notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        }
        await self._send_notification(initialized_notification)
    
    def _next_id(self) -> int:
        """Get next request ID."""
        self._request_id += 1
        return self._request_id
    
    async def _send_request_raw(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send a raw JSON-RPC request and return response."""
        if self._process is None or self._process.stdin is None:
            raise ToolExecutionError("MCP server not started")
        
        request_json = json.dumps(request) + "\n"
        request_bytes = request_json.encode()
        
        async with self._lock:
            try:
                self._process.stdin.write(request_bytes)
                await self._process.stdin.drain()
                
                # Read response line
                response_line = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=self._startup_timeout
                )
                
                if not response_line:
                    raise ToolExecutionError("MCP server closed connection")
                
                response = json.loads(response_line.decode())
                return response
                
            except asyncio.TimeoutError as e:
                raise ToolExecutionError(
                    f"MCP request timeout after {self._startup_timeout}s"
                ) from e
            except json.JSONDecodeError as e:
                raise ToolExecutionError(
                    f"Invalid JSON from MCP server: {e}"
                ) from e
    
    async def _send_notification(self, notification: dict[str, Any]) -> None:
        """Send a notification (no response expected)."""
        if self._process is None or self._process.stdin is None:
            raise ToolExecutionError("MCP server not started")
        
        notification_json = json.dumps(notification) + "\n"
        notification_bytes = notification_json.encode()
        
        async with self._lock:
            self._process.stdin.write(notification_bytes)
            await self._process.stdin.drain()
    
    async def list_tools(self) -> list[MCPToolInfo]:
        """
        Get list of available tools from server.
        
        Returns:
            List of MCPToolInfo objects
            
        Raises:
            ToolExecutionError: If request fails
        """
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {}
        }
        
        response = await self._send_request_raw(request)
        
        if response.get("error"):
            raise ToolExecutionError(
                f"Failed to list tools: {response['error']}"
            )
        
        tools = response.get("result", {}).get("tools", [])
        return [MCPToolInfo(**tool) for tool in tools]
    
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPCallResult:
        """
        Invoke a tool on the server.
        
        Args:
            name: Tool name
            arguments: Tool arguments
            
        Returns:
            MCPCallResult with tool output
            
        Raises:
            ToolExecutionError: If invocation fails
        """
        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments
            }
        }
        
        response = await self._send_request_raw(request)
        
        if response.get("error"):
            error = response["error"]
            stderr_logs = self.get_stderr_logs()
            error_msg = error.get('message', 'Unknown error')
            
            # Include error data if available (traceback, etc.)
            error_data = error.get('data')
            if error_data and isinstance(error_data, dict):
                if 'traceback' in error_data:
                    error_msg += f"\n\nTraceback:\n{error_data['traceback']}"
            
            if stderr_logs:
                error_msg += f"\n\nServer stderr:\n{stderr_logs}"
            return MCPCallResult(
                isError=True,
                error=error_msg
            )
        
        result = response.get("result", {})
        return MCPCallResult(**result)
    
    async def stop(self) -> None:
        """Stop the server process gracefully."""
        # Cancel stderr reader task
        if self._stderr_task is not None:
            self._stderr_task.cancel()
            try:
                await self._stderr_task
            except asyncio.CancelledError:
                pass
            self._stderr_task = None
        
        if self._process is None:
            return
        
        try:
            # Send shutdown request if possible
            shutdown_request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "shutdown"
            }
            try:
                await asyncio.wait_for(
                    self._send_request_raw(shutdown_request),
                    timeout=5.0
                )
            except Exception:
                pass  # Ignore shutdown errors
            
            # Terminate process
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
                
        finally:
            self._process = None
    
    @property
    def is_running(self) -> bool:
        """Check if server process is running."""
        if self._process is None:
            return False
        return self._process.returncode is None
