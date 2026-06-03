"""
Tool executor for invoking tools from the registry.

Provides a clean interface for executing tools with validation and error handling.
Supports both synchronous and asynchronous tools.
"""

from __future__ import annotations

import asyncio
import inspect
import time

from audio_agent.core.schemas import ToolCallRequest, ToolResult
from audio_agent.core.errors import ToolExecutionError, ToolRegistryError
from audio_agent.core.logging import log_tool_call, log_tool_result, log_error
from audio_agent.tools.registry import ToolRegistry


class ToolExecutor:
    """
    Executor for running tools from a registry.
    
    Handles:
    - Tool lookup
    - Request validation
    - Execution timing
    - Error wrapping
    - Both sync and async tool invoke methods
    """
    
    def __init__(self, registry: ToolRegistry) -> None:
        """
        Initialize executor with a tool registry.
        
        Args:
            registry: ToolRegistry containing available tools
        """
        if registry is None:
            raise ValueError("ToolExecutor requires a non-None registry")
        self._registry = registry
    
    async def execute(self, request: ToolCallRequest) -> ToolResult:
        """
        Execute a tool call request asynchronously.
        
        This method handles both synchronous and asynchronous tools:
        - Async tools: Awaited directly
        - Sync tools: Run in thread pool to avoid blocking
        
        Args:
            request: ToolCallRequest specifying tool and arguments
        
        Returns:
            ToolResult from the tool
        
        Raises:
            ToolExecutionError: If tool not found or execution fails
        """
        if request is None:
            raise ToolExecutionError("Cannot execute None request")
        
        tool_name = request.tool_name
        log_tool_call(tool_name, request.args)
        
        # Look up tool
        try:
            tool = self._registry.get(tool_name)
        except ToolRegistryError as e:
            raise ToolExecutionError(
                f"Failed to find tool '{tool_name}'",
                details={"original_error": str(e)}
            ) from e
        
        # Execute with timing
        start_time = time.time()
        try:
            # Check if invoke is async
            if inspect.iscoroutinefunction(tool.invoke):
                # Async tool - await directly
                result = await tool.invoke(request)
            else:
                # Sync tool - run in thread pool to avoid blocking
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,  # Uses default executor
                    tool.invoke,
                    request
                )
        except ToolExecutionError:
            raise
        except Exception as e:
            log_error("ToolExecutor", e, {"tool_name": tool_name})
            raise ToolExecutionError(
                f"Tool '{tool_name}' raised unexpected error: {type(e).__name__}: {e}",
                details={"tool_name": tool_name, "error_type": type(e).__name__}
            ) from e
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        # Validate result
        if result is None:
            raise ToolExecutionError(
                f"Tool '{tool_name}' returned None instead of ToolResult",
                details={"tool_name": tool_name}
            )
        
        if not isinstance(result, ToolResult):
            raise ToolExecutionError(
                f"Tool '{tool_name}' returned {type(result).__name__} instead of ToolResult",
                details={"tool_name": tool_name, "actual_type": type(result).__name__}
            )
        
        # Update execution time if not set
        if result.execution_time_ms == 0.0:
            result = ToolResult(
                tool_name=result.tool_name,
                success=result.success,
                output=result.output,
                error_message=result.error_message,
                execution_time_ms=elapsed_ms,
                timestamp=result.timestamp,
            )
        
        log_tool_result(
            tool_name,
            result.success,
            f"Output keys: {list(result.output.keys())}" if result.output else "Empty output"
        )
        
        return result
    
    def execute_sync(self, request: ToolCallRequest) -> ToolResult:
        """
        Execute a tool synchronously.
        
        This is a convenience method for sync-only contexts.
        For async contexts, use execute() instead.
        
        Note: MCP tools require async, so this will fail for MCP tools.
        Use execute() for full compatibility.
        
        Args:
            request: ToolCallRequest specifying tool and arguments
        
        Returns:
            ToolResult from the tool
        """
        if request is None:
            raise ToolExecutionError("Cannot execute None request")
        
        tool_name = request.tool_name
        tool = self._registry.get(tool_name)
        
        # Check if tool is async
        if inspect.iscoroutinefunction(tool.invoke):
            raise ToolExecutionError(
                f"Tool '{tool_name}' is async. Use execute() instead of execute_sync().",
                details={"tool_name": tool_name}
            )
        
        log_tool_call(tool_name, request.args)
        
        start_time = time.time()
        try:
            result = tool.invoke(request)
        except Exception as e:
            log_error("ToolExecutor", e, {"tool_name": tool_name})
            raise ToolExecutionError(
                f"Tool '{tool_name}' raised unexpected error: {type(e).__name__}: {e}",
                details={"tool_name": tool_name, "error_type": type(e).__name__}
            ) from e
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        if result is None:
            raise ToolExecutionError(
                f"Tool '{tool_name}' returned None instead of ToolResult",
                details={"tool_name": tool_name}
            )
        
        if not isinstance(result, ToolResult):
            raise ToolExecutionError(
                f"Tool '{tool_name}' returned {type(result).__name__} instead of ToolResult",
                details={"tool_name": tool_name, "actual_type": type(result).__name__}
            )
        
        if result.execution_time_ms == 0.0:
            result = ToolResult(
                tool_name=result.tool_name,
                success=result.success,
                output=result.output,
                error_message=result.error_message,
                execution_time_ms=elapsed_ms,
                timestamp=result.timestamp,
            )
        
        log_tool_result(
            tool_name,
            result.success,
            f"Output keys: {list(result.output.keys())}" if result.output else "Empty output"
        )
        
        return result
