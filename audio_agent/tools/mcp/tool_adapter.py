"""
MCPToolAdapter - Adapts MCP tools to the BaseTool interface.
"""

from __future__ import annotations

from audio_agent.core.schemas import ToolSpec, ToolCallRequest, ToolResult
from audio_agent.core.errors import ToolExecutionError
from audio_agent.tools.base import BaseTool
from audio_agent.tools.mcp.schemas import MCPToolInfo
from audio_agent.tools.mcp.server_manager import MCPServerManager


class MCPToolAdapter(BaseTool):
    """
    Adapts an MCP tool to the BaseTool interface.
    
    This allows MCP tools to be used seamlessly with the existing
    tool registry and executor.
    """
    
    def __init__(
        self,
        server_name: str,
        tool_info: MCPToolInfo,
        server_manager: MCPServerManager,
    ):
        """
        Initialize MCP tool adapter.
        
        Args:
            server_name: Name of the MCP server hosting this tool
            tool_info: MCP tool information from server
            server_manager: Server manager for getting client
        """
        self._server_name = server_name
        self._tool_info = tool_info
        self._server_manager = server_manager
        self._spec = self._build_spec()
    
    @property
    def spec(self) -> ToolSpec:
        """Return the tool specification."""
        return self._spec
    
    def _build_spec(self) -> ToolSpec:
        """
        Convert MCP tool info to ToolSpec.
        
        The MCP inputSchema maps directly to our input_schema.
        Output is always a dict with content.
        """
        return ToolSpec(
            name=self._tool_info.name,
            description=self._tool_info.description,
            input_schema=self._tool_info.inputSchema,
            output_schema={
                "type": "object",
                "properties": {
                    "content": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "text": {"type": "string"},
                            }
                        }
                    },
                    "isError": {"type": "boolean"},
                    "error": {"type": "string"},
                }
            },
            tags=["mcp", self._server_name],
        )
    
    async def invoke(self, request: ToolCallRequest) -> ToolResult:
        """
        Invoke the MCP tool via the server.
        
        This is an async method that communicates with the MCP server.
        
        Args:
            request: Tool call request with arguments
            
        Returns:
            ToolResult with the tool output
        """
        # Validate request
        self.validate_request(request)

        client = None
        try:
            # Get client for server
            client = await self._server_manager.get_client(self._server_name)
            
            # Call the tool
            result = await client.call_tool(
                name=self._tool_info.name,
                arguments=request.args
            )
            
            # Convert MCP result to ToolResult
            if result.isError:
                return ToolResult(
                    tool_name=self.spec.name,
                    success=False,
                    output={},
                    error_message=result.error or "MCP tool returned error",
                )
            
            # Extract text content from result
            output = {"content": []}
            for item in result.content:
                if hasattr(item, "model_dump"):
                    output["content"].append(item.model_dump())
                else:
                    output["content"].append(item)
            
            # Combine text for convenience
            text_parts = []
            for item in result.content:
                if item.type == "text":
                    text_parts.append(item.text)
            output["text"] = "\n".join(text_parts)
            
            # Try to extract structured data from text (e.g., JSON output)
            # This handles tools that return JSON with output_path, etc.
            self._extract_structured_output(output)
            
            return ToolResult(
                tool_name=self.spec.name,
                success=True,
                output=output,
            )
            
        except Exception as e:
            return ToolResult(
                tool_name=self.spec.name,
                success=False,
                output={},
                error_message=f"MCP tool invocation failed: {e}",
            )
        finally:
            # per_call clients are ephemeral and not tracked by the manager, so stop them
            # here to avoid leaking a subprocess on every invocation.
            if client is not None and self._server_manager.is_per_call(self._server_name):
                try:
                    await client.stop()
                except Exception:
                    pass
    
    def _extract_structured_output(self, output: dict) -> None:
        """
        Extract structured data from text content.
        
        Parses JSON from text fields to extract fields like:
        - output_path: Set as generated_audio_path for audio processing tools
        - duration, sample_rate, channels: Audio metadata
        """
        import json
        
        # Try to parse JSON from the combined text
        text = output.get("text", "")
        if not text:
            return
        
        try:
            # Try parsing the entire text as JSON
            data = json.loads(text)
            if isinstance(data, dict):
                # If there's an output_path, also set it as generated_audio_path
                # This allows tool_executor_node to track the generated audio
                if "output_path" in data and "generated_audio_path" not in output:
                    output["generated_audio_path"] = data["output_path"]
                
                # Copy other common audio metadata fields
                for key in ["duration", "sample_rate", "channels", "format"]:
                    if key in data and key not in output:
                        output[key] = data[key]
                
                # Store the parsed data for convenience
                output["parsed_data"] = data
        except json.JSONDecodeError:
            # Not valid JSON, ignore
            pass
    
    def validate_request(self, request: ToolCallRequest) -> None:
        """
        Validate a request before execution.
        
        Raises:
            ToolExecutionError: If request is invalid
        """
        if request.tool_name != self.spec.name:
            raise ToolExecutionError(
                f"Tool name mismatch: expected '{self.spec.name}', got '{request.tool_name}'",
                details={"expected": self.spec.name, "actual": request.tool_name}
            )
