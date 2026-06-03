#!/usr/bin/env python3
"""
Template MCP Server

This is a template for implementing an MCP (Model Context Protocol) server.
Copy this file and customize for your tool.

The server communicates via stdin/stdout using JSON-RPC messages.
"""

from __future__ import annotations

import json
import sys
import os
from typing import Any


class TemplateMCPServer:
    """
    Template MCP Server implementation.
    
    Handles:
    - JSON-RPC message parsing
    - MCP protocol methods (initialize, tools/list, tools/call)
    - Tool execution
    """
    
    def __init__(self):
        """Initialize the server."""
        self._initialized = False
        self._tools = [
            {
                "name": "example_tool",
                "description": "An example tool that echoes input",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "input_text": {
                            "type": "string",
                            "description": "The input text to process"
                        }
                    },
                    "required": ["input_text"]
                }
            }
        ]
    
    def run(self) -> None:
        """Run the server, reading from stdin and writing to stdout."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            
            try:
                request = json.loads(line)
                response = self._handle_request(request)
                if response:
                    self._send_response(response)
            except json.JSONDecodeError as e:
                self._send_error(None, -32700, f"Parse error: {e}")
            except Exception as e:
                request_id = request.get("id") if isinstance(request, dict) else None
                self._send_error(request_id, -32603, f"Internal error: {e}")
    
    def _handle_request(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """
        Handle a JSON-RPC request.
        
        Returns:
            Response dict, or None for notifications
        """
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})
        
        if method == "initialize":
            return self._handle_initialize(request_id, params)
        
        elif method == "notifications/initialized":
            # Notification, no response needed
            return None
        
        elif method == "tools/list":
            return self._handle_tools_list(request_id)
        
        elif method == "tools/call":
            return self._handle_tools_call(request_id, params)
        
        elif method == "shutdown":
            return self._handle_shutdown(request_id)
        
        else:
            return self._error_response(request_id, -32601, f"Method not found: {method}")
    
    def _handle_initialize(self, request_id: Any, params: dict) -> dict[str, Any]:
        """Handle initialize request."""
        self._initialized = True
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {
                    "tools": {}
                },
                "serverInfo": {
                    "name": "template-tool-server",
                    "version": "1.0.0"
                }
            }
        }
    
    def _handle_tools_list(self, request_id: Any) -> dict[str, Any]:
        """Handle tools/list request."""
        if not self._initialized:
            return self._error_response(request_id, -32001, "Server not initialized")
        
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "tools": self._tools
            }
        }
    
    def _handle_tools_call(self, request_id: Any, params: dict) -> dict[str, Any]:
        """Handle tools/call request."""
        if not self._initialized:
            return self._error_response(request_id, -32001, "Server not initialized")
        
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        
        try:
            result = self._execute_tool(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": result
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Error: {e}"
                        }
                    ],
                    "isError": True,
                    "error": str(e)
                }
            }
    
    def _execute_tool(self, tool_name: str, arguments: dict) -> dict[str, Any]:
        """
        Execute a tool with given arguments.
        
        Implement your tool logic here.
        """
        if tool_name == "example_tool":
            input_text = arguments.get("input_text", "")
            
            # Your tool logic here
            result_text = f"Processed: {input_text}"
            
            return {
                "content": [
                    {
                        "type": "text",
                        "text": result_text
                    }
                ],
                "isError": False
            }
        
        else:
            raise ValueError(f"Unknown tool: {tool_name}")
    
    def _handle_shutdown(self, request_id: Any) -> dict[str, Any]:
        """Handle shutdown request."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {}
        }
    
    def _error_response(self, request_id: Any, code: int, message: str) -> dict[str, Any]:
        """Create an error response."""
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": code,
                "message": message
            }
        }
    
    def _send_response(self, response: dict[str, Any]) -> None:
        """Send a response to stdout."""
        response_json = json.dumps(response) + "\n"
        sys.stdout.write(response_json)
        sys.stdout.flush()
    
    def _send_error(self, request_id: Any, code: int, message: str) -> None:
        """Send an error response."""
        self._send_response(self._error_response(request_id, code, message))


def main():
    """Main entry point."""
    # Optional: Load configuration from environment
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    
    # Create and run server
    server = TemplateMCPServer()
    server.run()


if __name__ == "__main__":
    main()
