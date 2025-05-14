import asyncio
import json
import os
import sys
import requests
from modelcontextprotocol.server import Server, StdioServerTransport
from modelcontextprotocol.types import (
    CallToolRequestSchema,
    CallToolResponseSchema,
    ErrorCode,
    ListToolsRequestSchema,
    ListToolsResponseSchema,
    McpError,
    TextContentSchema,
)

# Ensure the src directory is in the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Configuration ---
SERVER_NAME = "mcp-litellm-batch-py"
SERVER_VERSION = "0.1.0"
TOOL_NAME = "litellm_batch_ask"
# Use localhost as the MCP server runs alongside the FastAPI service (likely in the same container eventually, or on the host)
FASTAPI_ENDPOINT = os.environ.get("MCP_LITELLM_FASTAPI_ENDPOINT", "http://localhost:8000/ask")

# --- MCP Server Implementation ---

class LiteLLMMcpServer:
    """
    MCP Server that acts as a bridge to the LiteLLM FastAPI service.
    """
    def __init__(self):
        self.server = Server(
            {"name": SERVER_NAME, "version": SERVER_VERSION},
            {"capabilities": {"tools": {}}}, # Announce tool capability
        )
        self._setup_handlers()
        self.server.onerror = self._handle_error

    def _handle_error(self, error: McpError):
        """Logs MCP errors."""
        print(f"[MCP Error] Code: {error.code}, Message: {error.message}", file=sys.stderr)
        if error.data:
            print(f"  Data: {error.data}", file=sys.stderr)

    def _setup_handlers(self):
        """Sets up handlers for MCP requests."""
        self.server.setRequestHandler(ListToolsRequestSchema, self._handle_list_tools)
        self.server.setRequestHandler(CallToolRequestSchema, self._handle_call_tool)

    async def _handle_list_tools(self, request) -> ListToolsResponseSchema:
        """Handles requests to list available tools."""
        return {
            "tools": [
                {
                    "name": TOOL_NAME,
                    "description": "Sends a batch of questions to the LiteLLM service for processing.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": {"type": "object"}, # Keep it flexible, FastAPI validates
                                "description": "A list of question objects for batch processing."
                            }
                        },
                        "required": ["questions"],
                    },
                }
            ]
        }

    async def _handle_call_tool(self, request) -> CallToolResponseSchema:
        """Handles requests to call a specific tool."""
        if request.params.name != TOOL_NAME:
            raise McpError(ErrorCode.MethodNotFound, f"Unknown tool: {request.params.name}")

        tool_args = request.params.arguments
        if not isinstance(tool_args, dict) or "questions" not in tool_args:
             raise McpError(ErrorCode.InvalidParams, f"Invalid arguments for {TOOL_NAME}. Expected JSON object with 'questions' key.")

        try:
            # Call the FastAPI endpoint
            response = requests.post(FASTAPI_ENDPOINT, json=tool_args, timeout=120) # Increased timeout for potentially long LLM calls
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

            # Return the successful response from FastAPI
            return {
                "content": [
                    {
                        "type": "text",
                        "text": response.text # Return the raw JSON response text
                    }
                ]
            }
        except requests.exceptions.RequestException as e:
            # Handle network/request errors
            error_message = f"Error calling FastAPI service at {FASTAPI_ENDPOINT}: {e}"
            print(error_message, file=sys.stderr)
            # Return error information via MCP
            return {
                 "content": [
                    {
                        "type": "text",
                        "text": error_message
                    }
                ],
                "isError": True
            }
        except Exception as e:
             # Handle unexpected errors
            error_message = f"Unexpected error processing tool call: {e}"
            print(error_message, file=sys.stderr)
            raise McpError(ErrorCode.InternalError, error_message)


    async def run(self):
        """Connects to the transport and runs the server."""
        transport = StdioServerTransport()
        await self.server.connect(transport)
        print(f"{SERVER_NAME} v{SERVER_VERSION} running on stdio, proxying to {FASTAPI_ENDPOINT}", file=sys.stderr)
        # Keep the server running indefinitely
        await asyncio.Event().wait()

# --- Main Execution ---
if __name__ == "__main__":
    server_instance = LiteLLMMcpServer()
    try:
        asyncio.run(server_instance.run())
    except KeyboardInterrupt:
        print("\nShutting down MCP server...", file=sys.stderr)