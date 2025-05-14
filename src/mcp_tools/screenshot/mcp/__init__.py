"""
MCP Layer for Screenshot Module

This package contains the MCP (Model Context Protocol) layer for the screenshot functionality,
providing wrappers for the core functionality to be used by AI agents.

The MCP layer is designed to:
1. Expose core functions as MCP tools
2. Handle MCP-specific protocol requirements
3. Manage server startup and configuration
4. Implement MCP-compatible error handling

Usage:
    # Start the MCP server
    python -m mcp_tools.screenshot.mcp.mcp_server start
    
    # Use the MCP server in Python
    from mcp_tools.screenshot.mcp import create_mcp_server
    mcp = create_mcp_server()
    mcp.run()
"""

# MCP server creation
from mcp_tools.screenshot.mcp.mcp_tools import create_mcp_server

# MCP server entry point
from mcp_tools.screenshot.mcp.mcp_server import (
    main,
    health_check,
    get_server_info,
    configure_logging
)

# MCP wrappers
from mcp_tools.screenshot.mcp.wrappers import (
    screenshot_wrapper,
    describe_screenshot_wrapper,
    regions_wrapper,
    describe_image_wrapper,
    format_mcp_response
)

__all__ = [
    # MCP server
    'create_mcp_server',
    'main',
    'health_check',
    'get_server_info',
    'configure_logging',
    
    # MCP wrappers
    'screenshot_wrapper',
    'describe_screenshot_wrapper',
    'regions_wrapper',
    'describe_image_wrapper',
    'format_mcp_response'
]

# Example configuration for .mcp.json
EXAMPLE_MCP_CONFIG = """
{
  "mcpServers": {
    "screenshot": {
      "command": "python",
      "args": [
        "-m",
        "mcp_tools.screenshot.integration.mcp_server",
        "start"
      ]
    }
  }
}
"""

if __name__ == "__main__":
    print("""
Example usage of the screenshot MCP server:

# Start the MCP server
python -m mcp_tools.screenshot.integration.mcp_server start

# Run a health check
python -m mcp_tools.screenshot.integration.mcp_server health

# Show server information
python -m mcp_tools.screenshot.integration.mcp_server info

# Show API schema
python -m mcp_tools.screenshot.integration.mcp_server schema

# Example .mcp.json configuration:
""")
    print(EXAMPLE_MCP_CONFIG)
