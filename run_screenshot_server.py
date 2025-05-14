#!/usr/bin/env python3
"""
MCP Server Entry Point for Screenshot Tools

This is the main entry point for the screenshot MCP server, designed to be
directly referenced in the .mcp.json configuration.

When this file is executed directly, it creates and starts the MCP server
with all screenshot and image description tools registered.
"""

import os
import sys
from loguru import logger

# Configure logger
logger.remove()  # Remove default handler that outputs to stderr
logger.add(
    "logs/mcp_server.log",
    rotation="10 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Import the create_mcp_server function from the package
from mcp_tools.screenshot.mcp_tools import create_mcp_server


def main():
    """Create and run the MCP server with screenshot tools."""
    logger.info("Starting MCP server for screenshot tools")
    
    # Create the MCP server with screenshot tools
    mcp = create_mcp_server("Screenshot Tools")
    
    # Run the server
    mcp.run()
    
    return 0


if __name__ == "__main__":
    """
    Direct entry point for the screenshot MCP server.
    This file is designed to be referenced in .mcp.json.
    """
    sys.exit(main())