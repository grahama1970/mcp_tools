"""MCP integration layer for gitget module.

This module provides integration with the MCP (Model Control Protocol) for the gitget
module, allowing it to be used as a tool within Claude and other MCP-compatible assistants.

Links to third-party package documentation:
- FastMCP: https://fastmcp.readthedocs.io/en/latest/

Exports:
- handler: Main MCP request handler function
- mcp_app: FastMCP application instance
- schema: JSON schema definitions for MCP

Usage example:
    >>> from mcp_tools.gitget.mcp import handler
    >>> response = handler({"name": "gitget", "input": {...}})
"""

from .wrapper import handler, mcp_app
from . import schema

__all__ = ["handler", "mcp_app", "schema"]