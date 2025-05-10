#!/usr/bin/env python3

try:
    from mcp.server.fastmcp import FastMCP
    print("MCP is installed")
except ImportError as e:
    print(f"MCP is NOT installed: {e}")