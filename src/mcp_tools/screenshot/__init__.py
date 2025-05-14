"""
Screenshot MCP Tool

A modular, robust MCP tool for capturing, processing, and describing screenshots
with a clean three-layer architecture.

This package implements a clean three-layer architecture for maximum maintainability,
testability, and extensibility:

1. Core Layer: Pure business logic
2. Presentation Layer: CLI interface with rich formatting
3. Integration Layer: MCP wrapper for AI agent usage

Usage:
    # Direct API usage (Core Layer)
    from mcp_tools.screenshot.core import capture_screenshot, describe_image_content
    result = capture_screenshot(quality=70, region="right_half")
    description = describe_image_content(image_path=result["file"])
    
    # CLI usage (Presentation Layer)
    # python -m mcp_tools.screenshot.presentation.cli screenshot --quality 70
    
    # MCP server usage (Integration Layer)
    # python -m mcp_tools.screenshot.integration.mcp_server start
"""

import sys

# Core functionality
from mcp_tools.screenshot.core import (
    capture_screenshot,
    describe_image_content,
    get_screen_regions
)

# CLI layer
from mcp_tools.screenshot.cli import app as cli_app

# MCP layer
from mcp_tools.screenshot.mcp import create_mcp_server

__version__ = "1.0.0"
__author__ = "Anthropic"

__all__ = [
    # Core functions
    'capture_screenshot',
    'describe_image_content',
    'get_screen_regions',
    
    # CLI entrypoint
    'cli_app',
    
    # MCP server
    'create_mcp_server',
    
    # Version info
    '__version__',
    '__author__'
]

if __name__ == "__main__":
    print("""
Screenshot MCP Tool - Command Help

Usage:
  # Take a screenshot
  python -m mcp_tools.screenshot.presentation.cli screenshot --quality 70
  
  # Get a description
  python -m mcp_tools.screenshot.presentation.cli describe --region right_half
  
  # Start MCP server
  python -m mcp_tools.screenshot.integration.mcp_server start
  
  # Show help
  python -m mcp_tools.screenshot.presentation.cli --help
""")
