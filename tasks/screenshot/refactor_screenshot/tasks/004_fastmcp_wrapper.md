# Task 4: Create FastMCP Wrapper

**Objective**: Implement a FastMCP wrapper to integrate the screenshot tool with MCP.

**Pre-Task Setup**:
- Review existing MCP implementation for compatibility

**Implementation Steps**:
- [ ] 4.1 Create mcp_tools.py module with FastMCP integration
- [ ] 4.2 Implement server creation function with debug option
- [ ] 4.3 Implement tool registration functions
- [ ] 4.4 Adapt CLI functions for MCP interface
- [ ] 4.5 Add detailed logging for debugging
- [ ] 4.6 Implement parameter conversion between CLI and MCP
- [ ] 4.7 Add stand-alone server test in `__main__` section
- [ ] 4.8 Test MCP tools registration and function
- [ ] 4.9 Git commit verified MCP wrapper with message "Add: FastMCP wrapper integrating with CLI functions"

**Technical Specifications**:
- Reuse CLI functions where possible to avoid duplication
- Ensure parameter types are compatible between MCP and CLI
- Add detailed logging at each step for debugging
- Maintain the same functionality as original MCP implementation
- Implement debug mode for enhanced logging

**Verification Method**:
- Run module directly to test MCP server: `python -m mcp_tools.screenshot.mcp_tools`
- Check logs for expected registration and function events
- Verify tool registration works correctly

**Acceptance Criteria**:
- MCP tools register correctly with FastMCP
- CLI functions are properly utilized
- Parameter conversion works correctly
- Debug mode provides enhanced logging
- Stand-alone test verifies basic functionality

## Example Implementation Approach

```python
import logging
from typing import Dict, Any, Optional, List
import json
import os
from fastmcp import FastMCP, serve_tools

# Import CLI functions to reuse
from mcp_tools.screenshot.cli import screenshot_func, describe_func

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mcp_screenshot")

def create_server(debug: bool = False) -> FastMCP:
    """
    Create a FastMCP server with screenshot tools.
    
    Args:
        debug: Enable debug mode for extra logging
        
    Returns:
        FastMCP server instance
    """
    if debug:
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled for MCP Screenshot server")
    
    server = FastMCP(
        name="screenshot",
        description="Tools for capturing and describing screenshots",
        version="1.0.0"
    )
    
    logger.info("Registering screenshot tools with MCP server")
    register_tools(server)
    
    return server

def register_tools(server: FastMCP) -> None:
    """
    Register all screenshot tools with the MCP server.
    
    Args:
        server: FastMCP server instance
    """
    # Register the screenshot tool
    server.add_tool(
        name="screenshot",
        description="Capture a screenshot of the desktop or a specific region",
        function=mcp_screenshot,
        parameters={
            "quality": {
                "type": "integer",
                "description": "JPEG compression quality (1-100)",
                "default": 30,
                "minimum": 1,
                "maximum": 100
            },
            "region": {
                "type": "string",
                "description": "Screen region to capture: 'full', 'right_half', or coordinates (format: 'x,y,width,height')",
                "default": "full"
            }
        },
        returns={
            "type": "object",
            "properties": {
                "file": {"type": "string", "description": "Path to the saved screenshot file"},
                "dimensions": {"type": "array", "description": "Width and height of the image"},
                "size": {"type": "integer", "description": "Size in bytes of the saved file"}
            }
        }
    )
    
    # Register additional tools...
    
    logger.debug("All tools registered successfully")

def mcp_screenshot(quality: int = 30, region: str = "full") -> Dict[str, Any]:
    """
    MCP wrapper for screenshot CLI function.
    
    Args:
        quality: JPEG compression quality (1-100)
        region: Screen region to capture
        
    Returns:
        Screenshot result dictionary
    """
    logger.debug(f"MCP screenshot called with quality={quality}, region={region}")
    
    try:
        # Call the CLI function with the same parameters
        result = screenshot_func(
            quality=quality,
            region=region,
            output_dir="screenshots",  # Use default output directory
        )
        
        logger.debug(f"Screenshot result: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error in mcp_screenshot: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    """
    Standalone test for MCP server.
    
    Run with: python -m mcp_tools.screenshot.mcp_tools
    """
    print("Starting MCP Screenshot server in test mode...")
    server = create_server(debug=True)
    
    # Serve tools in a separate thread for testing
    serve_tools(server, host="127.0.0.1", port=8000)
    
    print(f"MCP Screenshot server running on http://127.0.0.1:8000")
    print("Available tools:")
    for tool_name in server.tools:
        print(f" - {tool_name}")
    
    print("\nPress Ctrl+C to stop")
    
    try:
        # Keep the main thread alive
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Server stopped.")
```
