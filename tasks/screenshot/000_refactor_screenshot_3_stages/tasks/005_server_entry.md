# Task 5: Update Server Entry Point and Configuration

**Objective**: Create an enhanced server entry point with debugging capabilities and update configuration.

**Pre-Task Setup**:
- Review existing .mcp.json configuration

**Implementation Steps**:
- [ ] 5.1 Update run_screenshot_server.py with enhanced options
- [ ] 5.2 Add command-line arguments for debug mode
- [ ] 5.3 Configure logging based on arguments
- [ ] 5.4 Update .mcp.json to reference the entry point
- [ ] 5.5 Add version information and usage examples
- [ ] 5.6 Test server start-up with various options
- [ ] 5.7 Git commit verified entry point with message "Update: Enhanced server entry point with debug capabilities"

**Technical Specifications**:
- Use argparse to handle command-line options
- Set up logging based on verbosity level
- Ensure proper error handling on startup
- Maintain compatibility with .mcp.json format
- Add version and help information

**Verification Method**:
- Start server with various options: `python run_screenshot_server.py --debug`
- Check logs for startup information
- Verify .mcp.json configuration works

**Acceptance Criteria**:
- Server starts correctly with various options
- Logging is properly configured
- Version and help information is displayed
- Compatible with .mcp.json configuration

## Example Implementation Approach

```python
#!/usr/bin/env python3
"""
Screenshot MCP Server Entry Point

This script starts the Screenshot MCP server with configurable options.
"""

import argparse
import logging
import os
import sys
import json
from typing import Dict, Any

# Import the server creation function
from mcp_tools.screenshot.mcp_tools import create_server, serve_tools

# Version information
VERSION = "1.0.0"

def setup_logging(verbosity: int) -> None:
    """
    Configure logging based on verbosity level.
    
    Args:
        verbosity: Verbosity level (0-3)
    """
    log_levels = [
        logging.WARNING,  # 0: Only warnings and errors
        logging.INFO,     # 1: Info, warnings, and errors
        logging.DEBUG,    # 2: Debug and all above
        logging.NOTSET    # 3: All logging
    ]
    
    # Set the level based on verbosity, capped at the highest level
    level = log_levels[min(verbosity, len(log_levels) - 1)]
    
    # Configure the root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def parse_arguments() -> Dict[str, Any]:
    """
    Parse command-line arguments.
    
    Returns:
        Dictionary of parsed arguments
    """
    parser = argparse.ArgumentParser(
        description=f"Screenshot MCP Server v{VERSION}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=9001,
        help="Port to run the server on"
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Host address to bind the server to"
    )
    
    parser.add_argument(
        "-v", "--verbose", 
        action="count", 
        default=0,
        help="Increase verbosity (can be used multiple times)"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode (equivalent to -vv)"
    )
    
    parser.add_argument(
        "--version", 
        action="version",
        version=f"Screenshot MCP Server v{VERSION}"
    )
    
    args = parser.parse_args()
    
    # If debug flag is set, ensure verbosity is at least 2
    if args.debug and args.verbose < 2:
        args.verbose = 2
    
    return vars(args)

def update_mcp_config(host: str, port: int) -> None:
    """
    Update the .mcp.json configuration file if it exists.
    
    Args:
        host: Server host address
        port: Server port
    """
    config_path = os.path.expanduser("~/.mcp.json")
    
    try:
        # Create or update the configuration
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {}
        
        # Update the screenshot server configuration
        if "servers" not in config:
            config["servers"] = {}
        
        config["servers"]["screenshot"] = {
            "url": f"http://{host}:{port}"
        }
        
        # Write the updated configuration
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logging.info(f"Updated MCP configuration at {config_path}")
    
    except Exception as e:
        logging.warning(f"Failed to update MCP configuration: {str(e)}")

def main() -> int:
    """
    Main entry point for the server.
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Set up logging based on verbosity
        setup_logging(args["verbose"])
        
        # Create the server
        server = create_server(debug=args["debug"])
        
        # Update the MCP configuration
        update_mcp_config(args["host"], args["port"])
        
        # Print server information
        print(f"Starting Screenshot MCP Server v{VERSION}")
        print(f"Host: {args['host']}")
        print(f"Port: {args['port']}")
        print(f"Debug mode: {'Enabled' if args['debug'] else 'Disabled'}")
        print(f"Verbosity level: {args['verbose']}")
        print()
        print("Available tools:")
        for tool_name in server.tools:
            print(f" - {tool_name}")
        print()
        print("Press Ctrl+C to stop the server")
        
        # Start the server
        serve_tools(server, host=args["host"], port=args["port"])
        
        # Keep the main thread alive
        import time
        while True:
            time.sleep(1)
            
        return 0
        
    except KeyboardInterrupt:
        print("\nServer stopped.")
        return 0
        
    except Exception as e:
        logging.error(f"Server error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

## .mcp.json Configuration Example

```json
{
  "servers": {
    "screenshot": {
      "command": "python",
      "args": [
        "-m", "mcp_tools.screenshot.run_screenshot_server",
        "--port", "9001"
      ]
    }
  }
}
```
