#!/usr/bin/env python3
"""
MCP Server Entry Point for Screenshot Tools

This is the main entry point for the screenshot MCP server, designed to be
directly referenced in the .mcp.json configuration.

This module is part of the Integration Layer and connects the MCP functionality
to the application core.
"""

import os
import sys
import argparse
import json
from typing import Dict, Any, Optional
from loguru import logger

from mcp_tools.screenshot.mcp.mcp_tools import create_mcp_server


def ensure_log_directory() -> None:
    """Ensure log directory exists"""
    os.makedirs("logs", exist_ok=True)


def configure_logging(level: str = "INFO") -> None:
    """
    Configure logging with proper format and level.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    ensure_log_directory()
    
    # Remove default handlers
    logger.remove()
    
    # Add file logger
    logger.add(
        "logs/mcp_server.log",
        rotation="10 MB",
        retention="1 week",
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    )
    
    # Add stderr logger for visible output
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | {message}",
        level=level,
        colorize=True
    )


def get_server_info() -> Dict[str, Any]:
    """
    Get server information.
    
    Returns:
        Dict[str, Any]: Server information
    """
    return {
        "name": "Claude MCP Screenshot Server",
        "version": "1.0.0",
        "description": "A modular screenshot capture and description tool for Claude MCP",
        "author": "Anthropic",
        "github": "https://github.com/anthropics/claude-mcp-configs",
    }


def health_check() -> Dict[str, Any]:
    """
    Perform a health check.
    
    Returns:
        Dict[str, Any]: Health check results
    """
    import platform
    import mss
    
    try:
        # Check if key dependencies are available
        import PIL
        import litellm
        
        # Capture a small test screenshot
        with mss.mss() as sct:
            sct.grab(sct.monitors[1])
            
        return {
            "status": "healthy",
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "mss_version": getattr(mss, "__version__", "unknown"),
            "pil_version": getattr(PIL, "__version__", "unknown"),
            "litellm_version": getattr(litellm, "__version__", "unknown"),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


def main() -> int:
    """
    Main entry point for the MCP server.
    
    Returns:
        int: Exit code
    """
    parser = argparse.ArgumentParser(description="Claude MCP Screenshot Server")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Start server command
    start_parser = subparsers.add_parser("start", help="Start the MCP server")
    start_parser.add_argument("--host", type=str, default="localhost", help="Host to listen on")
    start_parser.add_argument("--port", type=int, default=3000, help="Port to listen on")
    start_parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Check server health")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Display server information")
    
    # Schema command
    schema_parser = subparsers.add_parser("schema", help="Display server schema")
    schema_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    if args.command == "start":
        # Configure logging
        log_level = "DEBUG" if args.debug else "INFO"
        configure_logging(log_level)
        
        # Log startup info
        logger.info(f"Starting MCP server for screenshot tools")
        logger.info(f"Host: {args.host}, Port: {args.port}, Debug: {args.debug}")
        
        try:
            # Create and run the MCP server
            mcp = create_mcp_server(
                name="Screenshot Tools",
                host=args.host,
                port=args.port,
                log_level=log_level
            )
            
            # Run the server
            mcp.run()
            
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
            return 0
        except Exception as e:
            logger.error(f"Server failed to start: {str(e)}", exc_info=True)
            return 1
    
    elif args.command == "health":
        # Run health check
        result = health_check()
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "healthy" else 1
    
    elif args.command == "info":
        # Show server info
        info = get_server_info()
        print(json.dumps(info, indent=2))
        return 0
    
    elif args.command == "schema":
        # Create server to get schema
        mcp = create_mcp_server()
        
        # Get schema
        schema = mcp.get_schema()
        
        if args.json:
            print(json.dumps(schema, indent=2))
        else:
            for function_name, function_info in schema["functions"].items():
                print(f"Function: {function_name}")
                print(f"  Description: {function_info.get('description', 'No description')}")
                print("  Parameters:")
                for param_name, param_info in function_info.get("parameters", {}).get("properties", {}).items():
                    print(f"    {param_name}: {param_info.get('type', 'unknown')} - {param_info.get('description', 'No description')}")
                print()
        
        return 0
    
    return 0


if __name__ == "__main__":
    """
    Direct entry point for the screenshot MCP server.
    This file is designed to be referenced in .mcp.json.
    
    Usage:
      python -m mcp_tools.screenshot.integration.mcp_server start [--host HOST] [--port PORT] [--debug]
      python -m mcp_tools.screenshot.integration.mcp_server health
      python -m mcp_tools.screenshot.integration.mcp_server info
      python -m mcp_tools.screenshot.integration.mcp_server schema [--json]
    """
    sys.exit(main())