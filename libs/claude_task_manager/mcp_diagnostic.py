#!/usr/bin/env python
"""
MCP diagnostic tool for Claude Task Manager.

This script provides utilities to diagnose MCP-related issues with the Task Manager.
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
import socket
from typing import Dict, List, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MCP-Diagnostic")


def check_mcp_config() -> Tuple[bool, str]:
    """
    Check if MCP configuration exists and contains task_manager entry.
    
    Returns:
        Tuple with success status and config path
    """
    # Get MCP config path
    mcp_config_path = os.environ.get('MCP_CONFIG_PATH')
    if mcp_config_path:
        config_path = Path(mcp_config_path)
    else:
        config_path = Path.home() / '.mcp.json'
    
    # Check if config exists
    if not config_path.exists():
        logger.error(f"MCP config not found at {config_path}")
        return False, str(config_path)
    
    # Load config
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError:
        logger.error(f"MCP config at {config_path} is not valid JSON")
        return False, str(config_path)
    
    # Check if task_manager is configured
    if 'mcpServers' not in config:
        logger.error("No mcpServers section in MCP config")
        return False, str(config_path)
    
    if 'task_manager' not in config['mcpServers']:
        logger.error("No task_manager entry in MCP config")
        return False, str(config_path)
    
    tm_config = config['mcpServers']['task_manager']
    if 'command' not in tm_config or 'args' not in tm_config:
        logger.error("task_manager config is incomplete")
        return False, str(config_path)
    
    logger.info(f"MCP config found at {config_path} with task_manager entry")
    return True, str(config_path)


def check_claude_code() -> bool:
    """
    Check if Claude Code is available.
    
    Returns:
        True if Claude Code is available, False otherwise
    """
    try:
        result = subprocess.run(
            ["claude", "--version"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        if result.returncode != 0:
            logger.error("Claude Code is installed but returned an error")
            logger.error(f"Error: {result.stderr}")
            return False
        
        logger.info(f"Claude Code found: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        logger.error("Claude Code not found in PATH")
        return False


def check_desktop_commander() -> bool:
    """
    Check if Desktop Commander is installed and running.
    
    Returns:
        True if Desktop Commander is detected, False otherwise
    """
    # This is a simplified check - there's no direct way to check for Desktop Commander
    # Instead, we'll try to check if the directory exists where it would typically be installed
    
    # Common installation paths
    paths = [
        Path.home() / '.config' / 'desktop-commander',
        Path.home() / 'Library' / 'Application Support' / 'desktop-commander'
    ]
    
    # Check if any of the paths exist
    found = any(path.exists() for path in paths)
    
    if found:
        logger.info("Desktop Commander installation detected")
    else:
        logger.error("Desktop Commander installation not detected")
        logger.error("Please install Desktop Commander and ensure it's running")
    
    return found


def check_dependencies() -> bool:
    """
    Check if all required dependencies are installed.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    try:
        import fastmcp
        logger.info("FastMCP is installed")
        fastmcp_ok = True
    except ImportError:
        logger.error("FastMCP is not installed")
        logger.error("Please install it with: pip install fastmcp")
        fastmcp_ok = False
    
    try:
        import typer
        logger.info("Typer is installed")
        typer_ok = True
    except ImportError:
        logger.error("Typer is not installed")
        logger.error("Please install it with: pip install typer")
        typer_ok = False
    
    try:
        import rich
        logger.info("Rich is installed")
        rich_ok = True
    except ImportError:
        logger.error("Rich is not installed")
        logger.error("Please install it with: pip install rich")
        rich_ok = False
    
    return fastmcp_ok and typer_ok and rich_ok


def check_port(port: int) -> bool:
    """
    Check if a port is available.
    
    Args:
        port: Port number to check
        
    Returns:
        True if port is available, False otherwise
    """
    try:
        # Try to bind to the port
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))
            logger.info(f"Port {port} is available")
            return True
    except socket.error:
        logger.error(f"Port {port} is already in use")
        return False


def check_server(host: str, port: int) -> bool:
    """
    Check if the server is running at the specified host and port.
    
    Args:
        host: Host to check
        port: Port to check
        
    Returns:
        True if server is running, False otherwise
    """
    try:
        # Try to connect to the server
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1)
            s.connect((host, port))
            logger.info(f"Server is running at {host}:{port}")
            return True
    except (socket.error, socket.timeout):
        logger.error(f"Server is not running at {host}:{port}")
        return False


def run_diagnostics(run_server_check: bool = False, host: str = "localhost", port: int = 3000) -> Dict[str, Any]:
    """
    Run all diagnostics and return results.
    
    Args:
        run_server_check: Whether to check if server is running
        host: Host to check
        port: Port to check
        
    Returns:
        Dictionary with diagnostic results
    """
    results = {
        "mcp_config": {
            "status": False,
            "path": None
        },
        "claude_code": {
            "status": False
        },
        "desktop_commander": {
            "status": False
        },
        "dependencies": {
            "status": False
        }
    }
    
    # Check MCP config
    mcp_config_ok, config_path = check_mcp_config()
    results["mcp_config"]["status"] = mcp_config_ok
    results["mcp_config"]["path"] = config_path
    
    # Check Claude Code
    claude_code_ok = check_claude_code()
    results["claude_code"]["status"] = claude_code_ok
    
    # Check Desktop Commander
    desktop_commander_ok = check_desktop_commander()
    results["desktop_commander"]["status"] = desktop_commander_ok
    
    # Check dependencies
    dependencies_ok = check_dependencies()
    results["dependencies"]["status"] = dependencies_ok
    
    # Check server if requested
    if run_server_check:
        results["server"] = {
            "status": check_server(host, port),
            "host": host,
            "port": port
        }
    
    # Check port if not checking server
    else:
        results["port"] = {
            "status": check_port(port),
            "port": port
        }
    
    # Overall status
    results["overall"] = {
        "status": all([
            mcp_config_ok, 
            claude_code_ok, 
            desktop_commander_ok, 
            dependencies_ok
        ])
    }
    
    return results


def format_results(results: Dict[str, Any]) -> None:
    """
    Format and print diagnostic results.
    
    Args:
        results: Diagnostic results
    """
    print("\n===== Claude Task Manager MCP Diagnostics =====\n")
    
    # MCP Config
    if results["mcp_config"]["status"]:
        print("✅ MCP Config: Found")
        print(f"   Path: {results['mcp_config']['path']}")
    else:
        print("❌ MCP Config: Not found or invalid")
        print(f"   Expected path: {results['mcp_config']['path']}")
    
    # Claude Code
    if results["claude_code"]["status"]:
        print("✅ Claude Code: Installed and accessible")
    else:
        print("❌ Claude Code: Not installed or not in PATH")
    
    # Desktop Commander
    if results["desktop_commander"]["status"]:
        print("✅ Desktop Commander: Detected")
    else:
        print("❌ Desktop Commander: Not detected")
    
    # Dependencies
    if results["dependencies"]["status"]:
        print("✅ Dependencies: All installed")
    else:
        print("❌ Dependencies: Some missing")
    
    # Server or port
    if "server" in results:
        if results["server"]["status"]:
            print(f"✅ Server: Running at {results['server']['host']}:{results['server']['port']}")
        else:
            print(f"❌ Server: Not running at {results['server']['host']}:{results['server']['port']}")
    elif "port" in results:
        if results["port"]["status"]:
            print(f"✅ Port {results['port']['port']}: Available")
        else:
            print(f"❌ Port {results['port']['port']}: Already in use")
    
    # Overall status
    print("\n===== Overall Status =====")
    if results["overall"]["status"]:
        print("✅ All checks passed. MCP integration should work correctly.")
    else:
        print("❌ Some checks failed. MCP integration may not work correctly.")
    
    # Print troubleshooting tips if there are failures
    if not results["overall"]["status"]:
        print("\n===== Troubleshooting Tips =====")
        if not results["mcp_config"]["status"]:
            print("• Create a .mcp.json file in your home directory with the task_manager entry")
            print("  See README.md for the correct format")
        if not results["claude_code"]["status"]:
            print("• Ensure Claude Code is installed and in your PATH")
            print("  Try running 'claude --version' to verify")
        if not results["desktop_commander"]["status"]:
            print("• Install Desktop Commander using the instructions in README.md")
            print("• Ensure Claude Desktop is running with the hammer icon visible")
        if not results["dependencies"]["status"]:
            print("• Install missing dependencies with pip:")
            print("  pip install fastmcp typer rich")
        if "server" in results and not results["server"]["status"]:
            print(f"• Start the server with: python run_task_manager_server.py --host {results['server']['host']} --port {results['server']['port']}")
        elif "port" in results and not results["port"]["status"]:
            print(f"• Port {results['port']['port']} is already in use. Choose a different port")
            print(f"  python run_task_manager_server.py --port <different-port>")


def main():
    """Command-line interface for MCP diagnostics."""
    parser = argparse.ArgumentParser(description="MCP diagnostic tool for Task Manager")
    parser.add_argument(
        "--check-server",
        action="store_true",
        help="Check if server is running (default: check if port is available)"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to check (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3000,
        help="Port to check (default: 3000)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    args = parser.parse_args()
    
    # Run diagnostics
    results = run_diagnostics(args.check_server, args.host, args.port)
    
    # Output results
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        format_results(results)


if __name__ == "__main__":
    main()
