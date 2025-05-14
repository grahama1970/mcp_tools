#!/usr/bin/env python
"""
Server entry point for Claude Task Manager.

This script initializes and runs the FastMCP server for the Claude Task Manager,
allowing it to be used as an MCP tool.
"""

import os
import sys
import json
import logging
import signal
import socket
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add project root to path if needed
sys.path.append(str(Path(__file__).parent))

# Import the FastMCP wrapper
from claude_task_manager.fast_mcp_wrapper import TaskManagerMCP

# Create a Typer app
app = typer.Typer(
    name="task-manager-server",
    help="Claude Task Manager MCP Server",
    add_completion=False,
)

# Create a rich console for pretty output
console = Console()

# Global variable for the MCP instance
mcp_instance = None


def signal_handler(sig, frame):
    """Handle termination signals."""
    logger = logging.getLogger("TaskManagerServer")
    logger.info(f"Received signal {sig}, shutting down gracefully...")
    sys.exit(0)


def check_port_availability(host: str, port: int) -> bool:
    """
    Check if a port is available.
    
    Args:
        host: Host to bind to
        port: Port to check
        
    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except socket.error:
        return False


def check_claude_code() -> bool:
    """
    Check if Claude Code is available.
    
    Returns:
        True if Claude Code is available, False otherwise
    """
    try:
        import subprocess
        result = subprocess.run(
            ["claude", "--version"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


@app.command()
def start(
    host: str = typer.Option(
        "localhost", 
        "--host", 
        "-h", 
        help="Host to listen on"
    ),
    port: int = typer.Option(
        3000, 
        "--port", 
        "-p", 
        help="Port to listen on"
    ),
    base_dir: str = typer.Option(
        None, 
        "--base-dir", 
        "-b", 
        help="Base directory for task management"
    ),
    log_level: str = typer.Option(
        "INFO", 
        "--log-level", 
        "-l", 
        help="Set the logging level",
        case_sensitive=False,
        show_default=True,
    ),
    debug: bool = typer.Option(
        False, 
        "--debug", 
        "-d", 
        help="Enable debug mode"
    ),
    test: bool = typer.Option(
        False, 
        "--test", 
        "-t", 
        help="Run in test mode (doesn't require Claude Code)"
    ),
    skip_checks: bool = typer.Option(
        False, 
        "--skip-checks", 
        "-s", 
        help="Skip dependency checks"
    ),
):
    """Start the Claude Task Manager MCP server."""
    global mcp_instance
    
    # Use default base directory if not specified
    if base_dir is None:
        base_dir = os.environ.get('CLAUDE_TASKS_DIR', str(Path.home() / 'claude_tasks'))
    
    # Override log level if debug is enabled
    if debug:
        log_level = "DEBUG"
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger("TaskManagerServer")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Log startup information
    logger.info("Starting Claude Task Manager MCP Server")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Server address: {host}:{port}")
    
    # Run checks if not skipped
    if not skip_checks:
        # Check port availability
        if not check_port_availability(host, port):
            console.print(f"[red]Error:[/red] Port {port} is already in use")
            logger.error(f"Port {port} is already in use")
            raise typer.Exit(code=1)
        
        # Check Claude Code availability
        if not test and not check_claude_code():
            console.print("[red]Error:[/red] Claude Code is not available")
            console.print("To start the server without Claude Code (for testing), use --test")
            logger.error("Claude Code is not available")
            raise typer.Exit(code=1)
    
    try:
        # Initialize TaskManagerMCP
        mcp_instance = TaskManagerMCP(base_dir=base_dir, log_level=log_level)
        
        # Display server info
        panel = Panel(
            f"[green]Server running at:[/green] {host}:{port}\n"
            f"[green]Base directory:[/green] {base_dir}\n"
            f"[green]Log level:[/green] {log_level}\n\n"
            "Press Ctrl+C to stop the server",
            title="Claude Task Manager MCP Server",
            border_style="green",
        )
        console.print(panel)
        
        # Start the server
        mcp_instance.run(host=host, port=port)
    except Exception as e:
        console.print(f"[red]Error starting server:[/red] {e}", style="bold red")
        logger.error(f"Error starting server: {e}")
        raise typer.Exit(code=1)


@app.command()
def diagnostic(
    host: str = typer.Option(
        "localhost", 
        "--host", 
        "-h", 
        help="Host to check"
    ),
    port: int = typer.Option(
        3000, 
        "--port", 
        "-p", 
        help="Port to check"
    ),
    check_server: bool = typer.Option(
        False, 
        "--check-server", 
        "-c", 
        help="Check if server is running"
    ),
    json_output: bool = typer.Option(
        False, 
        "--json", 
        "-j", 
        help="Output as JSON"
    ),
):
    """Run diagnostics for the Claude Task Manager MCP server."""
    # Import here to avoid circular imports
    try:
        from claude_task_manager.mcp_diagnostic import run_diagnostics, format_results
        results = run_diagnostics(check_server, host, port)
        
        if json_output:
            typer.echo(json.dumps(results, indent=2))
        else:
            format_results(results)
    except ImportError:
        console.print("[red]Error:[/red] Could not import mcp_diagnostic module")
        console.print("Make sure claude_task_manager is installed")
        raise typer.Exit(code=1)


@app.command()
def schema(
    output: Optional[str] = typer.Option(
        None, 
        "--output", 
        "-o", 
        help="Output file path"
    ),
    format: str = typer.Option(
        "json", 
        "--format", 
        "-f", 
        help="Output format (json or yaml)"
    ),
):
    """Output the MCP schema."""
    try:
        from claude_task_manager.mcp_schema import generate_mcp_schema
        schema = generate_mcp_schema()
        
        if format.lower() == "yaml":
            try:
                import yaml
                schema_str = yaml.dump(schema, sort_keys=False, default_flow_style=False)
            except ImportError:
                console.print("[yellow]PyYAML not installed. Falling back to JSON format.[/yellow]")
                schema_str = json.dumps(schema, indent=2)
        else:
            schema_str = json.dumps(schema, indent=2)
        
        if output:
            with open(output, 'w') as f:
                f.write(schema_str)
            console.print(f"[green]Schema saved to:[/green] {output}")
        else:
            typer.echo(schema_str)
    except ImportError:
        console.print("[red]Error:[/red] Could not import mcp_schema module")
        console.print("Make sure claude_task_manager is installed")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
