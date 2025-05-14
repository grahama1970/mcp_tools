"""Main Typer application for ArangoDB CLI.

This module defines the main Typer application and subcommands for the
ArangoDB CLI, providing a user-friendly interface for database operations.

Links to third-party documentation:
- Typer: https://typer.tiangolo.com/
- Rich: https://rich.readthedocs.io/
"""

import typer
from rich.console import Console
from typing import Optional

app = typer.Typer(
    name="arangodb-cli",
    help="ArangoDB interaction CLI",
    rich_markup_mode="rich"
)
console = Console()

# Import subcommands
from mcp_tools.arangodb.cli.commands.search import app as search_app
from mcp_tools.arangodb.cli.commands.database import app as database_app

# Define other subcommand apps that haven't been refactored yet
memory_app = typer.Typer(name="memory", help="Memory agent commands")

# Add subcommands to main app
app.add_typer(search_app, name="search", help="Search operations")
app.add_typer(database_app, name="db", help="Database operations")
app.add_typer(memory_app, name="memory", help="Memory agent commands")

# Main callback for global options
@app.callback()
def main(
    log_level: str = typer.Option(
        "INFO", "--log-level", "-l",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
):
    """ArangoDB CLI for database operations, search, and graph management."""
    import logging
    logging.basicConfig(level=getattr(logging, log_level.upper()))


if __name__ == "__main__":
    import sys
    try:
        # Simple validation by displaying help
        app(["--help"])
        print("✅ VALIDATION PASSED - CLI application is properly defined")
        sys.exit(0)
    except Exception as e:
        print(f"❌ VALIDATION FAILED: {str(e)}")
        sys.exit(1)