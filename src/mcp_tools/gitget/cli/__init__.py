"""CLI layer for gitget module.

This module provides the command-line interface for the gitget functionality,
using Typer for command definitions and Rich for formatted console output.

Links to third-party package documentation:
- Typer: https://typer.tiangolo.com/
- Rich: https://rich.readthedocs.io/en/latest/

Exports:
- app: The main Typer application instance
- formatters: Rich-based formatters for console output
- validators: Input validation helpers
- schemas: Pydantic schemas for CLI input and output

Usage example:
    >>> from mcp_tools.gitget.cli import app
    >>> app()  # Run the CLI application
"""

from .app import app
from . import formatters
from . import validators
from . import schemas

__all__ = ["app", "formatters", "validators", "schemas"]