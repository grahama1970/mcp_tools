"""Command implementations for the ArangoDB CLI.

This package contains the implementation of various command groups
for interacting with ArangoDB.
"""

from mcp_tools.arangodb.cli.commands.search import app as search_app
from mcp_tools.arangodb.cli.commands.database import app as database_app

__all__ = ["search_app", "database_app"]