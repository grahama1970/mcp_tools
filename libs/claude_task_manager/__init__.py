"""
Claude Task Manager - A system for managing complex task execution with context isolation

This package provides tools for breaking down and executing complex Claude Code tasks 
with proper context isolation (Boomerang-style).
"""

__version__ = "1.0.0"
__author__ = "Anthropic"
__license__ = "MIT"

# Core components
from claude_task_manager.task_manager import TaskManager

# Make type definitions available
from claude_task_manager.types import (
    ProjectInfo, ProjectResult, TaskBreakdownResult, TaskRunResult,
    ProjectRunResult, ProjectListResult, TaskListResult, ErrorResult,
    SchemaCommandResult, LogLevel, OutputFormat
)

# Optional MCP integration
try:
    from claude_task_manager.fast_mcp_wrapper import TaskManagerMCP
except ImportError:
    # FastMCP dependency might be missing
    pass
