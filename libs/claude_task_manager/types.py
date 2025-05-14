"""
Type definitions for Claude Task Manager.

This module defines TypedDict classes and other type annotations used throughout the package
to provide more specific typing information.
"""

from typing import TypedDict, List, Dict, Any, Optional, Union, Literal


class ProjectInfo(TypedDict):
    """Information about a project."""
    name: str
    created_at: str
    tasks_dir: str
    results_dir: str
    source_file: Optional[str]


class ProjectResult(TypedDict):
    """Result of a project creation operation."""
    success: bool
    project_name: str
    project_dir: str
    source_file: Optional[str]


class TaskBreakdownResult(TypedDict):
    """Result of a task breakdown operation."""
    success: bool
    project_name: str
    project_dir: str
    source_file: str
    created_files: List[str]
    file_count: int
    next_step_command: str


class TaskRunResult(TypedDict):
    """Result of a task execution operation."""
    success: bool
    project_name: str
    task_filename: str
    result_file: str


class ProjectRunResult(TypedDict):
    """Result of a project execution operation."""
    success: bool
    project_name: str
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    tasks: List[Dict[str, Any]]
    results: List[str]


class ProjectListResult(TypedDict):
    """Result of a project listing operation."""
    success: bool
    base_dir: str
    projects: List[str]
    count: int


class TaskListResult(TypedDict):
    """Result of a task listing operation."""
    success: bool
    project_name: str
    tasks: List[str]
    count: int


class ErrorResult(TypedDict):
    """Error result with details."""
    success: bool
    error: str
    context: Dict[str, Any]


class SchemaCommandResult(TypedDict):
    """Schema command return type."""
    name: str
    version: str
    description: str
    commands: Dict[str, Dict[str, Any]]
    options: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]


# Define literal types for common parameters
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
OutputFormat = Literal["json", "yaml"]
MergeMethod = Literal["merge", "squash", "rebase"]


# MCP-specific types
class MCPParameter(TypedDict):
    """MCP parameter description."""
    type: str
    description: str
    required: bool
    default: Optional[Any]
    examples: Optional[List[Any]]


class MCPFunctionReturns(TypedDict):
    """MCP function return type description."""
    type: str
    description: str
    properties: Dict[str, Dict[str, Any]]


class MCPFunctionExample(TypedDict):
    """MCP function example."""
    description: str
    input: Dict[str, Any]
    output: Dict[str, Any]


class MCPFunction(TypedDict):
    """MCP function description."""
    name: str
    description: str
    parameters: Dict[str, MCPParameter]
    returns: Dict[str, Any]
    examples: List[MCPFunctionExample]


class MCPMetadata(TypedDict):
    """MCP metadata."""
    name: str
    version: str
    description: str
    requires_desktop_commander: bool
    requires_claude_code: bool
    mcp_compatible: bool
    documentation_url: Optional[str]
    author: Optional[str]


class MCPSchema(TypedDict):
    """MCP schema output."""
    functions: List[MCPFunction]
    metadata: MCPMetadata
