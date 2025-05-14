"""
MCP schema generator for Claude Task Manager.

This module provides utilities to generate MCP-compatible schema
for the Task Manager functions.
"""

import json
import sys
import argparse
from typing import Dict, List, Any, Optional

from claude_task_manager.types import MCPSchema, MCPFunction, MCPParameter


def generate_mcp_schema() -> MCPSchema:
    """
    Generate a complete MCP schema for all Task Manager functions.
    
    Returns:
        MCPSchema with all function definitions
    """
    # Define the schema metadata
    schema: MCPSchema = {
        "functions": [],
        "metadata": {
            "name": "claude-task-manager",
            "version": "1.0.0",
            "description": "Claude Task Manager - Manage complex tasks with context isolation",
            "requires_desktop_commander": True,
            "requires_claude_code": True,
            "mcp_compatible": True,
            "documentation_url": "https://github.com/your-repo/claude_task_manager",
            "author": "Anthropic"
        }
    }
    
    # Add create_project function
    schema["functions"].append({
        "name": "create_project",
        "description": "Create a new project structure",
        "parameters": {
            "project_name": {
                "type": "string",
                "description": "Name of the project",
                "required": True,
                "default": None
            },
            "source_file": {
                "type": "string",
                "description": "Path to source task file (optional)",
                "required": False,
                "default": None
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether the operation was successful"
                },
                "project_name": {
                    "type": "string",
                    "description": "Name of the created project"
                },
                "project_dir": {
                    "type": "string",
                    "description": "Path to the created project directory"
                }
            }
        },
        "examples": [
            {
                "description": "Create a new empty project",
                "input": {"project_name": "my_project"},
                "output": {
                    "success": True,
                    "project_name": "my_project",
                    "project_dir": "/home/user/claude_tasks/my_project"
                }
            },
            {
                "description": "Create a project with a source task file",
                "input": {
                    "project_name": "my_project", 
                    "source_file": "/path/to/task_list.md"
                },
                "output": {
                    "success": True,
                    "project_name": "my_project",
                    "project_dir": "/home/user/claude_tasks/my_project",
                    "source_file": "/path/to/task_list.md"
                }
            }
        ]
    })
    
    # Add break_task function
    schema["functions"].append({
        "name": "break_task",
        "description": "Break down a task into individual task files",
        "parameters": {
            "project_name": {
                "type": "string",
                "description": "Name of the project",
                "required": True,
                "default": None
            },
            "source_file": {
                "type": "string",
                "description": "Path to source task file",
                "required": True,
                "default": None
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether the operation was successful"
                },
                "project_name": {
                    "type": "string",
                    "description": "Name of the project"
                },
                "project_dir": {
                    "type": "string",
                    "description": "Path to the project directory"
                },
                "created_files": {
                    "type": "array",
                    "description": "List of created task files",
                    "items": {
                        "type": "string"
                    }
                },
                "file_count": {
                    "type": "integer",
                    "description": "Number of files created"
                }
            }
        },
        "examples": [
            {
                "description": "Break down a task file",
                "input": {
                    "project_name": "my_project", 
                    "source_file": "/path/to/task_list.md"
                },
                "output": {
                    "success": True,
                    "project_name": "my_project",
                    "project_dir": "/home/user/claude_tasks/my_project",
                    "created_files": [
                        "000_project_overview.md",
                        "001_first_task.md",
                        "002_second_task.md"
                    ],
                    "file_count": 3
                }
            }
        ]
    })
    
    # Add run_task function
    schema["functions"].append({
        "name": "run_task",
        "description": "Run a single task with context isolation",
        "parameters": {
            "project_name": {
                "type": "string",
                "description": "Name of the project",
                "required": True,
                "default": None
            },
            "task_filename": {
                "type": "string",
                "description": "Name of the task file",
                "required": True,
                "default": None
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether the operation was successful"
                },
                "project_name": {
                    "type": "string",
                    "description": "Name of the project"
                },
                "task_filename": {
                    "type": "string",
                    "description": "Name of the task file"
                },
                "result_file": {
                    "type": "string",
                    "description": "Path to the result file"
                },
                "content": {
                    "type": "string",
                    "description": "Content of the result file"
                }
            }
        },
        "examples": [
            {
                "description": "Run a single task",
                "input": {
                    "project_name": "my_project", 
                    "task_filename": "001_first_task.md"
                },
                "output": {
                    "success": True,
                    "project_name": "my_project",
                    "task_filename": "001_first_task.md",
                    "result_file": "/home/user/claude_tasks/my_project/results/001_first_task.result",
                    "content": "Task execution output..."
                }
            }
        ]
    })
    
    # Add run_project function
    schema["functions"].append({
        "name": "run_project",
        "description": "Run all tasks in a project with context isolation",
        "parameters": {
            "project_name": {
                "type": "string",
                "description": "Name of the project",
                "required": True,
                "default": None
            },
            "skip_confirmation": {
                "type": "boolean",
                "description": "Skip confirmation prompt and run all tasks",
                "required": False,
                "default": True
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether the operation was successful"
                },
                "project_name": {
                    "type": "string",
                    "description": "Name of the project"
                },
                "total_tasks": {
                    "type": "integer",
                    "description": "Total number of tasks"
                },
                "completed_tasks": {
                    "type": "integer",
                    "description": "Number of tasks completed"
                },
                "pending_tasks": {
                    "type": "integer",
                    "description": "Number of tasks pending"
                },
                "results": {
                    "type": "array",
                    "description": "Task execution results",
                    "items": {
                        "type": "object"
                    }
                }
            }
        },
        "examples": [
            {
                "description": "Run all tasks in a project",
                "input": {"project_name": "my_project"},
                "output": {
                    "success": True,
                    "project_name": "my_project",
                    "total_tasks": 3,
                    "completed_tasks": 3,
                    "pending_tasks": 0,
                    "results": [
                        {
                            "task_file": "001_first_task.md",
                            "result_file": "/path/to/result1.result",
                            "status": "success",
                            "content": "Task execution output..."
                        },
                        {
                            "task_file": "002_second_task.md",
                            "result_file": "/path/to/result2.result",
                            "status": "success",
                            "content": "Task execution output..."
                        }
                    ]
                }
            }
        ]
    })
    
    # Add list_projects function
    schema["functions"].append({
        "name": "list_projects",
        "description": "List all projects",
        "parameters": {},
        "returns": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether the operation was successful"
                },
                "base_dir": {
                    "type": "string",
                    "description": "Base directory for task management"
                },
                "projects": {
                    "type": "array",
                    "description": "List of project names",
                    "items": {
                        "type": "string"
                    }
                },
                "count": {
                    "type": "integer",
                    "description": "Number of projects found"
                }
            }
        },
        "examples": [
            {
                "description": "List all projects",
                "input": {},
                "output": {
                    "success": True,
                    "base_dir": "/home/user/claude_tasks",
                    "projects": ["project1", "project2", "project3"],
                    "count": 3
                }
            }
        ]
    })
    
    # Add list_tasks function
    schema["functions"].append({
        "name": "list_tasks",
        "description": "List all tasks in a project",
        "parameters": {
            "project_name": {
                "type": "string",
                "description": "Name of the project",
                "required": True,
                "default": None
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether the operation was successful"
                },
                "project_name": {
                    "type": "string",
                    "description": "Name of the project"
                },
                "tasks": {
                    "type": "array",
                    "description": "List of task filenames",
                    "items": {
                        "type": "string"
                    }
                },
                "count": {
                    "type": "integer",
                    "description": "Number of tasks found"
                }
            }
        },
        "examples": [
            {
                "description": "List all tasks in a project",
                "input": {"project_name": "my_project"},
                "output": {
                    "success": True,
                    "project_name": "my_project",
                    "tasks": [
                        "000_project_overview.md",
                        "001_first_task.md",
                        "002_second_task.md"
                    ],
                    "count": 3
                }
            }
        ]
    })
    
    # Add check_status function
    schema["functions"].append({
        "name": "check_status",
        "description": "Check the status of a project",
        "parameters": {
            "project_name": {
                "type": "string",
                "description": "Name of the project",
                "required": True,
                "default": None
            }
        },
        "returns": {
            "type": "object",
            "properties": {
                "success": {
                    "type": "boolean",
                    "description": "Whether the operation was successful"
                },
                "project_name": {
                    "type": "string",
                    "description": "Name of the project"
                },
                "total_tasks": {
                    "type": "integer",
                    "description": "Total number of tasks"
                },
                "completed_tasks": {
                    "type": "integer",
                    "description": "Number of tasks completed"
                },
                "pending_tasks": {
                    "type": "integer",
                    "description": "Number of tasks pending"
                },
                "completion_percentage": {
                    "type": "number",
                    "description": "Percentage of completion"
                },
                "tasks": {
                    "type": "array",
                    "description": "Status of each task",
                    "items": {
                        "type": "object"
                    }
                }
            }
        },
        "examples": [
            {
                "description": "Check the status of a project",
                "input": {"project_name": "my_project"},
                "output": {
                    "success": True,
                    "project_name": "my_project",
                    "total_tasks": 3,
                    "completed_tasks": 1,
                    "pending_tasks": 2,
                    "completion_percentage": 33.33,
                    "tasks": [
                        {
                            "task_file": "000_project_overview.md",
                            "status": "completed",
                            "result_file": "/path/to/result.result"
                        },
                        {
                            "task_file": "001_first_task.md",
                            "status": "pending",
                            "result_file": None
                        }
                    ]
                }
            }
        ]
    })
    
    return schema


def save_schema_to_file(schema: MCPSchema, output_file: str) -> None:
    """
    Save the schema to a file in JSON format.
    
    Args:
        schema: The MCP schema to save
        output_file: Path to the output file
    """
    with open(output_file, 'w') as f:
        json.dump(schema, f, indent=2)
    print(f"Schema saved to {output_file}")


def main():
    """Command-line interface for schema generation."""
    parser = argparse.ArgumentParser(description="Generate MCP schema for Task Manager")
    parser.add_argument(
        "--output", "-o",
        default="task_manager_mcp_schema.json",
        help="Output file path (default: task_manager_mcp_schema.json)"
    )
    
    args = parser.parse_args()
    
    # Generate and save the schema
    schema = generate_mcp_schema()
    save_schema_to_file(schema, args.output)


if __name__ == "__main__":
    main()
