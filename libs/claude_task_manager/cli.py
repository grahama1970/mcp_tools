"""
Command-line interface for Claude Task Manager.

This module provides a command-line interface to interact with the TaskManager class,
using Typer for a more modern command-line experience.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal, Union, cast

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

from claude_task_manager.task_manager import TaskManager
from claude_task_manager.types import (
    ProjectInfo, ProjectResult, TaskBreakdownResult, TaskRunResult,
    ProjectRunResult, ProjectListResult, TaskListResult, ErrorResult,
    SchemaCommandResult, LogLevel, OutputFormat, MCPSchema, MCPFunction
)

# Initialize Typer app with metadata
app = typer.Typer(
    name="claude-tasks", 
    help="Claude Task Manager - Manage complex tasks with context isolation",
    add_completion=True,
)

# Create a rich console for pretty output
console = Console()

# Set up environment variables
DEFAULT_BASE_DIR = os.environ.get('CLAUDE_TASKS_DIR', str(Path.home() / 'claude_tasks'))


def get_manager(base_dir: str, log_level: str) -> TaskManager:
    """
    Create and return a TaskManager instance.
    
    Args:
        base_dir: Base directory for task management
        log_level: Logging level to use
        
    Returns:
        TaskManager instance
    """
    level = getattr(logging, log_level.upper())
    return TaskManager(base_dir, log_level=level)


@app.callback()
def common(
    ctx: typer.Context,
    base_dir: str = typer.Option(
        DEFAULT_BASE_DIR,
        "--base-dir",
        "-b",
        help="Base directory for task management",
        envvar="CLAUDE_TASKS_DIR",
    ),
    log_level: LogLevel = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Set the logging level",
        case_sensitive=False,
        show_default=True,
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output results as JSON (for machine consumption)",
    ),
):
    """
    Common parameters for all commands.
    """
    # Store parameters in context for use in commands
    ctx.obj = {
        "base_dir": base_dir,
        "log_level": log_level,
        "json_output": json_output,
    }


@app.command(name="create-project")
def create_project(
    project_name: str = typer.Argument(..., help="Name of the project"),
    source_file: Optional[Path] = typer.Argument(None, help="Path to source task file (optional)"),
    ctx: typer.Context = typer.Context,
):
    """
    Create a new project structure.
    """
    try:
        manager = get_manager(ctx.obj["base_dir"], ctx.obj["log_level"])
        
        # Convert source_file to string if provided
        source_file_str = str(source_file) if source_file else None
        
        # Create project
        project_dir = manager.create_project(project_name, source_file_str)
        
        # Prepare result data
        result: ProjectResult = {
            "success": True,
            "project_name": project_name,
            "project_dir": str(project_dir),
            "source_file": source_file_str,
        }
        
        # Output in JSON or human-readable format
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(result, indent=2))
        else:
            panel = Panel(
                f"[green]Project created at:[/green] {project_dir}",
                title="Project Creation Successful",
                border_style="green",
            )
            console.print(panel)
        
    except ValueError as e:
        # Handle validation errors
        error_data: ErrorResult = {
            "success": False,
            "error": str(e),
            "context": {"project_name": project_name}
        }
        
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(error_data, indent=2))
        else:
            console.print(f"[red]Validation error:[/red] {e}", style="bold red")
        
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        # Handle file not found errors
        error_data: ErrorResult = {
            "success": False,
            "error": str(e),
            "context": {"project_name": project_name, "source_file": source_file_str}
        }
        
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(error_data, indent=2))
        else:
            console.print(f"[red]File not found:[/red] {e}", style="bold red")
        
        raise typer.Exit(code=1)
    except Exception as e:
        # Handle unexpected errors
        error_data: ErrorResult = {
            "success": False,
            "error": str(e),
            "context": {"project_name": project_name}
        }
        
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(error_data, indent=2))
        else:
            console.print(f"[red]Error creating project:[/red] {e}", style="bold red")
        
        raise typer.Exit(code=1)


@app.command(name="break-task")
def break_task(
    project_name: str = typer.Argument(..., help="Name of the project"),
    source_file: str = typer.Argument(..., help="Path to source task file"),
    ctx: typer.Context = typer.Context,
):
    """
    Break down a task into individual task files.
    """
    try:
        manager = get_manager(ctx.obj["base_dir"], ctx.obj["log_level"])
        
        # Don't show spinner in JSON mode
        if ctx.obj.get("json_output", False):
            project_dir, created_files = manager.break_down_task(project_name, source_file)
        else:
            with console.status("[bold blue]Breaking down task file...[/]"):
                project_dir, created_files = manager.break_down_task(project_name, source_file)
        
        # Prepare result data
        result: TaskBreakdownResult = {
            "success": True,
            "project_name": project_name,
            "project_dir": str(project_dir),
            "source_file": source_file,
            "created_files": created_files,
            "file_count": len(created_files),
            "next_step_command": f"claude-tasks run-project {project_name}"
        }
        
        # Output in JSON or human-readable format
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(result, indent=2))
        else:
            # Create a pretty table with the results
            table = Table(title=f"Task Breakdown Results for {project_name}")
            table.add_column("Task File", style="cyan")
            table.add_column("Status", style="green")
            
            for filename in created_files:
                table.add_row(filename, "✅ Created")
            
            console.print(table)
            
            # Print next steps
            console.print("\n[bold]Next Steps:[/bold]")
            console.print(f"  Run the tasks with: [cyan]claude-tasks run-project {project_name}[/cyan]")
        
    except Exception as e:
        error_data: ErrorResult = {
            "success": False,
            "error": str(e),
            "context": {
                "project_name": project_name,
                "source_file": source_file
            }
        }
        
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(error_data, indent=2))
        else:
            console.print(f"[red]Error breaking down task:[/red] {e}", style="bold red")
        
        raise typer.Exit(code=1)


@app.command(name="run-task")
def run_task(
    project_name: str = typer.Argument(..., help="Name of the project"),
    task_filename: str = typer.Argument(..., help="Name of the task file"),
    ctx: typer.Context = typer.Context,
):
    """
    Run a single task with context isolation.
    """
    try:
        manager = get_manager(ctx.obj["base_dir"], ctx.obj["log_level"])
        
        # Don't show spinner in JSON mode
        if ctx.obj.get("json_output", False):
            result_file = manager.run_task(project_name, task_filename)
        else:
            with console.status(f"[bold blue]Running task: {task_filename}[/]"):
                result_file = manager.run_task(project_name, task_filename)
        
        # Prepare result data
        result: TaskRunResult = {
            "success": True,
            "project_name": project_name,
            "task_filename": task_filename,
            "result_file": result_file,
        }
        
        # Output in JSON or human-readable format
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(result, indent=2))
        else:
            panel = Panel(
                f"Task execution complete.\n[green]Result saved to:[/green] {result_file}",
                title=f"Task Execution Successful: {task_filename}",
                border_style="green",
            )
            console.print(panel)
        
    except Exception as e:
        error_data: ErrorResult = {
            "success": False,
            "error": str(e),
            "context": {
                "project_name": project_name,
                "task_filename": task_filename
            }
        }
        
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(error_data, indent=2))
        else:
            console.print(f"[red]Error running task:[/red] {e}", style="bold red")
        
        raise typer.Exit(code=1)


@app.command(name="run-project")
def run_project(
    project_name: str = typer.Argument(..., help="Name of the project"),
    skip_confirmation: bool = typer.Option(
        False, 
        "--yes", 
        "-y", 
        help="Skip confirmation prompt and run all tasks"
    ),
    ctx: typer.Context = typer.Context,
):
    """
    Run all tasks in a project with context isolation.
    """
    try:
        manager = get_manager(ctx.obj["base_dir"], ctx.obj["log_level"])
        json_output = ctx.obj.get("json_output", False)
        
        # Get task list first to show plan
        project_dir = Path(ctx.obj["base_dir"]) / project_name
        tasks_dir = project_dir / "tasks"
        sequence_file = project_dir / "task_sequence.txt"
        
        task_files = []
        if sequence_file.exists():
            with open(sequence_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        task_files.append(line)
        else:
            task_files = sorted([f.name for f in tasks_dir.glob('*.md')])
        
        # Check if we have any tasks
        if not task_files:
            error_data: ErrorResult = {
                "success": False,
                "error": "No tasks found in project",
                "context": {"project_name": project_name}
            }
            
            if json_output:
                typer.echo(json.dumps(error_data, indent=2))
            else:
                console.print(f"[red]Error:[/red] No tasks found in project {project_name}", style="bold red")
            
            raise typer.Exit(code=1)
        
        # Display execution plan
        if not json_output:
            table = Table(title=f"Task Execution Plan for {project_name}")
            table.add_column("#", style="cyan", justify="right")
            table.add_column("Task File", style="cyan")
            table.add_column("Status", style="yellow")
            
            for i, task in enumerate(task_files, 1):
                table.add_row(str(i), task, "⏳ Pending")
            
            console.print(table)
        
        # Confirm execution if needed
        if not skip_confirmation and not json_output:
            if not typer.confirm("\nExecute all tasks?", default=True):
                console.print("[yellow]Task execution aborted.[/yellow]")
                raise typer.Exit(code=0)
        
        # Execute tasks
        results = []
        failed_tasks = []
        task_results = []
        
        for i, task_file in enumerate(task_files, 1):
            try:
                if json_output:
                    result = manager.run_task(project_name, task_file)
                else:
                    with console.status(f"[bold blue]Running task {i}/{len(task_files)}: {task_file}[/]"):
                        result = manager.run_task(project_name, task_file)
                
                results.append(result)
                task_results.append({
                    "task_file": task_file,
                    "result_file": result,
                    "status": "success"
                })
                
                if not json_output:
                    console.print(f"[green]✓[/green] Task {i}/{len(task_files)} completed: {task_file}")
            
            except Exception as e:
                failed_tasks.append(task_file)
                task_results.append({
                    "task_file": task_file,
                    "error": str(e),
                    "status": "failed"
                })
                
                if not json_output:
                    console.print(f"[red]✗[/red] Task {i}/{len(task_files)} failed: {task_file} - {e}")
        
        # Prepare result data
        result_data: ProjectRunResult = {
            "success": len(failed_tasks) == 0,
            "project_name": project_name,
            "total_tasks": len(task_files),
            "completed_tasks": len(results),
            "failed_tasks": len(failed_tasks),
            "tasks": task_results,
            "results": results
        }
        
        # Output in JSON or human-readable format
        if json_output:
            typer.echo(json.dumps(result_data, indent=2))
        else:
            # Print summary
            console.print("\n[bold]Execution Summary:[/bold]")
            success_count = len(results)
            fail_count = len(failed_tasks)
            
            console.print(f"  [green]✓ {success_count} tasks completed successfully[/green]")
            if fail_count > 0:
                console.print(f"  [red]✗ {fail_count} tasks failed[/red]")
            
            console.print("\n[bold]Results saved to:[/bold]")
            for result in results:
                console.print(f"  {result}")
        
    except typer.Abort:
        if not ctx.obj.get("json_output", False):
            console.print("[yellow]Task execution aborted.[/yellow]")
        else:
            typer.echo(json.dumps({
                "success": False,
                "error": "Task execution aborted by user",
                "context": {"project_name": project_name}
            }, indent=2))
        
        raise typer.Exit(code=0)
    
    except Exception as e:
        error_data: ErrorResult = {
            "success": False,
            "error": str(e),
            "context": {"project_name": project_name}
        }
        
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(error_data, indent=2))
        else:
            console.print(f"[red]Error running project:[/red] {e}", style="bold red")
        
        raise typer.Exit(code=1)


@app.command(name="list-projects")
def list_projects(
    ctx: typer.Context = typer.Context,
):
    """
    List all projects.
    """
    try:
        manager = get_manager(ctx.obj["base_dir"], ctx.obj["log_level"])
        projects = manager.list_projects()
        
        # Prepare result data
        result: ProjectListResult = {
            "success": True,
            "base_dir": ctx.obj["base_dir"],
            "projects": projects,
            "count": len(projects)
        }
        
        # Output in JSON or human-readable format
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(result, indent=2))
        else:
            if projects:
                table = Table(title="Available Projects")
                table.add_column("Project Name", style="cyan")
                
                for project in projects:
                    table.add_row(project)
                
                console.print(table)
            else:
                console.print("[yellow]No projects found[/yellow]")
        
    except Exception as e:
        error_data: ErrorResult = {
            "success": False,
            "error": str(e),
            "context": {"base_dir": ctx.obj["base_dir"]}
        }
        
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(error_data, indent=2))
        else:
            console.print(f"[red]Error listing projects:[/red] {e}", style="bold red")
        
        raise typer.Exit(code=1)


@app.command(name="list-tasks")
def list_tasks(
    project_name: str = typer.Argument(..., help="Name of the project"),
    ctx: typer.Context = typer.Context,
):
    """
    List all tasks in a project.
    """
    try:
        manager = get_manager(ctx.obj["base_dir"], ctx.obj["log_level"])
        tasks = manager.list_tasks(project_name)
        
        # Prepare result data
        result: TaskListResult = {
            "success": True,
            "project_name": project_name,
            "tasks": tasks,
            "count": len(tasks)
        }
        
        # Output in JSON or human-readable format
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(result, indent=2))
        else:
            if tasks:
                table = Table(title=f"Tasks in Project: {project_name}")
                table.add_column("#", style="cyan", justify="right")
                table.add_column("Task File", style="cyan")
                
                for i, task in enumerate(tasks, 1):
                    table.add_row(str(i), task)
                
                console.print(table)
            else:
                console.print(f"[yellow]No tasks found in project {project_name}[/yellow]")
        
    except Exception as e:
        error_data: ErrorResult = {
            "success": False,
            "error": str(e),
            "context": {"project_name": project_name}
        }
        
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(error_data, indent=2))
        else:
            console.print(f"[red]Error listing tasks:[/red] {e}", style="bold red")
        
        raise typer.Exit(code=1)


@app.command(name="check-status")
def check_status(
    project_name: str = typer.Argument(..., help="Name of the project"),
    ctx: typer.Context = typer.Context,
):
    """
    Check the status of a project.
    """
    try:
        manager = get_manager(ctx.obj["base_dir"], ctx.obj["log_level"])
        status = manager.check_project_status(project_name)
        
        # Check if project exists
        if not status["exists"]:
            error_data: ErrorResult = {
                "success": False,
                "error": f"Project {project_name} not found",
                "context": {"project_name": project_name}
            }
            
            if ctx.obj.get("json_output", False):
                typer.echo(json.dumps(error_data, indent=2))
            else:
                console.print(f"[red]Error:[/red] Project {project_name} not found", style="bold red")
            
            raise typer.Exit(code=1)
        
        # Prepare result data
        result = {
            "success": True,
            "project_name": project_name,
            "total_tasks": status["total_tasks"],
            "completed_tasks": status["completed_tasks"],
            "pending_tasks": status["pending_tasks"],
            "completion_percentage": status["completion_percentage"]
        }
        
        # Output in JSON or human-readable format
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(result, indent=2))
        else:
            # Create progress display
            table = Table(title=f"Project Status: {project_name}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            
            table.add_row("Total Tasks", str(status["total_tasks"]))
            table.add_row("Completed", str(status["completed_tasks"]))
            table.add_row("Pending", str(status["pending_tasks"]))
            table.add_row("Completion", f"{status['completion_percentage']:.1f}%")
            
            console.print(table)
            
            # Progress bar
            completion = status["completion_percentage"] / 100
            bar_width = 50
            filled = int(completion * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            console.print(f"\n[bold]Progress:[/bold] [{get_progress_color(completion)}]{bar}[/] {completion:.1%}")
        
    except Exception as e:
        error_data: ErrorResult = {
            "success": False,
            "error": str(e),
            "context": {"project_name": project_name}
        }
        
        if ctx.obj.get("json_output", False):
            typer.echo(json.dumps(error_data, indent=2))
        else:
            console.print(f"[red]Error checking project status:[/red] {e}", style="bold red")
        
        raise typer.Exit(code=1)


def get_progress_color(completion: float) -> str:
    """Get color for progress bar based on completion percentage."""
    if completion < 0.3:
        return "red"
    elif completion < 0.7:
        return "yellow"
    else:
        return "green"


@app.command(name="schema")
def schema(
    ctx: typer.Context = typer.Context,
    format: OutputFormat = typer.Option(
        "json", 
        "--format", 
        "-f", 
        help="Output format (json or yaml)",
        case_sensitive=False,
    ),
    mcp_format: bool = typer.Option(
        False,
        "--mcp",
        "-m",
        help="Output schema in MCP-compatible format",
    ),
):
    """
    Output the CLI schema in a machine-readable format.
    
    This is useful for automated tooling, documentation generation, and MCP integration.
    """
    try:
        # Create standard CLI schema
        schema_dict: SchemaCommandResult = {
            "name": "claude-task-manager",
            "version": "1.0.0",
            "description": "Claude Task Manager - Manage complex tasks with context isolation",
            "commands": {
                "create-project": {
                    "description": "Create a new project structure",
                    "parameters": {
                        "project_name": {
                            "type": "string",
                            "description": "Name of the project",
                            "required": True
                        },
                        "source_file": {
                            "type": "string",
                            "description": "Path to source task file (optional)",
                            "required": False
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
                            },
                            "source_file": {
                                "type": "string",
                                "description": "Path to the source task file (if provided)"
                            }
                        }
                    },
                    "examples": [
                        {
                            "description": "Create a new empty project",
                            "command": "claude-tasks create-project my_project"
                        },
                        {
                            "description": "Create a project with a source task file",
                            "command": "claude-tasks create-project my_project /path/to/task_list.md"
                        }
                    ]
                },
                "break-task": {
                    "description": "Break down a task into individual task files",
                    "parameters": {
                        "project_name": {
                            "type": "string",
                            "description": "Name of the project",
                            "required": True
                        },
                        "source_file": {
                            "type": "string",
                            "description": "Path to source task file",
                            "required": True
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
                            },
                            "next_step_command": {
                                "type": "string",
                                "description": "Suggested next command to run"
                            }
                        }
                    },
                    "examples": [
                        {
                            "description": "Break down a task file",
                            "command": "claude-tasks break-task my_project /path/to/task_list.md"
                        }
                    ]
                },
                "run-task": {
                    "description": "Run a single task with context isolation",
                    "parameters": {
                        "project_name": {
                            "type": "string",
                            "description": "Name of the project",
                            "required": True
                        },
                        "task_filename": {
                            "type": "string",
                            "description": "Name of the task file",
                            "required": True
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
                            }
                        }
                    },
                    "examples": [
                        {
                            "description": "Run a single task",
                            "command": "claude-tasks run-task my_project 001_first_task.md"
                        }
                    ]
                },
                "run-project": {
                    "description": "Run all tasks in a project with context isolation",
                    "parameters": {
                        "project_name": {
                            "type": "string",
                            "description": "Name of the project",
                            "required": True
                        },
                        "skip_confirmation": {
                            "type": "boolean",
                            "description": "Skip confirmation prompt and run all tasks",
                            "required": False,
                            "default": False
                        }
                    },
                    "returns": {
                        "type": "object",
                        "properties": {
                            "success": {
                                "type": "boolean",
                                "description": "Whether all tasks completed successfully"
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
                                "description": "Number of tasks completed successfully"
                            },
                            "failed_tasks": {
                                "type": "integer",
                                "description": "Number of tasks that failed"
                            },
                            "tasks": {
                                "type": "array",
                                "description": "Details of each task execution",
                                "items": {
                                    "type": "object"
                                }
                            },
                            "results": {
                                "type": "array",
                                "description": "List of result file paths",
                                "items": {
                                    "type": "string"
                                }
                            }
                        }
                    },
                    "examples": [
                        {
                            "description": "Run all tasks in a project",
                            "command": "claude-tasks run-project my_project"
                        },
                        {
                            "description": "Run all tasks without confirmation",
                            "command": "claude-tasks run-project my_project --yes"
                        }
                    ]
                },
                "list-projects": {
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
                            "command": "claude-tasks list-projects"
                        }
                    ]
                },
                "list-tasks": {
                    "description": "List all tasks in a project",
                    "parameters": {
                        "project_name": {
                            "type": "string",
                            "description": "Name of the project",
                            "required": True
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
                            "command": "claude-tasks list-tasks my_project"
                        }
                    ]
                },
                "check-status": {
                    "description": "Check the status of a project",
                    "parameters": {
                        "project_name": {
                            "type": "string",
                            "description": "Name of the project",
                            "required": True
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
                                "description": "Percentage of tasks completed"
                            }
                        }
                    },
                    "examples": [
                        {
                            "description": "Check the status of a project",
                            "command": "claude-tasks check-status my_project"
                        }
                    ]
                },
                "schema": {
                    "description": "Output the CLI schema in a machine-readable format",
                    "parameters": {
                        "format": {
                            "type": "string",
                            "description": "Output format (json or yaml)",
                            "default": "json",
                            "enum": ["json", "yaml"]
                        },
                        "mcp_format": {
                            "type": "boolean",
                            "description": "Output schema in MCP-compatible format",
                            "default": False
                        }
                    },
                    "returns": {
                        "type": "object",
                        "description": "Schema definition"
                    },
                    "examples": [
                        {
                            "description": "Output schema in JSON format",
                            "command": "claude-tasks schema"
                        },
                        {
                            "description": "Output schema in YAML format",
                            "command": "claude-tasks schema --format yaml"
                        },
                        {
                            "description": "Output schema in MCP-compatible format",
                            "command": "claude-tasks schema --mcp"
                        }
                    ]
                }
            },
            "options": {
                "base_dir": {
                    "type": "string",
                    "description": "Base directory for task management",
                    "default": DEFAULT_BASE_DIR,
                    "env_var": "CLAUDE_TASKS_DIR"
                },
                "log_level": {
                    "type": "string",
                    "description": "Set the logging level",
                    "default": "INFO",
                    "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                },
                "json_output": {
                    "type": "boolean",
                    "description": "Output results as JSON (for machine consumption)",
                    "default": False
                }
            },
            "metadata": {
                "requires_desktop_commander": True,
                "requires_claude_code": True,
                "mcp_compatible": True,
                "fastmcp_ready": True,
                "version": "1.0.0",
                "author": "Anthropic",
                "license": "MIT"
            }
        }
        
        # If MCP format is requested, convert to MCP-compatible schema
        if mcp_format:
            mcp_schema: MCPSchema = {
                "functions": [],
                "metadata": {
                    "name": "claude-task-manager",
                    "version": "1.0.0",
                    "description": "Claude Task Manager - Manage complex tasks with context isolation",
                    "requires_desktop_commander": True,
                    "requires_claude_code": True,
                    "mcp_compatible": True
                }
            }
            
            # Convert each command to an MCP function
            for cmd_name, cmd_data in schema_dict["commands"].items():
                # Skip schema command in MCP format
                if cmd_name == "schema":
                    continue
                
                # Create MCP function
                mcp_function: MCPFunction = {
                    "name": cmd_name.replace("-", "_"),
                    "description": cmd_data["description"],
                    "parameters": {},
                    "returns": cmd_data["returns"],
                    "examples": cmd_data["examples"]
                }
                
                # Convert parameters
                for param_name, param_data in cmd_data["parameters"].items():
                    mcp_function["parameters"][param_name] = {
                        "type": param_data["type"],
                        "description": param_data["description"],
                        "required": param_data.get("required", False),
                        "default": param_data.get("default", None)
                    }
                
                # Add to functions list
                mcp_schema["functions"].append(mcp_function)
            
            # Output MCP schema
            if format.lower() == "yaml":
                try:
                    import yaml
                    yaml_output = yaml.dump(mcp_schema, sort_keys=False, default_flow_style=False)
                    typer.echo(yaml_output)
                except ImportError:
                    console.print("[yellow]PyYAML not installed. Falling back to JSON format.[/yellow]")
                    typer.echo(json.dumps(mcp_schema, indent=2))
            else:
                typer.echo(json.dumps(mcp_schema, indent=2))
                
        else:
            # Output standard schema
            if format.lower() == "yaml":
                try:
                    import yaml
                    yaml_output = yaml.dump(schema_dict, sort_keys=False, default_flow_style=False)
                    typer.echo(yaml_output)
                except ImportError:
                    console.print("[yellow]PyYAML not installed. Falling back to JSON format.[/yellow]")
                    typer.echo(json.dumps(schema_dict, indent=2))
            else:
                typer.echo(json.dumps(schema_dict, indent=2))
        
    except Exception as e:
        console.print(f"[red]Error generating schema:[/red] {e}", style="bold red")
        raise typer.Exit(code=1)


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    app()
