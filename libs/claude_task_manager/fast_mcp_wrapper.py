"""
FastMCP wrapper for Claude Task Manager.

This module provides the FastMCP integration, allowing the TaskManager to be used
as an MCP server with proper function mapping and parameter conversion.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, cast

# Import FastMCP
from fastmcp import FastMCP

# Import TaskManager
from claude_task_manager.task_manager import TaskManager
from claude_task_manager.types import (
    ProjectInfo, ProjectResult, TaskBreakdownResult, TaskRunResult,
    ProjectRunResult, ProjectListResult, TaskListResult, ErrorResult
)


class TaskManagerMCP:
    """
    MCP wrapper for TaskManager class.
    
    This class provides the FastMCP integration, mapping MCP functions to TaskManager methods.
    """
    
    def __init__(self, base_dir: Optional[str] = None, log_level: str = "INFO"):
        """
        Initialize TaskManagerMCP with base directory and log level.
        
        Args:
            base_dir: Base directory for task management (default: $HOME/claude_tasks)
            log_level: Logging level (default: INFO)
            
        Raises:
            RuntimeError: If Claude Code is not available
        """
        # Setup logging first for proper initialization messages
        self.log_level = getattr(logging, log_level.upper())
        logging.basicConfig(
            level=self.log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("TaskManagerMCP")
        self.logger.setLevel(self.log_level)
        
        # Check if Claude Code is available
        self._check_dependencies()
        
        # Set base directory
        if base_dir is None:
            base_dir = os.environ.get('CLAUDE_TASKS_DIR', str(Path.home() / 'claude_tasks'))
            
        self.base_dir = base_dir
        
        # Initialize TaskManager
        self.task_manager = TaskManager(self.base_dir, self.log_level)
        
        # Initialize FastMCP
        self.app = FastMCP(name="Claude Task Manager")
        
        # Register functions
        self._register_functions()
        
        self.logger.info(f"TaskManagerMCP initialized with base directory: {self.base_dir}")
        
    def _check_dependencies(self) -> None:
        """
        Check if all required dependencies are available.
        
        Raises:
            RuntimeError: If any dependencies are missing
        """
        # Check for Claude Code
        try:
            import subprocess
            result = subprocess.run(["claude", "--version"], 
                                   capture_output=True, text=True, check=False)
            if result.returncode != 0:
                self.logger.error("Claude Code is not available or not in PATH")
                raise RuntimeError("Claude Code is not available. Please install Claude Desktop Commander.")
            self.logger.debug(f"Claude Code version: {result.stdout.strip()}")
        except FileNotFoundError:
            self.logger.error("Claude Code is not available or not in PATH")
            raise RuntimeError("Claude Code is not available. Please install Claude Desktop Commander.")
    
    def _register_functions(self) -> None:
        """Register all MCP functions."""
        # Project management functions
        self.app.function(self.create_project, name="create_project", 
                         description="Create a new project structure")
        self.app.function(self.list_projects, name="list_projects",
                         description="List all projects")
        self.app.function(self.check_status, name="check_status",
                         description="Check the status of a project")
        
        # Task management functions
        self.app.function(self.break_task, name="break_task",
                         description="Break down a task into individual task files")
        self.app.function(self.list_tasks, name="list_tasks",
                         description="List all tasks in a project")
        
        # Task execution functions
        self.app.function(self.run_task, name="run_task",
                         description="Run a single task with context isolation")
        self.app.function(self.run_project, name="run_project",
                         description="Run all tasks in a project with context isolation")
    
    def create_project(self, project_name: str, source_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new project structure.
        
        Args:
            project_name: Name of the project
            source_file: Optional path to a source task file
            
        Returns:
            Dict with project creation result
        """
        self.logger.info(f"MCP: Creating project {project_name}")
        try:
            project_dir = self.task_manager.create_project(project_name, source_file)
            
            result: ProjectResult = {
                "success": True,
                "project_name": project_name,
                "project_dir": str(project_dir),
                "source_file": source_file
            }
            return result
            
        except ValueError as e:
            # Handle validation errors
            self.logger.error(f"Validation error creating project: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "validation_error",
                "context": {"project_name": project_name}
            }
        except FileNotFoundError as e:
            # Handle file not found errors
            self.logger.error(f"File not found error creating project: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "file_not_found",
                "context": {"project_name": project_name, "source_file": source_file}
            }
        except Exception as e:
            # Handle unexpected errors
            self.logger.error(f"Error creating project: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "unexpected_error",
                "context": {"project_name": project_name}
            }
    
    def break_task(self, project_name: str, source_file: str) -> Dict[str, Any]:
        """
        Break down a task into individual task files.
        
        Args:
            project_name: Name of the project
            source_file: Path to source task file
            
        Returns:
            Dict with task breakdown result
        """
        self.logger.info(f"MCP: Breaking down task file for project {project_name}")
        try:
            project_dir, created_files = self.task_manager.break_down_task(project_name, source_file)
            
            result: TaskBreakdownResult = {
                "success": True,
                "project_name": project_name,
                "project_dir": str(project_dir),
                "source_file": source_file,
                "created_files": created_files,
                "file_count": len(created_files),
                "next_step_command": f"claude-tasks run-project {project_name}"
            }
            return result
            
        except Exception as e:
            self.logger.error(f"Error breaking down task: {e}")
            return {
                "success": False,
                "error": str(e),
                "context": {
                    "project_name": project_name,
                    "source_file": source_file
                }
            }
    
    def run_task(self, project_name: str, task_filename: str) -> Dict[str, Any]:
        """
        Run a single task with context isolation.
        
        Args:
            project_name: Name of the project
            task_filename: Name of the task file
            
        Returns:
            Dict with task execution result
        """
        self.logger.info(f"MCP: Running task {task_filename} in project {project_name}")
        try:
            result_file = self.task_manager.run_task(project_name, task_filename)
            
            # Read result file content
            with open(result_file, 'r') as f:
                content = f.read()
            
            result: Dict[str, Any] = {
                "success": True,
                "project_name": project_name,
                "task_filename": task_filename,
                "result_file": result_file,
                "content": content
            }
            return result
            
        except Exception as e:
            self.logger.error(f"Error running task: {e}")
            return {
                "success": False,
                "error": str(e),
                "context": {
                    "project_name": project_name,
                    "task_filename": task_filename
                }
            }
    
    def run_project(self, project_name: str, skip_confirmation: bool = True) -> Dict[str, Any]:
        """
        Run all tasks in a project with context isolation.
        
        Args:
            project_name: Name of the project
            skip_confirmation: Skip confirmation prompt and run all tasks
            
        Returns:
            Dict with project execution result
        """
        self.logger.info(f"MCP: Running all tasks in project {project_name}")
        try:
            # In MCP, we always skip confirmation
            results = self.task_manager.run_project(project_name)
            
            # Get task status
            status = self.task_manager.check_project_status(project_name)
            
            # Collect result contents
            task_results = []
            for result_file in results:
                task_file = os.path.basename(result_file).replace('.result', '.md')
                try:
                    with open(result_file, 'r') as f:
                        content = f.read()
                    task_results.append({
                        "task_file": task_file,
                        "result_file": result_file,
                        "status": "success",
                        "content": content
                    })
                except Exception as e:
                    task_results.append({
                        "task_file": task_file,
                        "result_file": result_file,
                        "status": "error",
                        "error": str(e)
                    })
            
            result: Dict[str, Any] = {
                "success": True,
                "project_name": project_name,
                "total_tasks": status["total_tasks"],
                "completed_tasks": status["completed_tasks"],
                "pending_tasks": status["pending_tasks"],
                "completion_percentage": status["completion_percentage"],
                "results": task_results
            }
            return result
            
        except Exception as e:
            self.logger.error(f"Error running project: {e}")
            return {
                "success": False,
                "error": str(e),
                "context": {"project_name": project_name}
            }
    
    def list_projects(self) -> Dict[str, Any]:
        """
        List all projects.
        
        Returns:
            Dict with project list result
        """
        self.logger.info("MCP: Listing all projects")
        try:
            projects = self.task_manager.list_projects()
            
            result: ProjectListResult = {
                "success": True,
                "base_dir": self.base_dir,
                "projects": projects,
                "count": len(projects)
            }
            return result
            
        except Exception as e:
            self.logger.error(f"Error listing projects: {e}")
            return {
                "success": False,
                "error": str(e),
                "context": {"base_dir": self.base_dir}
            }
    
    def list_tasks(self, project_name: str) -> Dict[str, Any]:
        """
        List all tasks in a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dict with task list result
        """
        self.logger.info(f"MCP: Listing tasks in project {project_name}")
        try:
            tasks = self.task_manager.list_tasks(project_name)
            
            result: TaskListResult = {
                "success": True,
                "project_name": project_name,
                "tasks": tasks,
                "count": len(tasks)
            }
            return result
            
        except Exception as e:
            self.logger.error(f"Error listing tasks: {e}")
            return {
                "success": False,
                "error": str(e),
                "context": {"project_name": project_name}
            }
    
    def check_status(self, project_name: str) -> Dict[str, Any]:
        """
        Check the status of a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dict with project status
        """
        self.logger.info(f"MCP: Checking status of project {project_name}")
        try:
            status = self.task_manager.check_project_status(project_name)
            
            if not status["exists"]:
                return {
                    "success": False,
                    "error": f"Project {project_name} not found",
                    "context": {"project_name": project_name}
                }
            
            # Get the tasks and results for more detailed status
            tasks = self.task_manager.list_tasks(project_name)
            results = self.task_manager.get_task_results(project_name)
            
            # Create a detailed task status list
            task_status = []
            for task in tasks:
                if task in results:
                    task_status.append({
                        "task_file": task,
                        "status": "completed",
                        "result_file": results[task]
                    })
                else:
                    task_status.append({
                        "task_file": task,
                        "status": "pending",
                        "result_file": None
                    })
            
            # Prepare comprehensive status response
            result = {
                "success": True,
                "project_name": project_name,
                "total_tasks": status["total_tasks"],
                "completed_tasks": status["completed_tasks"],
                "pending_tasks": status["pending_tasks"],
                "completion_percentage": status["completion_percentage"],
                "tasks": task_status,
                "base_dir": self.base_dir,
                "project_dir": str(Path(self.base_dir) / project_name)
            }
            return result
            
        except Exception as e:
            self.logger.error(f"Error checking project status: {e}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "unexpected_error",
                "context": {"project_name": project_name}
            }
    
    def run(self, host: str = "localhost", port: int = 3000) -> None:
        """
        Run the FastMCP server.
        
        Args:
            host: Host to listen on (default: localhost)
            port: Port to listen on (default: 3000)
        """
        self.logger.info(f"Starting FastMCP server on {host}:{port}")
        self.app.run(host=host, port=port)


# Default instance for simple import
default_mcp = TaskManagerMCP()

if __name__ == "__main__":
    # If run directly, start the server
    default_mcp.run()
