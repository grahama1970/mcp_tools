"""
Core task manager module - provides the main TaskManager class that handles projects, tasks, and execution.

This module requires Desktop Commander to be installed and enabled in Claude Desktop for file system access.
See: https://desktopcommander.app/#installation
"""

import os
import sys
import subprocess
import shutil
import re
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple, cast

from claude_task_manager.types import (
    ProjectInfo, ProjectResult, TaskBreakdownResult, TaskRunResult,
    ProjectRunResult, ProjectListResult, TaskListResult, ErrorResult,
    LogLevel as LogLevelType
)


class TaskManager:
    """
    Main class for managing Claude Code task execution with context isolation.
    
    This class handles the entire workflow:
    1. Creating project structures
    2. Breaking down large tasks into individual task files
    3. Executing tasks with context isolation
    4. Managing results
    """
    
    def __init__(self, base_dir: str, log_level: int = logging.INFO):
        """
        Initialize the TaskManager with a base directory.
        
        Args:
            base_dir: Base directory for all projects
            log_level: Logging level to use
        """
        self.base_dir = Path(base_dir)
        self.setup_logging(log_level)
        
        # Create base directory if it doesn't exist
        self.base_dir.mkdir(exist_ok=True, parents=True)
        
        self.logger.info(f"Task Manager initialized at {self.base_dir}")
    
    def setup_logging(self, log_level: int) -> None:
        """
        Set up logging configuration.
        
        Args:
            log_level: Logging level to use
        """
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("TaskManager")
        self.logger.setLevel(log_level)
    
    def create_project(self, project_name: str, source_file: Optional[str] = None) -> Path:
        """
        Create a new project structure.
        
        Args:
            project_name: Name of the project
            source_file: Optional path to a source task file
            
        Returns:
            Path to the created project directory
        
        Raises:
            ValueError: If the project name is invalid
            FileNotFoundError: If the source file doesn't exist
        """
        self.logger.info(f"Creating project: {project_name}")
        
        # Validate project name
        if not project_name or not project_name.strip():
            raise ValueError("Project name cannot be empty")
            
        if os.path.sep in project_name or project_name.startswith('.'):
            raise ValueError(f"Invalid project name: {project_name}")
            
        # Create project directory structure
        project_dir = self.base_dir / project_name
        tasks_dir = project_dir / "tasks"
        results_dir = project_dir / "results"
        temp_dir = project_dir / "temp"
        
        for dir_path in [project_dir, tasks_dir, results_dir, temp_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)
        
        # Initialize sequence file
        sequence_file = project_dir / "task_sequence.txt"
        with open(sequence_file, 'w') as f:
            f.write(f"# Task sequence for project: {project_name}\n")
            f.write("# Format: one task filename per line (relative to tasks directory)\n")
            f.write("# Lines starting with # are comments and will be ignored\n")
            f.write("# Empty lines are also ignored\n\n")
        
        # Initialize project metadata
        project_info: ProjectInfo = {
            "name": project_name,
            "created_at": str(Path.ctime(Path.home())),
            "tasks_dir": str(tasks_dir),
            "results_dir": str(results_dir),
            "source_file": None
        }
        
        with open(project_dir / "project.json", 'w') as f:
            json.dump(project_info, f, indent=2)
        
        # Copy source file if provided
        if source_file:
            source_path = Path(source_file)
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_file}")
                
            self.logger.info(f"Copying source file: {source_file}")
            target_path = project_dir / "task_list.md"
            shutil.copy2(source_file, target_path)
            project_info["source_file"] = str(target_path)
            
            # Update project metadata
            with open(project_dir / "project.json", 'w') as f:
                json.dump(project_info, f, indent=2)
        
        self.logger.info(f"Project created at {project_dir}")
        return project_dir
    
    def break_down_task(self, project_name: str, source_file: str) -> Tuple[Path, List[str]]:
        """
        Break down a large task file into individual task files using Claude Code.
        
        Args:
            project_name: Name of the project
            source_file: Path to the source task file
            
        Returns:
            Tuple containing:
            - Path to the project directory
            - List of created task files
            
        Raises:
            FileNotFoundError: If the source file doesn't exist
            RuntimeError: If the task breakdown fails
        """
        self.logger.info(f"Breaking down task file: {source_file} for project: {project_name}")
        
        # Validate source file
        source_path = Path(source_file)
        if not source_path.exists():
            raise FileNotFoundError(f"Source file not found: {source_file}")
        
        # Ensure project exists or create it
        project_dir = self.base_dir / project_name
        if not project_dir.exists():
            project_dir = self.create_project(project_name, source_file)
        
        tasks_dir = project_dir / "tasks"
        temp_dir = project_dir / "temp"
        temp_dir.mkdir(exist_ok=True)
        
        # Create instructions for Claude
        instructions_file = temp_dir / "breakdown_instructions.md"
        result_file = temp_dir / "breakdown_result.md"
        
        with open(instructions_file, 'w') as f:
            f.write("# Task Breakdown Instructions\n\n")
            f.write("I need you to analyze the following task list and break it down into individual task files that can be executed independently.\n\n")
            f.write("## Source Task List\n\n")
            
            # Read and include the source file content
            with open(source_file, 'r') as src:
                f.write(src.read())
            
            f.write("\n\n## Instructions\n\n")
            f.write("1. Analyze the task list and identify logical divisions for independent tasks\n")
            f.write("2. For each task, create a separate markdown file named with a numerical prefix (e.g., 001_task_name.md)\n")
            f.write("3. Ensure each task file is self-contained with all necessary context\n")
            f.write("4. The first file should be 000_project_overview.md with a summary of the entire project\n")
            f.write("5. For each task file, include:\n")
            f.write("   - Clear title and objective\n")
            f.write("   - Required context or background\n")
            f.write("   - Implementation steps\n")
            f.write("   - Verification methods\n")
            f.write("   - Acceptance criteria\n")
            f.write("6. Generate a task_sequence.txt file listing the tasks in execution order\n\n")
            f.write("## Output Format\n\n")
            f.write("For each task file, provide the filename and content in this format:\n\n")
            f.write("### [FILENAME: 000_project_overview.md]\n")
            f.write("# Project Overview\n")
            f.write("...content...\n\n")
            f.write("### [FILENAME: 001_first_task.md]\n")
            f.write("# Task 1: First Task\n")
            f.write("...content...\n\n")
            f.write("And so on. After listing all task files, provide a task_sequence.txt file with all filenames in execution order.\n\n")
            f.write("This breakdown should ensure each task can be executed in isolation without requiring context from other tasks.\n")
        
        # Run Claude Code to perform the breakdown
        self.logger.info("Running Claude Code to analyze and break down the task list...")
        
        cmd = ["claude", "code", "--input", str(instructions_file), "--output", str(result_file)]
        try:
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            self.logger.debug(f"Claude Code output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Claude Code failed with error: {e}")
            self.logger.error(f"STDOUT: {e.stdout}")
            self.logger.error(f"STDERR: {e.stderr}")
            raise RuntimeError("Task breakdown failed") from e
        
        self.logger.info("Task breakdown complete. Extracting task files...")
        
        # Process the breakdown result
        created_files = self._extract_task_files(result_file, tasks_dir, project_dir)
        
        self.logger.info(f"Created {len(created_files)} task files")
        return project_dir, created_files
    
    def _extract_task_files(self, result_file: Path, tasks_dir: Path, project_dir: Path) -> List[str]:
        """
        Extract task files from Claude's breakdown result.
        
        Args:
            result_file: Path to the breakdown result file
            tasks_dir: Directory to write task files to
            project_dir: Root project directory
            
        Returns:
            List of created task filenames
            
        Raises:
            FileNotFoundError: If the result file doesn't exist
        """
        if not result_file.exists():
            raise FileNotFoundError(f"Result file not found: {result_file}")
            
        created_files: List[str] = []
        
        # Process the breakdown result
        with open(result_file, 'r') as f:
            content = f.read()
        
        # Extract task files using regex
        file_pattern = r'### \[FILENAME: ([^\]]+)\](.*?)(?=### \[FILENAME:|$)'
        matches = re.findall(file_pattern, content, re.DOTALL)
        
        for filename, file_content in matches:
            filename = filename.strip()
            if filename == "task_sequence.txt":
                # Handle task sequence file separately
                with open(project_dir / "task_sequence.txt", 'w') as f:
                    f.write(file_content.strip())
            else:
                # Handle regular task files
                file_path = tasks_dir / filename
                with open(file_path, 'w') as f:
                    f.write(file_content.strip())
                created_files.append(filename)
        
        # If no task sequence file was provided, generate one
        if not any(f == "task_sequence.txt" for f, _ in matches):
            self.logger.info("Generating task_sequence.txt based on filenames...")
            with open(project_dir / "task_sequence.txt", 'a') as f:
                f.write("\n# Generated task sequence\n")
                for filename in sorted(created_files):
                    f.write(f"{filename}\n")
        
        return created_files
    
    def run_task(self, project_name: str, task_filename: str) -> str:
        """
        Run a single task with context isolation.
        
        Args:
            project_name: Name of the project
            task_filename: Name of the task file
            
        Returns:
            Path to the result file
            
        Raises:
            FileNotFoundError: If the project or task file doesn't exist
            RuntimeError: If the task execution fails
        """
        self.logger.info(f"Running task: {task_filename} in project: {project_name}")
        
        project_dir = self.base_dir / project_name
        tasks_dir = project_dir / "tasks"
        results_dir = project_dir / "results"
        
        # Ensure directories exist
        for dir_path in [project_dir, tasks_dir, results_dir]:
            if not dir_path.exists():
                self.logger.error(f"Directory not found: {dir_path}")
                raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        # Construct task file path
        task_file = tasks_dir / task_filename
        if not task_file.exists():
            self.logger.error(f"Task file not found: {task_file}")
            raise FileNotFoundError(f"Task file not found: {task_file}")
        
        # Prepare result file
        result_filename = task_filename.replace('.md', '.result')
        result_file = results_dir / result_filename
        
        # Create temporary file for Claude input with clear command
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp:
            tmp.write("/clear\n")
            tmp.write("# Focus on this task only\n")
            tmp.write("Please focus exclusively on the task described in this file.\n")
            tmp.write("Do not attempt to access previous context or tasks.\n\n")
            
            # Append task content
            with open(task_file, 'r') as f:
                tmp.write(f.read())
            
            tmp_path = tmp.name
        
        # Run Claude Code
        self.logger.info(f"Starting Claude Code for task: {task_filename}")
        cmd = ["claude", "code", "--input", tmp_path, "--output", str(result_file)]
        
        try:
            result = subprocess.run(cmd, check=True, text=True, capture_output=True)
            self.logger.debug(f"Claude Code output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Claude Code failed with error: {e}")
            self.logger.error(f"STDOUT: {e.stdout}")
            self.logger.error(f"STDERR: {e.stderr}")
            raise RuntimeError(f"Task execution failed for {task_filename}") from e
        finally:
            # Clean up temporary file
            os.unlink(tmp_path)
        
        self.logger.info(f"Task completed: {task_filename}")
        self.logger.info(f"Result saved to: {result_file}")
        
        return str(result_file)
    
    def run_project(self, project_name: str) -> List[str]:
        """
        Run all tasks in a project in sequence.
        
        Args:
            project_name: Name of the project
            
        Returns:
            List of result file paths
            
        Raises:
            FileNotFoundError: If the project directory doesn't exist
        """
        self.logger.info(f"Running all tasks for project: {project_name}")
        
        project_dir = self.base_dir / project_name
        tasks_dir = project_dir / "tasks"
        sequence_file = project_dir / "task_sequence.txt"
        
        # Ensure project exists
        if not project_dir.exists():
            self.logger.error(f"Project directory not found: {project_dir}")
            raise FileNotFoundError(f"Project directory not found: {project_dir}")
        
        # Get task sequence
        task_files: List[str] = []
        if sequence_file.exists():
            self.logger.info(f"Using task sequence from: {sequence_file}")
            with open(sequence_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        task_files.append(line)
        else:
            self.logger.warning(f"No task sequence file found at {sequence_file}")
            self.logger.info("Using alphabetical order for task execution")
            task_files = sorted([f.name for f in tasks_dir.glob('*.md')])
        
        # Display execution plan
        self.logger.info(f"Task execution plan for {project_name}:")
        for i, task in enumerate(task_files, 1):
            self.logger.info(f"  {i}. {task}")
        
        # Run each task
        results: List[str] = []
        for task_file in task_files:
            try:
                result = self.run_task(project_name, task_file)
                results.append(result)
                
                # Add a small delay between tasks
                time.sleep(2)
            except Exception as e:
                self.logger.error(f"Error running task {task_file}: {e}")
                # Continue with next task
        
        self.logger.info(f"All tasks for project {project_name} completed")
        return results
    
    def list_projects(self) -> List[str]:
        """
        List all projects in the base directory.
        
        Returns:
            List of project names
        """
        projects: List[str] = []
        for path in self.base_dir.iterdir():
            if path.is_dir() and (path / "project.json").exists():
                projects.append(path.name)
        
        return sorted(projects)
    
    def list_tasks(self, project_name: str) -> List[str]:
        """
        List all tasks in a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            List of task filenames
        """
        project_dir = self.base_dir / project_name
        tasks_dir = project_dir / "tasks"
        
        if not tasks_dir.exists():
            return []
        
        return sorted([f.name for f in tasks_dir.glob('*.md')])
        
    def get_task_results(self, project_name: str) -> Dict[str, str]:
        """
        Get all task results for a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dictionary mapping task filenames to result file paths
        """
        project_dir = self.base_dir / project_name
        results_dir = project_dir / "results"
        
        if not results_dir.exists():
            return {}
            
        results: Dict[str, str] = {}
        for result_file in results_dir.glob('*.result'):
            task_filename = result_file.name.replace('.result', '.md')
            results[task_filename] = str(result_file)
            
        return results
        
    def check_project_status(self, project_name: str) -> Dict[str, Any]:
        """
        Check the status of a project.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Dictionary with project status information
        """
        project_dir = self.base_dir / project_name
        
        if not project_dir.exists():
            return {"exists": False}
            
        tasks = self.list_tasks(project_name)
        results = self.get_task_results(project_name)
        
        completed_tasks = [task for task in tasks if task in results]
        pending_tasks = [task for task in tasks if task not in results]
        
        return {
            "exists": True,
            "total_tasks": len(tasks),
            "completed_tasks": len(completed_tasks),
            "pending_tasks": len(pending_tasks),
            "completion_percentage": 100 * len(completed_tasks) / len(tasks) if tasks else 0
        }
