#!/usr/bin/env python3
"""
Claude Task Manager integration for the screenshot tool.

This script provides a convenient way to use the Claude Task Manager
specifically for the screenshot tool refactoring project.
"""

import os
import sys
import argparse
from pathlib import Path

# Try to import the TaskManager directly
try:
    from claude_task_manager import TaskManager
except ImportError:
    # If not installed, try to add the libs directory to Python path
    libs_dir = Path(__file__).parent.parent.parent / "libs"
    sys.path.append(str(libs_dir))
    try:
        from claude_task_manager import TaskManager
    except ImportError:
        print("Error: claude_task_manager package not found.")
        print(f"Please install it with: python {libs_dir.parent}/setup_libs.py task-manager")
        print("Or install directly with: pip install -e libs/claude_task_manager")
        sys.exit(1)

# Constants
BASE_DIR = Path(__file__).parent
TASKS_DIR = BASE_DIR / "tasks"
SOURCE_FILE = BASE_DIR / "000_refactor_screenshot.md.bak"

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Claude Task Manager for Screenshot Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # init command
    init_parser = subparsers.add_parser("init", help="Initialize a new project")
    init_parser.add_argument(
        "project_name", 
        help="Name of the project",
        nargs="?",
        default="refactor_screenshot",
    )
    
    # breakdown command
    breakdown_parser = subparsers.add_parser("breakdown", help="Break down a task file into subtasks")
    breakdown_parser.add_argument(
        "project_name", 
        help="Name of the project",
        nargs="?",
        default="refactor_screenshot",
    )
    breakdown_parser.add_argument(
        "--source", 
        "-s",
        help="Path to source task file",
        default=str(SOURCE_FILE),
    )
    
    # run command
    run_parser = subparsers.add_parser("run", help="Run a project")
    run_parser.add_argument(
        "project_name", 
        help="Name of the project",
        nargs="?",
        default="refactor_screenshot",
    )
    
    # run-task command
    run_task_parser = subparsers.add_parser("run-task", help="Run a single task")
    run_task_parser.add_argument(
        "project_name", 
        help="Name of the project",
        nargs="?",
        default="refactor_screenshot",
    )
    run_task_parser.add_argument(
        "task_filename", 
        help="Name of the task file",
    )
    
    # list command
    list_parser = subparsers.add_parser("list", help="List projects or tasks")
    list_parser.add_argument(
        "project_name", 
        help="Name of the project (optional)",
        nargs="?",
    )
    
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Create task manager
    manager = TaskManager(str(BASE_DIR))
    
    # Execute the command
    if args.command == "init":
        print(f"Initializing project: {args.project_name}")
        project_dir = manager.create_project(args.project_name)
        print(f"Project created at: {project_dir}")
        
    elif args.command == "breakdown":
        print(f"Breaking down task file for project: {args.project_name}")
        source_file = args.source
        project_dir, created_files = manager.break_down_task(args.project_name, source_file)
        print(f"Created {len(created_files)} task files:")
        for filename in created_files:
            print(f"  - {filename}")
        
    elif args.command == "run":
        print(f"Running all tasks for project: {args.project_name}")
        results = manager.run_project(args.project_name)
        print(f"All tasks completed. Results saved to:")
        for result in results:
            print(f"  - {result}")
        
    elif args.command == "run-task":
        print(f"Running task {args.task_filename} for project: {args.project_name}")
        result = manager.run_task(args.project_name, args.task_filename)
        print(f"Task completed. Result saved to: {result}")
        
    elif args.command == "list":
        if args.project_name:
            # List tasks in a project
            tasks = manager.list_tasks(args.project_name)
            print(f"Tasks in project {args.project_name}:")
            for task in tasks:
                print(f"  - {task}")
        else:
            # List projects
            projects = manager.list_projects()
            print("Available projects:")
            for project in projects:
                print(f"  - {project}")
    
    else:
        print("No command specified. Use -h for help.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
