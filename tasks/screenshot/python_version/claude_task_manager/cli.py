"""
Command-line interface for Claude Task Manager.

This module provides a command-line interface to interact with the TaskManager class.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from .task_manager import TaskManager


def setup_argparse() -> argparse.ArgumentParser:
    """Set up the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Claude Task Manager - Manage complex tasks with context isolation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new project
  claude-tasks create-project my_project /path/to/task_list.md
  
  # Break down a task into individual tasks
  claude-tasks break-task my_project /path/to/task_list.md
  
  # Run a single task
  claude-tasks run-task my_project 001_first_task.md
  
  # Run all tasks in a project
  claude-tasks run-project my_project
  
  # List available projects
  claude-tasks list-projects
  
  # List tasks in a project
  claude-tasks list-tasks my_project
"""
    )
    
    parser.add_argument(
        '--base-dir', 
        default=os.environ.get('CLAUDE_TASKS_DIR', str(Path.home() / 'claude_tasks')),
        help="Base directory for task management (default: ~/claude_tasks or CLAUDE_TASKS_DIR env var)"
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help="Set the logging level (default: INFO)"
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # create-project command
    create_parser = subparsers.add_parser('create-project', help='Create a new project')
    create_parser.add_argument('project_name', help='Name of the project')
    create_parser.add_argument(
        'source_file',
        nargs='?',
        default=None,
        help='Path to source task file (optional)'
    )
    
    # break-task command
    break_parser = subparsers.add_parser('break-task', help='Break down a task into individual tasks')
    break_parser.add_argument('project_name', help='Name of the project')
    break_parser.add_argument('source_file', help='Path to source task file')
    
    # run-task command
    run_task_parser = subparsers.add_parser('run-task', help='Run a single task')
    run_task_parser.add_argument('project_name', help='Name of the project')
    run_task_parser.add_argument('task_filename', help='Name of the task file')
    
    # run-project command
    run_project_parser = subparsers.add_parser('run-project', help='Run all tasks in a project')
    run_project_parser.add_argument('project_name', help='Name of the project')
    
    # list-projects command
    list_projects_parser = subparsers.add_parser('list-projects', help='List all projects')
    
    # list-tasks command
    list_tasks_parser = subparsers.add_parser('list-tasks', help='List all tasks in a project')
    list_tasks_parser.add_argument('project_name', help='Name of the project')
    
    return parser


def main():
    """Main entry point for the CLI."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Set up logging
    log_level = getattr(logging, args.log_level)
    
    # Create task manager
    try:
        manager = TaskManager(args.base_dir, log_level=log_level)
    except Exception as e:
        print(f"Error initializing task manager: {e}")
        return 1
    
    # Execute the requested command
    try:
        if args.command == 'create-project':
            project_dir = manager.create_project(args.project_name, args.source_file)
            print(f"Project created at: {project_dir}")
            
        elif args.command == 'break-task':
            project_dir, created_files = manager.break_down_task(args.project_name, args.source_file)
            print(f"Task breakdown complete. Created {len(created_files)} task files:")
            for filename in created_files:
                print(f"  - {filename}")
            print(f"\nYou can now run the tasks with:")
            print(f"claude-tasks run-project {args.project_name}")
            
        elif args.command == 'run-task':
            result_file = manager.run_task(args.project_name, args.task_filename)
            print(f"Task execution complete. Result saved to: {result_file}")
            
        elif args.command == 'run-project':
            results = manager.run_project(args.project_name)
            print(f"Project execution complete. Results saved to:")
            for result in results:
                print(f"  - {result}")
            
        elif args.command == 'list-projects':
            projects = manager.list_projects()
            if projects:
                print("Available projects:")
                for project in projects:
                    print(f"  - {project}")
            else:
                print("No projects found")
            
        elif args.command == 'list-tasks':
            tasks = manager.list_tasks(args.project_name)
            if tasks:
                print(f"Tasks in project {args.project_name}:")
                for task in tasks:
                    print(f"  - {task}")
            else:
                print(f"No tasks found in project {args.project_name}")
            
        else:
            parser.print_help()
            return 1
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
