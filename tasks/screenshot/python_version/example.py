#!/usr/bin/env python3
"""
Example script demonstrating how to use the Claude Task Manager.
"""

import os
import sys
import logging
from pathlib import Path

from claude_task_manager.task_manager import TaskManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Example")

def main():
    """Example workflow using the task manager."""
    # Define the base directory and source task file
    base_dir = Path('/Users/robert/claude_mcp_configs/tasks/screenshot/python_projects')
    source_file = Path('/Users/robert/claude_mcp_configs/tasks/screenshot/000_refactor_screenshot.md.bak')
    project_name = 'refactor_screenshot_py'
    
    # Check if source file exists
    if not source_file.exists():
        logger.error(f"Source file not found: {source_file}")
        return 1
    
    try:
        # Initialize the task manager
        logger.info("Initializing task manager...")
        manager = TaskManager(str(base_dir))
        
        # List existing projects
        projects = manager.list_projects()
        logger.info(f"Existing projects: {projects}")
        
        # Break down the task if project doesn't exist
        if project_name not in projects:
            logger.info(f"Breaking down task into project: {project_name}")
            project_dir, task_files = manager.break_down_task(project_name, str(source_file))
            logger.info(f"Created {len(task_files)} task files in {project_dir}")
        else:
            logger.info(f"Project {project_name} already exists")
        
        # List tasks in the project
        tasks = manager.list_tasks(project_name)
        logger.info(f"Tasks in project: {tasks}")
        
        # Run the first task
        if tasks:
            first_task = tasks[0]
            logger.info(f"Running first task: {first_task}")
            result = manager.run_task(project_name, first_task)
            logger.info(f"Task result saved to: {result}")
            
            # Uncomment to run all tasks
            # logger.info(f"Running all tasks in project: {project_name}")
            # results = manager.run_project(project_name)
            # logger.info(f"All tasks completed. Results: {results}")
        else:
            logger.warning(f"No tasks found in project {project_name}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
