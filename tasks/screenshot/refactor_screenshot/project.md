# Project: refactor_screenshot

This project uses Claude Code with context isolation to execute a series of tasks.

## Structure

- **Project Directory**: `/Users/robert/claude_mcp_configs/tasks/screenshot/refactor_screenshot`
  - Contains this project.md file and task_sequence.txt controlling execution order
  
- **Tasks Directory**: `/Users/robert/claude_mcp_configs/tasks/screenshot/refactor_screenshot/tasks`
  - Contains the individual task files to be executed in isolation
  
- **Results Directory**: `/Users/robert/claude_mcp_configs/tasks/screenshot/refactor_screenshot/results`
  - Where task execution results will be stored

## Running the Tasks

To execute all tasks in this project:

```bash
cd /Users/robert/claude_mcp_configs/tasks/screenshot
./run_project.sh refactor_screenshot
```

To run a specific task from this project:

```bash
./run_task.sh refactor_screenshot <task_filename>
```

## Task Sequence

The execution order is controlled by `task_sequence.txt`, which lists the task files in the order they should be executed.
