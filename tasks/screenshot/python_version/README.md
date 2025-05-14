# Claude Task Manager

A Python package for managing complex Claude Code tasks with context isolation, implementing a "Boomerang" style approach.

## Features

- **Task Breakdown**: Analyze large task files and break them into individual tasks
- **Context Isolation**: Execute each task with a clean context window
- **Project Management**: Organize tasks into projects
- **Flexible Execution**: Run individual tasks or entire projects
- **Clean CLI**: Well-documented command-line interface

## Installation

```bash
# Install directly from the directory
pip install -e .

# Or install from GitHub (once published)
# pip install git+https://github.com/yourusername/claude-task-manager.git
```

## Usage

### Command Line Interface

```bash
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
```

### Python API

```python
from claude_task_manager import TaskManager

# Initialize the task manager
manager = TaskManager('/path/to/base/directory')

# Create a new project
project_dir = manager.create_project('my_project', '/path/to/task_list.md')

# Break down a task into individual tasks
project_dir, created_files = manager.break_down_task('my_project', '/path/to/task_list.md')

# Run a single task
result_file = manager.run_task('my_project', '001_first_task.md')

# Run all tasks in a project
results = manager.run_project('my_project')

# List available projects
projects = manager.list_projects()

# List tasks in a project
tasks = manager.list_tasks('my_project')
```

## How It Works

1. **Task Breakdown**: The system uses Claude Code to analyze a large task file and break it into smaller, independent task files.
2. **Context Isolation**: Each task is executed in a separate Claude Code session with a clean context window, using the `/clear` command.
3. **Project Organization**: Tasks are organized into projects, each with its own directory structure.
4. **Execution Flow**: Tasks can be executed individually or in sequence, with proper context isolation between them.

## Project Structure

```
/base_directory/
├── project_name/
│   ├── project.json          # Project metadata
│   ├── task_list.md          # Original task list
│   ├── task_sequence.txt     # Task execution sequence
│   ├── tasks/                # Individual task files
│   │   ├── 000_project_overview.md
│   │   ├── 001_first_task.md
│   │   └── ...
│   ├── results/              # Execution results
│   │   ├── 000_project_overview.result
│   │   ├── 001_first_task.result
│   │   └── ...
│   └── temp/                 # Temporary files
└── ...
```

## Requirements

- Python 3.7+
- Claude Desktop with `claude` command-line tool accessible
- Desktop Commander MCP for file system access

## License

MIT License
