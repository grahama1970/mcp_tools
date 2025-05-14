# Claude Code Task Management System

This system provides a structured approach for managing complex task lists by breaking them down into individual tasks that can be executed by Claude Code in isolation. This approach effectively mimics Roo's Boomerang mode by ensuring each task has a clean context window, preventing context overflow and maintaining focus.

## Rationale: Why This System Works

### The Context Limit Problem

Claude Code, like other LLMs, has a finite context window (approximately 200K tokens). When working with complex projects, the entire context can quickly become overwhelmed, leading to:

1. "Context limit exceeded" errors
2. Degraded performance as context fills up
3. Loss of focus on the current task
4. Inability to complete complex projects

### The Boomerang Pattern Solution

This system implements a "Boomerang" pattern inspired by Roo Code's Boomerang mode:

1. **Initial Task Breakdown**: Use Claude Code first to analyze and break down a large task into individual task files
2. **Task Isolation**: Each task runs in a completely separate context
3. **Clean Context Initialization**: Every task begins with `/clear` to reset context
4. **Independent Execution**: Tasks don't depend on context from previous tasks
5. **Result Centralization**: All results flow back to a central location

This approach is especially effective because:

- It prevents context overflow by compartmentalizing tasks
- It maintains focus on a single responsibility for each task
- It allows for complex multi-stage projects that would otherwise exceed context limits
- It enables easy rerunning of specific tasks if needed

## Project Structure

```
/tasks/screenshot/
├── README.md              # This file
├── orchestrator.sh        # Script to run a single task in isolation
├── project_template.sh    # Script to create a new project structure
├── run_project.sh         # Script to run all tasks in a project
├── run_task.sh            # Script to run a single task from a project
├── task_breakdown.sh      # Script to break down a large task into individual tasks
├── [project_name]/        # Project directory (e.g., "refactor_screenshot")
│   ├── project.md         # Project metadata and instructions
│   ├── task_list.md       # Original full task list (optional)
│   ├── task_sequence.txt  # Task execution sequence
│   ├── run_all.sh         # Project-specific runner script
│   ├── tasks/             # Individual task files
│   │   ├── 000_project_overview.md
│   │   ├── 001_first_task.md
│   │   ├── 002_second_task.md
│   │   └── ...
│   └── results/           # Execution results
│       ├── 000_project_overview.result
│       ├── 001_first_task.result
│       └── ...
└── ...
```

## Core Scripts

The system includes several key scripts:

1. **task_breakdown.sh**: The first script to use - breaks down a large task into individual task files
   ```
   ./task_breakdown.sh <project_name> <source_task_file>
   ```

2. **project_template.sh**: Creates a new project structure (used by task_breakdown.sh)
   ```
   ./project_template.sh <project_name> [source_task_file]
   ```

3. **run_project.sh**: Executes all tasks in a project in sequence
   ```
   ./run_project.sh <project_name>
   ```

4. **run_task.sh**: Executes a single task from a project
   ```
   ./run_task.sh <project_name> <task_filename>
   ```

5. **orchestrator.sh**: Core script that runs a single task with a clean context
   ```
   ./orchestrator.sh <task_file_path> <results_directory>
   ```

## Step-by-Step Implementation Guide

### 1. Breaking Down a Task List (FIRST STEP)

The first and most crucial step is to use Claude Code to analyze a large task list and break it down into individual task files:

```bash
./task_breakdown.sh <project_name> <source_task_file>
```

This command:
- Creates a new project structure if needed
- Uses Claude Code to analyze the source task file
- Breaks it down into individual task files
- Creates a task_sequence.txt file for execution order

Example:
```bash
./task_breakdown.sh ui_refactor /path/to/big_task.md
```

During this process, Claude Code will:
1. Analyze the task list to identify logical divisions
2. Create properly structured individual task files
3. Ensure each task has all necessary context to run in isolation
4. Generate a project overview and execution sequence

This initial breakdown step is essential for managing context properly in Claude Code.

### 2. Setting Up a New Project (if not done in step 1)

Create a new project structure:

```bash
./project_template.sh <project_name> [source_task_file]
```

This command:
- Creates the project directory structure
- Sets up task_sequence.txt for execution order
- Creates a project-specific runner script
- Optionally processes a source task file

Example:
```bash
./project_template.sh ui_refactor /path/to/big_task.md
```

### 3. Breaking Down Tasks

For optimal context isolation:

1. Each task file should be self-contained with all necessary context
2. Tasks should be focused on a single responsibility
3. Split large tasks into multiple smaller tasks
4. Use clear naming conventions (e.g., numerical prefixes)

Example task structure:
- `000_project_overview.md`: General project description
- `001_core_functions.md`: First implementation task
- `002_rich_formatters.md`: Second implementation task
- etc.

### 4. Setting Up Task Sequence

Edit the `task_sequence.txt` file to control execution order:

```
# Task sequence
000_project_overview.md
001_core_functions.md
002_rich_formatters.md
...
```

### 5. Running Tasks

#### Run All Tasks in a Project:

```bash
./run_project.sh <project_name>
```

Example:
```bash
./run_project.sh refactor_screenshot
```

Alternatively, from within the project directory:
```bash
./refactor_screenshot/run_all.sh
```

#### Run a Single Task:

```bash
./run_task.sh <project_name> <task_filename>
```

Example:
```bash
./run_task.sh refactor_screenshot 001_core_functions.md
```

### 6. Context Management Within Tasks

Claude Code provides built-in commands for context management:

- `/clear`: Completely clears conversation context
  - Used automatically at the start of each task
  - Can be used during a task to reset context if needed

- `/compact`: Intelligently compresses the context
  - Useful during task execution if context becomes full
  - Preserves important information while saving space

### 7. Reviewing Results

After execution, each task's output is saved to:
```
/tasks/screenshot/<project_name>/results/<task_name>.result
```

## Working with Existing Projects

To convert an existing project to this format:

1. Create a new project structure:
   ```bash
   ./project_template.sh <project_name> <original_task_file>
   ```

2. Copy existing task files to the new tasks directory:
   ```bash
   cp /path/to/original/tasks/*.md /tasks/screenshot/<project_name>/tasks/
   ```

3. Edit the task_sequence.txt file to set the execution order.

## Benefits of This Approach

1. **Prevents context overflow**: Each task runs with a fresh context
2. **Improves focus**: Claude Code deals with one specific task at a time
3. **Enhances modularity**: Tasks can be rearranged or rerun individually
4. **Enables complex projects**: Break down projects that would otherwise exceed context limits
5. **Preserves results**: Each task's output is saved separately
6. **Maintains organization**: Clear project structure for easy management
7. **Facilitates collaboration**: Project structure is easy to share and understand

This system effectively solves the context limitation problems in Claude Code while providing a structured approach to managing complex tasks, similar to how Roo's Boomerang mode operates.
