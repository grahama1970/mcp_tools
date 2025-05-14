# Claude Code Task Runner System

## Overview

This document describes the Task Runner system implemented for Claude Code. This system is inspired by the Roo Boomerang mode and solves the context limitation problems in Claude Code by isolating each task in its own context window.

## Problem Addressed

When working with Claude Code on large projects with multiple tasks, the context window can become overwhelmed, leading to:
- "Input length and max_tokens exceed context limit" errors
- Degraded coding performance as context grows
- Difficulty maintaining focus on specific tasks

## Solution: Task Isolation System

The Task Runner system solves these problems by:
1. Breaking complex projects into discrete task files
2. Running each task in a completely isolated Claude Code session
3. Maintaining clean separation between task contexts
4. Collecting results in a structured manner

## Components

The system consists of the following components:

### 1. Task Files
Individual markdown files in the `/Users/robert/claude_mcp_configs/tasks/screenshot/` directory. Each file should focus on a single task or feature. Files are named with numeric prefixes to control execution order (e.g., `000_refactor_screenshot.md`).

### 2. Orchestrator Script (`orchestrator.sh`)
Executes a single task with Claude Code in isolation. It:
- Takes a task file as input
- Runs Claude Code with just that file
- Ensures a fresh context for each run using `--no-history`
- Saves the result to a designated output location

### 3. Task Runner Script (`run_tasks.sh`)
Coordinates the execution of all tasks by:
- Finding all markdown task files in the directory
- Sorting them to ensure proper execution order
- Running each through the orchestrator sequentially
- Creating a clean results directory for outputs

### 4. CLAUDE.md
Contains instructions for Claude Code on how to approach tasks:
- Focus only on the current task
- Maintain context isolation
- Provide clear task completion indicators

## Usage Instructions

### Running the System

1. Navigate to the task directory:
   ```bash
   cd /Users/robert/claude_mcp_configs/tasks/screenshot
   ```

2. Execute the task runner:
   ```bash
   ./run_tasks.sh
   ```

3. View results in the `task_results` directory after completion.

Each task runs in its own isolated Claude Code session, using the `/clear` command to ensure a clean context for each task. This prevents context overflow between tasks.

### Adding New Tasks

1. Create new task files in the directory using the naming convention:
   ```
   NNN_descriptive_name.md
   ```
   Where NNN is a numeric prefix (e.g., 001, 002) for ordering.

2. Write clear, focused instructions in each task file.

3. Run the task runner to process the new tasks.

### Modifying Existing Tasks

If a task needs modification:
1. Edit the task file directly
2. Re-run just that task with:
   ```bash
   ./orchestrator.sh /path/to/task_file.md
   ```

## Best Practices

1. **Keep tasks focused** - Each task should do one thing well
2. **Include necessary context** - Each task should include or reference any required context
3. **Clear instructions** - Be specific about what should be accomplished
4. **Minimize dependencies** - Reduce dependencies between tasks when possible
5. **Review results individually** - Check each task's output before moving to integration

## Implementation Details

This system uses Desktop Commander MCP to access the file system and run commands. It creates a Boomerang-like experience by ensuring each Claude Code session has a fresh context window, preventing context overflow.

The approach is similar to Roo Boomerang mode's context isolation where:
- A parent "orchestrator" manages task delegation
- Individual tasks execute in isolation using the `/clear` command
- Results flow back to a central location
- Context never overflows between tasks

We use Claude Code's built-in context management capabilities:
- The `/clear` command at the start of each task ensures a clean context
- Each task runs in a separate Claude Code session
- Results are captured to their own output files

## Troubleshooting

- **Task fails to run**: Ensure the script has proper permissions (`chmod +x *.sh`)
- **Claude Code not found**: Verify Claude Code is in your PATH
- **Context errors persist**: Try using the `/compact` command if `/clear` is insufficient
- **Task files not found**: Check path and naming conventions
- **Context bleeding between tasks**: Ensure each task starts with the `/clear` command
