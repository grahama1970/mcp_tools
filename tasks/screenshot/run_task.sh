#!/bin/bash
# run_task.sh - Executes a single task from a project in isolation

# Check if project and task names were provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "ERROR: Please provide both project name and task filename"
  echo "Usage: $0 <project_name> <task_filename>"
  echo "Example: $0 refactor_screenshot 001_core_functions.md"
  exit 1
fi

PROJECT_NAME="$1"
TASK_FILENAME="$2"
BASE_DIR="/Users/robert/claude_mcp_configs/tasks/screenshot"
PROJECT_DIR="$BASE_DIR/$PROJECT_NAME"
TASKS_DIR="$PROJECT_DIR/tasks"
RESULTS_DIR="$PROJECT_DIR/results"

# Check if the project exists
if [ ! -d "$PROJECT_DIR" ]; then
  echo "ERROR: Project '$PROJECT_NAME' not found in $BASE_DIR"
  echo "Available projects:"
  find "$BASE_DIR" -maxdepth 1 -type d -not -path "$BASE_DIR" -not -path "*/\.*" | xargs -n1 basename | sort
  exit 1
fi

# Construct the task file path
TASK_FILE="$TASKS_DIR/$TASK_FILENAME"

# Check if the task file exists
if [ ! -f "$TASK_FILE" ]; then
  echo "ERROR: Task file '$TASK_FILENAME' not found in $TASKS_DIR"
  echo "Available tasks:"
  find "$TASKS_DIR" -name "*.md" -type f | xargs -n1 basename | sort
  exit 1
fi

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

echo "=== Running Single Task for Project: $PROJECT_NAME ==="
echo "Task file: $TASK_FILE"
echo "Results directory: $RESULTS_DIR"
echo ""

# Run the orchestrator with this task file
"$BASE_DIR/orchestrator.sh" "$TASK_FILE" "$RESULTS_DIR"

echo ""
echo "=== Task execution complete ==="
echo "Result saved to: $RESULTS_DIR/$(basename "$TASK_FILE" .md).result"
