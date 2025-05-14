#!/bin/bash
# run_project.sh - Executes all tasks from a project in isolation

# Check if a project name was provided
if [ -z "$1" ]; then
  echo "ERROR: Please provide a project name"
  echo "Usage: $0 <project_name>"
  echo "Example: $0 refactor_screenshot"
  
  echo ""
  echo "Available projects:"
  find "/Users/robert/claude_mcp_configs/tasks/screenshot" -maxdepth 1 -type d -not -path "/Users/robert/claude_mcp_configs/tasks/screenshot" -not -path "*/\.*" | xargs -n1 basename | sort
  exit 1
fi

PROJECT_NAME="$1"
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

# Check if the tasks directory exists
if [ ! -d "$TASKS_DIR" ]; then
  echo "ERROR: Tasks directory for project '$PROJECT_NAME' not found at $TASKS_DIR"
  exit 1
fi

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

echo "=== Starting Claude Code Task Runner for Project: $PROJECT_NAME ==="
echo "Tasks directory: $TASKS_DIR"
echo "Results directory: $RESULTS_DIR"
echo ""

# Read the task sequence from the task sequence file (if exists)
TASK_SEQUENCE_FILE="$PROJECT_DIR/task_sequence.txt"
if [ -f "$TASK_SEQUENCE_FILE" ]; then
  echo "Using task sequence from: $TASK_SEQUENCE_FILE"
  TASK_FILES=()
  while IFS= read -r task_file; do
    # Skip comments and empty lines
    if [[ -n "$task_file" && ! "$task_file" =~ ^[[:space:]]*# && ! "$task_file" =~ ^[[:space:]]*$ ]]; then
      TASK_FILES+=("$TASKS_DIR/$task_file")
    fi
  done < "$TASK_SEQUENCE_FILE"
  
  # If no valid task files were found in sequence file
  if [ ${#TASK_FILES[@]} -eq 0 ]; then
    echo "WARNING: No valid task files found in sequence file. Using alphabetical order."
    TASK_FILES=($(find "$TASKS_DIR" -name "*.md" -not -name "README.md" | sort))
  fi
else
  # If no sequence file exists, find all markdown task files and sort them
  echo "WARNING: No task sequence file found at $TASK_SEQUENCE_FILE"
  echo "Using alphabetical order for task execution"
  TASK_FILES=($(find "$TASKS_DIR" -name "*.md" -not -name "README.md" | sort))
fi

# Display the task execution plan
echo "Task execution plan:"
for ((i=0; i<${#TASK_FILES[@]}; i++)); do
  echo "  $((i+1)). $(basename "${TASK_FILES[$i]}")"
done
echo ""

# Process each task in sequence
for task_file in "${TASK_FILES[@]}"; do
  if [ -f "$task_file" ]; then
    TASK_NAME=$(basename "$task_file")
    echo "Processing task: $TASK_NAME"
    
    # Run the orchestrator with this task file
    "$BASE_DIR/orchestrator.sh" "$task_file" "$RESULTS_DIR"
    
    echo "----------------------------------------"
    echo ""
    
    # Optional: add a pause between tasks to prevent rate limiting
    sleep 2
  else
    echo "WARNING: Task file not found: $task_file"
  fi
done

echo "=== All tasks for project $PROJECT_NAME completed ==="
echo "Results saved to: $RESULTS_DIR"
