#!/bin/bash
# task_breakdown.sh - Uses Claude Code to break down a large task file into individual tasks

# Check if required parameters were provided
if [ -z "$1" ] || [ -z "$2" ]; then
  echo "ERROR: Please provide both project name and source task file"
  echo "Usage: $0 <project_name> <source_task_file>"
  echo "Example: $0 new_project /path/to/large_task.md"
  exit 1
fi

PROJECT_NAME="$1"
SOURCE_TASK_FILE="$2"
BASE_DIR="/Users/robert/claude_mcp_configs/tasks/screenshot"
PROJECT_DIR="$BASE_DIR/$PROJECT_NAME"

# Check if the source task file exists
if [ ! -f "$SOURCE_TASK_FILE" ]; then
  echo "ERROR: Source task file '$SOURCE_TASK_FILE' not found"
  exit 1
fi

# Check if the project already exists
if [ ! -d "$PROJECT_DIR" ]; then
  echo "Creating new project structure for '$PROJECT_NAME'..."
  "$BASE_DIR/project_template.sh" "$PROJECT_NAME" "$SOURCE_TASK_FILE"
else
  echo "Project '$PROJECT_NAME' already exists, continuing with task breakdown..."
fi

TASKS_DIR="$PROJECT_DIR/tasks"
TEMP_DIR="$PROJECT_DIR/temp"
BREAKDOWN_RESULT="$TEMP_DIR/task_breakdown_result.md"

# Create a temporary directory
mkdir -p "$TEMP_DIR"

echo "=== Using Claude Code to break down task list ==="
echo "Source task file: $SOURCE_TASK_FILE"
echo "Project: $PROJECT_NAME"
echo ""

# Create the task breakdown instructions
cat > "$TEMP_DIR/breakdown_instructions.md" << EOF
# Task Breakdown Instructions

I need you to analyze the following task list and break it down into individual task files that can be executed independently.

## Source Task List

$(cat "$SOURCE_TASK_FILE")

## Instructions

1. Analyze the task list and identify logical divisions for independent tasks
2. For each task, create a separate markdown file named with a numerical prefix (e.g., 001_task_name.md)
3. Ensure each task file is self-contained with all necessary context
4. The first file should be 000_project_overview.md with a summary of the entire project
5. For each task file, include:
   - Clear title and objective
   - Required context or background
   - Implementation steps
   - Verification methods
   - Acceptance criteria
6. Generate a task_sequence.txt file listing the tasks in execution order

## Output Format

For each task file, provide the filename and content in this format:

### [FILENAME: 000_project_overview.md]
# Project Overview
...content...

### [FILENAME: 001_first_task.md]
# Task 1: First Task
...content...

And so on. After listing all task files, provide a task_sequence.txt file with all filenames in execution order.

This breakdown should ensure each task can be executed in isolation without requiring context from other tasks.
EOF

# Run Claude Code to perform the task breakdown
echo "Running Claude Code to analyze and break down the task list..."
claude code --input "$TEMP_DIR/breakdown_instructions.md" --output "$BREAKDOWN_RESULT"

echo "Task breakdown complete. Results saved to: $BREAKDOWN_RESULT"
echo ""

# Process the breakdown result to extract individual task files
echo "Extracting individual task files from the breakdown..."

# Ensure the tasks directory exists
mkdir -p "$TASKS_DIR"

# Extract task files using awk
awk '/^### \[FILENAME: /{
  if (file != "") {
    close(file)
  }
  match($0, /^### \[FILENAME: ([^]]+)\]/, arr)
  file = "'$TASKS_DIR'/" arr[1]
  next
}
file != "" {
  print $0 > file
}' "$BREAKDOWN_RESULT"

# Extract task_sequence.txt if present
if grep -q "task_sequence.txt" "$BREAKDOWN_RESULT"; then
  echo "Extracting task_sequence.txt..."
  awk '/^### \[FILENAME: task_sequence.txt\]/{flag=1;next} /^### \[FILENAME:/{flag=0} flag' "$BREAKDOWN_RESULT" > "$PROJECT_DIR/task_sequence.txt"
else
  # Generate task_sequence.txt based on filenames
  echo "Generating task_sequence.txt based on filenames..."
  echo "# Task sequence for project: $PROJECT_NAME" > "$PROJECT_DIR/task_sequence.txt"
  echo "# Format: one task filename per line (relative to tasks directory)" >> "$PROJECT_DIR/task_sequence.txt"
  echo "" >> "$PROJECT_DIR/task_sequence.txt"
  find "$TASKS_DIR" -name "*.md" | sort | xargs -n1 basename >> "$PROJECT_DIR/task_sequence.txt"
fi

echo "Task breakdown and extraction complete!"
echo ""
echo "Created the following task files:"
ls -1 "$TASKS_DIR"
echo ""
echo "Task sequence file updated. You can now run the tasks using:"
echo "./run_project.sh $PROJECT_NAME"
echo ""
echo "To view or edit individual tasks, check the files in: $TASKS_DIR"
