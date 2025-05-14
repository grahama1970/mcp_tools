#!/bin/bash
# project_template.sh - Creates a new project structure with task management

# Check if a project name was provided
if [ -z "$1" ]; then
  echo "ERROR: Please provide a project name"
  echo "Usage: $0 <project_name> [source_task_file]"
  echo "Example: $0 ui_refactor /path/to/big_task.md"
  exit 1
fi

PROJECT_NAME="$1"
SOURCE_TASK_FILE="$2"
BASE_DIR="/Users/robert/claude_mcp_configs/tasks/screenshot"
PROJECT_DIR="$BASE_DIR/${PROJECT_NAME}"
TASKS_DIR="$PROJECT_DIR/tasks"
RESULTS_DIR="$PROJECT_DIR/results"

# Create directory structure
mkdir -p "$PROJECT_DIR"
mkdir -p "$TASKS_DIR"
mkdir -p "$RESULTS_DIR"

echo "=== Creating new project: $PROJECT_NAME ==="
echo "Project directory: $PROJECT_DIR"
echo "Tasks directory: $TASKS_DIR"
echo "Results directory: $RESULTS_DIR"
echo ""

# Create a project.md file for metadata and instructions
cat > "$PROJECT_DIR/project.md" << EOF
# Project: $PROJECT_NAME

This project uses Claude Code with context isolation to execute a series of tasks.

## Structure

- **Project Directory**: \`$PROJECT_DIR\`
  - Contains this project.md file and task_sequence.txt controlling execution order
  
- **Tasks Directory**: \`$TASKS_DIR\`
  - Contains the individual task files to be executed in isolation
  
- **Results Directory**: \`$RESULTS_DIR\`
  - Where task execution results will be stored

## Running the Tasks

To execute all tasks in this project:

\`\`\`bash
cd $BASE_DIR
./run_project.sh $PROJECT_NAME
\`\`\`

To run a specific task from this project:

\`\`\`bash
./run_task.sh $PROJECT_NAME <task_filename>
\`\`\`

## Task Sequence

The execution order is controlled by \`task_sequence.txt\`, which lists the task files in the order they should be executed.
EOF

# Create an empty task_sequence.txt file
cat > "$PROJECT_DIR/task_sequence.txt" << EOF
# Task sequence for project: $PROJECT_NAME
# Format: one task filename per line (relative to tasks directory)
# Lines starting with # are comments and will be ignored
# Empty lines are also ignored

# Example:
# 000_project_overview.md
# 001_first_task.md
# 002_second_task.md
EOF

# Create project-specific run script
cat > "$PROJECT_DIR/run_all.sh" << EOF
#!/bin/bash
# Run all tasks in $PROJECT_NAME project

cd "$(dirname "\$0")/.."
./run_project.sh "$PROJECT_NAME"
EOF
chmod +x "$PROJECT_DIR/run_all.sh"

# If a source task file was provided, process it
if [ -n "$SOURCE_TASK_FILE" ] && [ -f "$SOURCE_TASK_FILE" ]; then
  echo "Setting up project based on source task file: $SOURCE_TASK_FILE"
  
  # Copy the source task file to the project directory
  cp "$SOURCE_TASK_FILE" "$PROJECT_DIR/task_list.md"
  
  # Create a default overview task
  cat > "$TASKS_DIR/000_project_overview.md" << EOF
# Project Overview: $PROJECT_NAME

This file provides an overview of the entire project.

$(head -n 20 "$SOURCE_TASK_FILE" | grep -v "^#" | grep -v "^$" | head -n 5)

...

See the individual task files for detailed instructions on each step of the process.
EOF
  echo "000_project_overview.md" >> "$PROJECT_DIR/task_sequence.txt"
  
  echo "NOTE: Task breakdown requires manual refinement."
  echo "      Please edit the generated tasks in: $TASKS_DIR"
  echo "      And update the sequence in: $PROJECT_DIR/task_sequence.txt"
else
  # Create sample task files
  cat > "$TASKS_DIR/000_project_overview.md" << EOF
# Project Overview: $PROJECT_NAME

This file provides an overview of the entire project.

## Objective

[Describe the main objective of this project]

## Steps

1. First step
2. Second step
3. Third step

See the individual task files for detailed instructions on each step of the process.
EOF
  echo "000_project_overview.md" >> "$PROJECT_DIR/task_sequence.txt"
  
  cat > "$TASKS_DIR/001_first_task.md" << EOF
# Task 1: First Task

**Objective**: [Describe the objective of this task]

## Implementation Steps

- [ ] Step 1
- [ ] Step 2
- [ ] Step 3

## Acceptance Criteria

- Criterion 1
- Criterion 2
- Criterion 3
EOF
  echo "001_first_task.md" >> "$PROJECT_DIR/task_sequence.txt"
  
  cat > "$TASKS_DIR/002_second_task.md" << EOF
# Task 2: Second Task

**Objective**: [Describe the objective of this task]

## Implementation Steps

- [ ] Step 1
- [ ] Step 2
- [ ] Step 3

## Acceptance Criteria

- Criterion 1
- Criterion 2
- Criterion 3
EOF
  echo "002_second_task.md" >> "$PROJECT_DIR/task_sequence.txt"
  
  echo "Created sample project structure."
  echo "Please edit the task files in: $TASKS_DIR"
  echo "And update the sequence in: $PROJECT_DIR/task_sequence.txt"
fi

echo ""
echo "=== Project creation complete ==="
echo "To run all tasks in this project: ./run_project.sh $PROJECT_NAME"
echo "Or from within the project directory: $PROJECT_DIR/run_all.sh"
