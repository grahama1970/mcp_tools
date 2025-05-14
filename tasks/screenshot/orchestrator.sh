#!/bin/bash
# orchestrator.sh - Runs a single Claude Code task with isolated context

TASK_FILE=$1
RESULTS_DIR="${2:-/Users/robert/claude_mcp_configs/tasks/screenshot/task_results}"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# Get the filename without path for use in the output file
TASK_NAME=$(basename "$TASK_FILE")

echo "=== Starting task: $TASK_NAME ==="

# Create a new separate Claude Code session for this task
# Each task will start with a clean context window
claude code --input "$TASK_FILE" --output "$RESULTS_DIR/${TASK_NAME%.md}.result" <<EOF
/clear
# Focus on this task only
Please focus exclusively on the task described in this file.
Do not attempt to access previous context or tasks.
EOF

echo "=== Task complete: $TASK_NAME ==="
echo "Result saved to: $RESULTS_DIR/${TASK_NAME%.md}.result"
