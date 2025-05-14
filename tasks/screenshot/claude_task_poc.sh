#!/bin/bash

# Claude Task Manager POC
# This script demonstrates using claude to process tasks with context isolation
# Complete workflow: task list → individual tasks → process with context isolation

# Configuration
TASKS_DIR="$HOME/claude_mcp_configs/tasks"
RESULTS_DIR="$HOME/claude_mcp_configs/results"
LOG_FILE="$HOME/claude_mcp_configs/task_manager.log"
CLAUDE_PATH="/Users/robert/.npm-global/bin/claude"
TIMEOUT=300  # 5-minute timeout for Claude tasks

# Create directories if they don't exist
mkdir -p "$TASKS_DIR" "$RESULTS_DIR"

# Initialize log file
echo "[$(date)] Starting Claude Task Manager POC" > "$LOG_FILE"

# Example task list (this would normally be a file)
TASK_LIST="# Task List

## Task 1: Data Analysis

### Objective
Analyze the following data and provide key insights.

### Data
Here is a sample dataset of sales figures:

| Month  | Sales (USD) |
|--------|-------------|
| Jan    | 10,200      |
| Feb    | 11,500      |
| Mar    | 9,800       |
| Apr    | 12,300      |

### Questions
1. What is the total sales amount?
2. What is the average monthly sales?
3. Which month had the highest sales?
4. What recommendations would you make based on this data?

---

## Task 2: Code Review

### Objective
Review the following Python code and suggest improvements.

### Code
\`\`\`python
def process_data(data_list):
    result = []
    for i in range(len(data_list)):
        if data_list[i] > 0:
            result.append(data_list[i] * 2)
        else:
            result.append(0)
    return result
    
# Example usage
data = [5, -3, 10, 0, 8, -2]
processed = process_data(data)
print(processed)
\`\`\`

### Questions
1. What improvements would you suggest for this code?
2. Are there any potential bugs or edge cases?
3. How would you make this code more efficient?
4. Provide a rewritten version with your improvements.

---

## Task 3: Context Isolation Test

### Objective
This task tests whether Claude Code maintains context isolation between tasks.

### Previous Tasks
If you can access information from previous tasks, please mention it. If not, please state that you cannot access previous task contexts.

### Questions
1. Can you recall the sales figure for January from the first task?
2. Can you recall the Python function name from the second task?
3. Please explain whether context isolation is working properly based on your answers above.
"

# Function to log messages
log_message() {
    local level="$1"
    local message="$2"
    echo "[$(date)] [$level] $message" | tee -a "$LOG_FILE"
}

# Function to kill all Claude related processes
kill_claude_processes() {
    log_message "INFO" "Killing all Claude related processes"
    echo "🔪 Terminating all Claude processes..."
    
    # Use pkill to kill all processes with Claude in the name (case insensitive)
    pkill -f -9 "[cC]laude" 2>/dev/null
    
    # Also try to kill specific process patterns
    pkill -f -9 "Claude.app" 2>/dev/null
    pkill -f -9 "Claude Helper" 2>/dev/null
    
    # Wait a moment to ensure processes are terminated
    sleep 2
    
    # Check if any Claude processes remain
    if pgrep -f "[cC]laude" > /dev/null; then
        log_message "WARN" "Some Claude processes still running after pkill"
        local remaining_processes=$(ps -ef | grep -i "claude" | grep -v grep | wc -l)
        echo "⚠️ $remaining_processes Claude processes still running"
        
        # Try more forceful termination for stubborn processes
        log_message "INFO" "Attempting more forceful termination of remaining processes"
        ps -ef | grep -i "claude" | grep -v grep | awk '{print $2}' | xargs -I{} kill -9 {} 2>/dev/null
        
        # Check again
        sleep 1
        if pgrep -f "[cC]laude" > /dev/null; then
            log_message "ERROR" "Failed to kill all Claude processes"
            echo "❌ Failed to terminate all Claude processes"
        else
            log_message "INFO" "Successfully terminated all remaining Claude processes"
            echo "✅ All Claude processes terminated"
        fi
    else
        log_message "INFO" "All Claude processes successfully terminated"
        echo "✅ All Claude processes terminated"
    fi
}

# Function to split task list into individual task files
split_task_list() {
    log_message "INFO" "Splitting task list into individual task files"
    
    # Clear existing task files
    rm -f "$TASKS_DIR"/*.md
    
    # Create a temporary file with the task list
    local task_list_file="$HOME/claude_mcp_configs/task_list.md"
    echo "$TASK_LIST" > "$task_list_file"
    
    # Extract task sections using markers (---) and task headers (## Task)
    local task_count=0
    local current_task=""
    local task_title=""
    
    # Read the task list line by line
    while IFS= read -r line || [ -n "$line" ]; do
        # Check if this is a task header
        if [[ "$line" =~ ^"## Task "([0-9]+): ]]; then
            # If we have a previous task, save it
            if [ -n "$current_task" ] && [ -n "$task_title" ]; then
                task_count=$((task_count + 1))
                task_file=$(printf "%s/00%d_%s.md" "$TASKS_DIR" "$task_count" "$(echo "$task_title" | tr '[:upper:] ' '[:lower:]_')")
                echo "$current_task" > "$task_file"
                log_message "INFO" "Created task file: $(basename "$task_file")"
            fi
            
            # Start a new task
            task_title=$(echo "$line" | sed 's/^## Task [0-9]\+: //')
            current_task="# $task_title\n\n"
        elif [ -n "$task_title" ]; then
            # If we're inside a task, add the line to the current task
            if [[ "$line" == "---" ]]; then
                # Skip separator lines
                continue
            else
                current_task+="$line\n"
            fi
        fi
    done < "$task_list_file"
    
    # Save the last task
    if [ -n "$current_task" ] && [ -n "$task_title" ]; then
        task_count=$((task_count + 1))
        task_file=$(printf "%s/00%d_%s.md" "$TASKS_DIR" "$task_count" "$(echo "$task_title" | tr '[:upper:] ' '[:lower:]_')")
        echo -e "$current_task" > "$task_file"
        log_message "INFO" "Created task file: $(basename "$task_file")"
    fi
    
    log_message "INFO" "Created $task_count task files in $TASKS_DIR"
    
    # Remove temporary task list file
    rm -f "$task_list_file"
}

# Check if claude is installed and available
check_claude() {
    log_message "INFO" "Checking for claude at specific path: $CLAUDE_PATH"
    
    # Check if claude is at the specified path
    if [ ! -f "$CLAUDE_PATH" ] || [ ! -x "$CLAUDE_PATH" ]; then
        log_message "WARN" "claude not found at $CLAUDE_PATH or not executable"
        echo "⚠️ WARN: claude not found at $CLAUDE_PATH or not executable"
        return 1
    fi
    
    log_message "INFO" "claude is available at $CLAUDE_PATH"
    echo "✅ claude is available at $CLAUDE_PATH"
    return 0
}

# Function to run a task
run_task() {
    local task_file="$1"
    local task_name=$(basename "$task_file" .md)
    local result_file="$RESULTS_DIR/${task_name}.result"
    local error_file="$RESULTS_DIR/${task_name}.error"
    
    log_message "INFO" "Processing task: $task_name"
    echo "📋 Processing task: $task_name"
    
    # Kill any existing Claude processes before starting
    kill_claude_processes
    
    log_message "INFO" "Starting claude for task: $task_name"
    echo "🚀 Starting claude for task: $task_name"
    
    # Run Claude with timeout to prevent hanging
    log_message "INFO" "Running claude with content from: $task_file"
    timeout $TIMEOUT bash -c "cat \"$task_file\" | \"$CLAUDE_PATH\" --print > \"$result_file\" 2> \"$error_file\""
    local exit_code=$?
    
    # Check if process timed out
    if [ $exit_code -eq 124 ]; then
        log_message "ERROR" "Task $task_name timed out after $TIMEOUT seconds"
        echo "⏱️ Task timed out after $TIMEOUT seconds"
        echo "Task execution for $task_name failed (timeout)" > "$result_file"
    elif [ $exit_code -eq 0 ]; then
        log_message "INFO" "Task completed successfully: $task_name"
        echo "✅ Task completed successfully: $task_name"
        echo "📝 Result saved to: $result_file"
        
        # Show a preview of the result
        echo ""
        echo "📄 Result preview (first 5 lines):"
        echo "---------------------------------------------"
        head -n 5 "$result_file"
        echo "... (truncated for brevity) ..."
        echo "---------------------------------------------"
        echo ""
    else
        log_message "ERROR" "Task failed with exit code $exit_code: $task_name"
        echo "❌ Task failed with exit code $exit_code: $task_name"
        
        # Provide detailed error information
        if [ -s "$error_file" ]; then
            log_message "ERROR" "Error details: $(cat "$error_file")"
            echo "📋 Error details:"
            cat "$error_file"
        fi
    fi
    
    # Always kill Claude processes after task completion
    kill_claude_processes
    
    return $exit_code
}

# Function to list all tasks
list_tasks() {
    echo "Available tasks:"
    ls -1 "$TASKS_DIR"/*.md 2>/dev/null | sed 's/.*\///' | sed 's/\.md$//' | nl
}

# Function to process all tasks sequentially
process_all_tasks() {
    log_message "INFO" "Processing all tasks..."
    echo "🔄 Processing all tasks..."
    
    # Check if there are any tasks
    if [ ! "$(ls -A "$TASKS_DIR"/*.md 2>/dev/null)" ]; then
        log_message "ERROR" "No tasks found in $TASKS_DIR"
        echo "❌ No tasks found in $TASKS_DIR"
        return 1
    fi
    
    # Kill any existing Claude processes before starting
    kill_claude_processes
    
    # Track success/failure counts
    local total=0
    local success=0
    local failed=0
    
    for task_file in "$TASKS_DIR"/*.md; do
        ((total++))
        
        # Run the task
        run_task "$task_file"
        
        if [ $? -eq 0 ]; then
            ((success++))
        else
            ((failed++))
        fi
        
        # Add a delay between tasks to ensure clean state
        sleep 5
    done
    
    log_message "INFO" "All tasks completed. Success: $success, Failed: $failed, Total: $total"
    echo "🎉 All tasks completed. Success: $success, Failed: $failed, Total: $total"
}

# At script start, kill any existing Claude processes
kill_claude_processes

# Display menu (for information only)
echo "====================================="
echo "   Claude Task Manager POC"
echo "====================================="
echo "1. Split task list into individual tasks"
echo "2. List available tasks"
echo "3. Process all tasks"
echo "4. Exit"
echo "====================================="

# Check if claude is available
if ! check_claude; then
    log_message "ERROR" "Cannot proceed without claude"
    echo "❌ Cannot proceed without claude"
    echo "Please ensure Claude Code is installed correctly and try again"
    exit 1
fi

# Split the task list into individual task files
echo "🔪 Splitting task list into individual files..."
split_task_list

# Display available tasks
echo ""
echo "📋 Available tasks after splitting:"
list_tasks
echo ""

# Automatically run process_all_tasks without requiring input
echo "🚀 Auto-selecting: Process all tasks"
echo ""
process_all_tasks

# Final cleanup - make sure all Claude processes are terminated
kill_claude_processes

log_message "INFO" "Script completed"
echo "Script completed. Exiting..."
exit 0