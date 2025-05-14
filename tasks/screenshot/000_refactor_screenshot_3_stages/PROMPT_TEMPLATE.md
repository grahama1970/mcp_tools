I need to process a large multi-task project by breaking it down and executing each task with proper context isolation. Please help me by running the following shell commands using Desktop Commander:

# 1. First, break down the large task into individual task files
cd /Users/robert/claude_mcp_configs/tasks/screenshot
./task_breakdown.sh refactor_screenshot_new /Users/robert/claude_mcp_configs/tasks/screenshot/000_refactor_screenshot.md.bak

# 2. Now execute all tasks in sequence, with proper context isolation between tasks
./run_project.sh refactor_screenshot_new