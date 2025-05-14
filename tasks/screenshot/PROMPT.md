# Claude Task Manager Using Desktop Commander

I need to process a large multi-task project by breaking it down and executing each task with proper context isolation. 

## Prerequisites
This process requires Desktop Commander to be installed and working in Claude Desktop. I can see the hammer icon in the chat interface, indicating Desktop Commander is properly connected.

## Running the Tasks
Please help me by running the following Python script using Desktop Commander:

```bash
cd /Users/robert/claude_mcp_configs/tasks/screenshot
python claude_tasks.py breakdown refactor_screenshot_new --source /Users/robert/claude_mcp_configs/tasks/screenshot/000_refactor_screenshot.md.bak
python claude_tasks.py run refactor_screenshot_new
```

The claude_tasks.py script handles:
1. Breaking down the large task into individual task files
2. Running each task in sequence with proper context isolation (/clear between tasks)
3. Saving results to a dedicated results directory

This approach uses a Python implementation of the Boomerang pattern to maintain context isolation between tasks.
