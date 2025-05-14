# Task Context Management Instructions

## IMPORTANT: FOCUS ONLY ON THE CURRENT TASK

When working on tasks through the task runner system:

1. **Focus EXCLUSIVELY on the current task file provided**
   - Do not attempt to access or reference other tasks
   - Treat each task as if it's the only task in the repository

2. **Maintain context isolation**
   - Consider only the information in the current file
   - If you need additional context, explicitly request it or look only at files directly referenced in the task

3. **When complete:**
   - Summarize your changes concisely at the end of your response
   - Clearly indicate when the task is complete with "TASK COMPLETE" marker

4. **Memory Management:**
   - If context becomes full during task execution, use `/compact` to preserve important information
   - Use `/clear` command when switching between major phases of the task
   - Remember to preserve key context when using these commands

## Context Management Commands

Claude Code offers these commands to help manage context:

- `/clear` - Completely clears conversation context (use with caution)
- `/compact` - Intelligently compresses the context to save space while preserving key information

## Important Notice

These instructions are designed to help Claude Code provide the best possible answers with optimal use of the available context window. By focusing exclusively on one task at a time, you'll produce higher quality results than if you try to hold the entire project in memory at once.
