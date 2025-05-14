# Screenshot Tool Refactoring Project Overview

**Objective**: Refactor the MCP screenshot tool with a layered architecture that ensures each function is independently debuggable, creates an MCP-quality CLI, and wraps it with FastMCP.

**Requirements**:
1. Each Python function must be independently debuggable with clear sample I/O
2. CLI must follow MCP-level quality standards (schema, rich help, etc.)
3. Separate core functions from presentation and MCP concerns
4. Use Rich tables for formatted output
5. Wrap CLI with FastMCP for seamless integration

## Architecture Overview

The refactoring will implement a three-layer architecture:
1. **Core Layer**: Pure Python functions with no dependencies on CLI/MCP
2. **Presentation Layer**: Rich formatters and Typer CLI with MCP-quality documentation
3. **Integration Layer**: FastMCP wrapper that leverages the CLI functions

This design ensures each component is independently testable and debuggable, while providing a seamless user experience across interfaces.

## Task List

The project has been broken down into the following tasks:

1. **Task 1: Refactor Core Functions** (001_core_functions.md)
   - Create independently debuggable core functions
   - Implement error handling and validation
   - Add debug demonstrations

2. **Task 2: Create Rich Formatters** (002_rich_formatters.md)
   - Implement Rich tables for output formatting
   - Create error and result formatters
   - Ensure consistent styling

3. **Task 3: Implement MCP-Quality CLI** (003_mcp_quality_cli.md)
   - Create Typer-based CLI with rich help
   - Implement machine-readable schema
   - Add comprehensive examples

4. **Task 4: Create FastMCP Wrapper** (004_fastmcp_wrapper.md)
   - Integrate with FastMCP
   - Adapt CLI functions for MCP interface
   - Implement parameter conversion

5. **Task 5: Update Server Entry Point** (005_server_entry.md)
   - Create enhanced server entry point
   - Add debug and configuration options
   - Update .mcp.json configuration

6. **Task 6: Integration Testing** (006_integration_testing.md)
   - Test with MCP Inspector
   - Verify all functionality
   - Document issues and results

## Usage Table

| Command / Function | Description | Example Usage | Expected Output |
|-------------------|-------------|---------------|-----------------|
| Core Functions |
| `capture_screenshot()` | Capture a screenshot | `capture_screenshot(quality=50, region="right_half")` | `{"file": "path/to/file.jpg", "dimensions": [800, 600], "size": 102400}` |
| `describe_image()` | Describe an image | `describe_image("path/to/image.jpg")` | `{"description": "...", "confidence": 4}` |
| CLI Commands |
| `screenshot` | Take a screenshot | `screenshot --quality 50 --region right_half` | Rich table with screenshot details |
| `describe` | Describe an image | `describe path/to/image.jpg` | Rich table with description |
| `screenshot_and_describe` | Capture & describe | `screenshot_and_describe --region right_half` | Rich tables with screenshot and description |
| `schema` | Get machine-readable docs | `schema` | JSON schema of all commands |
| MCP Tools |
| `screenshot` | MCP screenshot tool | N/A (used via MCP Inspector) | Screenshot result with optional description |
| `describe_screenshot` | MCP describe tool | N/A (used via MCP Inspector) | Screenshot with description |

## Version Control Plan

- **Initial Commit**: Create before starting implementation with message "Task: Begin screenshot tool refactoring"
- **Function Commits**: After each major function is implemented with descriptive messages
- **Task Commits**: After completing each major task as specified in the steps
- **Final Tag**: Create git tag "screenshot-tool-refactored-v1.0" upon completion
