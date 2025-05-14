# Task 3: Implement MCP-Quality CLI

**Objective**: Create a command-line interface that meets MCP-level documentation and usability standards.

**Pre-Task Setup**:
- Review the MCP-quality CLI requirements from our conversation

**Implementation Steps**:
- [ ] 3.1 Create cli.py module with Typer app setup and imports
- [ ] 3.2 Implement `screenshot` command with rich help documentation
- [ ] 3.3 Implement `describe` command with parameter annotations
- [ ] 3.4 Implement `screenshot_and_describe` combined command
- [ ] 3.5 Add `schema` command for machine-readable documentation
- [ ] 3.6 Add `regions` utility command
- [ ] 3.7 Implement consistent error handling across commands
- [ ] 3.8 Add comprehensive examples to help text
- [ ] 3.9 Test CLI commands with various parameters
- [ ] 3.10 Git commit verified CLI with message "Add: MCP-quality CLI with rich documentation and schema"

**Technical Specifications**:
- Follow MCP-level documentation standards:
  - Use Annotated[] with typer.Option for all parameters
  - Add rich_help_panel grouping for related options
  - Include min/max validation where appropriate
  - Add detailed examples in docstrings
- Implement structured JSON output mode for all commands
- Ensure consistent error handling with appropriate exit codes
- Add schema command that outputs MCP-compatible JSON schema

**Verification Method**:
- Test CLI help: `python -m mcp_tools.screenshot.cli --help`
- Test each command with `--help` flag
- Test schema output: `python -m mcp_tools.screenshot.cli schema`
- Verify JSON output mode works correctly

**Acceptance Criteria**:
- CLI provides rich help documentation comparable to MCP
- All commands function correctly with proper parameters
- Schema command outputs valid, MCP-compatible JSON
- Error handling is consistent across all commands
- CLI can be used by both humans and automated tools

## MCP-Quality CLI Implementation

The CLI will follow MCP-level documentation standards:

**Rich Help Documentation Example**:
```python
@app.command()
def screenshot(
    quality: Annotated[int, typer.Option(
        help="JPEG compression quality (1-100). Higher values = larger files.",
        min=1, max=100,
        rich_help_panel="Image Options"
    )] = 30,
    region: Annotated[str, typer.Option(
        help="Screen region to capture: 'full', 'right_half', or coordinates (format: 'x,y,width,height')",
        rich_help_panel="Image Options"
    )] = "full",
    output_dir: Annotated[str, typer.Option(
        help="Directory to save the screenshot",
        rich_help_panel="Output Options"
    )] = "screenshots",
    json_output: Annotated[bool, typer.Option(
        help="Output results as JSON instead of rich text",
        rich_help_panel="Output Options"
    )] = False,
):
    """
    Capture a screenshot of the desktop or a specific region.
    
    The screenshot is saved as a JPEG file with the specified quality.
    
    Examples:
        screenshot --quality 50 --region right_half
        screenshot --region "100,100,400,300" --output-dir my_screenshots
        screenshot --json-output
    """
```

**Machine-Readable Schema Example**:
```python
@app.command(hidden=True)
def schema():
    """Output the CLI schema in a machine-readable format."""
    schema = {
        "name": "screenshot_cli",
        "version": "1.0.0",
        "commands": {
            "screenshot": {
                "description": "Capture a screenshot of the desktop or a specific region.",
                "parameters": {
                    "quality": {
                        "type": "integer",
                        "description": "JPEG compression quality (1-100). Higher values = larger files.",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 100
                    },
                    "region": {
                        "type": "string",
                        "description": "Screen region to capture: 'full', 'right_half', or coordinates (format: 'x,y,width,height')",
                        "default": "full"
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save the screenshot",
                        "default": "screenshots"
                    }
                },
                "returns": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "Path to the saved screenshot file"},
                        "dimensions": {"type": "array", "description": "Width and height of the image"},
                        "size": {"type": "integer", "description": "Size in bytes of the saved file"}
                    }
                },
                "examples": [
                    {"description": "Capture right half of screen", "command": "screenshot --quality 50 --region right_half"},
                    {"description": "Capture custom region", "command": "screenshot --region \"100,100,400,300\""}
                ]
            }
        }
    }
    typer.echo(json.dumps(schema, indent=2))
```
