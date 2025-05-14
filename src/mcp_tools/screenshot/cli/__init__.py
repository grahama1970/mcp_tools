"""
CLI Layer for Screenshot Module

This package contains the CLI (Command Line Interface) layer for the screenshot functionality,
providing a rich interface for human users.

The CLI layer is designed to:
1. Handle user interaction concerns
2. Format outputs for human readability
3. Parse and validate command-line arguments
4. Implement CLI-specific error handling

Usage:
    from mcp_tools.screenshot.cli import app as screenshot_app
    
    # Run the CLI app
    screenshot_app()
    
    # Alternative: use formatters directly
    from mcp_tools.screenshot.cli.formatters import print_screenshot_result
    result = capture_screenshot()
    print_screenshot_result(result)
"""

# CLI application
from mcp_tools.screenshot.cli.cli import app

# Formatters for rich output
from mcp_tools.screenshot.cli.formatters import (
    print_screenshot_result,
    print_description_result,
    print_regions_table,
    print_error,
    print_warning,
    print_info,
    print_json,
    create_progress,
    console
)

# CLI validators
from mcp_tools.screenshot.cli.validators import (
    validate_quality_option,
    validate_region_option,
    validate_output_dir,
    validate_model_option,
    validate_prompt_option,
    validate_credentials_file,
    validate_file_exists,
    validate_json_output
)

# Schema definitions and validation
from mcp_tools.screenshot.cli.schemas import (
    generate_cli_schema,
    convert_to_mcp_schema,
    format_cli_response,
    validate_output_against_schema
)

__all__ = [
    # CLI application
    'app',
    
    # Formatters
    'print_screenshot_result',
    'print_description_result',
    'print_regions_table',
    'print_error',
    'print_warning',
    'print_info',
    'print_json',
    'create_progress',
    'console',
    
    # Validators
    'validate_quality_option',
    'validate_region_option',
    'validate_output_dir',
    'validate_model_option',
    'validate_prompt_option',
    'validate_credentials_file',
    'validate_file_exists',
    'validate_json_output',
    
    # Schemas
    'generate_cli_schema',
    'convert_to_mcp_schema',
    'format_cli_response',
    'validate_output_against_schema'
]

# Example usage
if __name__ == "__main__":
    print("""
Example usage of the screenshot CLI:

# Take a screenshot
python -m mcp_tools.screenshot.presentation.cli screenshot --quality 70

# Take a screenshot of a specific region
python -m mcp_tools.screenshot.presentation.cli capture region right_half

# Take a screenshot and get AI description
python -m mcp_tools.screenshot.presentation.cli describe --prompt "What's in this image?"

# Show available screen regions
python -m mcp_tools.screenshot.presentation.cli tools regions

# Show version information
python -m mcp_tools.screenshot.presentation.cli tools version

# Output in JSON format (for all commands)
python -m mcp_tools.screenshot.presentation.cli screenshot --json
""")
