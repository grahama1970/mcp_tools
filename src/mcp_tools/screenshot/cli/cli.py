#!/usr/bin/env python3
"""
Command Line Interface for Screenshot Module

This module provides a CLI for the screenshot functionality using Typer and Rich,
allowing users to capture screenshots and obtain image descriptions.

This module is part of the Presentation Layer and should only depend on
Core Layer components, not on Integration Layer.

Sample input:
- CLI commands with options

Expected output:
- Formatted console output of operation results
- Files saved to disk
- Structured JSON output for machine consumption
- Schema information for automated tools
"""

import os
import sys
import json
from typing import Optional, List, Union, Dict, Any, Tuple

import typer
from rich.prompt import Prompt, Confirm
from loguru import logger

from mcp_tools.screenshot.core.constants import (
    IMAGE_SETTINGS, 
    REGION_PRESETS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT
)
from mcp_tools.screenshot.core.capture import capture_screenshot, get_screen_regions
from mcp_tools.screenshot.core.description import describe_image_content, find_credentials_file
from mcp_tools.screenshot.cli.formatters import (
    print_screenshot_result,
    print_description_result,
    print_regions_table,
    print_error,
    print_warning,
    print_info,
    print_json,
    create_progress
)
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
from mcp_tools.screenshot.cli.schemas import (
    generate_cli_schema,
    convert_to_mcp_schema,
    format_cli_response
)


# Initialize typer app with command groups
app = typer.Typer(
    help="Claude MCP Screenshot Tool with MSS",
    rich_markup_mode="rich",
    add_completion=False
)

# Create command groups
capture_app = typer.Typer(help="Screenshot capture commands", rich_markup_mode="rich")
vision_app = typer.Typer(help="Vision and description commands", rich_markup_mode="rich")
tools_app = typer.Typer(help="Utility tools", rich_markup_mode="rich")

app.add_typer(capture_app, name="capture", help="Screenshot capture commands")
app.add_typer(vision_app, name="vision", help="Vision and description commands")
app.add_typer(tools_app, name="tools", help="Utility tools")


@app.callback()
def main(
    ctx: typer.Context,
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output results as JSON",
        callback=validate_json_output
    ),
):
    """
    Claude MCP Screenshot Tool - Captures and describes screenshots

    This tool provides commands for capturing screenshots and getting AI-powered
    descriptions of screen content.
    """
    # Initialize context object to store shared state
    ctx.ensure_object(dict)
    ctx.obj["json_output"] = json_output


@app.command("screenshot")
def screenshot_command(
    ctx: typer.Context,
    quality: int = typer.Option(
        IMAGE_SETTINGS["DEFAULT_QUALITY"], 
        "--quality", "-q",
        help="JPEG compression quality (1-100)",
        callback=validate_quality_option
    ),
    region: Optional[str] = typer.Option(
        None, 
        "--region", "-r",
        help="Screen region to capture: 'full', 'right_half', 'left_half', etc. or 'x,y,width,height'",
        callback=validate_region_option
    ),
    output: Optional[str] = typer.Option(
        None, 
        "--output", "-o",
        help="Output file path. If not provided, saves to screenshots directory.",
        callback=validate_output_dir
    ),
    raw: bool = typer.Option(
        False, 
        "--raw",
        help="Save raw PNG in addition to compressed JPEG"
    )
):
    """
    Take a screenshot of the screen or a specific region.
    
    This command captures the screen or a region and saves it as a JPEG image.
    """
    try:
        json_output = ctx.obj.get("json_output", False)
        
        # Create progress indicator if not in JSON mode
        if not json_output:
            with create_progress("Taking screenshot") as progress:
                task = progress.add_task("Capturing screen...", total=100)
                progress.update(task, completed=30)
                
                # Take screenshot
                result = capture_screenshot(
                    quality=quality, 
                    region=region, 
                    output_dir=output,
                    include_raw=raw
                )
                
                progress.update(task, completed=100, description="Screenshot captured")
        else:
            # Take screenshot without progress indicator
            result = capture_screenshot(
                quality=quality, 
                region=region, 
                output_dir=output,
                include_raw=raw
            )
            
        # Check for errors
        if "error" in result:
            if json_output:
                print_json(result)
            else:
                print_error(result["error"])
            sys.exit(1)
            
        # Format result
        if json_output:
            # In JSON mode, exclude large base64 content
            if "content" in result:
                del result["content"]
            response = format_cli_response(True, data=result)
            print_json(response)
        else:
            print_screenshot_result(result)
            
    except Exception as e:
        logger.error(f"Screenshot command failed: {str(e)}")
        if json_output:
            response = format_cli_response(False, error=str(e))
            print_json(response)
        else:
            print_error(f"Screenshot failed: {str(e)}")
        sys.exit(1)


@app.command("describe")
def describe_command(
    ctx: typer.Context,
    quality: int = typer.Option(
        IMAGE_SETTINGS["DEFAULT_QUALITY"], 
        "--quality", "-q",
        help="JPEG compression quality (1-100)",
        callback=validate_quality_option
    ),
    region: Optional[str] = typer.Option(
        None, 
        "--region", "-r",
        help="Screen region to capture: 'full', 'right_half', 'left_half', etc. or 'x,y,width,height'",
        callback=validate_region_option
    ),
    prompt: str = typer.Option(
        DEFAULT_PROMPT, 
        "--prompt", "-p",
        help="Prompt for the AI description",
        callback=validate_prompt_option
    ),
    model: str = typer.Option(
        DEFAULT_MODEL, 
        "--model", "-m",
        help="AI model to use for description",
        callback=validate_model_option
    ),
    credentials: Optional[str] = typer.Option(
        None,
        "--credentials", "-c",
        help="Path to API credentials file",
        callback=validate_credentials_file
    )
):
    """
    Take a screenshot and get an AI description of the content.
    
    This command captures the screen or a region and uses an AI model to
    describe the content.
    """
    try:
        json_output = ctx.obj.get("json_output", False)
        
        # Create progress indicator if not in JSON mode
        if not json_output:
            with create_progress("Processing screenshot") as progress:
                task = progress.add_task("Capturing screen...", total=100)
                
                # Take screenshot
                screenshot_result = capture_screenshot(quality=quality, region=region)
                progress.update(task, completed=30, description="Processing image...")
                
                if "error" in screenshot_result:
                    progress.update(task, completed=100, description="Error")
                    print_error(f"Error taking screenshot: {screenshot_result['error']}")
                    sys.exit(1)
                
                # Get credentials file
                if not credentials:
                    credentials = find_credentials_file()
                    if not credentials:
                        progress.update(task, completed=100, description="Error")
                        print_warning("Credentials file not found. Description may fail.")
                
                progress.update(task, completed=60, description="Getting description...")
                
                # Get description
                description_result = describe_image_content(
                    image_path=screenshot_result["file"],
                    model=model,
                    prompt=prompt,
                    credentials_file=credentials
                )
                
                progress.update(task, completed=100, description="Completed")
        else:
            # Take screenshot without progress indicator
            screenshot_result = capture_screenshot(quality=quality, region=region)
            
            if "error" in screenshot_result:
                print_json({"error": f"Error taking screenshot: {screenshot_result['error']}"})
                sys.exit(1)
            
            # Get credentials file
            if not credentials:
                credentials = find_credentials_file()
            
            # Get description
            description_result = describe_image_content(
                image_path=screenshot_result["file"],
                model=model,
                prompt=prompt,
                credentials_file=credentials
            )
        
        # Check for errors in description
        if "error" in description_result:
            if json_output:
                result = {
                    "error": f"Error getting description: {description_result['error']}",
                    "file": screenshot_result.get("file")
                }
                print_json(result)
            else:
                print_error(f"Error getting description: {description_result['error']}")
            sys.exit(1)
        
        # Format result
        if json_output:
            result = {
                "description": description_result,
                "file": screenshot_result.get("file")
            }
            response = format_cli_response(True, data=result)
            print_json(response)
        else:
            print_description_result(description_result)
            print_info(f"Screenshot saved to: {screenshot_result['file']}")
            
    except Exception as e:
        logger.error(f"Describe command failed: {str(e)}")
        if json_output:
            response = format_cli_response(False, error=str(e))
            print_json(response)
        else:
            print_error(f"Description failed: {str(e)}")
        sys.exit(1)


@capture_app.command("full")
def capture_full(
    ctx: typer.Context,
    quality: int = typer.Option(
        IMAGE_SETTINGS["DEFAULT_QUALITY"], 
        "--quality", "-q",
        help="JPEG compression quality (1-100)",
        callback=validate_quality_option
    ),
    output: Optional[str] = typer.Option(
        None, 
        "--output", "-o",
        help="Output file path. If not provided, saves to screenshots directory.",
        callback=validate_output_dir
    ),
    raw: bool = typer.Option(
        False, 
        "--raw",
        help="Save raw PNG in addition to compressed JPEG"
    )
):
    """
    Take a screenshot of the entire screen (all monitors).
    """
    # Call main screenshot command with full screen
    screenshot_command(ctx, quality, None, output, raw)


@capture_app.command("region")
def capture_region(
    ctx: typer.Context,
    region: str = typer.Argument(
        ...,
        help="Screen region to capture: 'right_half', 'left_half', etc. or 'x,y,width,height'",
        callback=validate_region_option
    ),
    quality: int = typer.Option(
        IMAGE_SETTINGS["DEFAULT_QUALITY"], 
        "--quality", "-q",
        help="JPEG compression quality (1-100)",
        callback=validate_quality_option
    ),
    output: Optional[str] = typer.Option(
        None, 
        "--output", "-o",
        help="Output file path. If not provided, saves to screenshots directory.",
        callback=validate_output_dir
    ),
    raw: bool = typer.Option(
        False, 
        "--raw",
        help="Save raw PNG in addition to compressed JPEG"
    )
):
    """
    Take a screenshot of a specific screen region.
    """
    # Call main screenshot command with specific region
    screenshot_command(ctx, quality, region, output, raw)


@vision_app.command("describe")
def vision_describe(
    ctx: typer.Context,
    image_path: str = typer.Argument(
        ...,
        help="Path to image file to describe",
        callback=validate_file_exists
    ),
    prompt: str = typer.Option(
        DEFAULT_PROMPT, 
        "--prompt", "-p",
        help="Prompt for the AI description",
        callback=validate_prompt_option
    ),
    model: str = typer.Option(
        DEFAULT_MODEL, 
        "--model", "-m",
        help="AI model to use for description",
        callback=validate_model_option
    ),
    credentials: Optional[str] = typer.Option(
        None,
        "--credentials", "-c",
        help="Path to API credentials file",
        callback=validate_credentials_file
    )
):
    """
    Get an AI description of an existing image file.
    """
    try:
        json_output = ctx.obj.get("json_output", False)
        
        # Get credentials file
        if not credentials:
            credentials = find_credentials_file()
            if not credentials and not json_output:
                print_warning("Credentials file not found. Description may fail.")
        
        # Create progress indicator if not in JSON mode
        if not json_output:
            with create_progress("Getting image description") as progress:
                task = progress.add_task("Processing image...", total=100)
                progress.update(task, completed=30)
                
                # Get description
                result = describe_image_content(
                    image_path=image_path,
                    model=model,
                    prompt=prompt,
                    credentials_file=credentials
                )
                
                progress.update(task, completed=100, description="Completed")
        else:
            # Get description without progress indicator
            result = describe_image_content(
                image_path=image_path,
                model=model,
                prompt=prompt,
                credentials_file=credentials
            )
        
        # Check for errors
        if "error" in result:
            if json_output:
                print_json({"error": f"Error getting description: {result['error']}"})
            else:
                print_error(f"Error getting description: {result['error']}")
            sys.exit(1)
        
        # Format result
        if json_output:
            response = format_cli_response(True, data={"description": result})
            print_json(response)
        else:
            print_description_result(result)
            
    except Exception as e:
        logger.error(f"Vision describe command failed: {str(e)}")
        if json_output:
            response = format_cli_response(False, error=str(e))
            print_json(response)
        else:
            print_error(f"Description failed: {str(e)}")
        sys.exit(1)


@tools_app.command("regions")
def show_regions(ctx: typer.Context):
    """
    Show available screen region presets and dimensions.
    """
    try:
        json_output = ctx.obj.get("json_output", False)
        
        # Get screen regions
        regions = get_screen_regions()
        
        # Format result
        if json_output:
            response = format_cli_response(True, data={"regions": regions})
            print_json(response)
        else:
            print_regions_table(regions)
            
    except Exception as e:
        logger.error(f"Show regions command failed: {str(e)}")
        if json_output:
            response = format_cli_response(False, error=str(e))
            print_json(response)
        else:
            print_error(f"Failed to get screen regions: {str(e)}")
        sys.exit(1)


@tools_app.command("version")
def show_version(ctx: typer.Context):
    """
    Show version information.
    """
    # Version information
    version_info = {
        "name": "Claude MCP Screenshot Tool",
        "version": "1.0.0",
        "description": "A modular screenshot capture and description tool.",
        "author": "Anthropic",
    }
    
    json_output = ctx.obj.get("json_output", False)
    
    if json_output:
        response = format_cli_response(True, data=version_info)
        print_json(response)
    else:
        print_info(
            f"Name: {version_info['name']}\n"
            f"Version: {version_info['version']}\n"
            f"Description: {version_info['description']}\n"
            f"Author: {version_info['author']}"
        )


@tools_app.command("serve")
def server_command(ctx: typer.Context):
    """
    Start the MCP server for screenshot tools.
    
    This command launches the MCP server that exposes screenshot functionality
    to Claude and other AI agents.
    """
    print_info(
        "Starting MCP server requires the MCP layer.\n"
        "Please use: python -m mcp_tools.screenshot.mcp.mcp_server"
    )


@tools_app.command("schema")
def schema_command(
    ctx: typer.Context,
    format: str = typer.Option(
        "human", 
        "--format", "-f",
        help="Output format: human, json, or mcp"
    )
):
    """
    Show CLI schema with all commands and options.
    
    This command outputs a complete schema of all available commands
    and their parameters, useful for documentation and tool integration.
    
    Format options:
    - human: Human-readable text format
    - json: CLI schema in JSON format
    - mcp: MCP-compatible schema format
    """
    # Generate schema
    cli_schema = generate_cli_schema(app)
    
    json_output = ctx.obj.get("json_output", False)
    
    if format == "json" or json_output:
        response = format_cli_response(True, data={"schema": cli_schema})
        print_json(response)
    elif format == "mcp":
        # Convert to MCP format
        mcp_schema = convert_to_mcp_schema(cli_schema)
        response = format_cli_response(True, data={"schema": mcp_schema})
        print_json(response)
    else:
        # Human readable format
        print_info("Screenshot CLI Schema")
        print_info("===================")
        
        # Print main commands
        for cmd_name, cmd in cli_schema["commands"].items():
            if "commands" in cmd:
                # Command group
                print_info(f"\n[Command Group] {cmd_name}: {cmd.get('help', '')}")
                
                for subcmd_name, subcmd in cmd["commands"].items():
                    print_info(f"  {subcmd_name}: {subcmd.get('help', '')}")
                    
                    for param_name, param in subcmd.get("parameters", {}).items():
                        required = " (required)" if param.get("required", False) else ""
                        default = f" (default: {param.get('default')})" if param.get("default") else ""
                        print_info(f"    --{param_name}: {param.get('type', 'string')}{required}{default}")
                        print_info(f"      {param.get('help', '')}")
            else:
                # Normal command
                print_info(f"\n[Command] {cmd_name}: {cmd.get('help', '')}")
                
                for param_name, param in cmd.get("parameters", {}).items():
                    required = " (required)" if param.get("required", False) else ""
                    default = f" (default: {param.get('default')})" if param.get("default") else ""
                    print_info(f"  --{param_name}: {param.get('type', 'string')}{required}{default}")
                    print_info(f"    {param.get('help', '')}")


if __name__ == "__main__":
    """
    CLI entry point for the screenshot module.
    
    Examples:
      python -m mcp_tools.screenshot.presentation.cli screenshot --quality 70
      python -m mcp_tools.screenshot.presentation.cli describe --region right_half
      python -m mcp_tools.screenshot.presentation.cli capture full
      python -m mcp_tools.screenshot.presentation.cli tools regions
    """
    # Configure logger if running as main script
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<level>{level}: {message}</level>",
        level="WARNING",
        colorize=True
    )
    
    app()