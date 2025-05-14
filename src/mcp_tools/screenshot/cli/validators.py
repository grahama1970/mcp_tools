#!/usr/bin/env python3
"""
Validators for Screenshot Module CLI

This module provides validation functions for CLI inputs.
It includes validators for quality, regions, and other parameters.

This module is part of the Presentation Layer and should only depend on
Core Layer components, not on Integration Layer.

Sample input:
- CLI parameter values

Expected output:
- Validated and processed parameter values
- Friendly error messages
"""

import os
import sys
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

import typer
from loguru import logger

from mcp_tools.screenshot.core.constants import IMAGE_SETTINGS, REGION_PRESETS
from mcp_tools.screenshot.core.utils import validate_quality, validate_region
from mcp_tools.screenshot.cli.formatters import print_error, print_warning


def validate_quality_option(ctx: typer.Context, value: int) -> int:
    """
    Typer callback for validating quality option.
    
    Args:
        ctx: Typer context
        value: Quality value from CLI
        
    Returns:
        int: Validated quality value
    """
    try:
        return validate_quality(
            value, 
            IMAGE_SETTINGS["MIN_QUALITY"], 
            IMAGE_SETTINGS["MAX_QUALITY"]
        )
    except Exception as e:
        logger.error(f"Quality validation error: {str(e)}")
        print_error(
            f"Invalid quality value: {value}. Must be between "
            f"{IMAGE_SETTINGS['MIN_QUALITY']} and {IMAGE_SETTINGS['MAX_QUALITY']}."
        )
        raise typer.Exit(1)


def validate_region_option(ctx: typer.Context, value: Optional[str]) -> Optional[Union[str, List[int]]]:
    """
    Typer callback for validating region option.
    
    Args:
        ctx: Typer context
        value: Region value from CLI
        
    Returns:
        Optional[Union[str, List[int]]]: Validated region value
    """
    if value is None or value == "full":
        return None
        
    # Check if it's a preset
    if value in REGION_PRESETS:
        return value
        
    # Check if it's a custom region in x,y,w,h format
    if "," in value:
        try:
            coords = [int(x.strip()) for x in value.split(",")]
            is_valid, error = validate_region(coords)
            if not is_valid:
                print_error(error or "Invalid region format")
                raise typer.Exit(1)
            return coords
        except ValueError:
            print_error(f"Invalid region format: {value}. Expected 'x,y,width,height'")
            raise typer.Exit(1)
    
    # Not a valid preset or custom region
    valid_presets = ", ".join(REGION_PRESETS.keys())
    print_error(
        f"Invalid region: {value}. Must be one of {valid_presets} "
        f"or a custom region in 'x,y,width,height' format."
    )
    raise typer.Exit(1)


def validate_output_dir(ctx: typer.Context, value: Optional[str]) -> str:
    """
    Typer callback for validating output directory.
    
    Args:
        ctx: Typer context
        value: Output directory from CLI
        
    Returns:
        str: Validated output directory
    """
    if value is None:
        return "screenshots"
        
    # Check if dir exists or can be created
    try:
        os.makedirs(value, exist_ok=True)
        return value
    except Exception as e:
        print_error(f"Cannot create output directory: {value}. Error: {str(e)}")
        raise typer.Exit(1)


def validate_model_option(ctx: typer.Context, value: str) -> str:
    """
    Typer callback for validating model option.
    
    Args:
        ctx: Typer context
        value: Model name from CLI
        
    Returns:
        str: Validated model name
    """
    # List of supported models (could be moved to constants)
    supported_models = [
        "vertex_ai/gemini-2.5-pro-preview-05-06",
        "vertex_ai/gemini-pro-vision",
        "gpt-4-vision-preview",
        "claude-3-opus-20240229",
        "claude-3-sonnet-20240229"
    ]
    
    if "/" in value and value not in supported_models:
        print_warning(
            f"Model '{value}' is not in the list of known supported models. "
            f"This may work if the model supports multimodal inputs, but is not guaranteed."
        )
        
    return value


def validate_prompt_option(ctx: typer.Context, value: str) -> str:
    """
    Typer callback for validating prompt option.
    
    Args:
        ctx: Typer context
        value: Prompt from CLI
        
    Returns:
        str: Validated prompt
    """
    if len(value) < 3:
        print_warning("Prompt is very short. Consider providing a more detailed prompt for better results.")
        
    return value


def validate_credentials_file(ctx: typer.Context, value: Optional[str]) -> Optional[str]:
    """
    Typer callback for validating credentials file.
    
    Args:
        ctx: Typer context
        value: Credentials file path from CLI
        
    Returns:
        Optional[str]: Validated credentials file path
    """
    if value is None:
        return None
        
    if not os.path.exists(value):
        print_error(f"Credentials file not found: {value}")
        raise typer.Exit(1)
        
    try:
        import json
        with open(value, "r") as f:
            json.load(f)
        return value
    except Exception as e:
        print_error(f"Invalid credentials file: {value}. Error: {str(e)}")
        raise typer.Exit(1)


def validate_file_exists(ctx: typer.Context, value: str) -> str:
    """
    Typer callback for validating a file exists.
    
    Args:
        ctx: Typer context
        value: File path from CLI
        
    Returns:
        str: Validated file path
    """
    if not os.path.exists(value):
        print_error(f"File not found: {value}")
        raise typer.Exit(1)
        
    if not os.path.isfile(value):
        print_error(f"Not a file: {value}")
        raise typer.Exit(1)
        
    return value


def validate_json_output(ctx: typer.Context, value: bool) -> bool:
    """
    Typer callback for validating JSON output option.
    
    Args:
        ctx: Typer context
        value: JSON output flag from CLI
        
    Returns:
        bool: Validated JSON output flag
    """
    # Store in context for other callbacks to access
    ctx.ensure_object(dict)
    ctx.obj["json_output"] = value
    return value


if __name__ == "__main__":
    """Demonstrate and validate validator functions"""
    import sys
    from rich.console import Console
    
    console = Console()
    console.print("[bold]Validator Test Mode[/bold]")
    console.print("This module provides validation functions for CLI parameters.")
    console.print("In a real application, these are used as Typer callbacks.")
    console.print("Since Typer callbacks require a Typer context, they cannot be directly tested here.")
    console.print("\nFor more complete validation, see the CLI module tests.")
