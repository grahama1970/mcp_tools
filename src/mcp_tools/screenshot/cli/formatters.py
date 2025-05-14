#!/usr/bin/env python3
"""
Formatters for Screenshot Module CLI

This module provides rich formatting utilities for the CLI presentation layer.
It includes tables, panels, and progress indicators for a better user experience.

This module is part of the Presentation Layer and should only depend on
Core Layer components, not on Integration Layer.

Sample input:
- Screenshot result dictionary
- Image description results
- Error messages

Expected output:
- Rich formatted tables, panels, and progress indicators
"""

import os
import json
from typing import Dict, List, Any, Optional, Union

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.style import Style
from rich.text import Text

from loguru import logger


# Initialize console
console = Console()


# Color scheme
COLORS = {
    "success": "green",
    "error": "red",
    "warning": "yellow",
    "info": "blue",
    "path": "cyan",
    "highlight": "magenta",
    "dim": "grey70",
}


def print_screenshot_result(result: Dict[str, Any]) -> None:
    """
    Format and print screenshot result to the console.
    
    Args:
        result: Screenshot result dictionary
    """
    if "error" in result:
        print_error(result["error"])
        return
    
    # Create a panel
    file_path = result.get("file", "Unknown")
    filename = os.path.basename(file_path)
    directory = os.path.dirname(file_path)
    
    file_info = Text()
    file_info.append("Filename: ", style=COLORS["dim"])
    file_info.append(f"{filename}\n", style=COLORS["path"])
    file_info.append("Directory: ", style=COLORS["dim"])
    file_info.append(f"{directory}\n", style=COLORS["path"])
    
    # Add size information if available
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        size_kb = size_bytes / 1024
        file_info.append("Size: ", style=COLORS["dim"])
        file_info.append(f"{size_kb:.1f} KB", style=COLORS["info"])
    
    # Add raw file if available
    if "raw_file" in result:
        raw_path = result["raw_file"]
        raw_filename = os.path.basename(raw_path)
        file_info.append("\n\nRaw image: ", style=COLORS["dim"])
        file_info.append(f"{raw_filename}", style=COLORS["path"])
    
    panel = Panel(
        file_info,
        title="[bold green]Screenshot Captured Successfully",
        border_style=COLORS["success"],
        padding=(1, 2)
    )
    
    console.print(panel)


def print_description_result(result: Dict[str, Any]) -> None:
    """
    Format and print image description result to the console.
    
    Args:
        result: Image description result dictionary
    """
    if "error" in result:
        print_error(result["error"])
        return
    
    # Create description panel
    description = result.get("description", "No description available")
    confidence = result.get("confidence", 0)
    filename = result.get("filename", "Unknown")
    
    # Create confidence indicator
    confidence_text = Text("Confidence: ")
    confidence_bar = "●" * confidence + "○" * (5 - confidence)
    confidence_color = "green" if confidence >= 4 else "yellow" if confidence >= 2 else "red"
    confidence_text.append(confidence_bar, style=confidence_color)
    
    # Create content 
    content = Text()
    content.append(f"{description}\n\n")
    content.append(confidence_text)
    
    panel = Panel(
        content,
        title=f"[bold blue]Image Description: {filename}",
        border_style=COLORS["info"],
        padding=(1, 2)
    )
    
    console.print(panel)


def print_error(message: str, title: str = "Error") -> None:
    """
    Format and print error message to the console.
    
    Args:
        message: Error message
        title: Panel title
    """
    panel = Panel(
        Text(message, style=COLORS["error"]),
        title=f"[bold {COLORS['error']}]{title}",
        border_style=COLORS["error"],
        padding=(1, 2)
    )
    
    console.print(panel)


def print_warning(message: str, title: str = "Warning") -> None:
    """
    Format and print warning message to the console.
    
    Args:
        message: Warning message
        title: Panel title
    """
    panel = Panel(
        Text(message, style=COLORS["warning"]),
        title=f"[bold {COLORS['warning']}]{title}",
        border_style=COLORS["warning"],
        padding=(1, 2)
    )
    
    console.print(panel)


def print_info(message: str, title: str = "Info") -> None:
    """
    Format and print info message to the console.
    
    Args:
        message: Info message
        title: Panel title
    """
    panel = Panel(
        Text(message, style=COLORS["info"]),
        title=f"[bold {COLORS['info']}]{title}",
        border_style=COLORS["info"],
        padding=(1, 2)
    )
    
    console.print(panel)


def print_json(data: Dict[str, Any], title: str = "JSON Output") -> None:
    """
    Format and print JSON data to the console.
    
    Args:
        data: JSON data
        title: Panel title
    """
    json_str = json.dumps(data, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
    
    panel = Panel(
        syntax,
        title=f"[bold {COLORS['info']}]{title}",
        border_style=COLORS["info"],
        padding=(1, 2)
    )
    
    console.print(panel)


def print_regions_table(regions: Dict[str, Dict[str, int]]) -> None:
    """
    Format and print screen regions as a table.
    
    Args:
        regions: Dictionary of screen regions
    """
    table = Table(title="Available Screen Regions")
    
    table.add_column("Region Name", style=COLORS["highlight"])
    table.add_column("Top", justify="right", style=COLORS["info"])
    table.add_column("Left", justify="right", style=COLORS["info"])
    table.add_column("Width", justify="right", style=COLORS["info"])
    table.add_column("Height", justify="right", style=COLORS["info"])
    
    for name, region in regions.items():
        table.add_row(
            name,
            str(region["top"]),
            str(region["left"]),
            str(region["width"]),
            str(region["height"])
        )
    
    console.print(table)


def create_progress(description: str = "Processing") -> Progress:
    """
    Create a progress indicator.
    
    Args:
        description: Progress description
        
    Returns:
        Progress: Rich progress indicator
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[bold green]{task.completed}/{task.total}"),
        TimeElapsedColumn(),
    )


if __name__ == "__main__":
    """Demonstrate and validate formatters"""
    import sys
    
    # Sample data
    screenshot_result = {
        "file": "/tmp/screenshots/screenshot_1621012345.jpeg",
    }
    
    error_result = {
        "error": "Failed to capture screenshot: Permission denied"
    }
    
    description_result = {
        "description": "The screenshot shows a computer desktop with several windows open. There's a web browser showing a news website, a file explorer window, and a terminal. The desktop has a dark background with a mountain landscape.",
        "filename": "screenshot_1621012345.jpeg",
        "confidence": 4
    }
    
    regions = {
        "full": {"top": 0, "left": 0, "width": 1920, "height": 1080},
        "right_half": {"top": 0, "left": 960, "width": 960, "height": 1080},
        "left_half": {"top": 0, "left": 0, "width": 960, "height": 1080},
        "top_half": {"top": 0, "left": 0, "width": 1920, "height": 540},
        "bottom_half": {"top": 540, "left": 0, "width": 1920, "height": 540},
    }
    
    # Print examples
    console.print("\n[bold]Screenshot Result Example:[/bold]")
    print_screenshot_result(screenshot_result)
    
    console.print("\n[bold]Error Result Example:[/bold]")
    print_screenshot_result(error_result)
    
    console.print("\n[bold]Description Result Example:[/bold]")
    print_description_result(description_result)
    
    console.print("\n[bold]Regions Table Example:[/bold]")
    print_regions_table(regions)
    
    console.print("\n[bold]Warning Example:[/bold]")
    print_warning("This is a warning message")
    
    console.print("\n[bold]Info Example:[/bold]")
    print_info("This is an information message")
    
    console.print("\n[bold]JSON Example:[/bold]")
    print_json({"name": "Screenshot Tool", "version": "1.0.0", "regions": regions})
    
    console.print("\n[bold]Progress Example:[/bold]")
    with create_progress("Screenshot processing") as progress:
        task = progress.add_task("Taking screenshot...", total=100)
        for i in range(101):
            progress.update(task, completed=i)
            if i == 30:
                progress.update(task, description="Processing image...")
            if i == 70:
                progress.update(task, description="Saving file...")
            import time
            time.sleep(0.01)
    
    console.print("[bold green]All formatters demonstrated successfully![/bold green]")
