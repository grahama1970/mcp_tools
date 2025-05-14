"""
Memory Agent CLI Commands

This module contains the CLI command implementations for the Memory Agent functionality.
These commands are registered in the main CLI app under the 'memory' group.
"""

import json
import typer
from rich.console import Console
from rich.table import Table
from rich.json import JSON
from typing import List, Optional, Any, Dict, Union
from loguru import logger
from datetime import datetime

from complexity.arangodb.memory_agent import MemoryAgent
from complexity.arangodb.config import (
    MEMORY_MESSAGE_COLLECTION,
    MEMORY_COLLECTION,
    MEMORY_EDGE_COLLECTION,
    MEMORY_VIEW_NAME,
)

console = Console()

def memory_display_results(results_data: Dict[str, Any], title: str = "Memory Results"):
    """Display memory search results in a formatted table."""
    results = results_data.get("results", [])
    total = results_data.get("total", len(results))

    console.print(f"\n[bold blue]--- {title} (Found {len(results)} of {total}) ---[/bold blue]")

    if not results:
        console.print("[yellow]No memory items found matching the criteria.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta", expand=True, title=title)
    table.add_column("#", style="dim", width=3, no_wrap=True, justify="right")
    table.add_column("Score", justify="right", width=8)
    table.add_column("Memory Key", style="cyan", no_wrap=True, width=38)
    table.add_column("Content (Preview)", style="green", overflow="fold", min_width=30)
    table.add_column("Timestamp", style="yellow", width=12)

    for i, result_item in enumerate(results, start=1):
        if not isinstance(result_item, dict):
            continue

        # Extract fields from the result item
        score = result_item.get("rrf_score", 0.0)
        doc = result_item.get("doc", {})
        memory_key = doc.get("_key", "N/A")
        
        # Get content and create a preview
        content = doc.get("content", "")
        content_preview = content[:100] + "..." if len(content) > 100 else content
        
        # Format timestamp
        timestamp_str = doc.get("timestamp", "")
        if timestamp_str:
            try:
                # Try to parse ISO format timestamp
                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                timestamp_display = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError):
                timestamp_display = timestamp_str[:10]  # Just take first part as fallback
        else:
            timestamp_display = "N/A"
        
        # Add the row to the table
        table.add_row(
            str(i),
            f"{score:.4f}",
            memory_key,
            content_preview,
            timestamp_display,
        )

    console.print(table)

def memory_display_related(related_results: List[Dict[str, Any]], 
                         title: str = "Related Memories"):
    """Display related memories in a formatted table."""
    if not related_results:
        console.print("[yellow]No related memories found.[/yellow]")
        return

    console.print(f"\n[bold blue]--- {title} (Found {len(related_results)}) ---[/bold blue]")

    table = Table(show_header=True, header_style="bold magenta", expand=True, title=title)
    table.add_column("#", style="dim", width=3, no_wrap=True, justify="right")
    table.add_column("Memory Key", style="cyan", no_wrap=True, width=38)
    table.add_column("Relationship", style="blue", width=15)
    table.add_column("Strength", justify="right", width=8)
    table.add_column("Content (Preview)", style="green", overflow="fold", min_width=30)
    table.add_column("Rationale", style="yellow", overflow="fold", min_width=20)

    for i, result in enumerate(related_results, start=1):
        if not isinstance(result, dict):
            continue

        # Extract fields from the result
        memory_doc = result.get("memory", {})
        relationship = result.get("relationship", {})
        
        memory_key = memory_doc.get("_key", "N/A")
        rel_type = relationship.get("type", "unknown")
        strength = relationship.get("strength", 0.0)
        
        # Get content and create a preview
        content = memory_doc.get("content", "")
        content_preview = content[:80] + "..." if len(content) > 80 else content
        
        # Get rationale
        rationale = relationship.get("rationale", "N/A")
        
        # Add the row to the table
        table.add_row(
            str(i),
            memory_key,
            rel_type,
            f"{strength:.2f}",
            content_preview,
            rationale,
        )

    console.print(table)

def memory_display_conversation(messages: List[Dict[str, Any]], 
                              conversation_id: str,
                              title: str = "Conversation Context"):
    """Display conversation messages in a formatted table."""
    if not messages:
        console.print("[yellow]No messages found for this conversation.[/yellow]")
        return

    console.print(f"\n[bold blue]--- {title} (Conversation ID: {conversation_id}) ---[/bold blue]")

    table = Table(show_header=True, header_style="bold magenta", expand=True, title=title)
    table.add_column("#", style="dim", width=3, no_wrap=True, justify="right")
    table.add_column("Type", style="blue", width=10)
    table.add_column("Timestamp", style="yellow", width=12)
    table.add_column("Content", style="green", overflow="fold", min_width=50)

    # Sort messages by timestamp if present
    try:
        messages = sorted(messages, key=lambda x: x.get("timestamp", ""))
    except Exception:
        # If sorting fails, use original order
        pass

    for i, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            continue

        # Extract message details
        message_type = message.get("message_type", "unknown")
        # Format type for display
        type_display = "USER" if message_type == "user" else "AGENT" if message_type == "agent" else message_type
        
        # Format timestamp
        timestamp_str = message.get("timestamp", "")
        if timestamp_str:
            try:
                dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                timestamp_display = dt.strftime("%H:%M:%S")
            except (ValueError, TypeError):
                timestamp_display = timestamp_str[:10]
        else:
            timestamp_display = "N/A"
        
        # Get content
        content = message.get("content", "")
        
        # Style based on message type
        type_style = "bright_green" if type_display == "USER" else "bright_blue"
        
        # Add the row to the table
        table.add_row(
            str(i),
            f"[{type_style}]{type_display}[/{type_style}]",
            timestamp_display,
            content,
        )

    console.print(table)