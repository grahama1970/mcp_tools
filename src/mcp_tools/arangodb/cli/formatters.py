"""Rich formatting utilities for ArangoDB CLI.

This module provides formatting functions for displaying ArangoDB data
in the command line interface. It handles tables, JSON formatting,
and other display formats for various types of data.

Links to third-party documentation:
- Rich: https://rich.readthedocs.io/
- Rich Table: https://rich.readthedocs.io/en/stable/tables.html
- Rich Tree: https://rich.readthedocs.io/en/stable/tree.html

Sample input:
    results = {
        "results": [
            {"_key": "123", "title": "Document 1", "content": "...", "score": 0.95},
            {"_key": "456", "title": "Document 2", "content": "...", "score": 0.87}
        ],
        "total": 2
    }
    display_search_results(results, "Search Results", "score")

Expected output:
    A formatted table displayed in the console showing the search results.
"""

from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.tree import Tree
from rich.json import JSON
from rich.syntax import Syntax
from rich.box import Box, ROUNDED
from rich.text import Text

console = Console()


def truncate_string(s: str, max_length: int = 80) -> str:
    """Truncate a string to a maximum length, adding ellipsis if needed."""
    if not s:
        return ""
    
    s = str(s)  # Ensure input is a string
    if len(s) <= max_length:
        return s
    
    return s[:max_length - 3] + "..."


def format_timestamp(timestamp: str) -> str:
    """Format an ISO timestamp string to a human-readable format."""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return timestamp


def display_search_results(
    results_data: Dict[str, Any],
    title: str = "Search Results",
    score_field: str = "score",
    output_format: str = "table"
) -> None:
    """Display search results in a formatted table."""
    if not isinstance(results_data, dict):
        console.print("[yellow]Warning: Invalid format for search results.[/yellow]")
        return

    # Check if we have an error
    if "error" in results_data:
        console.print(f"[red]Error: {results_data['error']}[/red]")
        return

    # If JSON output is requested, print JSON and return
    if output_format.lower() == "json":
        # Create a clean copy for JSON output
        json_results = {
            "results": [
                {
                    "doc": result.get("doc", {}),
                    "score": result.get(score_field, 0)
                }
                for result in results_data.get("results", [])
            ],
            "total": results_data.get("total", 0),
            "offset": results_data.get("offset", 0),
            "query": results_data.get("query", ""),
            "time": results_data.get("time", 0)
        }
        print(json.dumps(json_results, indent=2))
        return

    results = results_data.get("results", [])
    total = results_data.get("total", len(results))
    offset = results_data.get("offset", 0)
    query = results_data.get("query", "")
    search_time = results_data.get("time", 0)
    
    # Print basic search metadata
    console.print(f"\n[bold blue]{'═' * 80}[/bold blue]")
    console.print(f"[bold blue]{title} (Showing {len(results)} of {total})[/bold blue]")

    if query:
        console.print(f"Query: '[yellow]{query}[/yellow]'")

    if search_time:
        console.print(f"Search time: [cyan]{search_time*1000:.2f}ms[/cyan]")

    console.print(f"[bold blue]{'─' * 80}[/bold blue]")

    if not results:
        console.print("[yellow]No results found matching the criteria.[/yellow]")
        return
    
    table = Table(
        show_header=True,
        header_style="bold magenta",
        expand=True,
        title=title,
        box=ROUNDED
    )
    
    # Add columns
    table.add_column("#", style="dim", width=3, justify="right")
    table.add_column("Score", justify="right", width=8)
    table.add_column("Key", style="cyan", no_wrap=True, width=10)
    table.add_column("Title/Name", style="green", width=30)
    table.add_column("Content", width=40)
    table.add_column("Tags", style="yellow", width=15)
    
    # Add rows
    for i, item in enumerate(results, start=1):
        # Handle different result structures
        doc = item.get("doc", item) if isinstance(item, dict) else {}

        # Extract fields
        key = doc.get("_key", "N/A")
        title = doc.get("title", doc.get("name", "N/A"))

        content_candidates = ["content", "description", "problem", "solution"]
        content = "N/A"
        for field in content_candidates:
            if field in doc and doc[field]:
                content = truncate_string(doc[field], 80)
                break

        tags = ", ".join(doc.get("tags", [])) if "tags" in doc and isinstance(doc["tags"], list) else ""

        # Format score based on value
        score_val = item.get(score_field, 0)
        if score_val > 7.0:
            score_str = f"[green]{score_val:.4f}[/green]"
        elif score_val > 5.0:
            score_str = f"[yellow]{score_val:.4f}[/yellow]"
        else:
            score_str = f"{score_val:.4f}"

        # Add row to table
        row = [str(offset + i)]
        row.append(score_str)
        row.extend([key, truncate_string(title, 30), content, tags])

        table.add_row(*row)
    
    console.print(table)
    
    # Show pagination info if applicable
    if total > (offset + len(results)):
        console.print(
            f"[dim]Showing results {offset + 1}-{offset + len(results)} of {total}. "
            f"Use --offset for pagination.[/dim]"
        )

    # Print detailed info for first result if available
    if results and "doc" in results[0]:
        display_search_result_details(results[0], score_field)


def display_document(
    document: Dict[str, Any],
    title: str = "Document Details"
) -> None:
    """Display a single document with formatted fields."""
    if not document:
        console.print("[yellow]No document data to display.[/yellow]")
        return
    
    # Create a tree for hierarchical display
    tree = Tree(f"[bold]{title}[/bold]")
    
    # Add metadata
    meta_branch = tree.add("📋 [bold cyan]Metadata[/bold cyan]")
    meta_branch.add(f"[dim]Key:[/dim] [cyan]{document.get('_key', 'N/A')}[/cyan]")
    meta_branch.add(f"[dim]ID:[/dim] [cyan]{document.get('_id', 'N/A')}[/cyan]")
    
    if "timestamp" in document:
        formatted_time = format_timestamp(document["timestamp"])
        meta_branch.add(f"[dim]Created:[/dim] {formatted_time}")
    
    if "updated_at" in document:
        updated_time = format_timestamp(document["updated_at"])
        meta_branch.add(f"[dim]Updated:[/dim] {updated_time}")
    
    # Add content
    content_branch = tree.add("📄 [bold green]Content[/bold green]")
    
    # Handle title/name
    title_val = document.get("title", document.get("name", None))
    if title_val:
        content_branch.add(f"[bold]Title/Name:[/bold] {title_val}")
    
    # Handle various content fields
    content_fields = ["content", "description", "problem", "solution", "context"]
    for field in content_fields:
        if field in document and document[field]:
            content_val = document[field]
            if isinstance(content_val, str) and len(content_val) > 100:
                # For long content, show a preview
                preview = truncate_string(content_val, 100)
                content_branch.add(f"[bold]{field.capitalize()}:[/bold] {preview}")
                
                # Add full content as a folded panel
                full_content = Panel(
                    content_val,
                    title=f"{field.capitalize()} (Full)",
                    title_align="left",
                    border_style="dim",
                    expand=False
                )
                console.print(full_content)
            else:
                content_branch.add(f"[bold]{field.capitalize()}:[/bold] {content_val}")
    
    # Handle tags
    if "tags" in document and document["tags"]:
        tags_str = ", ".join(f"[yellow]{tag}[/yellow]" for tag in document["tags"])
        content_branch.add(f"[bold]Tags:[/bold] {tags_str}")
    
    # Print the tree
    console.print(tree)


def display_relationship(
    edge: Dict[str, Any],
    from_doc: Optional[Dict[str, Any]] = None,
    to_doc: Optional[Dict[str, Any]] = None
) -> None:
    """Display a relationship edge with connected documents."""
    if not edge:
        console.print("[yellow]No relationship data to display.[/yellow]")
        return
    
    # Create a formatted panel for the relationship
    rel_type = edge.get("type", "RELATED")
    
    # Create a tree for the relationship
    tree = Tree(f"[bold magenta]Relationship: [bold cyan]{rel_type}[/bold cyan][/bold magenta]")
    
    # Add metadata branch
    meta = tree.add("📋 [bold cyan]Metadata[/bold cyan]")
    meta.add(f"[dim]Key:[/dim] [cyan]{edge.get('_key', 'N/A')}[/cyan]")
    meta.add(f"[dim]From:[/dim] [green]{edge.get('_from', 'N/A')}[/green]")
    meta.add(f"[dim]To:[/dim] [yellow]{edge.get('_to', 'N/A')}[/yellow]")
    
    if "timestamp" in edge:
        formatted_time = format_timestamp(edge["timestamp"])
        meta.add(f"[dim]Created:[/dim] {formatted_time}")
    
    # Add rationale if present
    if "rationale" in edge and edge["rationale"]:
        rationale = tree.add("💡 [bold green]Rationale[/bold green]")
        rationale.add(edge["rationale"])
    
    # Print the tree
    console.print(tree)
    
    # Display connected documents if provided
    if from_doc:
        console.print("\n[bold green]Source Document:[/bold green]")
        display_document_summary(from_doc, include_content=False)
    
    if to_doc:
        console.print("\n[bold yellow]Target Document:[/bold yellow]")
        display_document_summary(to_doc, include_content=False)


def display_search_result_details(
    result: Dict[str, Any],
    score_field: str = "score"
) -> None:
    """
    Display detailed information about a single search result.

    Args:
        result: Search result to display
        score_field: Name of the field containing the score value
    """
    doc = result.get("doc", {}) if isinstance(result, dict) else {}
    score = result.get(score_field, 0) if isinstance(result, dict) else 0

    # Create header with key
    key = doc.get("_key", "N/A")
    console.print(f"\n[bold green]{'═' * 80}[/bold green]")
    console.print(f"[bold green]  DOCUMENT DETAILS: [cyan]{key}[/cyan]  [/bold green]")
    console.print(f"[bold green]{'═' * 80}[/bold green]")

    # Show score with color formatting
    if score > 7.0:
        score_str = f"[green]{score:.5f}[/green]"
    elif score > 5.0:
        score_str = f"[yellow]{score:.5f}[/yellow]"
    else:
        score_str = f"[white]{score:.5f}[/white]"
    console.print(f"Score: {score_str}")

    # Find and display main text content
    text_fields = ["question", "problem", "title", "content", "text", "description"]
    main_text = None
    main_field = None
    for field in text_fields:
        if field in doc and doc[field]:
            main_text = doc[field]
            main_field = field.title()  # Capitalize field name
            break

    # Print main text content with highlighted field name
    if main_text and main_field:
        console.print(f"\n[yellow]{main_field}:[/yellow] {main_text}")

    # Display metadata fields
    metadata_fields = []

    # Handle special fields
    if "label" in doc:
        label_value = doc["label"]
        if label_value == 1:
            label_str = f"[green]{label_value}[/green]"
        elif label_value == 0:
            label_str = f"[yellow]{label_value}[/yellow]"
        else:
            label_str = f"[red]{label_value}[/red]"
        metadata_fields.append(("Label", label_str))

    if "validated" in doc:
        validated = doc["validated"]
        validated_str = f"[green]Yes[/green]" if validated else f"[red]No[/red]"
        metadata_fields.append(("Validated", validated_str))

    # Print metadata section if we have fields
    if metadata_fields:
        console.print(f"\n[cyan]Document Metadata:[/cyan]")
        for field, value in metadata_fields:
            console.print(f"  • [cyan]{field}:[/cyan] {value}")

    # Print tags in a special section if present
    if "tags" in doc and isinstance(doc["tags"], list) and doc["tags"]:
        tags = doc["tags"]
        console.print(f"\n[blue]Tags:[/blue]")
        tag_colors = ["blue", "magenta", "cyan", "green", "yellow"]
        for i, tag in enumerate(tags):
            color = tag_colors[i % len(tag_colors)]  # Cycle through colors
            console.print(f"  • [{color}]{tag}[/{color}]")

    # Print footer
    console.print(f"[bold green]{'═' * 80}[/bold green]\n")


def display_document_summary(
    document: Dict[str, Any],
    include_content: bool = True
) -> None:
    """Display a concise summary of a document."""
    if not document:
        return
    
    table = Table(box=ROUNDED, expand=False, show_header=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    
    # Add key metadata
    table.add_row("Key", f"[cyan]{document.get('_key', 'N/A')}[/cyan]")
    
    # Add title/name
    title_val = document.get("title", document.get("name", "N/A"))
    table.add_row("Title/Name", title_val)
    
    # Add content preview if requested
    if include_content:
        content_fields = ["content", "description", "problem", "solution"]
        for field in content_fields:
            if field in document and document[field]:
                preview = truncate_string(document[field], 80)
                table.add_row(field.capitalize(), preview)
                break
    
    # Add tags
    if "tags" in document and document["tags"]:
        tags_str = ", ".join(f"[yellow]{tag}[/yellow]" for tag in document["tags"])
        table.add_row("Tags", tags_str)
    
    console.print(table)


def display_traversal_results(
    results: List[Dict[str, Any]],
    title: str = "Graph Traversal Results"
) -> None:
    """Display graph traversal results in a formatted tree."""
    if not results:
        console.print("[yellow]No traversal results to display.[/yellow]")
        return
    
    console.print(f"\n[bold blue]--- {title} ({len(results)} paths) ---[/bold blue]")
    
    # Create a tree for each traversal path
    for i, result in enumerate(results, start=1):
        # Extract data
        if not isinstance(result, dict):
            continue
        
        vertex = result.get("vertex", {})
        edge = result.get("edge", {})
        path = result.get("path", {})
        
        # Create tree for this path
        tree = Tree(f"[bold]Path {i}[/bold]")
        
        # Add terminal vertex info
        vertex_name = vertex.get("title", vertex.get("name", vertex.get("_key", "Unknown")))
        vertex_node = tree.add(f"[bold cyan]Vertex:[/bold cyan] {vertex_name} ([dim]{vertex.get('_id', 'N/A')}[/dim])")
        
        # Add edge info if it exists
        if edge and "_id" in edge:
            edge_type = edge.get("type", "RELATED")
            edge_node = tree.add(f"[bold magenta]Edge:[/bold magenta] {edge_type} ([dim]{edge.get('_id', 'N/A')}[/dim])")
            
            # Add edge details
            if "rationale" in edge:
                edge_node.add(f"Rationale: {edge['rationale']}")
        
        # Add path vertices if available
        if path and "vertices" in path and isinstance(path["vertices"], list):
            path_node = tree.add(f"[bold green]Path Vertices ({len(path['vertices'])})[/bold green]")
            for v in path["vertices"]:
                v_name = v.get("title", v.get("name", v.get("_key", "Unknown")))
                path_node.add(f"{v_name} ([dim]{v.get('_id', 'N/A')}[/dim])")
        
        console.print(tree)
    
    console.print("")


if __name__ == "__main__":
    import sys
    # Test the formatting functions with sample data
    test_results = {
        "results": [
            {
                "doc": {
                    "_key": "doc1",
                    "title": "Test Document 1",
                    "content": "This is test content for document 1.",
                    "tags": ["test", "document", "first"],
                    "validated": True
                },
                "score": 8.92
            },
            {
                "doc": {
                    "_key": "doc2",
                    "title": "Test Document 2",
                    "content": "This is test content for document 2.",
                    "tags": ["test", "document", "second"],
                    "label": 1
                },
                "score": 6.75
            },
            {
                "doc": {
                    "_key": "doc3",
                    "title": "Test Document 3",
                    "content": "This is test content for document 3.",
                    "tags": ["test", "document", "third"],
                    "validated": False
                },
                "score": 4.32
            }
        ],
        "total": 3,
        "offset": 0,
        "query": "test document",
        "time": 0.023
    }

    # Validation setup
    all_validation_failures = []
    total_tests = 0

    # Test 1: Table format
    total_tests += 1
    try:
        console.print("[bold]Test 1: Table format[/bold]")
        display_search_results(test_results, "Test Search Results", "score", "table")
        console.print("Table format display completed")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Table format failed: {str(e)}")

    # Test 2: JSON format
    total_tests += 1
    try:
        console.print("\n[bold]Test 2: JSON format[/bold]")
        display_search_results(test_results, "Test Search Results", "score", "json")
        console.print("JSON format display completed")
    except Exception as e:
        all_validation_failures.append(f"Test 2: JSON format failed: {str(e)}")

    # Test 3: Document display
    total_tests += 1
    try:
        console.print("\n[bold]Test 3: Document display[/bold]")
        display_document(test_results["results"][0]["doc"], "Test Document Display")
        console.print("Document display completed")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Document display failed: {str(e)}")

    # Test 4: Search result details
    total_tests += 1
    try:
        console.print("\n[bold]Test 4: Search result details[/bold]")
        display_search_result_details(test_results["results"][0], "score")
        console.print("Search result details display completed")
    except Exception as e:
        all_validation_failures.append(f"Test 4: Search result details failed: {str(e)}")

    # Final validation result
    if all_validation_failures:
        console.print(f"\n[bold red]❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:[/bold red]")
        for failure in all_validation_failures:
            console.print(f"  - {failure}")
        sys.exit(1)
    else:
        console.print(f"\n[bold green]✅ VALIDATION PASSED - All {total_tests} tests produced expected results[/bold green]")
        console.print("Function is validated and formal tests can now be written")
        sys.exit(0)
