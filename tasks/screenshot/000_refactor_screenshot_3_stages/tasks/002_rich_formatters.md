# Task 2: Create Rich Formatters Module

**Objective**: Implement formatters for presenting screenshot tool output with Rich tables.

**Pre-Task Setup**:
- Review Rich library table documentation for best practices

**Implementation Steps**:
- [ ] 2.1 Create formatters.py module skeleton with imports
- [ ] 2.2 Implement screenshot table formatter with sample data
- [ ] 2.3 Implement description table formatter with sample data
- [ ] 2.4 Implement error panel formatter with sample cases
- [ ] 2.5 Add debug/test code in `__main__` section
- [ ] 2.6 Ensure consistent styling across all formatters
- [ ] 2.7 Test formatters with various data scenarios
- [ ] 2.8 Git commit verified formatters with message "Add: Rich table formatters module with self-testing capabilities"

**Technical Specifications**:
- Create formatters that match MCP quality display standards
- Use Rich tables with consistent styling
- Handle missing or partial data gracefully
- Include sample debug/test code in `__main__` section
- Format strings should escape Rich markup in user data

**Verification Method**:
- Run the module directly: `python -m mcp_tools.screenshot.formatters`
- Verify visual output matches expected styling
- Test with both complete and partial data

**Acceptance Criteria**:
- Formatters produce visually appealing tables/panels
- Error cases are handled gracefully with appropriate fallbacks
- Styling is consistent across all formatters
- Debug output demonstrates all formatter capabilities

## Example Implementation Approach

```python
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
from typing import Dict, Any, List, Optional

def format_screenshot_result(result: Dict[str, Any]) -> Table:
    """
    Format a screenshot result as a Rich table.
    
    Args:
        result: The screenshot result dictionary
        
    Returns:
        A Rich Table object
    """
    table = Table(title="Screenshot Results", box=box.ROUNDED)
    
    # Add columns
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    # Check for error
    if "error" in result:
        table.add_row("Status", "[red]Error[/red]")
        table.add_row("Error Message", f"[red]{result['error']}[/red]")
        return table
    
    # Add rows for successful result
    table.add_row("File", result.get("file", "N/A"))
    
    # Format dimensions if present
    dimensions = result.get("dimensions", [])
    if dimensions and len(dimensions) >= 2:
        table.add_row("Dimensions", f"{dimensions[0]} × {dimensions[1]} pixels")
    else:
        table.add_row("Dimensions", "N/A")
    
    # Format file size if present
    size = result.get("size", 0)
    if size > 0:
        size_kb = size / 1024
        size_mb = size_kb / 1024
        if size_mb >= 1:
            table.add_row("File Size", f"{size_mb:.2f} MB")
        else:
            table.add_row("File Size", f"{size_kb:.2f} KB")
    else:
        table.add_row("File Size", "N/A")
    
    return table

if __name__ == "__main__":
    """Module self-test with sample inputs"""
    console = Console()
    
    # Test 1: Successful result
    print("\nTest 1: Successful screenshot result")
    sample_result = {
        "file": "/tmp/screenshot_2025-05-11_123456.jpg",
        "dimensions": [1920, 1080],
        "size": 246822  # bytes
    }
    console.print(format_screenshot_result(sample_result))
    
    # Test 2: Error result
    print("\nTest 2: Error result")
    error_result = {
        "error": "Failed to capture screenshot: Permission denied"
    }
    console.print(format_screenshot_result(error_result))
    
    # Test 3: Partial data
    print("\nTest 3: Partial data")
    partial_result = {
        "file": "/tmp/screenshot_partial.jpg",
        # Missing dimensions and size
    }
    console.print(format_screenshot_result(partial_result))
```
