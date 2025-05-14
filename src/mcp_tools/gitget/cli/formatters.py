"""Rich-based formatters for CLI output in the gitget module.

This module provides formatted console output for the gitget CLI using the Rich library.
It includes functions for standard output, errors, warnings, and formatted tables.

Links to third-party package documentation:
- Rich: https://rich.readthedocs.io/en/latest/
- Rich Tables: https://rich.readthedocs.io/en/latest/tables.html
- Rich Console: https://rich.readthedocs.io/en/latest/console.html

Sample input:
    print_success("Repository cloned successfully")
    print_error("Failed to clone repository")
    print_warning("Repository already exists")
    
    data = [
        {"name": "file.py", "path": "/path/to/file.py", "size": 1024},
        {"name": "other.py", "path": "/path/to/other.py", "size": 2048}
    ]
    print_files_table(data)

Expected output:
    ✅ Repository cloned successfully
    ❌ Error: Failed to clone repository
    ⚠️ Warning: Repository already exists
    
    ┏━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━┓
    ┃ Name    ┃ Path            ┃ Size ┃
    ┡━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━┩
    │ file.py │ /path/to/file.py│ 1 KB │
    │ other.py│ /path/to/other..│ 2 KB │
    └─────────┴─────────────────┴──────┘
"""

from typing import Any, Dict, List, Optional, Union
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.filesize import decimal
from loguru import logger

# Create console instance
console = Console()

def print_success(message: str) -> None:
    """Print a success message to the console with a green checkmark.
    
    Args:
        message: The success message to display
    """
    console.print(f"✅ [bold green]{message}[/]")
    logger.success(message)

def print_error(message: str) -> None:
    """Print an error message to the console with a red X.
    
    Args:
        message: The error message to display
    """
    console.print(f"❌ [bold red]Error:[/] {message}")
    logger.error(message)

def print_warning(message: str) -> None:
    """Print a warning message to the console with a yellow warning sign.
    
    Args:
        message: The warning message to display
    """
    console.print(f"⚠️ [bold yellow]Warning:[/] {message}")
    logger.warning(message)

def print_info(message: str) -> None:
    """Print an info message to the console with a blue info sign.
    
    Args:
        message: The info message to display
    """
    console.print(f"ℹ️ [bold blue]Info:[/] {message}")
    logger.info(message)

def print_files_table(files: List[Dict[str, Any]]) -> None:
    """Print a table of files with name, path, and size.
    
    Args:
        files: List of dictionaries containing file information
              Each dictionary should have 'name', 'path', and 'size' keys
    """
    if not files:
        print_warning("No files found")
        return
        
    table = Table(title="Repository Files")
    table.add_column("Name", style="cyan")
    table.add_column("Path", style="green")
    table.add_column("Size", justify="right", style="magenta")
    
    for file in files:
        size_str = decimal(file["size"]) if isinstance(file["size"], (int, float)) else str(file["size"])
        table.add_row(file["name"], file["path"], size_str)
    
    console.print(table)

def print_repository_summary(
    repo_url: str, 
    local_path: str, 
    file_count: int, 
    total_size: int,
    commit_count: Optional[int] = None
) -> None:
    """Print a summary panel for a repository.
    
    Args:
        repo_url: URL of the repository
        local_path: Local path where repository is cloned
        file_count: Number of files in the repository
        total_size: Total size of the repository in bytes
        commit_count: Optional number of commits in the repository
    """
    panel = Panel(
        Text.from_markup(
            f"[bold blue]Repository:[/] {repo_url}\n"
            f"[bold blue]Local Path:[/] {local_path}\n"
            f"[bold blue]Files:[/] {file_count}\n"
            f"[bold blue]Total Size:[/] {decimal(total_size)}\n"
            + (f"[bold blue]Commits:[/] {commit_count}\n" if commit_count is not None else "")
        ),
        title="Repository Summary",
        border_style="green"
    )
    console.print(panel)

def print_file_tree(root: str, files: List[str]) -> None:
    """Print a tree representation of files in a repository.
    
    Args:
        root: Root directory of the repository
        files: List of file paths relative to the root
    """
    if not files:
        print_warning("No files found")
        return
        
    tree = Tree(f"📁 [bold blue]{os.path.basename(root)}[/]")
    
    # Create directory structure
    directories = {}
    for file_path in sorted(files):
        parts = file_path.split(os.sep)
        current = tree
        
        # Build the directory structure
        for i, part in enumerate(parts):
            is_file = i == len(parts) - 1
            path_so_far = os.sep.join(parts[:i+1])
            
            if is_file:
                # Add file to current directory
                current.add(f"📄 [cyan]{part}[/]")
            else:
                # Create or retrieve directory
                if path_so_far not in directories:
                    directories[path_so_far] = current.add(f"📁 [bold blue]{part}[/]")
                current = directories[path_so_far]
    
    console.print(tree)

def print_code_syntax(
    code: str, 
    language: str = "python", 
    title: Optional[str] = None,
    line_numbers: bool = True
) -> None:
    """Print formatted code with syntax highlighting.
    
    Args:
        code: The code to display
        language: Programming language for syntax highlighting
        title: Optional title for the code block
        line_numbers: Whether to show line numbers
    """
    syntax = Syntax(
        code,
        language,
        theme="monokai",
        line_numbers=line_numbers,
        word_wrap=True
    )
    
    if title:
        console.print(Panel(syntax, title=title, border_style="cyan"))
    else:
        console.print(syntax)

def get_spinner(text: str) -> Progress:
    """Create and return a spinner progress indicator.
    
    Args:
        text: Text to display next to the spinner
        
    Returns:
        A Progress object that can be used in a context manager
    """
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]{text}[/]"),
        transient=True
    )

if __name__ == "__main__":
    # Validation function to test formatters with real data
    import sys
    import tempfile
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Basic message formatters
    total_tests += 1
    try:
        # Redirect console output to capture it
        with console.capture() as capture:
            print_success("Test success message")
            print_error("Test error message")
            print_warning("Test warning message")
            print_info("Test info message")
        
        output = capture.get()
        expected_markers = ["✅", "❌", "⚠️", "ℹ️"]
        
        if not all(marker in output for marker in expected_markers):
            missing = [marker for marker in expected_markers if marker not in output]
            all_validation_failures.append(f"Basic formatters: Missing expected markers: {missing}")
    except Exception as e:
        all_validation_failures.append(f"Basic formatters: Unexpected exception: {str(e)}")
    
    # Test 2: Files table
    total_tests += 1
    try:
        test_files = [
            {"name": "file1.py", "path": "/path/to/file1.py", "size": 1024},
            {"name": "file2.py", "path": "/path/to/file2.py", "size": 2048},
        ]
        
        with console.capture() as capture:
            print_files_table(test_files)
        
        output = capture.get()
        expected_content = ["file1.py", "file2.py", "1 KB", "2 KB"]
        
        if not all(content in output for content in expected_content):
            missing = [content for content in expected_content if content not in output]
            all_validation_failures.append(f"Files table: Missing expected content: {missing}")
    except Exception as e:
        all_validation_failures.append(f"Files table: Unexpected exception: {str(e)}")
    
    # Test 3: Repository summary
    total_tests += 1
    try:
        with console.capture() as capture:
            print_repository_summary(
                "https://github.com/user/repo",
                "/tmp/repo",
                10,
                10240,
                5
            )
        
        output = capture.get()
        expected_content = ["Repository:", "Local Path:", "Files:", "10 KB", "Commits:"]
        
        if not all(content in output for content in expected_content):
            missing = [content for content in expected_content if content not in output]
            all_validation_failures.append(f"Repository summary: Missing expected content: {missing}")
    except Exception as e:
        all_validation_failures.append(f"Repository summary: Unexpected exception: {str(e)}")
    
    # Test 4: File tree
    total_tests += 1
    try:
        test_files = [
            "dir1/file1.py",
            "dir1/file2.py",
            "dir2/subdir/file3.py",
        ]
        
        with console.capture() as capture:
            print_file_tree("/tmp/repo", test_files)
        
        output = capture.get()
        expected_content = ["dir1", "dir2", "file1.py", "file2.py", "file3.py", "subdir"]
        
        if not all(content in output for content in expected_content):
            missing = [content for content in expected_content if content not in output]
            all_validation_failures.append(f"File tree: Missing expected content: {missing}")
    except Exception as e:
        all_validation_failures.append(f"File tree: Unexpected exception: {str(e)}")
    
    # Test 5: Code syntax highlighting
    total_tests += 1
    try:
        test_code = """def hello_world():
    print("Hello, world!")
    
hello_world()"""
        
        with console.capture() as capture:
            print_code_syntax(test_code, title="Test Code")
        
        output = capture.get()
        expected_content = ["hello_world", "Hello, world!", "Test Code"]
        
        if not all(content in output for content in expected_content):
            missing = [content for content in expected_content if content not in output]
            all_validation_failures.append(f"Code syntax: Missing expected content: {missing}")
    except Exception as e:
        all_validation_failures.append(f"Code syntax: Unexpected exception: {str(e)}")
    
    # Test 6: Empty files handling
    total_tests += 1
    try:
        with console.capture() as capture:
            print_files_table([])
            print_file_tree("/tmp/repo", [])
        
        output = capture.get()
        expected_content = ["No files found"]
        
        if not all(content in output for content in expected_content):
            missing = [content for content in expected_content if content not in output]
            all_validation_failures.append(f"Empty files handling: Missing expected content: {missing}")
    except Exception as e:
        all_validation_failures.append(f"Empty files handling: Unexpected exception: {str(e)}")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)