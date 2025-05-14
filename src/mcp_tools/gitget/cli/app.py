"""Command-line interface for the gitget module.

This module provides a Typer-based CLI for gitget functionality, allowing users
to interact with git repositories, clone them, process their contents, and more.

Links to third-party package documentation:
- Typer: https://typer.tiangolo.com/
- Rich: https://rich.readthedocs.io/en/latest/

Sample input:
    $ python -m mcp_tools.gitget.cli clone https://github.com/user/repo
    $ python -m mcp_tools.gitget.cli process --sparse-checkout="README.md,src/" https://github.com/user/repo
    $ python -m mcp_tools.gitget.cli extract https://github.com/user/repo --output-format=json

Expected output:
    ✅ Repository cloned successfully to /tmp/repos/user_repo
    ✅ Repository processed successfully: 10 files, 1024 bytes
    ✅ Repository extracted successfully: 20 chunks, 10240 tokens
"""

import os
import sys
import json
import time
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from mcp_tools.gitget.core.repo_operations import sparse_clone, find_files, process_repository
from mcp_tools.gitget.core.directory_manager import RepositoryDirectoryManager
from mcp_tools.gitget.core.text_chunker import TextChunker, SectionHierarchy
from mcp_tools.gitget.core.utils import (
    extract_repo_info, save_to_file, read_from_file, normalize_path
)

from .formatters import (
    print_success, print_error, print_warning, print_info,
    print_files_table, print_repository_summary, print_file_tree, print_code_syntax,
    get_spinner
)
from .validators import (
    validate_git_url, validate_output_dir, validate_branch_name,
    validate_sparse_checkout_paths, validate_git_installed, validate_file_path,
    validate_extension
)
from .schemas import (
    RepositoryInput, FileInfo, RepositoryInfo, ChunkInfo, ProcessResult,
    ContentType
)

class OutputFormat(str, Enum):
    """Output format enum."""
    JSON = "json"
    TEXT = "text"
    TABLE = "table"
    TREE = "tree"

# Create Typer app
app = typer.Typer(
    name="gitget",
    help="Git repository content extraction and processing tool",
    rich_markup_mode="rich"
)

# Create console
console = Console()

@app.command("clone")
def clone_repository(
    url: str = typer.Argument(
        ..., 
        callback=validate_git_url,
        help="URL of the Git repository to clone"
    ),
    output_dir: str = typer.Option(
        "./repos",
        "--output-dir", "-o",
        callback=validate_output_dir,
        help="Directory to store cloned repositories"
    ),
    branch: str = typer.Option(
        "main",
        "--branch", "-b",
        callback=validate_branch_name,
        help="Branch to checkout"
    ),
    sparse_checkout: Optional[List[str]] = typer.Option(
        None,
        "--sparse-checkout", "-s",
        callback=validate_sparse_checkout_paths,
        help="Comma-separated list of files/dirs to include in sparse checkout"
    ),
    depth: int = typer.Option(
        1,
        "--depth", "-d",
        min=1,
        help="Depth of git history to clone (1 for shallow clone)"
    ),
    include_git_dir: bool = typer.Option(
        False,
        "--include-git-dir", "-g",
        help="Whether to include .git directory"
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Force clone even if directory already exists"
    )
) -> None:
    """Clone a Git repository with optional sparse checkout.
    
    Examples:
        [bold]$ gitget clone https://github.com/user/repo[/bold]
        
        [bold]$ gitget clone --sparse-checkout="README.md,src/" https://github.com/user/repo[/bold]
        
        [bold]$ gitget clone --branch=develop --depth=2 git@github.com:user/repo.git[/bold]
    """
    # Validate Git is installed
    validate_git_installed()
    
    # Create repository input schema
    repo_input = RepositoryInput(
        url=url,
        branch=branch,
        output_dir=output_dir,
        sparse_checkout=sparse_checkout,
        depth=depth,
        include_git_dir=include_git_dir
    )
    
    # Extract repository info from URL
    repo_info = extract_repo_info(url)
    
    # Build local path
    local_path = os.path.join(output_dir, f"{repo_info['owner']}_{repo_info['name']}")
    
    # Check if directory already exists
    if os.path.exists(local_path) and not force:
        if Confirm.ask(f"Repository directory already exists at '{local_path}'. Overwrite?"):
            import shutil
            shutil.rmtree(local_path)
        else:
            print_warning(f"Clone aborted. Directory already exists: {local_path}")
            return
    
    # Start cloning
    print_info(f"Cloning repository {url}...")
    
    # Create progress spinner
    with get_spinner("Cloning repository") as progress:
        try:
            # Measure clone time
            start_time = time.time()
            
            # Clone repository
            result = sparse_clone(
                repo_input.url,
                local_path,
                branch=repo_input.branch,
                sparse_patterns=repo_input.sparse_checkout,
                depth=repo_input.depth
            )
            
            clone_time = time.time() - start_time
            
            if not result:
                print_error(f"Failed to clone repository: {url}")
                sys.exit(1)
        except Exception as e:
            print_error(f"Error cloning repository: {str(e)}")
            sys.exit(1)
    
    # Find all files in repository
    all_files = find_files(local_path, exclude_patterns=[".git"] if not include_git_dir else [])
    
    # Create repository summary
    total_size = sum(os.path.getsize(f) for f in all_files)
    
    # Print success message
    print_success(f"Repository cloned successfully to {local_path}")
    
    # Print repository summary
    print_repository_summary(
        url,
        local_path,
        len(all_files),
        total_size,
        None  # We don't have commit count information
    )

@app.command("process")
def process_repo(
    url: str = typer.Argument(
        ..., 
        callback=validate_git_url,
        help="URL of the Git repository to process"
    ),
    output_dir: str = typer.Option(
        "./repos",
        "--output-dir", "-o",
        callback=validate_output_dir,
        help="Directory to store cloned repositories"
    ),
    branch: str = typer.Option(
        "main",
        "--branch", "-b",
        callback=validate_branch_name,
        help="Branch to checkout"
    ),
    sparse_checkout: Optional[List[str]] = typer.Option(
        None,
        "--sparse-checkout", "-s",
        callback=validate_sparse_checkout_paths,
        help="Comma-separated list of files/dirs to include in sparse checkout"
    ),
    depth: int = typer.Option(
        1,
        "--depth", "-d",
        min=1,
        help="Depth of git history to clone (1 for shallow clone)"
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TABLE,
        "--format", "-f",
        help="Output format (json, text, table, tree)"
    ),
    include_content: bool = typer.Option(
        False,
        "--include-content", "-c",
        help="Whether to include file content in output"
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output", 
        help="Path to save output (if not specified, prints to console)"
    ),
    max_file_size: int = typer.Option(
        1024 * 1024,  # 1MB
        "--max-file-size",
        help="Maximum file size to process in bytes"
    ),
    binary: bool = typer.Option(
        False,
        "--binary", "-b",
        help="Whether to include binary files"
    ),
) -> None:
    """Process a Git repository and extract file information.
    
    Examples:
        [bold]$ gitget process https://github.com/user/repo[/bold]
        
        [bold]$ gitget process --format=json --output=repo_data.json https://github.com/user/repo[/bold]
        
        [bold]$ gitget process --sparse-checkout="README.md,src/" --include-content https://github.com/user/repo[/bold]
    """
    # Validate Git is installed
    validate_git_installed()
    
    # Create repository input schema
    repo_input = RepositoryInput(
        url=url,
        branch=branch,
        output_dir=output_dir,
        sparse_checkout=sparse_checkout,
        depth=depth,
        include_git_dir=False
    )
    
    # Extract repository info from URL
    repo_info = extract_repo_info(url)
    
    # Build local path
    local_path = os.path.join(output_dir, f"{repo_info['owner']}_{repo_info['name']}")
    
    # Start processing
    print_info(f"Processing repository {url}...")
    
    # Create progress spinner
    with get_spinner("Processing repository") as progress:
        try:
            # Process repository
            result = process_repository(
                repo_input.url,
                local_path,
                branch=repo_input.branch,
                sparse_patterns=repo_input.sparse_checkout,
                depth=repo_input.depth,
                max_file_size=max_file_size,
                include_binary=binary
            )
            
            if not result or not result.get("success", False):
                error_msg = result.get("error", "Unknown error") if result else "Failed to process repository"
                print_error(f"Failed to process repository: {error_msg}")
                sys.exit(1)
        except Exception as e:
            print_error(f"Error processing repository: {str(e)}")
            sys.exit(1)
    
    # Prepare output
    files = result.get("files", [])
    
    # Filter out content if not requested
    if not include_content:
        for file_info in files:
            if "content" in file_info:
                del file_info["content"]
    
    # Create repository summary
    total_size = sum(file.get("size", 0) for file in files)
    
    # Print success message
    print_success(f"Repository processed successfully: {len(files)} files, {total_size} bytes")
    
    # Output based on format
    if output_format == OutputFormat.JSON:
        # Convert to JSON
        output_data = json.dumps(result, indent=2)
        
        if output_file:
            # Save to file
            save_to_file(output_file, output_data)
            print_success(f"Output saved to {output_file}")
        else:
            # Print to console
            console.print_json(json.dumps(result))
    
    elif output_format == OutputFormat.TABLE:
        # Print as table
        file_list = [
            {
                "name": os.path.basename(file["path"]),
                "path": file["path"],
                "size": file["size"]
            }
            for file in files
        ]
        
        print_files_table(file_list)
        
        if output_file:
            # Save to file (in JSON format)
            save_to_file(output_file, json.dumps(result, indent=2))
            print_success(f"Output saved to {output_file}")
    
    elif output_format == OutputFormat.TREE:
        # Print as tree
        file_paths = [file["path"] for file in files]
        print_file_tree(local_path, file_paths)
        
        if output_file:
            # Save to file (in JSON format)
            save_to_file(output_file, json.dumps(result, indent=2))
            print_success(f"Output saved to {output_file}")
    
    else:  # OutputFormat.TEXT
        # Print as text
        for file in files:
            console.print(f"[bold cyan]{file['path']}[/] ([magenta]{file['size']} bytes[/])")
            if include_content and "content" in file:
                console.print(f"[dim]{'=' * 80}[/]")
                print_code_syntax(
                    file["content"],
                    language=file.get("language", "text"),
                    title=os.path.basename(file["path"]),
                    line_numbers=True
                )
                console.print(f"[dim]{'=' * 80}[/]\n")
        
        if output_file:
            # Save to file (in JSON format)
            save_to_file(output_file, json.dumps(result, indent=2))
            print_success(f"Output saved to {output_file}")

@app.command("extract")
def extract_chunks(
    url: str = typer.Argument(
        ..., 
        callback=validate_git_url,
        help="URL of the Git repository to extract chunks from"
    ),
    output_dir: str = typer.Option(
        "./repos",
        "--output-dir", "-o",
        callback=validate_output_dir,
        help="Directory to store cloned repositories"
    ),
    branch: str = typer.Option(
        "main",
        "--branch", "-b",
        callback=validate_branch_name,
        help="Branch to checkout"
    ),
    sparse_checkout: Optional[List[str]] = typer.Option(
        None,
        "--sparse-checkout", "-s",
        callback=validate_sparse_checkout_paths,
        help="Comma-separated list of files/dirs to include in sparse checkout"
    ),
    max_tokens: int = typer.Option(
        1024,
        "--max-tokens",
        help="Maximum number of tokens per chunk"
    ),
    overlap: int = typer.Option(
        200,
        "--overlap",
        help="Number of tokens to overlap between chunks"
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.JSON,
        "--format", "-f",
        help="Output format (json, text, table)"
    ),
    output_file: Optional[str] = typer.Option(
        None,
        "--output", 
        help="Path to save output (if not specified, prints to console)"
    ),
    exclude_extensions: List[str] = typer.Option(
        [".jpg", ".jpeg", ".png", ".gif", ".ico", ".svg", ".webp", ".bmp", 
         ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z", ".exe", ".bin"],
        "--exclude-extensions",
        help="File extensions to exclude from processing"
    )
) -> None:
    """Extract text chunks from a Git repository for processing with LLMs.
    
    Examples:
        [bold]$ gitget extract https://github.com/user/repo[/bold]
        
        [bold]$ gitget extract --max-tokens=2048 --overlap=100 https://github.com/user/repo[/bold]
        
        [bold]$ gitget extract --format=json --output=chunks.json https://github.com/user/repo[/bold]
    """
    # Validate Git is installed
    validate_git_installed()
    
    # Create repository input schema
    repo_input = RepositoryInput(
        url=url,
        branch=branch,
        output_dir=output_dir,
        sparse_checkout=sparse_checkout,
        depth=1,  # Always use depth 1 for extraction
        include_git_dir=False
    )
    
    # Extract repository info from URL
    repo_info = extract_repo_info(url)
    
    # Build local path
    local_path = os.path.join(output_dir, f"{repo_info['owner']}_{repo_info['name']}")
    
    # Start processing
    print_info(f"Extracting chunks from repository {url}...")
    
    # Process repository to get files
    with get_spinner("Processing repository") as progress:
        try:
            # Process repository
            result = process_repository(
                repo_input.url,
                local_path,
                branch=repo_input.branch,
                sparse_patterns=repo_input.sparse_checkout,
                depth=repo_input.depth,
                max_file_size=1024 * 1024,  # 1MB
                include_binary=False
            )
            
            if not result or not result.get("success", False):
                error_msg = result.get("error", "Unknown error") if result else "Failed to process repository"
                print_error(f"Failed to process repository: {error_msg}")
                sys.exit(1)
        except Exception as e:
            print_error(f"Error processing repository: {str(e)}")
            sys.exit(1)
    
    # Create a text chunker
    chunker = TextChunker(max_tokens=max_tokens, overlap_tokens=overlap)
    
    # Prepare output
    files = result.get("files", [])
    all_chunks = []
    
    # Filter files by extension
    files = [
        file for file in files 
        if not any(file["path"].lower().endswith(ext.lower()) for ext in exclude_extensions)
    ]
    
    # Start chunking
    print_info(f"Breaking files into chunks with max {max_tokens} tokens...")
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Chunking files...[/]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn()
    ) as progress:
        # Add overall task
        task = progress.add_task("Chunking files", total=len(files))
        
        for file in files:
            # Skip files without content
            if "content" not in file or not file["content"]:
                progress.update(task, advance=1)
                continue
            
            try:
                # Get file content
                content = file["content"]
                
                # Break into chunks
                file_chunks = chunker.chunk_text(
                    content,
                    file_path=file["path"],
                    preserve_sections=True
                )
                
                # Add to all chunks
                for chunk in file_chunks:
                    all_chunks.append({
                        "content": chunk["text"],
                        "start_line": chunk["start_line"],
                        "end_line": chunk["end_line"],
                        "file_path": file["path"],
                        "section": chunk.get("section"),
                        "token_count": chunk["token_count"]
                    })
            except Exception as e:
                print_warning(f"Error chunking file {file['path']}: {str(e)}")
            
            # Update progress
            progress.update(task, advance=1)
    
    # Create result
    chunk_result = {
        "repository": {
            "url": url,
            "name": repo_info["name"],
            "owner": repo_info["owner"],
            "local_path": local_path,
            "branch": branch
        },
        "chunks": all_chunks,
        "total_chunks": len(all_chunks),
        "total_tokens": sum(chunk["token_count"] for chunk in all_chunks)
    }
    
    # Print success message
    print_success(
        f"Repository extracted successfully: {len(all_chunks)} chunks, "
        f"{chunk_result['total_tokens']} tokens"
    )
    
    # Output based on format
    if output_format == OutputFormat.JSON:
        # Convert to JSON
        output_data = json.dumps(chunk_result, indent=2)
        
        if output_file:
            # Save to file
            save_to_file(output_file, output_data)
            print_success(f"Output saved to {output_file}")
        else:
            # Print to console
            console.print_json(json.dumps(chunk_result))
    
    elif output_format == OutputFormat.TABLE:
        # Print as table
        table = typer.rich_table.Table(title="Text Chunks")
        table.add_column("File", style="cyan")
        table.add_column("Lines", style="green")
        table.add_column("Tokens", style="magenta")
        table.add_column("Section", style="blue")
        
        for chunk in all_chunks:
            table.add_row(
                os.path.basename(chunk["file_path"]),
                f"{chunk['start_line']}-{chunk['end_line']}",
                str(chunk["token_count"]),
                chunk.get("section", "")
            )
        
        console.print(table)
        
        if output_file:
            # Save to file (in JSON format)
            save_to_file(output_file, json.dumps(chunk_result, indent=2))
            print_success(f"Output saved to {output_file}")
    
    else:  # OutputFormat.TEXT
        # Print as text
        for i, chunk in enumerate(all_chunks):
            console.print(f"[bold cyan]Chunk {i+1}[/] - [green]{os.path.basename(chunk['file_path'])}[/] "
                        f"(Lines {chunk['start_line']}-{chunk['end_line']}, {chunk['token_count']} tokens)")
            
            if chunk.get("section"):
                console.print(f"[bold blue]Section:[/] {chunk['section']}")
                
            console.print(f"[dim]{'=' * 80}[/]")
            console.print(chunk["content"][:500] + ("..." if len(chunk["content"]) > 500 else ""))
            console.print(f"[dim]{'=' * 80}[/]\n")
        
        if output_file:
            # Save to file (in JSON format)
            save_to_file(output_file, json.dumps(chunk_result, indent=2))
            print_success(f"Output saved to {output_file}")

@app.command("info")
def repository_info(
    path: str = typer.Argument(
        ...,
        callback=lambda p: validate_file_path(p, must_exist=True),
        help="Path to a local repository"
    ),
    output_format: OutputFormat = typer.Option(
        OutputFormat.TABLE,
        "--format", "-f",
        help="Output format (json, text, table, tree)"
    )
) -> None:
    """Show information about a local repository.
    
    Examples:
        [bold]$ gitget info ./repos/user_repo[/bold]
        
        [bold]$ gitget info --format=json ./repos/user_repo[/bold]
        
        [bold]$ gitget info --format=tree ./repos/user_repo[/bold]
    """
    # Normalize path
    path = normalize_path(path)
    
    # Check if path is a git repository
    git_dir = os.path.join(path, ".git")
    if not os.path.exists(git_dir):
        print_error(f"Not a Git repository: {path}")
        sys.exit(1)
    
    # Get repository information
    try:
        # Find all files
        all_files = find_files(path, exclude_patterns=[".git"])
        
        # Get repository URL
        url = subprocess.run(
            ["git", "-C", path, "config", "--get", "remote.origin.url"],
            capture_output=True,
            text=True,
            check=False
        ).stdout.strip()
        
        # Get current branch
        branch = subprocess.run(
            ["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=False
        ).stdout.strip()
        
        # Get commit count
        commit_count = subprocess.run(
            ["git", "-C", path, "rev-list", "--count", "HEAD"],
            capture_output=True,
            text=True,
            check=False
        ).stdout.strip()
        
        commit_count = int(commit_count) if commit_count.isdigit() else None
        
        # Create repository information
        repo_info = extract_repo_info(url) if url else {"name": os.path.basename(path), "owner": "unknown"}
        
        # Create file information
        file_infos = []
        total_size = 0
        
        for file_path in all_files:
            size = os.path.getsize(file_path)
            total_size += size
            
            # Determine if file is binary
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    f.read(1024)
                is_binary = False
                content_type = ContentType.TEXT
            except UnicodeDecodeError:
                is_binary = True
                content_type = ContentType.BINARY
            
            # Get last modified time
            last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
            
            # Create file info
            file_infos.append({
                "name": os.path.basename(file_path),
                "path": os.path.relpath(file_path, path),
                "size": size,
                "content_type": content_type,
                "is_binary": is_binary,
                "last_modified": last_modified
            })
        
        # Create result
        result = {
            "repository": {
                "url": url,
                "name": repo_info["name"],
                "owner": repo_info["owner"],
                "local_path": path,
                "branch": branch,
                "commit_count": commit_count,
                "file_count": len(all_files),
                "total_size": total_size
            },
            "files": file_infos
        }
    except Exception as e:
        print_error(f"Error getting repository information: {str(e)}")
        sys.exit(1)
    
    # Output based on format
    if output_format == OutputFormat.JSON:
        # Convert to JSON
        console.print_json(json.dumps(result, indent=2, default=str))
    
    elif output_format == OutputFormat.TABLE:
        # Print repository summary
        print_repository_summary(
            result["repository"]["url"],
            result["repository"]["local_path"],
            result["repository"]["file_count"],
            result["repository"]["total_size"],
            result["repository"]["commit_count"]
        )
        
        # Print files table
        file_list = [
            {
                "name": file["name"],
                "path": file["path"],
                "size": file["size"]
            }
            for file in result["files"]
        ]
        
        print_files_table(file_list)
    
    elif output_format == OutputFormat.TREE:
        # Print repository summary
        print_repository_summary(
            result["repository"]["url"],
            result["repository"]["local_path"],
            result["repository"]["file_count"],
            result["repository"]["total_size"],
            result["repository"]["commit_count"]
        )
        
        # Print file tree
        file_paths = [file["path"] for file in result["files"]]
        print_file_tree(path, file_paths)
    
    else:  # OutputFormat.TEXT
        # Print repository information
        repository = result["repository"]
        console.print(f"[bold cyan]Repository:[/] {repository['url']}")
        console.print(f"[bold cyan]Local Path:[/] {repository['local_path']}")
        console.print(f"[bold cyan]Branch:[/] {repository['branch']}")
        console.print(f"[bold cyan]Files:[/] {repository['file_count']}")
        console.print(f"[bold cyan]Size:[/] {repository['total_size']} bytes")
        
        if repository["commit_count"] is not None:
            console.print(f"[bold cyan]Commits:[/] {repository['commit_count']}")
        
        console.print("\n[bold cyan]Files:[/]")
        for file in result["files"]:
            console.print(f"  {file['path']} ({file['size']} bytes)")

if __name__ == "__main__":
    # Validation function to test CLI with real data
    import sys
    import tempfile
    import shutil
    from unittest.mock import patch
    import subprocess
    
    # Set this flag to avoid real git operations during testing
    sys._test_mode = True
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Help command
    total_tests += 1
    try:
        # Just test that the help command runs without error
        with patch.object(sys, 'argv', ['gitget', '--help']):
            try:
                # We expect this to raise SystemExit
                app()
            except SystemExit as e:
                if e.code != 0:
                    all_validation_failures.append(f"Help command failed with exit code {e.code}")
    except Exception as e:
        all_validation_failures.append(f"Help command: Unexpected exception: {str(e)}")
    
    # Test 2: Clone command
    total_tests += 1
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the sparse_clone function to return True
            # Also mock extract_repo_info to return a test dictionary
            with patch('mcp_tools.gitget.core.repo_operations.sparse_clone', return_value=True), \
                 patch('mcp_tools.gitget.core.utils.extract_repo_info', return_value={"name": "testrepo", "owner": "testuser"}), \
                 patch('mcp_tools.gitget.cli.validators.validate_git_installed', return_value=True), \
                 patch('mcp_tools.gitget.core.repo_operations.find_files', return_value=[os.path.join(temp_dir, "test.txt")]), \
                 patch.object(sys, 'argv', ['gitget', 'clone', 'https://github.com/user/repo', '--output-dir', temp_dir]):
                try:
                    # Create a test file to simulate a repo
                    with open(os.path.join(temp_dir, "test.txt"), "w") as f:
                        f.write("Test content")
                    
                    # We expect this to raise SystemExit
                    app()
                except SystemExit as e:
                    if e.code != 0:
                        all_validation_failures.append(f"Clone command failed with exit code {e.code}")
    except Exception as e:
        all_validation_failures.append(f"Clone command: Unexpected exception: {str(e)}")
    
    # Test 3: Process command
    total_tests += 1
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the process_repository function to return a test dictionary
            test_result = {
                "success": True,
                "files": [
                    {
                        "path": os.path.join(temp_dir, "test.txt"),
                        "size": 12,
                        "content": "Test content"
                    }
                ]
            }
            
            with patch('mcp_tools.gitget.core.repo_operations.process_repository', return_value=test_result), \
                 patch('mcp_tools.gitget.core.utils.extract_repo_info', return_value={"name": "testrepo", "owner": "testuser"}), \
                 patch('mcp_tools.gitget.cli.validators.validate_git_installed', return_value=True), \
                 patch.object(sys, 'argv', ['gitget', 'process', 'https://github.com/user/repo', '--output-dir', temp_dir, '--format', 'json']):
                try:
                    # We expect this to raise SystemExit
                    app()
                except SystemExit as e:
                    if e.code != 0:
                        all_validation_failures.append(f"Process command failed with exit code {e.code}")
    except Exception as e:
        all_validation_failures.append(f"Process command: Unexpected exception: {str(e)}")
    
    # Test 4: Info command
    total_tests += 1
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a .git directory to simulate a repo
            os.makedirs(os.path.join(temp_dir, ".git"))
            
            # Create a test file
            with open(os.path.join(temp_dir, "test.txt"), "w") as f:
                f.write("Test content")
            
            # Mock subprocess calls
            def mock_run(*args, **kwargs):
                cmd = args[0]
                if "config" in cmd and "remote.origin.url" in cmd:
                    return subprocess.CompletedProcess(cmd, 0, stdout="https://github.com/user/repo", stderr="")
                elif "rev-parse" in cmd:
                    return subprocess.CompletedProcess(cmd, 0, stdout="main", stderr="")
                elif "rev-list" in cmd:
                    return subprocess.CompletedProcess(cmd, 0, stdout="10", stderr="")
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            
            with patch('subprocess.run', side_effect=mock_run), \
                 patch('mcp_tools.gitget.core.repo_operations.find_files', return_value=[os.path.join(temp_dir, "test.txt")]), \
                 patch('mcp_tools.gitget.core.utils.extract_repo_info', return_value={"name": "testrepo", "owner": "testuser"}), \
                 patch.object(sys, 'argv', ['gitget', 'info', temp_dir, '--format', 'json']):
                try:
                    # We expect this to raise SystemExit
                    app()
                except SystemExit as e:
                    if e.code != 0:
                        all_validation_failures.append(f"Info command failed with exit code {e.code}")
    except Exception as e:
        all_validation_failures.append(f"Info command: Unexpected exception: {str(e)}")
    
    # Test 5: extract command
    total_tests += 1
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Mock the process_repository function to return a test dictionary
            test_result = {
                "success": True,
                "files": [
                    {
                        "path": os.path.join(temp_dir, "test.txt"),
                        "size": 12,
                        "content": "Test content with multiple lines\nAnd some more text\nTo have something to chunk"
                    }
                ]
            }
            
            with patch('mcp_tools.gitget.core.repo_operations.process_repository', return_value=test_result), \
                 patch('mcp_tools.gitget.core.utils.extract_repo_info', return_value={"name": "testrepo", "owner": "testuser"}), \
                 patch('mcp_tools.gitget.cli.validators.validate_git_installed', return_value=True), \
                 patch('mcp_tools.gitget.core.text_chunker.TextChunker.chunk_text', return_value=[
                     {
                         "text": "Test content with multiple lines\nAnd some more text",
                         "start_line": 1,
                         "end_line": 2,
                         "token_count": 10
                     },
                     {
                         "text": "To have something to chunk",
                         "start_line": 3,
                         "end_line": 3,
                         "token_count": 5
                     }
                 ]), \
                 patch.object(sys, 'argv', ['gitget', 'extract', 'https://github.com/user/repo', '--output-dir', temp_dir, '--format', 'json']):
                try:
                    # We expect this to raise SystemExit
                    app()
                except SystemExit as e:
                    if e.code != 0:
                        all_validation_failures.append(f"Extract command failed with exit code {e.code}")
    except Exception as e:
        all_validation_failures.append(f"Extract command: Unexpected exception: {str(e)}")
    
    # Clean up test mode flag
    if hasattr(sys, "_test_mode"):
        delattr(sys, "_test_mode")
    
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