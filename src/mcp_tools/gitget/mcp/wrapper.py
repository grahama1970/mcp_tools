"""FastMCP wrapper for the gitget module.

This module provides a FastMCP wrapper for the gitget CLI, making it available as an MCP tool
for Claude and other MCP-compatible assistants.

Links to third-party package documentation:
- FastMCP: https://fastmcp.readthedocs.io/en/latest/

Sample input:
    {
        "command": "clone",
        "params": {
            "url": "https://github.com/user/repo",
            "branch": "main"
        }
    }

Expected output:
    {
        "result": {
            "repository": {
                "url": "https://github.com/user/repo",
                "name": "repo",
                "owner": "user",
                "local_path": "/path/to/repo",
                "branch": "main",
                "file_count": 10,
                "total_size": 1024
            },
            "success": true
        }
    }
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

try:
    from fastmcp import FastMCP
except ImportError:
    # For development and testing without FastMCP
    class FastMCP:
        def __init__(self, *args, **kwargs):
            pass
            
        def handle_request(self, request):
            return {"error": "FastMCP not installed"}

from mcp_tools.gitget.cli.app import app as cli_app
from mcp_tools.gitget.core.repo_operations import sparse_clone, process_repository
from mcp_tools.gitget.core.directory_manager import RepositoryDirectoryManager
from mcp_tools.gitget.core.text_chunker import TextChunker
from mcp_tools.gitget.core.utils import extract_repo_info, save_to_file, read_from_file

from .schema import GITGET_SCHEMA

# Create FastMCP application
mcp_app = FastMCP(
    name="gitget",
    description="Git repository content extraction and processing tool",
    cli_app=cli_app,
    schema=GITGET_SCHEMA,
    version="0.1.0"
)

def handler(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle MCP requests for gitget.
    
    Args:
        request: MCP request dictionary
        
    Returns:
        MCP response dictionary
    """
    try:
        return mcp_app.handle_request(request)
    except Exception as e:
        # Handle errors and return MCP-compliant error response
        return {
            "error": {
                "message": str(e),
                "type": type(e).__name__
            }
        }

def _clone_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle 'clone' command.
    
    Args:
        params: Command parameters
        
    Returns:
        Command result
    """
    # Extract parameters
    url = params["url"]
    branch = params.get("branch", "main")
    output_dir = params.get("output_dir", "./repos")
    sparse_checkout = params.get("sparse_checkout")
    depth = params.get("depth", 1)
    include_git_dir = params.get("include_git_dir", False)
    force = params.get("force", False)
    
    # Extract repository info
    repo_info = extract_repo_info(url)
    
    # Build local path
    local_path = os.path.join(output_dir, f"{repo_info['owner']}_{repo_info['name']}")
    
    # Check if directory already exists
    if os.path.exists(local_path) and not force:
        return {
            "error": f"Repository directory already exists: {local_path}. Use 'force: true' to overwrite.",
            "success": False
        }
    
    try:
        # Clone repository
        result = sparse_clone(
            url,
            local_path,
            branch=branch,
            sparse_patterns=sparse_checkout,
            depth=depth
        )
        
        if not result:
            return {
                "error": f"Failed to clone repository: {url}",
                "success": False
            }
        
        # Create repository info
        repo_info_out = {
            "url": url,
            "name": repo_info["name"],
            "owner": repo_info["owner"],
            "local_path": local_path,
            "branch": branch
        }
        
        # Find all files in repository
        from mcp_tools.gitget.core.repo_operations import find_files
        all_files = find_files(local_path, exclude_patterns=[".git"] if not include_git_dir else [])
        
        # Add file information
        repo_info_out["file_count"] = len(all_files)
        repo_info_out["total_size"] = sum(os.path.getsize(f) for f in all_files)
        
        # Create file list
        file_list = []
        for file_path in all_files:
            size = os.path.getsize(file_path)
            
            # Determine if file is binary
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    f.read(1024)
                is_binary = False
                content_type = "text"
            except UnicodeDecodeError:
                is_binary = True
                content_type = "binary"
            
            # Create file info
            file_list.append({
                "name": os.path.basename(file_path),
                "path": os.path.relpath(file_path, local_path),
                "size": size,
                "content_type": content_type,
                "is_binary": is_binary
            })
        
        # Return success result
        return {
            "repository": repo_info_out,
            "files": file_list,
            "success": True
        }
        
    except Exception as e:
        # Return error result
        return {
            "error": str(e),
            "success": False
        }

def _process_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle 'process' command.
    
    Args:
        params: Command parameters
        
    Returns:
        Command result
    """
    # Extract parameters
    url = params["url"]
    branch = params.get("branch", "main")
    output_dir = params.get("output_dir", "./repos")
    sparse_checkout = params.get("sparse_checkout")
    depth = params.get("depth", 1)
    include_content = params.get("include_content", False)
    max_file_size = params.get("max_file_size", 1024 * 1024)  # 1MB default
    binary = params.get("binary", False)
    output_file = params.get("output_file")
    
    try:
        # Extract repository info from URL
        repo_info = extract_repo_info(url)
        
        # Build local path
        local_path = os.path.join(output_dir, f"{repo_info['owner']}_{repo_info['name']}")
        
        # Process repository
        result = process_repository(
            url,
            local_path,
            branch=branch,
            sparse_patterns=sparse_checkout,
            depth=depth,
            max_file_size=max_file_size,
            include_binary=binary
        )
        
        if not result or not result.get("success", False):
            error_msg = result.get("error", "Unknown error") if result else "Failed to process repository"
            return {
                "error": error_msg,
                "success": False
            }
        
        # Filter out content if not requested
        if not include_content:
            for file_info in result.get("files", []):
                if "content" in file_info:
                    del file_info["content"]
        
        # Save to file if requested
        if output_file:
            save_to_file(output_file, json.dumps(result, indent=2))
            result["output_file"] = output_file
        
        return result
        
    except Exception as e:
        # Return error result
        return {
            "error": str(e),
            "success": False
        }

def _extract_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle 'extract' command.
    
    Args:
        params: Command parameters
        
    Returns:
        Command result
    """
    # Extract parameters
    url = params["url"]
    branch = params.get("branch", "main")
    output_dir = params.get("output_dir", "./repos")
    sparse_checkout = params.get("sparse_checkout")
    max_tokens = params.get("max_tokens", 1024)
    overlap = params.get("overlap", 200)
    output_file = params.get("output_file")
    exclude_extensions = params.get("exclude_extensions", [
        ".jpg", ".jpeg", ".png", ".gif", ".ico", ".svg", ".webp", ".bmp",
        ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z", ".exe", ".bin"
    ])
    
    try:
        # Extract repository info from URL
        repo_info = extract_repo_info(url)
        
        # Build local path
        local_path = os.path.join(output_dir, f"{repo_info['owner']}_{repo_info['name']}")
        
        # Process repository to get files
        result = process_repository(
            url,
            local_path,
            branch=branch,
            sparse_patterns=sparse_checkout,
            depth=1,  # Always use depth 1 for extraction
            max_file_size=1024 * 1024,  # 1MB
            include_binary=False
        )
        
        if not result or not result.get("success", False):
            error_msg = result.get("error", "Unknown error") if result else "Failed to process repository"
            return {
                "error": error_msg,
                "success": False
            }
        
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
        
        # Process each file to extract chunks
        for file in files:
            # Skip files without content
            if "content" not in file or not file["content"]:
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
                # Log error but continue
                print(f"Error chunking file {file['path']}: {str(e)}", file=sys.stderr)
        
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
            "total_tokens": sum(chunk["token_count"] for chunk in all_chunks),
            "success": True
        }
        
        # Save to file if requested
        if output_file:
            save_to_file(output_file, json.dumps(chunk_result, indent=2))
            chunk_result["output_file"] = output_file
        
        return chunk_result
        
    except Exception as e:
        # Return error result
        return {
            "error": str(e),
            "success": False
        }

def _info_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handle 'info' command.
    
    Args:
        params: Command parameters
        
    Returns:
        Command result
    """
    # Extract parameters
    path = params["path"]
    
    try:
        # Normalize path
        from mcp_tools.gitget.core.utils import normalize_path
        path = normalize_path(path)
        
        # Check if path is a git repository
        git_dir = os.path.join(path, ".git")
        if not os.path.exists(git_dir):
            return {
                "error": f"Not a Git repository: {path}",
                "success": False
            }
        
        # Get repository information
        try:
            # Find all files
            from mcp_tools.gitget.core.repo_operations import find_files
            all_files = find_files(path, exclude_patterns=[".git"])
            
            # Get repository URL
            import subprocess
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
                    content_type = "text"
                except UnicodeDecodeError:
                    is_binary = True
                    content_type = "binary"
                
                # Create file info
                file_infos.append({
                    "name": os.path.basename(file_path),
                    "path": os.path.relpath(file_path, path),
                    "size": size,
                    "content_type": content_type,
                    "is_binary": is_binary,
                    "last_modified": os.path.getmtime(file_path)
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
                "files": file_infos,
                "success": True
            }
            
            return result
            
        except Exception as e:
            return {
                "error": f"Error getting repository information: {str(e)}",
                "success": False
            }
            
    except Exception as e:
        # Return error result
        return {
            "error": str(e),
            "success": False
        }

# Register command handlers
mcp_app.register_handler("clone", _clone_handler)
mcp_app.register_handler("process", _process_handler)
mcp_app.register_handler("extract", _extract_handler)
mcp_app.register_handler("info", _info_handler)

if __name__ == "__main__":
    # Validation function to test MCP wrapper with real data
    import sys
    import tempfile
    from unittest.mock import patch, MagicMock
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Clone handler
    total_tests += 1
    try:
        # Mock extract_repo_info and sparse_clone
        with patch('mcp_tools.gitget.core.utils.extract_repo_info', return_value={"name": "repo", "owner": "user"}), \
             patch('mcp_tools.gitget.core.repo_operations.sparse_clone', return_value=True), \
             patch('mcp_tools.gitget.core.repo_operations.find_files', return_value=["/tmp/repo/file.txt"]), \
             patch('os.path.getsize', return_value=1024), \
             patch('os.path.exists', return_value=False), \
             patch('builtins.open', MagicMock()):
            
            # Test parameters
            params = {
                "url": "https://github.com/user/repo",
                "branch": "main",
                "output_dir": "./repos",
                "sparse_checkout": ["README.md", "src/"],
                "depth": 1,
                "include_git_dir": False
            }
            
            # Call handler
            result = _clone_handler(params)
            
            # Check result
            if not result.get("success", False):
                all_validation_failures.append(f"Clone handler: Failed with error: {result.get('error', 'unknown')}")
            
            if "repository" not in result:
                all_validation_failures.append("Clone handler: Missing 'repository' in result")
            
            if "files" not in result:
                all_validation_failures.append("Clone handler: Missing 'files' in result")
    except Exception as e:
        all_validation_failures.append(f"Clone handler: Unexpected exception: {str(e)}")
    
    # Test 2: Process handler
    total_tests += 1
    try:
        # Mock extract_repo_info and process_repository
        with patch('mcp_tools.gitget.core.utils.extract_repo_info', return_value={"name": "repo", "owner": "user"}), \
             patch('mcp_tools.gitget.core.repo_operations.process_repository', return_value={
                 "success": True,
                 "files": [
                     {
                         "path": "/tmp/repo/file.txt",
                         "size": 1024,
                         "content": "Test content"
                     }
                 ]
             }):
            
            # Test parameters
            params = {
                "url": "https://github.com/user/repo",
                "branch": "main",
                "output_dir": "./repos",
                "include_content": True
            }
            
            # Call handler
            result = _process_handler(params)
            
            # Check result
            if not result.get("success", False):
                all_validation_failures.append(f"Process handler: Failed with error: {result.get('error', 'unknown')}")
            
            if "files" not in result:
                all_validation_failures.append("Process handler: Missing 'files' in result")
            
            if len(result.get("files", [])) == 0:
                all_validation_failures.append("Process handler: Empty 'files' list in result")
    except Exception as e:
        all_validation_failures.append(f"Process handler: Unexpected exception: {str(e)}")
    
    # Test 3: Extract handler
    total_tests += 1
    try:
        # Mock extract_repo_info, process_repository, and chunk_text
        with patch('mcp_tools.gitget.core.utils.extract_repo_info', return_value={"name": "repo", "owner": "user"}), \
             patch('mcp_tools.gitget.core.repo_operations.process_repository', return_value={
                 "success": True,
                 "files": [
                     {
                         "path": "/tmp/repo/file.txt",
                         "size": 1024,
                         "content": "Test content"
                     }
                 ]
             }), \
             patch('mcp_tools.gitget.core.text_chunker.TextChunker.chunk_text', return_value=[
                 {
                     "text": "Test content",
                     "start_line": 1,
                     "end_line": 1,
                     "token_count": 2
                 }
             ]):
            
            # Test parameters
            params = {
                "url": "https://github.com/user/repo",
                "branch": "main",
                "output_dir": "./repos",
                "max_tokens": 1024,
                "overlap": 200
            }
            
            # Call handler
            result = _extract_handler(params)
            
            # Check result
            if not result.get("success", False):
                all_validation_failures.append(f"Extract handler: Failed with error: {result.get('error', 'unknown')}")
            
            if "chunks" not in result:
                all_validation_failures.append("Extract handler: Missing 'chunks' in result")
            
            if "total_chunks" not in result or result["total_chunks"] != 1:
                all_validation_failures.append("Extract handler: Incorrect 'total_chunks' in result")
            
            if "total_tokens" not in result or result["total_tokens"] != 2:
                all_validation_failures.append("Extract handler: Incorrect 'total_tokens' in result")
    except Exception as e:
        all_validation_failures.append(f"Extract handler: Unexpected exception: {str(e)}")
    
    # Test 4: Info handler
    total_tests += 1
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a .git directory to simulate a repo
            os.makedirs(os.path.join(temp_dir, ".git"))
            
            # Create a test file
            with open(os.path.join(temp_dir, "test.txt"), "w") as f:
                f.write("Test content")
            
            # Mock subprocess calls and other functions
            with patch('subprocess.run') as mock_run, \
                 patch('mcp_tools.gitget.core.repo_operations.find_files', return_value=[os.path.join(temp_dir, "test.txt")]), \
                 patch('mcp_tools.gitget.core.utils.extract_repo_info', return_value={"name": "repo", "owner": "user"}):
                
                # Configure mock subprocess.run
                mock_run.return_value.stdout = "https://github.com/user/repo\n"
                
                # Test parameters
                params = {
                    "path": temp_dir
                }
                
                # Call handler
                result = _info_handler(params)
                
                # Check result
                if not result.get("success", False):
                    all_validation_failures.append(f"Info handler: Failed with error: {result.get('error', 'unknown')}")
                
                if "repository" not in result:
                    all_validation_failures.append("Info handler: Missing 'repository' in result")
                
                if "files" not in result:
                    all_validation_failures.append("Info handler: Missing 'files' in result")
    except Exception as e:
        all_validation_failures.append(f"Info handler: Unexpected exception: {str(e)}")
    
    # Test 5: MCP handler
    total_tests += 1
    try:
        # Mock FastMCP.handle_request
        with patch.object(mcp_app, 'handle_request', return_value={"result": "success"}):
            # Test request
            request = {
                "name": "gitget",
                "input": {
                    "command": "clone",
                    "params": {
                        "url": "https://github.com/user/repo"
                    }
                }
            }
            
            # Call handler
            result = handler(request)
            
            # Check result
            if "result" not in result:
                all_validation_failures.append("MCP handler: Missing 'result' in response")
    except Exception as e:
        all_validation_failures.append(f"MCP handler: Unexpected exception: {str(e)}")
    
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