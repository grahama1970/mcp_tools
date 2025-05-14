"""Input validators for the gitget CLI.

This module provides validation functions for CLI input parameters in the gitget module.
It helps ensure that user inputs are properly formatted and conform to expected formats.

Links to third-party package documentation:
- Typer: https://typer.tiangolo.com/
- Rich: https://rich.readthedocs.io/en/latest/

Sample input:
    is_valid = validate_git_url("https://github.com/user/repo")
    is_valid = validate_output_dir("/tmp/repos")
    is_valid = validate_file_path("/path/to/file.py")

Expected output:
    True  # If validation passes
    # OR raises a typer.BadParameter exception with a helpful error message
"""

import os
import re
import typer
from pathlib import Path
from typing import List, Optional, Union
import subprocess
from urllib.parse import urlparse

from .formatters import print_error, print_warning

def validate_git_url(url: str) -> str:
    """Validate a Git repository URL.
    
    Args:
        url: The repository URL to validate
        
    Returns:
        The validated URL
        
    Raises:
        typer.BadParameter: If the URL is invalid
    """
    # Check for empty URL
    if not url:
        raise typer.BadParameter("Repository URL cannot be empty")
    
    # HTTP(S) URL format
    if url.startswith(("http://", "https://")):
        parsed = urlparse(url)
        
        # Check for valid domain
        if not parsed.netloc:
            raise typer.BadParameter(f"Invalid Git URL: {url}. URL must include a domain name.")
        
        # Check for common Git hosting services
        if not any(host in parsed.netloc for host in ["github.com", "gitlab.com", "bitbucket.org"]):
            # Not a common host, but might still be valid - issue a warning
            print_warning(f"URL domain '{parsed.netloc}' is not a common Git hosting service")
        
        # Git URLs typically end with .git or have no extension
        if not (parsed.path.endswith(".git") or "." not in os.path.basename(parsed.path)):
            # Not ending with .git - could be a GitHub URL without .git extension
            if "github.com" in parsed.netloc and len(parsed.path.strip("/").split("/")) >= 2:
                # Likely a GitHub repository URL without .git extension
                pass
            else:
                print_warning(f"URL does not end with .git extension: {url}")
        
        return url
        
    # SSH URL format (git@github.com:user/repo.git)
    elif url.startswith("git@"):
        ssh_pattern = r'^git@([a-zA-Z0-9.-]+):([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)(.git)?$'
        
        if not re.match(ssh_pattern, url):
            raise typer.BadParameter(
                f"Invalid Git SSH URL: {url}. Format should be git@domain:user/repo.git"
            )
            
        return url
        
    # Neither HTTP(S) nor SSH format
    else:
        raise typer.BadParameter(
            f"Invalid Git URL: {url}. URL must start with 'http://', 'https://', or 'git@'"
        )

def validate_output_dir(path: str) -> str:
    """Validate and create an output directory if it doesn't exist.
    
    Args:
        path: The directory path to validate
        
    Returns:
        The validated directory path
        
    Raises:
        typer.BadParameter: If the path is invalid or cannot be created
    """
    # Check for empty path
    if not path:
        raise typer.BadParameter("Output directory path cannot be empty")
    
    # Expand user directory and environment variables
    expanded_path = os.path.expanduser(os.path.expandvars(path))
    
    # Create directory if it doesn't exist
    try:
        os.makedirs(expanded_path, exist_ok=True)
    except OSError as e:
        raise typer.BadParameter(f"Could not create directory '{expanded_path}': {e}")
    
    # Check if the directory is writable
    if not os.access(expanded_path, os.W_OK):
        raise typer.BadParameter(f"Directory '{expanded_path}' is not writable")
    
    return expanded_path

def validate_branch_name(branch: str) -> str:
    """Validate a Git branch name.
    
    Args:
        branch: The branch name to validate
        
    Returns:
        The validated branch name
        
    Raises:
        typer.BadParameter: If the branch name is invalid
    """
    # Check for empty branch name
    if not branch:
        raise typer.BadParameter("Branch name cannot be empty")
    
    # Git branch naming rules (simplified)
    # Cannot have: space, ~, ^, :, ?, *, [, \, control characters
    invalid_chars = [' ', '~', '^', ':', '?', '*', '[', '\\']
    if any(char in branch for char in invalid_chars):
        raise typer.BadParameter(
            f"Invalid branch name '{branch}'. Cannot contain any of these characters: {' '.join(invalid_chars)}"
        )
    
    # Cannot start with '/'
    if branch.startswith('/'):
        raise typer.BadParameter(f"Invalid branch name '{branch}'. Cannot start with '/'")
    
    # Cannot end with '/'
    if branch.endswith('/'):
        raise typer.BadParameter(f"Invalid branch name '{branch}'. Cannot end with '/'")
    
    # Cannot end with '.lock'
    if branch.endswith('.lock'):
        raise typer.BadParameter(f"Invalid branch name '{branch}'. Cannot end with '.lock'")
    
    # Cannot be a single '.'
    if branch == '.':
        raise typer.BadParameter("Invalid branch name '.'. Cannot be a single dot")
    
    # Cannot be a double '..'
    if branch == '..':
        raise typer.BadParameter("Invalid branch name '..'. Cannot be double dots")
    
    return branch

def validate_sparse_checkout_paths(paths: List[str]) -> List[str]:
    """Validate paths for sparse checkout.
    
    Args:
        paths: List of paths to validate
        
    Returns:
        The validated list of paths
        
    Raises:
        typer.BadParameter: If any path is invalid
    """
    # Check for empty list
    if not paths:
        return paths
    
    # Validate each path
    for path in paths:
        # Check for empty path
        if not path:
            raise typer.BadParameter("Sparse checkout path cannot be empty")
        
        # Don't allow absolute paths
        if os.path.isabs(path):
            raise typer.BadParameter(f"Sparse checkout path '{path}' cannot be absolute")
        
        # Don't allow paths that try to escape the repository
        if '..' in path.split('/'):
            raise typer.BadParameter(f"Sparse checkout path '{path}' cannot contain '..'")
    
    return paths

def validate_git_installed() -> bool:
    """Check if Git is installed on the system.
    
    Returns:
        True if Git is installed, False otherwise
        
    Raises:
        typer.BadParameter: If Git is not installed
    """
    try:
        result = subprocess.run(
            ["git", "--version"], 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        if result.returncode != 0:
            raise typer.BadParameter(
                "Git is not available on this system. Please install Git before using this tool."
            )
            
        return True
        
    except FileNotFoundError:
        raise typer.BadParameter(
            "Git is not installed on this system. Please install Git before using this tool."
        )

def validate_file_path(path: str, must_exist: bool = True) -> str:
    """Validate a file path.
    
    Args:
        path: The file path to validate
        must_exist: Whether the file must exist
        
    Returns:
        The validated file path
        
    Raises:
        typer.BadParameter: If the path is invalid or the file doesn't exist
    """
    # Check for empty path
    if not path:
        raise typer.BadParameter("File path cannot be empty")
    
    # Expand user directory and environment variables
    expanded_path = os.path.expanduser(os.path.expandvars(path))
    
    # Check if file exists (if required)
    if must_exist and not os.path.exists(expanded_path):
        raise typer.BadParameter(f"File '{expanded_path}' does not exist")
    
    return expanded_path

def validate_extension(path: str, allowed_extensions: List[str]) -> str:
    """Validate a file path has an allowed extension.
    
    Args:
        path: The file path to validate
        allowed_extensions: List of allowed file extensions (e.g., ['.py', '.txt'])
        
    Returns:
        The validated file path
        
    Raises:
        typer.BadParameter: If the file doesn't have an allowed extension
    """
    # Check for empty path
    if not path:
        raise typer.BadParameter("File path cannot be empty")
    
    # Get file extension
    _, ext = os.path.splitext(path)
    
    # Check if extension is allowed
    if ext.lower() not in [e.lower() for e in allowed_extensions]:
        raise typer.BadParameter(
            f"File '{path}' has invalid extension '{ext}'. "
            f"Allowed extensions: {', '.join(allowed_extensions)}"
        )
    
    return path

if __name__ == "__main__":
    # Validation function to test validators with real data
    import sys
    import tempfile
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Git URL validation - valid URLs
    total_tests += 1
    try:
        valid_urls = [
            "https://github.com/user/repo.git",
            "https://github.com/user/repo",
            "git@github.com:user/repo.git",
            "git@gitlab.com:user/repo.git",
            "https://bitbucket.org/user/repo.git"
        ]
        
        for url in valid_urls:
            try:
                result = validate_git_url(url)
                if result != url:
                    all_validation_failures.append(f"Git URL validation: Expected '{url}' to be returned, got '{result}'")
            except typer.BadParameter as e:
                all_validation_failures.append(f"Git URL validation: Valid URL '{url}' raised exception: {str(e)}")
    except Exception as e:
        all_validation_failures.append(f"Git URL validation (valid): Unexpected exception: {str(e)}")
    
    # Test 2: Git URL validation - invalid URLs
    total_tests += 1
    try:
        invalid_urls = [
            "",  # Empty URL
            "not-a-git-url",  # Not a URL
            "http://",  # No domain
            "git@",  # Incomplete SSH URL
            "ftp://github.com/user/repo.git"  # Wrong protocol
        ]
        
        for url in invalid_urls:
            try:
                validate_git_url(url)
                all_validation_failures.append(f"Git URL validation: Invalid URL '{url}' did not raise exception")
            except typer.BadParameter:
                # This is expected for invalid URLs
                pass
    except Exception as e:
        all_validation_failures.append(f"Git URL validation (invalid): Unexpected exception: {str(e)}")
    
    # Test 3: Output directory validation
    total_tests += 1
    try:
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Valid directory - exists
            result = validate_output_dir(temp_dir)
            if result != temp_dir:
                all_validation_failures.append(f"Output dir validation: Expected '{temp_dir}' to be returned, got '{result}'")
            
            # Valid directory - doesn't exist but can be created
            new_dir = os.path.join(temp_dir, "new_dir")
            result = validate_output_dir(new_dir)
            if result != new_dir or not os.path.exists(new_dir):
                all_validation_failures.append(f"Output dir validation: Directory '{new_dir}' was not created or returned")
            
            # Test expansion of user directory
            if "~" in os.path.expanduser("~"):
                all_validation_failures.append("Output dir validation: User directory expansion not working")
    except Exception as e:
        all_validation_failures.append(f"Output dir validation: Unexpected exception: {str(e)}")
    
    # Test 4: Branch name validation
    total_tests += 1
    try:
        valid_branches = [
            "main",
            "develop",
            "feature/new-feature",
            "fix-123",
            "v1.0.0"
        ]
        
        for branch in valid_branches:
            try:
                result = validate_branch_name(branch)
                if result != branch:
                    all_validation_failures.append(f"Branch validation: Expected '{branch}' to be returned, got '{result}'")
            except typer.BadParameter as e:
                all_validation_failures.append(f"Branch validation: Valid branch '{branch}' raised exception: {str(e)}")
                
        invalid_branches = [
            "",  # Empty
            "branch with space",  # Contains space
            "branch*",  # Contains wildcard
            "branch?",  # Contains question mark
            "/branch",  # Starts with slash
            "branch/",  # Ends with slash
            "branch.lock"  # Ends with .lock
        ]
        
        for branch in invalid_branches:
            try:
                validate_branch_name(branch)
                all_validation_failures.append(f"Branch validation: Invalid branch '{branch}' did not raise exception")
            except typer.BadParameter:
                # This is expected for invalid branches
                pass
    except Exception as e:
        all_validation_failures.append(f"Branch validation: Unexpected exception: {str(e)}")
    
    # Test 5: Sparse checkout paths validation
    total_tests += 1
    try:
        valid_paths = [
            ["README.md"],
            ["src/", "docs/"],
            ["file.txt", "dir/subdir/file.py"]
        ]
        
        for paths in valid_paths:
            try:
                result = validate_sparse_checkout_paths(paths)
                if result != paths:
                    all_validation_failures.append(f"Sparse checkout validation: Expected '{paths}' to be returned, got '{result}'")
            except typer.BadParameter as e:
                all_validation_failures.append(f"Sparse checkout validation: Valid paths '{paths}' raised exception: {str(e)}")
                
        invalid_paths = [
            ["/absolute/path"],  # Absolute path
            ["../escape/path"],  # Path with ..
            ["path", ""]  # Contains empty path
        ]
        
        for paths in invalid_paths:
            try:
                validate_sparse_checkout_paths(paths)
                all_validation_failures.append(f"Sparse checkout validation: Invalid paths '{paths}' did not raise exception")
            except typer.BadParameter:
                # This is expected for invalid paths
                pass
    except Exception as e:
        all_validation_failures.append(f"Sparse checkout validation: Unexpected exception: {str(e)}")
    
    # Test 6: File extension validation
    total_tests += 1
    try:
        valid_extensions = [
            ("file.py", [".py", ".txt"]),
            ("file.txt", [".py", ".txt"]),
            ("file.PY", [".py", ".txt"])  # Case insensitive
        ]
        
        for path, extensions in valid_extensions:
            try:
                result = validate_extension(path, extensions)
                if result != path:
                    all_validation_failures.append(f"Extension validation: Expected '{path}' to be returned, got '{result}'")
            except typer.BadParameter as e:
                all_validation_failures.append(f"Extension validation: Valid path '{path}' raised exception: {str(e)}")
                
        invalid_extensions = [
            ("file.js", [".py", ".txt"]),
            ("file", [".py", ".txt"]),
            ("file.py.bak", [".py", ".txt"])
        ]
        
        for path, extensions in invalid_extensions:
            try:
                validate_extension(path, extensions)
                all_validation_failures.append(f"Extension validation: Invalid path '{path}' did not raise exception")
            except typer.BadParameter:
                # This is expected for invalid paths
                pass
    except Exception as e:
        all_validation_failures.append(f"Extension validation: Unexpected exception: {str(e)}")
    
    # Test 7: Git installation check
    total_tests += 1
    try:
        # We'll just check that it doesn't raise an exception if Git is installed
        # If Git is not installed, this will fail, but that's an environment issue, not a code issue
        try:
            result = validate_git_installed()
            if not result:
                all_validation_failures.append("Git installation check: Expected True to be returned")
        except typer.BadParameter:
            # This could happen if Git is not installed
            # We'll assume it's a valid test result
            pass
    except Exception as e:
        all_validation_failures.append(f"Git installation check: Unexpected exception: {str(e)}")
    
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