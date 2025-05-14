#!/usr/bin/env python3
"""
Git Repository Operations Module

This module provides core functionality for working with Git repositories,
including sparse cloning, file operations, and repository analysis.

This module is part of the Core Layer and should have no dependencies on
Presentation or Integration layers.

Links to documentation:
- Git: https://git-scm.com/docs
- GitPython: https://gitpython.readthedocs.io/

Sample input:
- repo_url: "https://github.com/username/repository"
- extensions: ["md", "py"]
- clone_dir: "/path/to/output"
- files: ["README.md", "src/main.py"]
- dirs: ["docs/", "src/"]

Expected output:
- Dictionary with success status, clone directory, and list of found files
- Error information if the operation fails
"""

import os
import shutil
import subprocess
import tempfile
from typing import List, Dict, Any, Optional, Union, Tuple
import uuid

from loguru import logger

# Utility function for extracting repo name from URL
def extract_repo_name_from_url(repo_url: str) -> str:
    """
    Extract repository name from URL.
    
    Args:
        repo_url: Repository URL
        
    Returns:
        str: Repository name
    """
    return repo_url.rstrip('/').split('/')[-1]

def sanitize_repo_name(repo_name: str) -> str:
    """
    Sanitize repository name for use in file paths.
    
    Args:
        repo_name: Repository name
        
    Returns:
        str: Sanitized repository name
    """
    # Remove any characters that aren't allowed in file names
    return ''.join(c for c in repo_name if c.isalnum() or c in '._-')

def sparse_clone(
    repo_url: str,
    extensions: List[str],
    clone_dir: Optional[str] = None,
    files: Optional[List[str]] = None,
    dirs: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Sparse clone a Git repository, fetching only specified files.
    
    This function clones a Git repository with sparse checkout, which means
    only the specified files, directories, or file extensions are fetched.
    
    Args:
        repo_url: Repository URL to clone
        extensions: List of file extensions to include
        clone_dir: Directory to clone to (optional)
        files: Specific files to include (optional)
        dirs: Specific directories to include (optional)
        
    Returns:
        Dict[str, Any]: Result with the following keys:
            - success (bool): Whether the operation was successful
            - clone_dir (str): Path to the cloned repository
            - files (List[str]): List of files found
            - error (str): Error message if success is False
    """
    logger.info(f"Sparse cloning {repo_url} for extensions: {extensions}, files: {files}, dirs: {dirs}")
    
    # Set up clone directory if not provided
    if clone_dir is None:
        repo_name = sanitize_repo_name(extract_repo_name_from_url(repo_url))
        clone_dir = f"repos/{repo_name}_sparse"
    
    try:
        # Clean up existing directory if it exists
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        os.makedirs(clone_dir, exist_ok=True)
        
        # Initialize git repository
        subprocess.run(['git', 'init'], cwd=clone_dir, check=True)
        
        # Add remote
        subprocess.run(['git', 'remote', 'add', 'origin', repo_url], cwd=clone_dir, check=True)
        
        # Configure sparse checkout
        subprocess.run(['git', 'config', 'core.sparseCheckout', 'true'], cwd=clone_dir, check=True)
        
        # Create sparse checkout patterns
        sparse_patterns = []
        if files or dirs:
            if files:
                sparse_patterns.extend([f"{f}" for f in files])
            if dirs:
                sparse_patterns.extend([f"{d.rstrip('/')}/**/*" for d in dirs])
        else:
            for ext in extensions:
                sparse_patterns.append(f'**/*.{ext}')
                sparse_patterns.append(f'/*.{ext}')
        
        # Write sparse checkout file
        sparse_file = os.path.join(clone_dir, '.git', 'info', 'sparse-checkout')
        with open(sparse_file, 'w') as f:
            f.write('\n'.join(sparse_patterns) + '\n')
        
        # Pull repository
        subprocess.run(['git', 'pull', '--depth=1', 'origin', 'HEAD'], cwd=clone_dir, check=True)
        
        # Find files
        found_files = find_files(clone_dir, extensions, files, dirs)
        
        return {
            "success": True,
            "clone_dir": clone_dir,
            "files": found_files
        }
        
    except Exception as e:
        logger.error(f"Sparse clone failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Sparse clone failed: {str(e)}",
            "clone_dir": clone_dir if 'clone_dir' in locals() else None
        }

def find_files(
    clone_dir: str,
    extensions: List[str],
    files: Optional[List[str]] = None,
    dirs: Optional[List[str]] = None
) -> List[str]:
    """
    Find files matching the criteria in the cloned repository.
    
    Args:
        clone_dir: Directory containing the cloned repository
        extensions: List of file extensions to include
        files: Specific files to include (optional)
        dirs: Specific directories to include (optional)
        
    Returns:
        List[str]: List of files found
    """
    found = []
    
    # If specific files or directories were requested
    if files or dirs:
        requested_paths = set()
        if files:
            requested_paths.update(files)
        if dirs:
            for d in dirs:
                d_path = os.path.join(clone_dir, d)
                if os.path.exists(d_path) and os.path.isdir(d_path):
                    for root, _, filenames in os.walk(d_path):
                        for filename in filenames:
                            rel_path = os.path.relpath(os.path.join(root, filename), clone_dir)
                            requested_paths.add(rel_path)
        
        # Check each requested path
        for path in requested_paths:
            full_path = os.path.join(clone_dir, path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                rel_path = os.path.relpath(full_path, clone_dir)
                found.append(rel_path)
    else:
        # Find files by extension
        for ext in extensions:
            for root, _, filenames in os.walk(clone_dir):
                for filename in filenames:
                    if filename.lower().endswith(f'.{ext.lower()}'):
                        rel_path = os.path.relpath(os.path.join(root, filename), clone_dir)
                        # Skip files in .git directory
                        if not rel_path.startswith('.git/'):
                            found.append(rel_path)
    
    return sorted(found)

def build_tree(root_dir: str) -> str:
    """
    Build a tree representation of the directory structure.
    
    Args:
        root_dir: Root directory to build tree for
        
    Returns:
        str: Tree representation of the directory structure
    """
    logger.debug(f"Building tree for {root_dir}")
    tree_lines = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip .git directory
        if '.git' in dirpath:
            continue
            
        rel_dir = os.path.relpath(dirpath, root_dir)
        indent = "" if rel_dir == "." else "    " * rel_dir.count(os.sep)
        
        # Add directory name
        dir_name = os.path.basename(dirpath) if rel_dir != '.' else '.'
        tree_lines.append(f"{indent}{dir_name}/")
        
        # Add files
        for filename in sorted(filenames):
            tree_lines.append(f"{indent}    {filename}")
    
    return "\n".join(tree_lines)

def save_to_file(root_dir: str, filename: str, content: str) -> str:
    """
    Save content to a file in the root directory.
    
    Args:
        root_dir: Root directory
        filename: Filename
        content: Content to save
        
    Returns:
        str: Full path to the saved file
    """
    file_path = os.path.join(root_dir, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return file_path

def process_repository(
    clone_dir: str,
    extensions: List[str],
    files: Optional[List[str]] = None,
    dirs: Optional[List[str]] = None,
    code_metadata: bool = False
) -> Dict[str, Any]:
    """
    Process a cloned repository to generate summary and tree representation.
    
    Args:
        clone_dir: Path to the cloned repository
        extensions: List of file extensions to include
        files: Specific files to include (optional)
        dirs: Specific directories to include (optional)
        code_metadata: Whether to extract code metadata (optional)
        
    Returns:
        Dict[str, Any]: Result with the following keys:
            - success (bool): Whether the operation was successful
            - summary (str): Repository summary
            - tree (str): Tree representation of the repository
            - digest (str): Repository content digest
            - files (List[str]): List of processed files
            - error (str): Error message if success is False
    """
    try:
        logger.info(f"Processing repository in {clone_dir}")
        
        # Find files
        repo_files = find_files(clone_dir, extensions, files, dirs)
        
        if not repo_files:
            return {
                "success": False,
                "error": "No files found matching the criteria",
                "clone_dir": clone_dir
            }
        
        # Build tree
        tree = build_tree(clone_dir)
        
        # Create summary
        file_count = len(repo_files)
        total_bytes = 0
        digest_parts = []
        
        for file_path in repo_files:
            full_path = os.path.join(clone_dir, file_path)
            
            # Read file content
            with open(full_path, encoding="utf-8", errors="replace") as f:
                content = f.read()
                
            # Update total bytes
            total_bytes += len(content.encode("utf-8"))
            
            # Add to digest
            digest_parts.append("="*48)
            digest_parts.append(f"File: {file_path}")
            digest_parts.append("="*48)
            digest_parts.append(content)
            digest_parts.append("")
        
        # Create digest
        digest = "\n".join(digest_parts)
        
        # Create summary
        summary = (
            f"Directory: {clone_dir}\n"
            f"Files analyzed: {file_count}\n"
            f"Total bytes: {total_bytes}\n"
            f"Files included:\n" + "\n".join(repo_files)
        )
        
        return {
            "success": True,
            "summary": summary,
            "tree": tree,
            "digest": digest,
            "files": repo_files,
            "clone_dir": clone_dir
        }
        
    except Exception as e:
        logger.error(f"Repository processing failed: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Repository processing failed: {str(e)}",
            "clone_dir": clone_dir
        }

if __name__ == "__main__":
    """Validate repository operations with real data"""
    import sys
    import tempfile
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Extract repo name from URL
    total_tests += 1
    test_url = "https://github.com/username/repo-name"
    expected_name = "repo-name"
    actual_name = extract_repo_name_from_url(test_url)
    if actual_name != expected_name:
        all_validation_failures.append(f"Extract repo name test: Expected {expected_name}, got {actual_name}")
    
    # Test 2: Sanitize repo name
    total_tests += 1
    test_name = "repo-name$with@invalid!chars"
    expected_sanitized = "repo-namewithinvalidchars"
    actual_sanitized = sanitize_repo_name(test_name)
    if actual_sanitized != expected_sanitized:
        all_validation_failures.append(f"Sanitize repo name test: Expected {expected_sanitized}, got {actual_sanitized}")
    
    # Test 3: Build tree
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple directory structure
            os.makedirs(os.path.join(temp_dir, "dir1"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "dir2"), exist_ok=True)
            with open(os.path.join(temp_dir, "file1.txt"), "w") as f:
                f.write("File 1")
            with open(os.path.join(temp_dir, "dir1", "file2.txt"), "w") as f:
                f.write("File 2")
            
            # Build tree
            tree = build_tree(temp_dir)
            
            # Check if the tree contains expected entries
            expected_entries = ["./", "    file1.txt", "dir1/", "    file2.txt", "dir2/"]
            for entry in expected_entries:
                if entry not in tree:
                    all_validation_failures.append(f"Build tree test: Missing entry '{entry}' in tree")
                    break
    except Exception as e:
        all_validation_failures.append(f"Build tree test failed: {str(e)}")
    
    # Test 4: Save to file
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            content = "Test content"
            filename = "test.txt"
            file_path = save_to_file(temp_dir, filename, content)
            
            # Check if the file exists and contains the expected content
            if not os.path.exists(file_path):
                all_validation_failures.append(f"Save to file test: File not created at {file_path}")
            else:
                with open(file_path, "r") as f:
                    actual_content = f.read()
                if actual_content != content:
                    all_validation_failures.append(f"Save to file test: Expected content '{content}', got '{actual_content}'")
    except Exception as e:
        all_validation_failures.append(f"Save to file test failed: {str(e)}")
    
    # Test 5: Find files
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple directory structure with different file types
            os.makedirs(os.path.join(temp_dir, "dir1"), exist_ok=True)
            os.makedirs(os.path.join(temp_dir, "dir2"), exist_ok=True)
            with open(os.path.join(temp_dir, "file1.txt"), "w") as f:
                f.write("File 1")
            with open(os.path.join(temp_dir, "file2.md"), "w") as f:
                f.write("File 2")
            with open(os.path.join(temp_dir, "dir1", "file3.py"), "w") as f:
                f.write("File 3")
            with open(os.path.join(temp_dir, "dir2", "file4.txt"), "w") as f:
                f.write("File 4")
            
            # Test finding by extension
            found_txt = find_files(temp_dir, ["txt"])
            expected_txt = ["dir2/file4.txt", "file1.txt"]
            if sorted(found_txt) != expected_txt:
                all_validation_failures.append(f"Find files by extension test: Expected {expected_txt}, got {found_txt}")
            
            # Test finding by specific files
            found_specific = find_files(temp_dir, [], files=["file1.txt", "dir1/file3.py"])
            expected_specific = ["dir1/file3.py", "file1.txt"]
            if sorted(found_specific) != expected_specific:
                all_validation_failures.append(f"Find files by specific files test: Expected {expected_specific}, got {found_specific}")
            
            # Test finding by directory
            found_dir = find_files(temp_dir, [], dirs=["dir1"])
            expected_dir = ["dir1/file3.py"]
            if sorted(found_dir) != expected_dir:
                all_validation_failures.append(f"Find files by directory test: Expected {expected_dir}, got {found_dir}")
    except Exception as e:
        all_validation_failures.append(f"Find files test failed: {str(e)}")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Repository operations are validated and ready for use")
        sys.exit(0)  # Exit with success code