#!/usr/bin/env python3
"""
Utility Functions for Git Repository Operations

This module provides utility functions for working with Git repositories,
including file operations, URL parsing, and other common tasks.

This module is part of the Core Layer and should have no dependencies on
Presentation or Integration layers.

Links to documentation:
- os: https://docs.python.org/3/library/os.html
- pathlib: https://docs.python.org/3/library/pathlib.html

Sample input:
- repo_url: "https://github.com/username/repository"
- file_path: "/path/to/file.txt"
- content: "File content"

Expected output:
- Processed values: "repository", "/path/to/file.txt", "File content"
"""

import os
import re
import json
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from loguru import logger

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

def extract_repo_name_from_url(repo_url: str) -> str:
    """
    Extract repository name from URL.
    
    Args:
        repo_url: Repository URL
        
    Returns:
        str: Repository name
    """
    # Extract the repository name from the URL
    return repo_url.rstrip('/').split('/')[-1]

def save_to_file(file_path: str, content: str) -> str:
    """
    Save content to a file.
    
    Args:
        file_path: Path to the file
        content: Content to save
        
    Returns:
        str: File path
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return file_path

def read_from_file(file_path: str) -> str:
    """
    Read content from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: File content
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def save_json_to_file(file_path: str, data: Dict[str, Any]) -> str:
    """
    Save JSON data to a file.
    
    Args:
        file_path: Path to the file
        data: JSON data to save
        
    Returns:
        str: File path
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    return file_path

def read_json_from_file(file_path: str) -> Dict[str, Any]:
    """
    Read JSON data from a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict[str, Any]: JSON data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def is_binary_file(file_path: str) -> bool:
    """
    Check if a file is binary.
    
    Args:
        file_path: Path to the file
        
    Returns:
        bool: True if the file is binary, False otherwise
    """
    # Check if the file is a common binary format
    binary_extensions = [
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico',
        '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx',
        '.zip', '.tar', '.gz', '.bz2', '.7z',
        '.exe', '.dll', '.so', '.dylib',
        '.pyc', '.pyo', '.pyd',
    ]
    
    if any(file_path.lower().endswith(ext) for ext in binary_extensions):
        return True
    
    # Check if the file contains binary data
    try:
        with open(file_path, "rb") as f:
            chunk = f.read(1024)
            return b'\0' in chunk
    except Exception:
        # If we can't read the file, assume it's not binary
        return False

def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a path.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: File extension without the dot
    """
    ext = os.path.splitext(file_path)[1]
    return ext.lstrip('.').lower()

def generate_unique_id() -> str:
    """
    Generate a unique ID.
    
    Returns:
        str: Unique ID
    """
    return str(uuid.uuid4())

def clean_path(path: str) -> str:
    """
    Clean a path string.
    
    Args:
        path: Path string
        
    Returns:
        str: Cleaned path
    """
    # Normalize path separators
    path = path.replace('\\', '/')
    
    # Remove duplicate slashes
    path = re.sub(r'/+', '/', path)
    
    # Remove trailing slash
    path = path.rstrip('/')
    
    return path

if __name__ == "__main__":
    """Validate utility functions with real data"""
    import sys
    import tempfile
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Sanitize repo name
    total_tests += 1
    test_names = [
        ("repo-name", "repo-name"),
        ("repo name", "reponame"),
        ("repo/name", "reponame"),
        ("repo.name", "repo.name"),
        ("repo_name", "repo_name"),
        ("repo@name", "reponame"),
    ]
    
    for input_name, expected_name in test_names:
        actual_name = sanitize_repo_name(input_name)
        if actual_name != expected_name:
            all_validation_failures.append(f"Sanitize repo name test: Expected '{expected_name}' for input '{input_name}', got '{actual_name}'")
    
    # Test 2: Extract repo name from URL
    total_tests += 1
    test_urls = [
        ("https://github.com/username/repo-name", "repo-name"),
        ("https://github.com/username/repo-name/", "repo-name"),
        ("https://github.com/username/repo.name", "repo.name"),
        ("https://gitlab.com/username/repo_name", "repo_name"),
        ("https://bitbucket.org/username/repo-name", "repo-name"),
    ]
    
    for input_url, expected_name in test_urls:
        actual_name = extract_repo_name_from_url(input_url)
        if actual_name != expected_name:
            all_validation_failures.append(f"Extract repo name test: Expected '{expected_name}' for input '{input_url}', got '{actual_name}'")
    
    # Test 3: Save to file and read from file
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_content = "Test file content"
            test_file = os.path.join(temp_dir, "test.txt")
            
            # Save to file
            save_to_file(test_file, test_content)
            
            # Read from file
            actual_content = read_from_file(test_file)
            
            if actual_content != test_content:
                all_validation_failures.append(f"Save/read file test: Expected '{test_content}', got '{actual_content}'")
    except Exception as e:
        all_validation_failures.append(f"Save/read file test failed: {str(e)}")
    
    # Test 4: Save JSON to file and read JSON from file
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data = {"key": "value", "nested": {"key": "value"}, "list": [1, 2, 3]}
            test_file = os.path.join(temp_dir, "test.json")
            
            # Save JSON to file
            save_json_to_file(test_file, test_data)
            
            # Read JSON from file
            actual_data = read_json_from_file(test_file)
            
            if actual_data != test_data:
                all_validation_failures.append(f"Save/read JSON test: Expected {test_data}, got {actual_data}")
    except Exception as e:
        all_validation_failures.append(f"Save/read JSON test failed: {str(e)}")
    
    # Test 5: Get file extension
    total_tests += 1
    test_paths = [
        ("file.txt", "txt"),
        ("file.TXT", "txt"),
        ("file.tar.gz", "gz"),
        ("file", ""),
        ("/path/to/file.py", "py"),
        ("C:\\path\\to\\file.md", "md"),
    ]
    
    for input_path, expected_ext in test_paths:
        actual_ext = get_file_extension(input_path)
        if actual_ext != expected_ext:
            all_validation_failures.append(f"Get file extension test: Expected '{expected_ext}' for input '{input_path}', got '{actual_ext}'")
    
    # Test 6: Generate unique ID
    total_tests += 1
    id1 = generate_unique_id()
    id2 = generate_unique_id()
    
    if id1 == id2:
        all_validation_failures.append(f"Generate unique ID test: Generated IDs are not unique: {id1} and {id2}")
    
    # Test 7: Clean path
    total_tests += 1
    test_paths = [
        ("/path/to/file", "/path/to/file"),
        ("/path//to/file/", "/path/to/file"),
        ("path\\to\\file", "path/to/file"),
        ("path/to//file///", "path/to/file"),
    ]
    
    for input_path, expected_path in test_paths:
        actual_path = clean_path(input_path)
        if actual_path != expected_path:
            all_validation_failures.append(f"Clean path test: Expected '{expected_path}' for input '{input_path}', got '{actual_path}'")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Utility functions are validated and ready for use")
        sys.exit(0)  # Exit with success code