#!/usr/bin/env python3
"""
Repository Directory Manager Module

This module provides functionality for managing directory structures for
Git repository analysis, ensuring consistent organization of intermediate
files and final outputs.

This module is part of the Core Layer and should have no dependencies on
Presentation or Integration layers.

Links to documentation:
- pathlib: https://docs.python.org/3/library/pathlib.html

Sample input:
- base_dir: "/path/to/output"
- repo_name: "repository-name"

Expected output:
- RepositoryDirectoryManager instance with methods for getting file paths
- Directory structure with chunks, parsed, metadata, and output directories
"""

import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union
from loguru import logger

class RepositoryDirectoryManager:
    """
    Manages the directory structure for Git repository analysis.
    
    This class is responsible for creating and managing directories for:
    - chunks: Stored chunked text
    - parsed: Parsed markdown with section structure
    - metadata: Extracted code metadata
    - output: Final output files
    """
    
    def __init__(self, base_dir: str):
        """
        Initialize the directory manager.
        
        Args:
            base_dir: Base directory for all output
        """
        self.base_dir = Path(base_dir)
        self.dirs = {
            "chunks": self.base_dir / "chunks",
            "parsed": self.base_dir / "parsed",
            "metadata": self.base_dir / "metadata",
            "output": self.base_dir / "output"
        }
        
    def create_directory_structure(self) -> Dict[str, Path]:
        """
        Create the directory structure if it doesn't exist.
        
        Returns:
            Dictionary with directory paths
        """
        logger.info(f"Creating directory structure in {self.base_dir}")
        
        # Create base directory if it doesn't exist
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Create subdirectories
        for name, path in self.dirs.items():
            os.makedirs(path, exist_ok=True)
            logger.debug(f"Created directory: {path}")
            
        return self.dirs
    
    def get_chunk_path(self, file_path: str, chunk_id: Optional[str] = None) -> Path:
        """
        Get the path for a chunked file.
        
        Args:
            file_path: Original file path
            chunk_id: Optional chunk identifier
            
        Returns:
            Path to the chunked file
        """
        file_name = Path(file_path).name
        base_name = Path(file_name).stem
        suffix = f"_{chunk_id}" if chunk_id else ""
        return self.dirs["chunks"] / f"{base_name}{suffix}.json"
    
    def get_parsed_path(self, file_path: str) -> Path:
        """
        Get the path for a parsed file.
        
        Args:
            file_path: Original file path
            
        Returns:
            Path to the parsed file
        """
        file_name = Path(file_path).name
        base_name = Path(file_name).stem
        return self.dirs["parsed"] / f"{base_name}_parsed.json"
    
    def get_metadata_path(self, file_path: str) -> Path:
        """
        Get the path for a metadata file.
        
        Args:
            file_path: Original file path
            
        Returns:
            Path to the metadata file
        """
        file_name = Path(file_path).name
        base_name = Path(file_name).stem
        return self.dirs["metadata"] / f"{base_name}_metadata.json"
    
    def get_output_path(self, name: str) -> Path:
        """
        Get the path for an output file.
        
        Args:
            name: Name of the output file
            
        Returns:
            Path to the output file
        """
        return self.dirs["output"] / name
    
    def cleanup(self, keep_output: bool = True) -> None:
        """
        Clean up the directory structure.
        
        Args:
            keep_output: Whether to keep the output directory
        """
        logger.info(f"Cleaning up directory structure in {self.base_dir}")
        
        # Remove intermediate directories
        for name, path in self.dirs.items():
            if name != "output" or not keep_output:
                if os.path.exists(path):
                    shutil.rmtree(path)
                    logger.debug(f"Removed directory: {path}")
        
        # Remove base directory if it's empty and we're not keeping output
        if not keep_output and os.path.exists(self.base_dir) and not os.listdir(self.base_dir):
            os.rmdir(self.base_dir)
            logger.debug(f"Removed empty base directory: {self.base_dir}")

def create_repo_directory_structure(repo_name: str, base_dir: Optional[str] = None) -> RepositoryDirectoryManager:
    """
    Create a directory structure for a repository.
    
    Args:
        repo_name: Name of the repository
        base_dir: Optional base directory (defaults to repos/<repo_name>)
        
    Returns:
        RepositoryDirectoryManager instance
    """
    if base_dir is None:
        base_dir = os.path.join("repos", f"{repo_name}")
    
    manager = RepositoryDirectoryManager(base_dir)
    manager.create_directory_structure()
    return manager

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

def save_to_file(file_path: str, content: str) -> None:
    """
    Save content to a file.
    
    Args:
        file_path: Path to the file
        content: Content to save
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

if __name__ == "__main__":
    """Validate directory manager with real data"""
    import sys
    import tempfile
    import json
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Create directory structure
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a directory manager
            manager = RepositoryDirectoryManager(temp_dir)
            dirs = manager.create_directory_structure()
            
            # Check if all directories were created
            expected_dirs = ["chunks", "parsed", "metadata", "output"]
            missing_dirs = []
            
            for dir_name in expected_dirs:
                if not os.path.exists(dirs[dir_name]):
                    missing_dirs.append(dir_name)
            
            if missing_dirs:
                all_validation_failures.append(f"Create directory structure test: Missing directories: {missing_dirs}")
    except Exception as e:
        all_validation_failures.append(f"Create directory structure test failed: {str(e)}")
    
    # Test 2: Get chunk path
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RepositoryDirectoryManager(temp_dir)
            manager.create_directory_structure()
            
            file_path = "test/path/to/file.py"
            chunk_id = "chunk1"
            
            # Get chunk path
            chunk_path = manager.get_chunk_path(file_path, chunk_id)
            expected_path = Path(temp_dir) / "chunks" / "file_chunk1.json"
            
            if chunk_path != expected_path:
                all_validation_failures.append(f"Get chunk path test: Expected {expected_path}, got {chunk_path}")
    except Exception as e:
        all_validation_failures.append(f"Get chunk path test failed: {str(e)}")
    
    # Test 3: Get parsed path
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RepositoryDirectoryManager(temp_dir)
            manager.create_directory_structure()
            
            file_path = "test/path/to/file.md"
            
            # Get parsed path
            parsed_path = manager.get_parsed_path(file_path)
            expected_path = Path(temp_dir) / "parsed" / "file_parsed.json"
            
            if parsed_path != expected_path:
                all_validation_failures.append(f"Get parsed path test: Expected {expected_path}, got {parsed_path}")
    except Exception as e:
        all_validation_failures.append(f"Get parsed path test failed: {str(e)}")
    
    # Test 4: Get metadata path
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RepositoryDirectoryManager(temp_dir)
            manager.create_directory_structure()
            
            file_path = "test/path/to/file.py"
            
            # Get metadata path
            metadata_path = manager.get_metadata_path(file_path)
            expected_path = Path(temp_dir) / "metadata" / "file_metadata.json"
            
            if metadata_path != expected_path:
                all_validation_failures.append(f"Get metadata path test: Expected {expected_path}, got {metadata_path}")
    except Exception as e:
        all_validation_failures.append(f"Get metadata path test failed: {str(e)}")
    
    # Test 5: Get output path
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RepositoryDirectoryManager(temp_dir)
            manager.create_directory_structure()
            
            output_name = "SUMMARY.txt"
            
            # Get output path
            output_path = manager.get_output_path(output_name)
            expected_path = Path(temp_dir) / "output" / "SUMMARY.txt"
            
            if output_path != expected_path:
                all_validation_failures.append(f"Get output path test: Expected {expected_path}, got {output_path}")
    except Exception as e:
        all_validation_failures.append(f"Get output path test failed: {str(e)}")
    
    # Test 6: Create and save files
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RepositoryDirectoryManager(temp_dir)
            manager.create_directory_structure()
            
            # Create some sample files
            chunk_path = manager.get_chunk_path("test.py", "chunk1")
            with open(chunk_path, "w") as f:
                json.dump({"content": "Test chunk"}, f)
            
            parsed_path = manager.get_parsed_path("test.md")
            with open(parsed_path, "w") as f:
                json.dump({"sections": [{"title": "Test Section"}]}, f)
            
            metadata_path = manager.get_metadata_path("test.py")
            with open(metadata_path, "w") as f:
                json.dump({"functions": [{"name": "test_func"}]}, f)
            
            output_path = manager.get_output_path("SUMMARY.txt")
            with open(output_path, "w") as f:
                f.write("Test summary")
            
            # Check if all files exist
            if not os.path.exists(chunk_path):
                all_validation_failures.append(f"Create files test: Chunk file not created at {chunk_path}")
            if not os.path.exists(parsed_path):
                all_validation_failures.append(f"Create files test: Parsed file not created at {parsed_path}")
            if not os.path.exists(metadata_path):
                all_validation_failures.append(f"Create files test: Metadata file not created at {metadata_path}")
            if not os.path.exists(output_path):
                all_validation_failures.append(f"Create files test: Output file not created at {output_path}")
    except Exception as e:
        all_validation_failures.append(f"Create files test failed: {str(e)}")
    
    # Test 7: Cleanup
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = RepositoryDirectoryManager(temp_dir)
            manager.create_directory_structure()
            
            # Create a sample file in the output directory
            output_path = manager.get_output_path("SUMMARY.txt")
            with open(output_path, "w") as f:
                f.write("Test summary")
            
            # Cleanup (keep the output)
            manager.cleanup(keep_output=True)
            
            # Check that output directory still exists
            if not os.path.exists(manager.dirs["output"]):
                all_validation_failures.append("Cleanup test: Output directory should still exist")
            
            # Check that other directories were removed
            for name, path in manager.dirs.items():
                if name != "output" and os.path.exists(path):
                    all_validation_failures.append(f"Cleanup test: {name} directory should have been removed")
    except Exception as e:
        all_validation_failures.append(f"Cleanup test failed: {str(e)}")
    
    # Test 8: Create repo directory structure
    total_tests += 1
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Use the temp directory as base
            repo_name = "test-repo"
            base_dir = os.path.join(temp_dir, repo_name)
            
            # Create repo directory structure
            manager = create_repo_directory_structure(repo_name, base_dir)
            
            # Check if directories were created
            expected_dirs = ["chunks", "parsed", "metadata", "output"]
            for dir_name in expected_dirs:
                if not os.path.exists(manager.dirs[dir_name]):
                    all_validation_failures.append(f"Create repo directory structure test: Missing directory: {dir_name}")
    except Exception as e:
        all_validation_failures.append(f"Create repo directory structure test failed: {str(e)}")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Directory manager is validated and ready for use")
        sys.exit(0)  # Exit with success code