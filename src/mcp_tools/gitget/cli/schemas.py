"""Pydantic schemas for CLI input and output in the gitget module.

This module provides Pydantic models for validating CLI input and structuring output data
for the gitget module.

Links to third-party package documentation:
- Pydantic: https://docs.pydantic.dev/latest/
- Pydantic Types: https://docs.pydantic.dev/latest/api/types/

Sample input:
    repo_input = RepositoryInput(
        url="https://github.com/user/repo",
        branch="main",
        output_dir="/tmp/repos",
        sparse_checkout=["README.md", "src/"]
    )
    
    file_info = FileInfo(
        name="file.py",
        path="/path/to/file.py",
        size=1024,
        content_type="text/plain",
        is_binary=False
    )

Expected output:
    repo_input.model_dump()
    # {'url': 'https://github.com/user/repo', 'branch': 'main', ...}
    
    file_info.model_dump()
    # {'name': 'file.py', 'path': '/path/to/file.py', 'size': 1024, ...}
"""

from typing import Dict, List, Optional, Union, Any
from enum import Enum
import os
from pydantic import BaseModel, Field, HttpUrl, DirectoryPath, validator
from datetime import datetime
import sys

class ContentType(str, Enum):
    """Enum for file content types."""
    TEXT = "text"
    BINARY = "binary"
    DIRECTORY = "directory"
    UNKNOWN = "unknown"

class RepositoryInput(BaseModel):
    """Schema for repository input parameters."""
    url: str = Field(..., description="Repository URL (e.g., https://github.com/user/repo)")
    branch: str = Field("main", description="Branch to checkout")
    output_dir: str = Field("./repos", description="Directory to store cloned repositories")
    sparse_checkout: Optional[List[str]] = Field(None, description="List of files/dirs to include in sparse checkout")
    depth: Optional[int] = Field(1, description="Depth of git history to clone (1 for shallow clone)")
    include_git_dir: bool = Field(False, description="Whether to include .git directory")

    @validator("url")
    def validate_url(cls, v):
        """Validate that the URL is a git repository URL."""
        if not (v.startswith("https://") or v.startswith("git@")):
            raise ValueError("URL must start with 'https://' or 'git@'")
        if not any(domain in v for domain in ["github.com", "gitlab.com", "bitbucket.org"]):
            raise ValueError("URL must be from a known git provider (github.com, gitlab.com, bitbucket.org)")
        return v

    @validator("output_dir")
    def validate_output_dir(cls, v):
        """Create the output directory if it doesn't exist."""
        os.makedirs(v, exist_ok=True)
        return v

class FileInfo(BaseModel):
    """Schema for file information."""
    name: str = Field(..., description="Filename")
    path: str = Field(..., description="Full path to the file")
    size: int = Field(..., description="File size in bytes")
    content_type: ContentType = Field(ContentType.UNKNOWN, description="File content type")
    is_binary: bool = Field(False, description="Whether the file is binary")
    last_modified: Optional[datetime] = Field(None, description="Last modification time")
    
    @validator("path")
    def validate_path_exists(cls, v):
        """Validate that the file path exists."""
        if not os.path.exists(v) and not getattr(sys, "_called_from_test", False):
            raise ValueError(f"File path does not exist: {v}")
        return v

class RepositoryInfo(BaseModel):
    """Schema for repository information."""
    url: str = Field(..., description="Repository URL")
    name: str = Field(..., description="Repository name")
    owner: str = Field(..., description="Repository owner")
    local_path: str = Field(..., description="Local path to the repository")
    branch: str = Field(..., description="Current branch")
    file_count: int = Field(0, description="Number of files")
    total_size: int = Field(0, description="Total size in bytes")
    files: List[FileInfo] = Field([], description="List of file information")
    clone_time: Optional[float] = Field(None, description="Time taken to clone in seconds")

class ChunkInfo(BaseModel):
    """Schema for text chunk information."""
    content: str = Field(..., description="Chunk content")
    start_line: int = Field(..., description="Starting line number")
    end_line: int = Field(..., description="Ending line number")
    file_path: str = Field(..., description="Source file path")
    section: Optional[str] = Field(None, description="Section name")
    token_count: int = Field(0, description="Number of tokens in the chunk")

class ProcessResult(BaseModel):
    """Schema for processing result."""
    repository: RepositoryInfo = Field(..., description="Repository information")
    chunks: List[ChunkInfo] = Field([], description="List of text chunks")
    error: Optional[str] = Field(None, description="Error message if any")
    success: bool = Field(True, description="Whether the operation was successful")

if __name__ == "__main__":
    # Validation function to test schemas with real data
    import sys
    import tempfile
    import json
    from datetime import datetime, timezone
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: RepositoryInput validation
    total_tests += 1
    try:
        # Valid input
        repo_input = RepositoryInput(
            url="https://github.com/user/repo",
            branch="develop",
            output_dir=tempfile.mkdtemp(),
            sparse_checkout=["README.md", "src/"],
            depth=2
        )
        
        # Verify serialization
        data = repo_input.model_dump()
        expected_keys = ["url", "branch", "output_dir", "sparse_checkout", "depth", "include_git_dir"]
        
        if not all(key in data for key in expected_keys):
            missing = [key for key in expected_keys if key not in data]
            all_validation_failures.append(f"RepositoryInput: Missing expected keys: {missing}")
        
        # Test URL validation
        try:
            invalid_repo = RepositoryInput(url="invalid-url", branch="main", output_dir="./repos")
            all_validation_failures.append("RepositoryInput: URL validation failed to catch invalid URL")
        except ValueError:
            # This is expected
            pass
            
    except Exception as e:
        all_validation_failures.append(f"RepositoryInput: Unexpected exception: {str(e)}")
    
    # Test 2: FileInfo validation
    total_tests += 1
    try:
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"Test content")
        
        # Valid file info
        file_info = FileInfo(
            name=os.path.basename(temp_file.name),
            path=temp_file.name,
            size=os.path.getsize(temp_file.name),
            content_type=ContentType.TEXT,
            is_binary=False,
            last_modified=datetime.now(timezone.utc)
        )
        
        # Verify serialization
        data = file_info.model_dump()
        expected_keys = ["name", "path", "size", "content_type", "is_binary", "last_modified"]
        
        if not all(key in data for key in expected_keys):
            missing = [key for key in expected_keys if key not in data]
            all_validation_failures.append(f"FileInfo: Missing expected keys: {missing}")
        
        # Clean up
        os.unlink(temp_file.name)
            
    except Exception as e:
        all_validation_failures.append(f"FileInfo: Unexpected exception: {str(e)}")
    
    # Test 3: RepositoryInfo validation
    total_tests += 1
    try:
        # Valid repository info
        repo_info = RepositoryInfo(
            url="https://github.com/user/repo",
            name="repo",
            owner="user",
            local_path="/tmp/repos/user_repo",
            branch="main",
            file_count=10,
            total_size=1024,
            files=[],
            clone_time=1.5
        )
        
        # Verify serialization
        data = repo_info.model_dump()
        expected_keys = ["url", "name", "owner", "local_path", "branch", "file_count", "total_size", "files", "clone_time"]
        
        if not all(key in data for key in expected_keys):
            missing = [key for key in expected_keys if key not in data]
            all_validation_failures.append(f"RepositoryInfo: Missing expected keys: {missing}")
            
    except Exception as e:
        all_validation_failures.append(f"RepositoryInfo: Unexpected exception: {str(e)}")
    
    # Test 4: ChunkInfo validation
    total_tests += 1
    try:
        # Valid chunk info
        chunk_info = ChunkInfo(
            content="def hello_world():\n    print('Hello, world!')",
            start_line=1,
            end_line=2,
            file_path="/path/to/file.py",
            section="Function Definitions",
            token_count=20
        )
        
        # Verify serialization
        data = chunk_info.model_dump()
        expected_keys = ["content", "start_line", "end_line", "file_path", "section", "token_count"]
        
        if not all(key in data for key in expected_keys):
            missing = [key for key in expected_keys if key not in data]
            all_validation_failures.append(f"ChunkInfo: Missing expected keys: {missing}")
            
    except Exception as e:
        all_validation_failures.append(f"ChunkInfo: Unexpected exception: {str(e)}")
    
    # Test 5: ProcessResult validation
    total_tests += 1
    try:
        # Create minimal repository info
        repo_info = RepositoryInfo(
            url="https://github.com/user/repo",
            name="repo",
            owner="user",
            local_path="/tmp/repos/user_repo",
            branch="main"
        )
        
        # Valid process result
        result = ProcessResult(
            repository=repo_info,
            chunks=[],
            success=True
        )
        
        # Failed process result
        failed_result = ProcessResult(
            repository=repo_info,
            chunks=[],
            error="Failed to process repository",
            success=False
        )
        
        # Verify serialization
        data = result.model_dump()
        error_data = failed_result.model_dump()
        
        expected_keys = ["repository", "chunks", "error", "success"]
        
        if not all(key in data for key in expected_keys):
            missing = [key for key in expected_keys if key not in data]
            all_validation_failures.append(f"ProcessResult: Missing expected keys: {missing}")
        
        if not error_data["error"] or error_data["success"]:
            all_validation_failures.append(f"ProcessResult: Error state not correctly represented")
            
    except Exception as e:
        all_validation_failures.append(f"ProcessResult: Unexpected exception: {str(e)}")
    
    # Set this flag to allow validation without file existence check
    sys._called_from_test = True
    
    # Test 6: JSON serialization and deserialization
    total_tests += 1
    try:
        # Create a test repository input
        repo_input = RepositoryInput(
            url="https://github.com/user/repo",
            branch="main",
            output_dir="./repos"
        )
        
        # Serialize to JSON
        json_data = repo_input.model_dump_json()
        
        # Deserialize from JSON
        parsed_data = json.loads(json_data)
        deserialized = RepositoryInput(**parsed_data)
        
        if deserialized.url != repo_input.url or deserialized.branch != repo_input.branch:
            all_validation_failures.append(f"JSON serialization: Data mismatch after serialization/deserialization")
            
    except Exception as e:
        all_validation_failures.append(f"JSON serialization: Unexpected exception: {str(e)}")
    
    # Clean up
    if hasattr(sys, "_called_from_test"):
        delattr(sys, "_called_from_test")
    
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