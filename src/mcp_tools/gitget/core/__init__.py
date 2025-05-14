"""
Core Layer for Git Repository Operations Module

This package contains the core business logic for Git repository operations,
including sparse cloning, repository analysis, and text processing. It provides
pure functions and classes that can be used independently of UI or integration concerns.

The core layer is designed to be:
1. Independent of UI or integration concerns
2. Fully testable in isolation
3. Focused on business logic only
4. Self-validating through unit tests

Usage:
    from mcp_tools.gitget.core import sparse_clone, process_repository
    result = sparse_clone("https://github.com/org/repo", extensions=["md", "py"])
    digest = process_repository(result["clone_dir"])
"""

# Repository Operations
from mcp_tools.gitget.core.repo_operations import (
    sparse_clone,
    process_repository,
    build_tree
)

# Directory Management
from mcp_tools.gitget.core.directory_manager import (
    RepositoryDirectoryManager,
    create_repo_directory_structure
)

# Text Processing
from mcp_tools.gitget.core.text_chunker import (
    TextChunker,
    hash_string,
    SectionHierarchy,
    count_tokens_with_tiktoken
)

# Text Summarization
from mcp_tools.gitget.core.text_summarizer import (
    summarize_text,
    validate_summary
)

# Utility Functions
from mcp_tools.gitget.core.utils import (
    save_to_file,
    read_from_file,
    sanitize_repo_name,
    extract_repo_name_from_url
)

__all__ = [
    # Repository Operations
    'sparse_clone',
    'process_repository',
    'build_tree',
    
    # Directory Management
    'RepositoryDirectoryManager',
    'create_repo_directory_structure',
    
    # Text Processing
    'TextChunker',
    'hash_string',
    'SectionHierarchy',
    'count_tokens_with_tiktoken',
    
    # Text Summarization
    'summarize_text',
    'validate_summary',
    
    # Utilities
    'save_to_file',
    'read_from_file',
    'sanitize_repo_name',
    'extract_repo_name_from_url'
]