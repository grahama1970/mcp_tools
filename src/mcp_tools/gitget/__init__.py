"""Git repository content extraction and processing module.

This module provides tools for cloning, analyzing, and processing Git repositories,
with a focus on extracting and chunking content for use with LLMs.

Links to third-party package documentation:
- GitPython: https://gitpython.readthedocs.io/en/stable/
- tiktoken: https://github.com/openai/tiktoken
- spaCy: https://spacy.io/api/doc

Sample input:
    >>> from mcp_tools.gitget.core import sparse_clone, process_repository
    >>> sparse_clone("https://github.com/user/repo", "/tmp/repo")
    >>> result = process_repository("https://github.com/user/repo", "/tmp/repo")

Expected output:
    # Clone repository and analyze content
    True
    
    # Process repository contents
    {
        'repository': {...},
        'files': [...],
        'success': True
    }
"""

# Import submodules
from . import core
from . import cli
from . import mcp

# Version
__version__ = "0.1.0"

# Expose main functionality at package level
from .core.repo_operations import sparse_clone, process_repository, find_files
from .core.directory_manager import RepositoryDirectoryManager
from .core.text_chunker import TextChunker, SectionHierarchy
from .core.utils import extract_repo_info, save_to_file, read_from_file

# MCP handler
from .mcp.wrapper import handler as mcp_handler

__all__ = [
    # Submodules
    "core",
    "cli",
    "mcp",
    
    # Core functions
    "sparse_clone",
    "process_repository",
    "find_files",
    "RepositoryDirectoryManager",
    "TextChunker",
    "SectionHierarchy",
    "extract_repo_info",
    "save_to_file",
    "read_from_file",
    
    # MCP
    "mcp_handler",
    
    # Version
    "__version__"
]

if __name__ == "__main__":
    # Validation function to test package imports
    import sys
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Import core functionality
    total_tests += 1
    try:
        # Import core functions and classes
        from mcp_tools.gitget.core import sparse_clone, process_repository, find_files
        from mcp_tools.gitget.core import RepositoryDirectoryManager
        from mcp_tools.gitget.core import TextChunker, SectionHierarchy
        from mcp_tools.gitget.core import extract_repo_info, save_to_file, read_from_file
        
        # Verify they are callable
        if not callable(sparse_clone):
            all_validation_failures.append("Core import: sparse_clone is not callable")
        
        if not callable(process_repository):
            all_validation_failures.append("Core import: process_repository is not callable")
        
        if not callable(find_files):
            all_validation_failures.append("Core import: find_files is not callable")
    except ImportError as e:
        all_validation_failures.append(f"Core import: Failed to import core functionality: {str(e)}")
    except Exception as e:
        all_validation_failures.append(f"Core import: Unexpected exception: {str(e)}")
    
    # Test 2: Import CLI app
    total_tests += 1
    try:
        # Import CLI app
        from mcp_tools.gitget.cli import app
        
        # Verify it has the expected commands
        commands = [cmd.name for cmd in app.registered_commands]
        expected_commands = ["clone", "process", "extract", "info"]
        
        for cmd in expected_commands:
            if cmd not in commands:
                all_validation_failures.append(f"CLI import: Missing command '{cmd}' in CLI app")
    except ImportError as e:
        all_validation_failures.append(f"CLI import: Failed to import CLI app: {str(e)}")
    except Exception as e:
        all_validation_failures.append(f"CLI import: Unexpected exception: {str(e)}")
    
    # Test 3: Import MCP handler
    total_tests += 1
    try:
        # Import MCP handler
        from mcp_tools.gitget.mcp import handler, mcp_app
        
        # Verify it's callable
        if not callable(handler):
            all_validation_failures.append("MCP import: handler is not callable")
    except ImportError as e:
        all_validation_failures.append(f"MCP import: Failed to import MCP handler: {str(e)}")
    except Exception as e:
        all_validation_failures.append(f"MCP import: Unexpected exception: {str(e)}")
    
    # Test 4: Package-level imports
    total_tests += 1
    try:
        # Import from package root
        import mcp_tools.gitget
        
        # Check version
        if not hasattr(mcp_tools.gitget, "__version__"):
            all_validation_failures.append("Package import: Missing __version__ attribute")
        
        # Check top-level functions
        for func in ["sparse_clone", "process_repository", "find_files", "extract_repo_info"]:
            if not hasattr(mcp_tools.gitget, func):
                all_validation_failures.append(f"Package import: Missing top-level function '{func}'")
        
        # Check top-level classes
        for cls in ["RepositoryDirectoryManager", "TextChunker", "SectionHierarchy"]:
            if not hasattr(mcp_tools.gitget, cls):
                all_validation_failures.append(f"Package import: Missing top-level class '{cls}'")
        
        # Check MCP handler
        if not hasattr(mcp_tools.gitget, "mcp_handler"):
            all_validation_failures.append("Package import: Missing top-level mcp_handler")
    except ImportError as e:
        all_validation_failures.append(f"Package import: Failed to import package: {str(e)}")
    except Exception as e:
        all_validation_failures.append(f"Package import: Unexpected exception: {str(e)}")
    
    # Final validation result
    if all_validation_failures:
        print(f"L VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f" VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)