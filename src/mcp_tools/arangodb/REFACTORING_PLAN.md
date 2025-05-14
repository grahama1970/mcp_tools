# ArangoDB Module Refactoring Plan

This document provides a detailed plan for refactoring the ArangoDB module to align with the 3-layer architecture as defined in the [3-Layer Architecture Refactoring Guide](/Users/robert/claude_mcp_configs/docs/REFACTOR_3_LAYER.md).

## 1. Project Structure

### Target Directory Structure

```
src/mcp_tools/arangodb/
├── __init__.py                # Package exports
├── core/                      # Business logic layer
│   ├── __init__.py
│   ├── db_operations.py       # Generic CRUD operations
│   ├── message_operations.py  # Message history operations
│   ├── relationship_ops.py    # Graph relationship operations
│   └── search/                # Search functionality
│       ├── __init__.py
│       ├── bm25_search.py
│       ├── semantic_search.py
│       ├── hybrid_search.py
│       └── utils.py
├── cli/                       # Command-line interface layer
│   ├── __init__.py
│   ├── app.py                 # Typer application
│   ├── formatters.py          # Rich formatting functions
│   ├── validators.py          # Input validation
│   ├── schemas.py             # Pydantic models
│   └── commands/              # Command implementations
│       ├── __init__.py
│       ├── search.py          # Search commands
│       ├── crud.py            # CRUD commands
│       ├── graph.py           # Graph commands
│       └── memory.py          # Memory agent commands
└── mcp/                       # MCP integration layer
    ├── __init__.py
    ├── schema.py              # JSON schemas
    └── wrapper.py             # FastMCP wrapper
```

### Implementation Steps

1. Create the directory structure if it doesn't exist
2. Create empty `__init__.py` files in each directory
3. Refactor code in small, manageable chunks (outlined below)

## 2. Core Layer Refactoring

### 2.1 Refactor Database Operations

1. Extract core functionality from `db_operations.py` into:
   - `core/db_operations.py`: Generic CRUD operations
   - `core/message_operations.py`: Message-specific operations
   - `core/relationship_ops.py`: Graph relationship operations

2. Remove any presentation logic
   - Eliminate colorama, rich, and formatting code
   - Ensure functions return data structures, not formatted text
   - Move validation output to standard format

3. Add proper validation
   ```python
   if __name__ == "__main__":
       import sys
       
       # List to track all validation failures
       all_validation_failures = []
       total_tests = 0
       
       # Test cases here...
       
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
   ```

### 2.2 Refactor Search Functionality

1. Create `core/search` directory with specialized search modules:
   - `bm25_search.py`: Core BM25 search algorithm
   - `semantic_search.py`: Vector-based semantic search
   - `hybrid_search.py`: Combined search with re-ranking
   - `utils.py`: Common utilities

2. Remove UI/presentation logic from search code:
   - Remove `output_format` parameters
   - Remove print formatting with colorama/rich
   - Extract core search functionality only
   - Return clean data structures

3. Add proper validation to each search module

### 2.3 Remove Conditional Imports

1. Replace try/except import blocks with direct imports
2. Handle specific errors during usage, not during import
3. List all dependencies in `pyproject.toml`

## 3. CLI Layer Refactoring

### 3.1 Create CLI Foundation

1. Create `cli/app.py` with Typer application structure
   ```python
   import typer
   from rich.console import Console
   
   app = typer.Typer(
       name="arangodb-cli",
       help="ArangoDB interaction CLI",
       rich_markup_mode="rich"
   )
   console = Console()
   
   # Add subcommands
   search_app = typer.Typer(name="search", help="Search commands")
   crud_app = typer.Typer(name="crud", help="Document CRUD operations")
   graph_app = typer.Typer(name="graph", help="Graph operations")
   
   app.add_typer(search_app, name="search")
   app.add_typer(crud_app, name="crud")
   app.add_typer(graph_app, name="graph")
   
   # Entry point
   if __name__ == "__main__":
       app()
   ```

2. Create `cli/formatters.py` for UI formatting:
   - Extract Rich formatting from search_api and cli.py
   - Create standardized functions for tables, panels, etc.
   - Ensure consistent formatting across commands

3. Create `cli/schemas.py` with Pydantic models:
   ```python
   from pydantic import BaseModel, Field
   from typing import List, Optional, Dict, Any
   
   class SearchParams(BaseModel):
       query: str = Field(..., description="Search query text")
       top_n: int = Field(10, description="Maximum results to return")
       # Additional fields...
   ```

4. Create `cli/validators.py` for input validation:
   ```python
   from typing import Optional
   import typer
   import json
   
   def validate_json_data(data: Optional[str], data_file: Optional[str]) -> dict:
       """Validate JSON data from string or file."""
       # Implementation...
   ```

### 3.2 Implement Command Modules

1. Create command modules in `cli/commands/`:
   - `search.py`: Search commands (bm25, semantic, hybrid)
   - `crud.py`: Document operations (create, read, update, delete)
   - `graph.py`: Graph operations (relationships, traversal)
   - `memory.py`: Memory agent commands

2. Move CLI command implementations from `cli.py`:
   ```python
   # cli/commands/search.py
   import typer
   from ...core.search import bm25_search
   from ..formatters import display_search_results
   
   search_app = typer.Typer()
   
   @search_app.command("bm25")
   def search_bm25(
       query: str = typer.Argument(..., help="Search query"),
       # other parameters...
   ):
       """Search using BM25 algorithm."""
       # Implementation...
   ```

3. Use schemas and validators in commands:
   ```python
   from ..schemas import SearchParams
   from ..validators import validate_search_params
   
   # In command function
   params = validate_search_params(query, top_n, threshold)
   results = bm25_search(db, **params.dict())
   ```

## 4. MCP Layer Implementation

### 4.1 Create MCP Schema Definitions

1. Create `mcp/schema.py`:
   ```python
   # Define JSON schemas for MCP integration
   
   BM25_SEARCH_SCHEMA = {
       "type": "object",
       "properties": {
           "query": {
               "type": "string",
               "description": "Search query text"
           },
           # Additional properties...
       },
       "required": ["query"]
   }
   
   # Other schema definitions...
   ```

2. Ensure schemas match CLI parameters:
   - Same field names
   - Matching types and constraints
   - Consistent descriptions

### 4.2 Implement MCP Wrapper

1. Create `mcp/wrapper.py`:
   ```python
   from fastmcp import FastMCP
   from .schema import BM25_SEARCH_SCHEMA, CRUD_SCHEMAS
   from ..core.search import bm25_search
   from ..core.db_operations import create_document, get_document
   
   # Handler functions for each command
   def _bm25_search_handler(params):
       """Handle BM25 search MCP requests."""
       # Implementation...
   
   # FastMCP integration
   mcp_app = FastMCP(
       name="ArangoDB",
       description="ArangoDB integration for document storage and search",
       function_map={
           "bm25_search": _bm25_search_handler,
           # Other functions...
       },
       schemas={
           "bm25_search": BM25_SEARCH_SCHEMA,
           # Other schemas...
       }
   )
   
   def mcp_handler(request):
       """Handle MCP requests."""
       return mcp_app.handle_request(request)
   ```

2. Create handler functions for each command:
   - Map MCP parameters to core function parameters
   - Handle errors and return appropriate responses
   - Ensure consistent response structure

## 5. Package Integration

### 5.1 Update Main Package Exports

1. Update `__init__.py` with proper exports:
   ```python
   """
   ArangoDB integration for document storage, retrieval, and graph operations.
   
   This module provides a 3-layer architecture for working with ArangoDB:
   1. Core layer: Pure business logic functions
   2. CLI layer: Command-line interface
   3. MCP layer: Claude tool integration
   """
   
   __version__ = "0.1.0"
   
   # Core database operations
   from .core.db_operations import create_document, get_document, update_document, delete_document
   
   # Search functionality
   from .core.search.bm25_search import bm25_search
   from .core.search.semantic_search import semantic_search
   
   # MCP integration
   from .mcp.wrapper import mcp_handler
   
   __all__ = [
       "create_document",
       "get_document",
       "update_document",
       "delete_document",
       "bm25_search",
       "semantic_search",
       "mcp_handler",
       # Other exports...
   ]
   ```

2. Include high-level usage examples in module docstring

### 5.2 Add Documentation

1. Add comprehensive docstrings to all modules:
   ```python
   """
   Module: db_operations.py
   
   This module provides core database operations for ArangoDB including
   CRUD operations for documents.
   
   Third-Party Packages:
   - python-arango: https://python-driver.arangodb.com/
   
   Sample Input:
       document = {"title": "Test", "content": "Test content"}
       result = create_document(db, "collection", document)
       
   Expected Output:
       {
           "_key": "12345",
           "_id": "collection/12345",
           "title": "Test",
           "content": "Test content"
       }
   """
   ```

2. Ensure consistent documentation across all files:
   - Module purpose
   - Third-party package references
   - Sample input/output
   - Implementation notes

## 6. Implementation Sequence

### Phase 1: Core Layer

1. Create directory structure
2. Refactor database operations
3. Refactor search functionality
4. Add validation to all modules

### Phase 2: CLI Layer

1. Create CLI foundation
2. Implement formatters and validators
3. Create command modules
4. Migrate commands from existing CLI

### Phase 3: MCP Layer

1. Create schema definitions
2. Implement MCP wrapper
3. Add handler functions

### Phase 4: Integration

1. Update package exports
2. Add comprehensive documentation
3. Validate the entire implementation

## 7. Validation Approach

1. For each refactored module:
   - Test with real data
   - Verify expected results
   - Track and report failures
   - Exit with appropriate code

2. Test integration between layers:
   - CLI calling core functions
   - MCP handlers calling core functions
   - Package exports working correctly

3. Verify against 3-layer architecture requirements:
   - Clean separation of concerns
   - No UI dependencies in core layer
   - Proper documentation and validation
   - Consistent error handling

## 8. Post-Refactoring Verification

Create a verification checklist:

- [ ] Core layer contains only business logic (no UI dependencies)
- [ ] CLI layer provides human-friendly interface
- [ ] MCP layer enables Claude integration
- [ ] All modules have proper validation
- [ ] Documentation is comprehensive
- [ ] Type hints are consistent
- [ ] Package exports are well-organized
- [ ] Error handling is consistent
- [ ] Tests pass with real data

## Conclusion

This refactoring plan provides a comprehensive approach to aligning the ArangoDB module with the 3-layer architecture. By following this plan, the module will gain:

1. **Improved maintainability** through clear separation of concerns
2. **Better testability** with independent core functions
3. **Enhanced documentation** with consistent standards
4. **Cleaner interfaces** between layers
5. **Proper validation** throughout the codebase

The result will be a more robust, maintainable, and extensible module that follows best practices and project standards.