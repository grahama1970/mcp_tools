# ArangoDB Module Refactoring Plan

This document outlines the refactoring plan for the ArangoDB module to align with the 3-layer architecture pattern.

## Current Structure Analysis

The current module structure is flat with mixed responsibilities:

- `db_operations.py`: Core database operations (CRUD, message history, relationships)
- `cli.py`: Command-line interface with typer
- `search_api/`: Various search implementations without clear separation
- `memory_agent/`: Memory agent implementation
- `utils/`: Utility functions (display, embedding, etc.)

## New 3-Layer Structure

### 1. Core Layer

The core layer will contain pure business logic with no UI or framework dependencies:

```
arangodb/
└── core/
    ├── __init__.py
    ├── db_operations.py      # Generic CRUD operations
    ├── message_operations.py # Message-specific operations
    ├── relationship_ops.py   # Graph relationship operations
    ├── memory_agent.py       # Memory agent core functionality
    └── search/
        ├── __init__.py
        ├── bm25_search.py     # BM25 search algorithm
        ├── semantic_search.py # Semantic search algorithm
        ├── hybrid_search.py   # Combined search
        ├── graph_traverse.py  # Graph traversal
        └── utils.py           # Search utilities
```

### 2. CLI Layer

The CLI layer will handle user interface, formatting, and input validation:

```
arangodb/
└── cli/
    ├── __init__.py
    ├── app.py               # Typer application with command groups
    ├── formatters.py        # Rich formatting for console output
    ├── validators.py        # Input validation
    ├── schemas.py           # Pydantic models for input/output
    └── commands/
        ├── __init__.py
        ├── search.py        # Search commands
        ├── crud.py          # CRUD commands
        ├── graph.py         # Graph/relationship commands
        └── memory.py        # Memory agent commands
```

### 3. MCP Layer

The MCP layer will provide integration with Claude tools:

```
arangodb/
└── mcp/
    ├── __init__.py
    ├── schema.py            # JSON schema definitions
    └── wrapper.py           # FastMCP wrapper
```

## Implementation Plan

### Phase 1: Create Directory Structure

1. Create new directories (`core`, `cli`, `mcp` and subdirectories)
2. Add `__init__.py` files to all directories

### Phase 2: Implement Core Layer

1. Move core database operations to `core/db_operations.py`
2. Extract message operations to `core/message_operations.py`
3. Extract relationship operations to `core/relationship_ops.py`
4. Move memory agent core logic to `core/memory_agent.py`
5. Reorganize search implementations in `core/search/` subdirectories
6. Ensure all core functions have proper docstrings and validation

### Phase 3: Implement CLI Layer

1. Create Typer application structure in `cli/app.py`
2. Extract formatting functions to `cli/formatters.py`
3. Create input validation in `cli/validators.py`
4. Define data models in `cli/schemas.py`
5. Organize commands by feature area in `cli/commands/` subdirectories
6. Implement Rich-based output formatting

### Phase 4: Implement MCP Layer

1. Define JSON schemas in `mcp/schema.py`
2. Create FastMCP wrapper in `mcp/wrapper.py`
3. Map between MCP requests and core functions

### Phase 5: Update Module Exports

1. Update main `__init__.py` to export core functionality
2. Include version information
3. Document high-level usage examples

## Migration Strategy

1. Keep original files during refactoring for reference
2. Implement new structure in parallel
3. Test each layer independently
4. Verify integration between layers
5. Update imports in any dependent modules
6. Remove original files once refactoring is complete

## Validation Requirements

Each file will include a main block that:
1. Tests functionality with real data
2. Verifies expected results
3. Reports success/failure
4. Exits with appropriate code

## Timeline

Estimated effort:
- Phase 1: 0.5 day
- Phase 2: 2-3 days
- Phase 3: 2-3 days
- Phase 4: 1 day
- Phase 5: 0.5 day
- Testing & Validation: 1-2 days

Total: 7-10 days