"""
Utility functions for ArangoDB operations.

This package provides various utilities for working with ArangoDB:
- Connection utilities for database and collection management
- Embedding utilities for vector operations
- Logging utilities for formatting and truncating log output
"""

from mcp_tools.arangodb.core.utils.connection import (
    connect_arango,
    ensure_database,
    ensure_collection,
    ensure_memory_agent_collections,
    ensure_arangosearch_view
)

from mcp_tools.arangodb.core.utils.embedding_utils import (
    get_embedding,
    calculate_cosine_similarity
)

from mcp_tools.arangodb.core.utils.log_utils import (
    truncate_large_value,
    log_safe_results
)

__all__ = [
    # Connection utilities
    "connect_arango",
    "ensure_database", 
    "ensure_collection",
    "ensure_memory_agent_collections",
    "ensure_arangosearch_view",
    
    # Embedding utilities
    "get_embedding",
    "calculate_cosine_similarity",
    
    # Logging utilities
    "truncate_large_value",
    "log_safe_results"
]