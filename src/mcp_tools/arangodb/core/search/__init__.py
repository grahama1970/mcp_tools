"""Search functionality for ArangoDB.

This module provides various search algorithms including BM25, semantic search,
hybrid search, tag search, keyword search, glossary search, and graph traversal.
"""

from mcp_tools.arangodb.core.search.bm25_search import bm25_search
from mcp_tools.arangodb.core.search.semantic_search import semantic_search, get_cached_vector_results
from mcp_tools.arangodb.core.search.hybrid_search import hybrid_search
from mcp_tools.arangodb.core.search.tag_search import tag_search, filter_by_tags, create_tag_filter_expression
from mcp_tools.arangodb.core.search.keyword_search import keyword_search, validate_keyword_search_result
from mcp_tools.arangodb.core.search.glossary_search import (
    glossary_search,
    get_glossary_terms,
    add_glossary_terms,
    highlight_text_with_glossary,
    GlossaryManager,
    validate_glossary_search
)

__all__ = [
    "bm25_search",
    "semantic_search",
    "get_cached_vector_results",
    "hybrid_search",
    "tag_search",
    "filter_by_tags",
    "create_tag_filter_expression",
    "keyword_search",
    "validate_keyword_search_result",
    "glossary_search",
    "get_glossary_terms",
    "add_glossary_terms",
    "highlight_text_with_glossary",
    "GlossaryManager",
    "validate_glossary_search",
]
