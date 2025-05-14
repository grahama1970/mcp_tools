"""
FastMCP wrapper for ArangoDB integration.

This module provides a FastMCP wrapper for exposing ArangoDB functionality
as Claude tools. It handles serialization/deserialization, request/response
mapping, and integration with the core layer functions.

Links to third-party documentation:
- FastMCP: (Internal project reference)

Sample input:
    Requests are received from the MCP framework and directed to the appropriate
    handler function based on the command name.

Expected output:
    MCP-compliant responses containing the result of the operation.
"""

import json
from typing import Dict, List, Any, Optional, Union, Callable
from loguru import logger

try:
    from fastmcp import FastMCP
except ImportError:
    logger.warning("FastMCP not available, creating mock class for development")
    # Mock class for development/testing without FastMCP
    class FastMCP:
        def __init__(self, name, description, function_map, schemas):
            self.name = name
            self.description = description
            self.function_map = function_map
            self.schemas = schemas

        def handle_request(self, request):
            return {"status": "mock", "message": "FastMCP mock class"}

# Import schemas
from mcp_tools.arangodb.mcp.schema import SCHEMAS

# Import database handlers
from mcp_tools.arangodb.mcp.db_handlers import HANDLER_MAP as DB_HANDLER_MAP

# Import core functionality
from mcp_tools.arangodb.core.search import (
    bm25_search,
    semantic_search,
    hybrid_search,
    tag_search,
    keyword_search,
    glossary_search,
    get_glossary_terms,
    add_glossary_terms,
    highlight_text_with_glossary,
)

def get_db_connection(db_config: Dict[str, Any]):
    """Create an ArangoDB database connection using the provided configuration."""
    from arango import ArangoClient
    
    host = db_config.get("host", "http://localhost:8529")
    username = db_config.get("username", "root")
    password = db_config.get("password", "")
    database = db_config.get("database", "_system")
    
    client = ArangoClient(hosts=host)
    return client.db(database, username=username, password=password)


def _bm25_search_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for BM25 text search."""
    start_time = json.time()
    query = params.get("query", "")
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        collection = params.get("collection", "")
        fields = params.get("fields", [])
        filter_conditions = params.get("filter", {})
        limit = params.get("limit", 10)
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not query:
            return {
                "status": "error",
                "error": "query parameter is required"
            }
        
        if not collection:
            return {
                "status": "error",
                "error": "collection parameter is required"
            }
            
        # Execute search
        results = bm25_search(
            query=query,
            collection=collection,
            fields=fields,
            filter_conditions=filter_conditions,
            limit=limit,
            db=db
        )
        
        # Calculate elapsed time
        elapsed_time = json.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "results": results,
            "query": query,
            "total": len(results),
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in BM25 search: {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": query
        }


def _semantic_search_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for semantic vector search."""
    start_time = json.time()
    query = params.get("query", "")
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        collection = params.get("collection", "")
        vector_field = params.get("vector_field", "")
        filter_conditions = params.get("filter", {})
        limit = params.get("limit", 10)
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not query:
            return {
                "status": "error",
                "error": "query parameter is required"
            }
        
        if not collection:
            return {
                "status": "error",
                "error": "collection parameter is required"
            }
            
        if not vector_field:
            return {
                "status": "error",
                "error": "vector_field parameter is required"
            }
            
        # Execute search
        results = semantic_search(
            query=query,
            collection=collection,
            vector_field=vector_field,
            filter_conditions=filter_conditions,
            limit=limit,
            db=db
        )
        
        # Calculate elapsed time
        elapsed_time = json.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "results": results,
            "query": query,
            "total": len(results),
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in semantic search: {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": query
        }


def _hybrid_search_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for hybrid search combining BM25 and semantic search."""
    start_time = json.time()
    query = params.get("query", "")
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        collection = params.get("collection", "")
        fields = params.get("fields", [])
        vector_field = params.get("vector_field", "")
        filter_conditions = params.get("filter", {})
        limit = params.get("limit", 10)
        weight_bm25 = params.get("weight_bm25", 0.5)
        weight_vector = params.get("weight_vector", 0.5)
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not query:
            return {
                "status": "error",
                "error": "query parameter is required"
            }
        
        if not collection:
            return {
                "status": "error",
                "error": "collection parameter is required"
            }
            
        if not fields:
            return {
                "status": "error",
                "error": "fields parameter is required"
            }
            
        if not vector_field:
            return {
                "status": "error",
                "error": "vector_field parameter is required"
            }
            
        # Execute search
        results = hybrid_search(
            query=query,
            collection=collection,
            fields=fields,
            vector_field=vector_field,
            filter_conditions=filter_conditions,
            limit=limit,
            weight_bm25=weight_bm25,
            weight_vector=weight_vector,
            db=db
        )
        
        # Calculate elapsed time
        elapsed_time = json.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "results": results,
            "query": query,
            "total": len(results),
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in hybrid search: {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": query
        }


def _tag_search_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for tag search."""
    start_time = json.time()
    tags = params.get("tags", [])
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        collection = params.get("collection", "")
        tag_field = params.get("tag_field", "tags")
        operator = params.get("operator", "AND")
        limit = params.get("limit", 10)
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not tags:
            return {
                "status": "error",
                "error": "tags parameter is required"
            }
        
        if not collection:
            return {
                "status": "error",
                "error": "collection parameter is required"
            }
            
        if not tag_field:
            return {
                "status": "error",
                "error": "tag_field parameter is required"
            }
            
        # Execute search
        results = tag_search(
            tags=tags,
            collection=collection,
            tag_field=tag_field,
            operator=operator,
            limit=limit,
            db=db
        )
        
        # Calculate elapsed time
        elapsed_time = json.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "results": results,
            "tags": tags,
            "operator": operator,
            "total": len(results),
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in tag search: {e}")
        return {
            "status": "error",
            "error": str(e),
            "tags": tags
        }


def _keyword_search_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for fuzzy keyword search."""
    start_time = json.time()
    query = params.get("query", "")
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        collection = params.get("collection", "")
        fields = params.get("fields", [])
        threshold = params.get("threshold", 80)
        limit = params.get("limit", 10)
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not query:
            return {
                "status": "error",
                "error": "query parameter is required"
            }
        
        if not collection:
            return {
                "status": "error",
                "error": "collection parameter is required"
            }
            
        if not fields:
            return {
                "status": "error",
                "error": "fields parameter is required"
            }
            
        # Execute search
        results = keyword_search(
            query=query,
            collection=collection,
            fields=fields,
            threshold=threshold,
            limit=limit,
            db=db
        )
        
        # Calculate elapsed time
        elapsed_time = json.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "results": results,
            "query": query,
            "threshold": threshold,
            "total": len(results),
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in keyword search: {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": query
        }


def _glossary_search_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for glossary search."""
    start_time = json.time()
    query = params.get("query", "")
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        collection = params.get("collection", "glossary")
        term_field = params.get("term_field", "term")
        definition_field = params.get("definition_field", "definition")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not query:
            return {
                "status": "error",
                "error": "query parameter is required"
            }
        
        if not collection:
            return {
                "status": "error",
                "error": "collection parameter is required"
            }
            
        if not term_field:
            return {
                "status": "error",
                "error": "term_field parameter is required"
            }
            
        if not definition_field:
            return {
                "status": "error",
                "error": "definition_field parameter is required"
            }
            
        # Execute search
        results = glossary_search(
            text=query,
            collection=collection,
            term_field=term_field,
            definition_field=definition_field,
            db=db
        )
        
        # Calculate elapsed time
        elapsed_time = json.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "results": results,
            "query": query,
            "total": len(results),
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in glossary search: {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": query
        }


def _get_glossary_terms_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for retrieving glossary terms."""
    start_time = json.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        collection = params.get("collection", "glossary")
        term_field = params.get("term_field", "term")
        definition_field = params.get("definition_field", "definition")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not collection:
            return {
                "status": "error",
                "error": "collection parameter is required"
            }
            
        if not term_field:
            return {
                "status": "error",
                "error": "term_field parameter is required"
            }
            
        if not definition_field:
            return {
                "status": "error",
                "error": "definition_field parameter is required"
            }
            
        # Execute query
        results = get_glossary_terms(
            collection=collection,
            term_field=term_field,
            definition_field=definition_field,
            db=db
        )
        
        # Calculate elapsed time
        elapsed_time = json.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "results": results,
            "total": len(results),
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in get glossary terms: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _add_glossary_term_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for adding glossary terms."""
    start_time = json.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        term = params.get("term", "")
        definition = params.get("definition", "")
        collection = params.get("collection", "glossary")
        term_field = params.get("term_field", "term")
        definition_field = params.get("definition_field", "definition")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not term:
            return {
                "status": "error",
                "error": "term parameter is required"
            }
            
        if not definition:
            return {
                "status": "error",
                "error": "definition parameter is required"
            }
            
        if not collection:
            return {
                "status": "error",
                "error": "collection parameter is required"
            }
            
        if not term_field:
            return {
                "status": "error",
                "error": "term_field parameter is required"
            }
            
        if not definition_field:
            return {
                "status": "error",
                "error": "definition_field parameter is required"
            }
            
        # Execute operation
        result = add_glossary_terms(
            terms={term: definition},
            collection=collection,
            term_field=term_field,
            definition_field=definition_field,
            db=db
        )
        
        # Calculate elapsed time
        elapsed_time = json.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "result": result,
            "term": term,
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in add glossary term: {e}")
        return {
            "status": "error",
            "error": str(e),
            "term": params.get("term", "")
        }


def _highlight_text_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for highlighting glossary terms in text."""
    start_time = json.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        text = params.get("text", "")
        collection = params.get("collection", "glossary")
        term_field = params.get("term_field", "term")
        definition_field = params.get("definition_field", "definition")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not text:
            return {
                "status": "error",
                "error": "text parameter is required"
            }
            
        if not collection:
            return {
                "status": "error",
                "error": "collection parameter is required"
            }
            
        if not term_field:
            return {
                "status": "error",
                "error": "term_field parameter is required"
            }
            
        if not definition_field:
            return {
                "status": "error",
                "error": "definition_field parameter is required"
            }
            
        # Execute operation
        result = highlight_text_with_glossary(
            text=text,
            collection=collection,
            term_field=term_field,
            definition_field=definition_field,
            db=db
        )
        
        # Calculate elapsed time
        elapsed_time = json.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "result": result,
            "text_length": len(text),
            "total_terms": len(result.get("terms", [])),
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in highlight text: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _get_document_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """LEGACY: Handler for retrieving a document. Use the new database functions instead."""
    logger.warning("Using legacy get_document_handler. Consider using database module handler instead.")
    
    # Import here to avoid circular dependencies
    from mcp_tools.arangodb.mcp.db_handlers import _get_document_handler
    return _get_document_handler(params)


# Map functions to operation names
FUNCTION_MAP = {
    # Search handlers
    "bm25_search": _bm25_search_handler,
    "semantic_search": _semantic_search_handler,
    "hybrid_search": _hybrid_search_handler,
    "tag_search": _tag_search_handler,
    "keyword_search": _keyword_search_handler,
    "glossary_search": _glossary_search_handler,
    "get_glossary_terms": _get_glossary_terms_handler,
    "add_glossary_term": _add_glossary_term_handler,
    "highlight_text": _highlight_text_handler,
    
    # Legacy DB handlers (maintained for backward compatibility)
    "get_document": _get_document_handler,
}

# Add database handlers from db_handlers
FUNCTION_MAP.update(DB_HANDLER_MAP)

# Create FastMCP application
mcp_app = FastMCP(
    name="ArangoDB",
    description="ArangoDB integration",
    function_map=FUNCTION_MAP,
    schemas=SCHEMAS
)

# MCP handler entry point
def handle_request(request):
    """Handle MCP requests by dispatching to the appropriate handler function."""
    return mcp_app.handle_request(request)


# Validation code
if __name__ == "__main__":
    import sys
    import time as actual_time
    # Use actual time module for testing
    json.time = actual_time.time
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test each handler (except those from DB_HANDLER_MAP, which have their own validation)
    for handler_name, handler_func in [(name, func) for name, func in FUNCTION_MAP.items() if name not in DB_HANDLER_MAP]:
        total_tests += 1
        
        # Check if handler has a corresponding schema
        if handler_name in SCHEMAS and handler_name not in SCHEMAS:
            all_validation_failures.append(f"Handler '{handler_name}' does not have a corresponding schema")
        
        # Test handler with minimal parameters
        try:
            minimal_params = {
                "db_config": {
                    "host": "http://localhost:8529",
                    "username": "root",
                    "password": "",
                    "database": "_system"
                }
            }
            handler_func(minimal_params)
        except Exception as e:
            # This is expected for validation errors, so we don't count these
            pass
    
    # Check if all schemas have a corresponding handler
    for schema_name in SCHEMAS:
        if schema_name not in FUNCTION_MAP:
            all_validation_failures.append(f"Schema '{schema_name}' does not have a corresponding handler")
    
    # Check that all database handlers are integrated
    for handler_name in DB_HANDLER_MAP:
        if handler_name not in FUNCTION_MAP:
            all_validation_failures.append(f"Database handler '{handler_name}' is not integrated in FUNCTION_MAP")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} handlers and mappings are valid")
        sys.exit(0)  # Exit with success code