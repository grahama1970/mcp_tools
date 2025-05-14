"""
ArangoDB integration for document storage, retrieval, and graph operations.

This module provides a 3-layer architecture for working with ArangoDB:
1. Core layer: Pure business logic functions
2. CLI layer: Command-line interface
3. MCP layer: Claude tool integration

High-level usage examples:

1. Basic document operations:
   ```python
   from arango import ArangoClient
   from mcp_tools.arangodb import (
       create_document, 
       get_document, 
       update_document,
       delete_document
   )
   
   # Connect to ArangoDB
   client = ArangoClient(hosts="http://localhost:8529")
   db = client.db("_system", username="root", password="")
   
   # Create a document
   doc = {
       "title": "Sample Document",
       "content": "This is a test document",
       "tags": ["test", "sample"]
   }
   result = create_document(db, "documents", doc)
   doc_key = result["_key"]
   
   # Retrieve the document
   document = get_document(db, "documents", doc_key)
   
   # Update the document
   update_document(db, "documents", doc_key, {"status": "updated"})
   
   # Delete the document
   delete_document(db, "documents", doc_key)
   ```

2. Relationship operations:
   ```python
   from mcp_tools.arangodb import (
       create_relationship,
       get_relationships,
       traverse_graph
   )
   
   # Create a relationship between documents
   create_relationship(
       db,
       "doc1_key",
       "doc2_key",
       "RELATED",
       "These documents are related",
       "documents",
       "relationships"
   )
   
   # Traverse the graph
   results = traverse_graph(
       db,
       "doc1_key",
       "documents",
       "relationships",
       "default",
       min_depth=1,
       max_depth=2
   )
   ```

3. Search operations:
   ```python
   from mcp_tools.arangodb.core.search import (
       bm25_search,
       ensure_view
   )
   
   # Ensure search view exists
   ensure_view(db, "documents_view", "documents", ["title", "content"])
   
   # Search for documents
   results = bm25_search(
       db,
       "search query",
       "documents",
       "documents_view"
   )
   ```

4. Using as a Claude tool:
   The ArangoDB functionality is exposed through MCP for use in Claude.
   See the MCP documentation for details on how to use the tools.
"""

# Version information
__version__ = "0.1.0"

# Core database operations
from .core.db_operations import (
    create_document,
    get_document,
    update_document,
    delete_document,
    query_documents,
    bulk_import_documents,
    execute_aql
)

# Relationship operations
from .core.relationship_ops import (
    create_relationship,
    get_relationships,
    delete_relationship,
    delete_document_relationships,
    traverse_graph,
    ensure_graph
)

# Export core search functionality
from .core.search.bm25_search import bm25_search, ensure_view

# Export MCP functionality
from .mcp.wrapper import mcp_handler

# Core layer modules
__all__ = [
    # Version
    "__version__",
    
    # Core database operations
    "create_document",
    "get_document",
    "update_document",
    "delete_document",
    "query_documents",
    "bulk_import_documents",
    "execute_aql",
    
    # Relationship operations
    "create_relationship",
    "get_relationships",
    "delete_relationship",
    "delete_document_relationships",
    "traverse_graph",
    "ensure_graph",
    
    # Core search
    "bm25_search",
    "ensure_view",
    
    # MCP
    "mcp_handler"
]


if __name__ == "__main__":
    import sys
    from arango import ArangoClient
    import uuid
    
    # Validation setup
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Import check
    total_tests += 1
    try:
        # Check that all exported functions can be imported
        for func_name in __all__:
            if func_name == "__version__":
                continue
                
            # Check that the function exists in globals
            if func_name not in globals():
                raise ValueError(f"Function {func_name} is in __all__ but not imported")
                
        print("All exported functions are properly imported")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Import check failed: {str(e)}")
    
    # Test 2: Create temp database and basic functionality
    total_tests += 1
    try:
        # Create a temporary test database
        client = ArangoClient(hosts="http://localhost:8529")
        sys_db = client.db("_system", username="root", password="")
        
        test_db_name = f"test_db_{uuid.uuid4().hex[:8]}"
        if not sys_db.has_database(test_db_name):
            sys_db.create_database(test_db_name)
        
        db = client.db(test_db_name, username="root", password="")
        
        # Create test collections
        test_collection = "test_documents"
        test_edge_collection = "test_edges"
        
        if db.has_collection(test_collection):
            db.delete_collection(test_collection)
        db.create_collection(test_collection)
        
        if db.has_collection(test_edge_collection):
            db.delete_collection(test_edge_collection)
        db.create_collection(test_edge_collection, edge=True)
        
        # Test document creation
        doc = {
            "title": "Test Document",
            "content": "This is a test document for initialization validation"
        }
        
        result = create_document(db, test_collection, doc)
        
        # Check returned document
        if not result or "_key" not in result:
            raise ValueError("Document creation failed")
            
        doc_key = result["_key"]
        
        # Verify document exists
        fetched = get_document(db, test_collection, doc_key)
        
        if not fetched or fetched["title"] != "Test Document":
            raise ValueError("Document retrieval failed")
            
        print("Basic functionality works")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Basic functionality check failed: {str(e)}")
    finally:
        # Clean up - drop the test database
        try:
            if 'sys_db' in locals() and 'test_db_name' in locals():
                if sys_db.has_database(test_db_name):
                    sys_db.delete_database(test_db_name)
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    # Final validation result
    if all_validation_failures:
        print(f"\nL VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\n VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)