"""
Core relationship operations for ArangoDB graph functionality.

This module provides pure business logic for managing relationships between
documents in ArangoDB graphs. These functions handle the creation, retrieval,
and deletion of edges that represent relationships.

Links to third-party documentation:
- ArangoDB Graphs: https://www.arangodb.com/docs/stable/graphs.html
- ArangoDB Python Driver: https://docs.python-arango.com/en/main/

Sample input:
    db = connect_to_arango()
    result = create_relationship(
        db, 
        "source_key", 
        "target_key", 
        "RELATED", 
        "These documents are related",
        collection_name="my_collection",
        edge_collection_name="my_edges"
    )

Expected output:
    {
        "_id": "my_edges/12345",
        "_key": "12345",
        "_rev": "12345"
    }
"""

import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union, Set

from loguru import logger
from arango.database import StandardDatabase
from arango.exceptions import (
    GraphCreateError,
    DocumentInsertError,
    AQLQueryExecuteError
)

from .db_operations import create_document, delete_document, execute_aql


def create_relationship(
    db: StandardDatabase,
    from_doc_key: str,
    to_doc_key: str,
    relationship_type: str,
    rationale: str,
    collection_name: str,
    edge_collection_name: str,
    attributes: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Create an edge between two documents in a graph.

    Args:
        db: ArangoDB database handle
        from_doc_key: Key of the source document
        to_doc_key: Key of the target document
        relationship_type: Type/category of the relationship
        rationale: Explanation for the relationship
        collection_name: Name of the vertex collection
        edge_collection_name: Name of the edge collection
        attributes: Optional additional metadata for the edge

    Returns:
        Optional[Dict[str, Any]]: The created edge document if successful, None otherwise
    """
    try:
        # Create edge document
        edge = {
            "_from": f"{collection_name}/{from_doc_key}",
            "_to": f"{collection_name}/{to_doc_key}",
            "type": relationship_type,
            "rationale": rationale,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Add additional attributes if provided
        if attributes:
            edge.update(attributes)
            
        # Create the edge
        result = create_document(db, edge_collection_name, edge)
        if result:
            logger.info(f"Created relationship ({relationship_type}) from {from_doc_key} to {to_doc_key}")
        
        return result
        
    except Exception as e:
        logger.exception(f"Error creating relationship: {e}")
        return None


def get_relationships(
    db: StandardDatabase,
    document_key: str,
    collection_name: str,
    edge_collection_name: str,
    relationship_type: Optional[str] = None,
    direction: str = "ANY"
) -> List[Dict[str, Any]]:
    """
    Get relationships for a document.

    Args:
        db: ArangoDB database handle
        document_key: Key of the document
        collection_name: Name of the vertex collection
        edge_collection_name: Name of the edge collection
        relationship_type: Optional type of relationship to filter by
        direction: Direction of relationships to retrieve ("OUTBOUND", "INBOUND", or "ANY")

    Returns:
        List[Dict[str, Any]]: List of relationship edge documents
    """
    try:
        # Validate direction
        valid_directions = ["OUTBOUND", "INBOUND", "ANY"]
        if direction not in valid_directions:
            logger.error(f"Invalid direction: {direction}. Must be one of: {', '.join(valid_directions)}")
            return []
            
        # Build AQL query
        doc_id = f"{collection_name}/{document_key}"
        type_filter = "FILTER edge.type == @type" if relationship_type else ""
        
        aql = f"""
        FOR vertex, edge IN 1..1 {direction} @start_vertex GRAPH @graph_name
        {type_filter}
        RETURN edge
        """
        
        # Set bind variables
        bind_vars = {
            "start_vertex": doc_id,
            "graph_name": "default",  # This will be overridden when calling execute_aql
        }
        
        if relationship_type:
            bind_vars["type"] = relationship_type
            
        # Execute query
        edges = execute_aql(db, aql, bind_vars)
        logger.info(f"Retrieved {len(edges)} relationships for document {document_key}")
        
        return edges
        
    except Exception as e:
        logger.exception(f"Error getting relationships for document {document_key}: {e}")
        return []


def delete_relationship(
    db: StandardDatabase,
    edge_key: str,
    edge_collection_name: str
) -> bool:
    """
    Delete a relationship edge by its key.

    Args:
        db: ArangoDB database handle
        edge_key: The key of the edge document to delete
        edge_collection_name: Name of the edge collection

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        result = delete_document(db, edge_collection_name, edge_key, ignore_missing=True)
        if result:
            logger.info(f"Deleted relationship edge {edge_key}")
        return result
        
    except Exception as e:
        logger.exception(f"Error deleting relationship edge {edge_key}: {e}")
        return False


def delete_document_relationships(
    db: StandardDatabase,
    document_key: str,
    collection_name: str,
    edge_collection_name: str
) -> bool:
    """
    Delete all relationships associated with a document.

    Args:
        db: ArangoDB database handle
        document_key: Key of the document
        collection_name: Name of the vertex collection
        edge_collection_name: Name of the edge collection

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        doc_id = f"{collection_name}/{document_key}"
        
        # Find all edges connected to this document (both inbound and outbound)
        aql = f"""
        FOR edge IN {edge_collection_name}
        FILTER edge._from == @doc_id OR edge._to == @doc_id
        RETURN edge._key
        """
        
        edge_keys = execute_aql(db, aql, {"doc_id": doc_id})
        
        # Delete each edge
        success = True
        for edge in edge_keys:
            if isinstance(edge, str):
                # If edge is just the key string
                result = delete_document(db, edge_collection_name, edge, ignore_missing=True)
            else:
                # If edge is a dictionary with _key
                result = delete_document(db, edge_collection_name, edge.get("_key"), ignore_missing=True)
                
            if not result:
                success = False
                
        logger.info(f"Deleted {len(edge_keys)} relationships for document {document_key}")
        return success
        
    except Exception as e:
        logger.exception(f"Error deleting relationships for document {document_key}: {e}")
        return False


def traverse_graph(
    db: StandardDatabase,
    start_vertex_key: str,
    collection_name: str,
    edge_collection_name: str,
    graph_name: str,
    min_depth: int = 1,
    max_depth: int = 1,
    direction: str = "OUTBOUND",
    vertex_filter: Optional[str] = None,
    edge_filter: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Traverse a graph starting from a specific vertex.

    Args:
        db: ArangoDB database handle
        start_vertex_key: Key of the starting vertex
        collection_name: Name of the vertex collection
        edge_collection_name: Name of the edge collection
        graph_name: Name of the graph
        min_depth: Minimum traversal depth
        max_depth: Maximum traversal depth
        direction: Direction of traversal ("OUTBOUND", "INBOUND", or "ANY")
        vertex_filter: Optional AQL filter for vertices
        edge_filter: Optional AQL filter for edges

    Returns:
        List[Dict[str, Any]]: List of traversal results
    """
    try:
        # Validate direction
        valid_directions = ["OUTBOUND", "INBOUND", "ANY"]
        if direction not in valid_directions:
            logger.error(f"Invalid direction: {direction}. Must be one of: {', '.join(valid_directions)}")
            return []
            
        # Validate depth
        if min_depth < 0 or max_depth < min_depth:
            logger.error(f"Invalid depth parameters: min_depth={min_depth}, max_depth={max_depth}")
            return []
            
        # Build AQL query
        start_vertex = f"{collection_name}/{start_vertex_key}"
        
        # Add filters if provided
        v_filter = vertex_filter if vertex_filter else ""
        e_filter = edge_filter if edge_filter else ""
        
        aql = f"""
        FOR v, e, p IN {min_depth}..{max_depth} {direction} @start_vertex GRAPH @graph_name
        {v_filter}
        {e_filter}
        RETURN {{
            vertex: v,
            edge: e,
            path: p
        }}
        """
        
        # Set bind variables
        bind_vars = {
            "start_vertex": start_vertex,
            "graph_name": graph_name
        }
            
        # Execute query
        results = execute_aql(db, aql, bind_vars)
        logger.info(f"Traversal from {start_vertex_key} returned {len(results)} results")
        
        return results
        
    except Exception as e:
        logger.exception(f"Error traversing graph from {start_vertex_key}: {e}")
        return []


def ensure_graph(
    db: StandardDatabase,
    graph_name: str,
    edge_collection_name: str,
    vertex_collection_name: str
) -> bool:
    """
    Ensure a graph exists with the specified edge and vertex collections.

    Args:
        db: ArangoDB database handle
        graph_name: Name of the graph
        edge_collection_name: Name of the edge collection
        vertex_collection_name: Name of the vertex collection

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if the graph already exists
        graph_names = db.graphs()
        existing_graphs = [g["name"] for g in graph_names]
        
        if graph_name in existing_graphs:
            logger.debug(f"Graph {graph_name} already exists")
            return True
            
        # Create the graph
        edge_definitions = [{
            "edge_collection": edge_collection_name,
            "from_vertex_collections": [vertex_collection_name],
            "to_vertex_collections": [vertex_collection_name]
        }]
        
        db.create_graph(
            name=graph_name,
            edge_definitions=edge_definitions
        )
        
        logger.info(f"Created graph {graph_name} with edge collection {edge_collection_name}")
        return True
        
    except GraphCreateError as e:
        # Graph might already exist
        logger.warning(f"Graph create error: {e}")
        return True
    except Exception as e:
        logger.exception(f"Error ensuring graph {graph_name}: {e}")
        return False


if __name__ == "__main__":
    import sys
    from arango import ArangoClient
    import uuid
    
    # Validation setup
    all_validation_failures = []
    total_tests = 0
    
    # Setup test database connection
    try:
        # Connect to ArangoDB (this tries to use standard environment variables)
        # For testing, we use a temporary database that will be dropped after tests
        client = ArangoClient(hosts="http://localhost:8529")
        sys_db = client.db("_system", username="root", password="")
        
        # Create a temporary test database
        test_db_name = f"test_db_{uuid.uuid4().hex[:8]}"
        if not sys_db.has_database(test_db_name):
            sys_db.create_database(test_db_name)
        
        db = client.db(test_db_name, username="root", password="")
        
        # Create test collections
        test_collection = "test_vertices"
        test_edge_collection = "test_edges"
        test_graph = "test_graph"
        
        if db.has_collection(test_collection):
            db.delete_collection(test_collection)
        db.create_collection(test_collection)
        
        if db.has_collection(test_edge_collection):
            db.delete_collection(test_edge_collection)
        db.create_collection(test_edge_collection, edge=True)
        
        # Test 1: Ensure graph
        total_tests += 1
        result = ensure_graph(db, test_graph, test_edge_collection, test_collection)
        if not result:
            all_validation_failures.append("Test 1: Ensure graph failed")
            
        # Test 2: Create test documents
        doc1 = create_document(db, test_collection, {"name": "Document 1"})
        doc2 = create_document(db, test_collection, {"name": "Document 2"})
        doc3 = create_document(db, test_collection, {"name": "Document 3"})
        
        if not doc1 or not doc2 or not doc3:
            all_validation_failures.append("Test setup: Creating test documents failed")
            raise Exception("Test setup failed")
            
        doc1_key = doc1.get("_key")
        doc2_key = doc2.get("_key")
        doc3_key = doc3.get("_key")
        
        # Test 3: Create relationship
        total_tests += 1
        rel1 = create_relationship(
            db,
            doc1_key,
            doc2_key,
            "RELATED",
            "Test related documents",
            test_collection,
            test_edge_collection,
            {"weight": 0.8}
        )
        
        if not rel1 or "_key" not in rel1:
            all_validation_failures.append("Test 3: Create relationship failed")
        else:
            rel1_key = rel1["_key"]
        
        # Test 4: Create another relationship
        total_tests += 1
        rel2 = create_relationship(
            db,
            doc1_key,
            doc3_key,
            "DEPENDS_ON",
            "Test dependency relationship",
            test_collection,
            test_edge_collection
        )
        
        if not rel2 or "_key" not in rel2:
            all_validation_failures.append("Test 4: Create second relationship failed")
            
        # Test 5: Get relationships
        total_tests += 1
        relationships = get_relationships(
            db,
            doc1_key,
            test_collection,
            test_edge_collection,
            direction="OUTBOUND"
        )
        
        if not relationships or len(relationships) != 2:
            all_validation_failures.append("Test 5: Get relationships failed")
            
        # Test 6: Get relationships with type filter
        total_tests += 1
        filtered_relationships = get_relationships(
            db,
            doc1_key,
            test_collection,
            test_edge_collection,
            relationship_type="RELATED",
            direction="OUTBOUND"
        )
        
        if not filtered_relationships or len(filtered_relationships) != 1:
            all_validation_failures.append("Test 6: Get relationships with type filter failed")
            
        # Test 7: Traverse graph
        total_tests += 1
        traversal = traverse_graph(
            db,
            doc1_key,
            test_collection,
            test_edge_collection,
            test_graph,
            min_depth=1,
            max_depth=1,
            direction="OUTBOUND"
        )
        
        if not traversal or len(traversal) != 2:
            all_validation_failures.append("Test 7: Traverse graph failed")
            
        # Test 8: Delete relationship
        total_tests += 1
        delete_result = delete_relationship(db, rel1_key, test_edge_collection)
        if not delete_result:
            all_validation_failures.append("Test 8: Delete relationship failed")
            
        # Test 9: Delete all relationships for a document
        total_tests += 1
        all_deleted = delete_document_relationships(db, doc1_key, test_collection, test_edge_collection)
        if not all_deleted:
            all_validation_failures.append("Test 9: Delete document relationships failed")
        
        # Verify all relationships were deleted
        remaining = get_relationships(db, doc1_key, test_collection, test_edge_collection)
        if remaining and len(remaining) > 0:
            all_validation_failures.append("Test 9: Not all relationships were deleted")
            
    except Exception as e:
        print(f"Setup error: {e}")
        all_validation_failures.append(f"Database setup failed: {str(e)}")
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
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)