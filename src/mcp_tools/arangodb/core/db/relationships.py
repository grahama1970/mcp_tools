"""
ArangoDB Relationship Operations.

This module provides operations for managing relationships in ArangoDB, including:
- Creating and deleting relationships between documents
- Linking messages to documents
- Retrieving related documents or messages

These functions build on the core CRUD operations to provide graph traversal
and relationship management capabilities.

Links:
- ArangoDB Python Driver: https://docs.python-arango.com/
- ArangoDB Graph Features: https://www.arangodb.com/docs/stable/graphs.html

Sample input:
    create_relationship(db, 'messages/123', 'documents/456', 'relates_to', {'weight': 0.95})

Expected output:
    {'_id': 'relates_to/789', '_key': '789', '_rev': '789', '_from': 'messages/123', 
     '_to': 'documents/456', 'weight': 0.95, 'created_at': '2023-05-11T10:30:00.000000'}
"""

from typing import Dict, List, Any, Optional, Union
import datetime

from arango.database import Database
from arango.exceptions import (
    DocumentInsertError,
    AQLQueryExecuteError
)

from mcp_tools.arangodb.core.db.crud import (
    create_document,
    delete_document,
    query_documents
)


def create_relationship(
    db: Database,
    from_id: str,
    to_id: str,
    edge_collection: str = "relates_to",
    properties: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a relationship (edge) between two documents.

    Args:
        db: ArangoDB database connection.
        from_id: The source document ID (_id format: collection/key).
        to_id: The target document ID (_id format: collection/key).
        edge_collection: The name of the edge collection.
        properties: Optional properties for the relationship.

    Returns:
        dict: The created edge document.

    Raises:
        DocumentInsertError: If relationship creation fails.
    """
    # Create edge document
    edge_data = {
        "_from": from_id,
        "_to": to_id,
    }

    # Add properties if provided
    if properties:
        edge_data.update(properties)

    try:
        # Insert the edge
        edge = db.collection(edge_collection).insert(
            edge_data,
            return_new=True
        )
        if "new" in edge:
            return edge["new"]
        return edge
    except DocumentInsertError as e:
        error_msg = f"Failed to create relationship from '{from_id}' to '{to_id}': {str(e)}"
        raise DocumentInsertError(error_msg) from e


def delete_relationship_by_key(
    db: Database,
    edge_key: str,
    edge_collection: str = "relates_to",
) -> bool:
    """
    Delete a relationship by its key.

    Args:
        db: ArangoDB database connection.
        edge_key: The edge key.
        edge_collection: The name of the edge collection.

    Returns:
        bool: True if the relationship was deleted.
    """
    return delete_document(edge_collection, edge_key, db)


def delete_relationships_between(
    db: Database,
    from_id: str,
    to_id: str,
    edge_collection: str = "relates_to",
) -> int:
    """
    Delete all relationships between two documents.

    Args:
        db: ArangoDB database connection.
        from_id: The source document ID.
        to_id: The target document ID.
        edge_collection: The name of the edge collection.

    Returns:
        int: The number of deleted relationships.
    """
    query = """
    FOR edge IN @@edge_collection
        FILTER edge._from == @from_id AND edge._to == @to_id
        REMOVE edge IN @@edge_collection
        COLLECT WITH COUNT INTO count
        RETURN count
    """
    bind_vars = {
        "@edge_collection": edge_collection,
        "from_id": from_id,
        "to_id": to_id,
    }
    
    result = query_documents(query, db, bind_vars)
    return result[0] if result else 0


def link_message_to_document(
    db: Database,
    message_id: str,
    document_id: str,
    properties: Optional[Dict[str, Any]] = None,
    edge_collection: str = "relates_to",
) -> Dict[str, Any]:
    """
    Create a relationship between a message and a document.

    Args:
        db: ArangoDB database connection.
        message_id: The message ID (_id format: collection/key).
        document_id: The document ID (_id format: collection/key).
        properties: Optional properties for the relationship.
        edge_collection: The name of the edge collection.

    Returns:
        dict: The created edge document.
    """
    return create_relationship(db, message_id, document_id, edge_collection, properties)


def get_documents_for_message(
    db: Database,
    message_id: str,
    collection_filter: Optional[str] = None,
    edge_collection: str = "relates_to",
) -> List[Dict[str, Any]]:
    """
    Retrieve all documents related to a message.

    Args:
        db: ArangoDB database connection.
        message_id: The message ID (_id format: collection/key).
        collection_filter: Optional filter for document collections.
        edge_collection: The name of the edge collection.

    Returns:
        list: A list of related documents.
    """
    # Construct the collection filter if provided
    collection_filter_clause = ""
    if collection_filter:
        collection_filter_clause = f"FILTER PARSE_IDENTIFIER(document._id).collection == '{collection_filter}'"
    
    query = f"""
    FOR edge IN @@edge_collection
        FILTER edge._from == @message_id
        LET document = DOCUMENT(edge._to)
        {collection_filter_clause}
        RETURN {{ 
            document: document,
            edge: edge
        }}
    """
    
    bind_vars = {
        "@edge_collection": edge_collection,
        "message_id": message_id,
    }
    
    return query_documents(query, db, bind_vars)


def get_messages_for_document(
    db: Database,
    document_id: str,
    message_collection: str = "messages",
    edge_collection: str = "relates_to",
) -> List[Dict[str, Any]]:
    """
    Retrieve all messages related to a document.

    Args:
        db: ArangoDB database connection.
        document_id: The document ID (_id format: collection/key).
        message_collection: The name of the messages collection.
        edge_collection: The name of the edge collection.

    Returns:
        list: A list of related messages.
    """
    query = """
    FOR edge IN @@edge_collection
        FILTER edge._to == @document_id AND PARSE_IDENTIFIER(edge._from).collection == @message_collection
        LET message = DOCUMENT(edge._from)
        RETURN { 
            message: message,
            edge: edge
        }
    """
    
    bind_vars = {
        "@edge_collection": edge_collection,
        "document_id": document_id,
        "message_collection": message_collection,
    }
    
    return query_documents(query, db, bind_vars)


def get_related_documents(
    db: Database,
    document_id: str,
    collection_filter: Optional[str] = None,
    edge_collection: str = "relates_to",
    direction: str = "outbound",
    max_depth: int = 1,
) -> List[Dict[str, Any]]:
    """
    Retrieve documents related to a document through graph traversal.

    Args:
        db: ArangoDB database connection.
        document_id: The document ID (_id format: collection/key).
        collection_filter: Optional filter for document collections.
        edge_collection: The name of the edge collection.
        direction: The direction to traverse ("outbound", "inbound", or "any").
        max_depth: The maximum traversal depth.

    Returns:
        list: A list of related documents with traversal information.
    """
    # Validate direction
    if direction not in ["outbound", "inbound", "any"]:
        direction = "outbound"
    
    # Construct collection filter
    filter_clause = ""
    if collection_filter:
        filter_clause = f"FILTER IS_SAME_COLLECTION('{collection_filter}', vertex._id)"
        
    query = f"""
    FOR vertex, edge, path IN 1..@max_depth {direction} @start_vertex @@edge_collection
        {filter_clause}
        RETURN {{
            document: vertex,
            edge: edge,
            path: path
        }}
    """
    
    bind_vars = {
        "@edge_collection": edge_collection,
        "start_vertex": document_id,
        "max_depth": max_depth,
    }
    
    return query_documents(query, db, bind_vars)


if __name__ == "__main__":
    import sys
    from arango import ArangoClient
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Setup test database connection
    # NOTE: This assumes a local ArangoDB instance is running
    # If you need to test against a real instance, update these credentials
    try:
        client = ArangoClient(hosts="http://localhost:8529")
        db = client.db("_system", username="root", password="")
        
        # Create test collections if they don't exist
        if not db.has_collection("test_messages"):
            db.create_collection("test_messages")
            
        if not db.has_collection("test_documents"):
            db.create_collection("test_documents")
            
        if not db.has_collection("test_relates_to"):
            db.create_edge_collection("test_relates_to")
            
        message_collection = "test_messages"
        document_collection = "test_documents"
        relationship_collection = "test_relates_to"
        
        # Create test message
        message = db.collection(message_collection).insert({
            "content": "Test message",
            "role": "user",
            "conversation_id": "test_conversation"
        }, return_new=True)["new"]
        
        message_id = message["_id"]
        
        # Create test documents
        doc1 = db.collection(document_collection).insert({
            "title": "Test Document 1",
            "content": "This is test document 1"
        }, return_new=True)["new"]
        
        doc2 = db.collection(document_collection).insert({
            "title": "Test Document 2",
            "content": "This is test document 2"
        }, return_new=True)["new"]
        
        doc1_id = doc1["_id"]
        doc2_id = doc2["_id"]
        
        # Test 1: Create relationship
        total_tests += 1
        relationship = create_relationship(
            db=db,
            from_id=message_id,
            to_id=doc1_id,
            edge_collection=relationship_collection,
            properties={"weight": 0.95}
        )
        
        if not relationship or not isinstance(relationship, dict):
            all_validation_failures.append("Relationship creation failed: No result returned")
        elif relationship.get("_from") != message_id:
            all_validation_failures.append(f"Relationship creation failed: Incorrect source ID")
        elif relationship.get("_to") != doc1_id:
            all_validation_failures.append(f"Relationship creation failed: Incorrect target ID")
        elif relationship.get("weight") != 0.95:
            all_validation_failures.append(f"Relationship creation failed: Properties not stored")
            
        relationship_key = relationship.get("_key", "")
        
        # Test 2: Link message to document
        total_tests += 1
        link = link_message_to_document(
            db=db,
            message_id=message_id,
            document_id=doc2_id,
            properties={"relevance": 0.85},
            edge_collection=relationship_collection
        )
        
        if not link or not isinstance(link, dict):
            all_validation_failures.append("Message-document link failed: No result returned")
        elif link.get("_from") != message_id:
            all_validation_failures.append(f"Message-document link failed: Incorrect message ID")
        elif link.get("_to") != doc2_id:
            all_validation_failures.append(f"Message-document link failed: Incorrect document ID")
        elif link.get("relevance") != 0.85:
            all_validation_failures.append(f"Message-document link failed: Properties not stored")
        
        # Test 3: Get documents for message
        total_tests += 1
        related_docs = get_documents_for_message(
            db=db,
            message_id=message_id,
            edge_collection=relationship_collection
        )
        
        if not related_docs or not isinstance(related_docs, list):
            all_validation_failures.append("Get documents for message failed: No results returned")
        elif len(related_docs) != 2:
            all_validation_failures.append(f"Get documents for message failed: Expected 2 documents, got {len(related_docs)}")
            
        # Test 4: Get documents for message with collection filter
        total_tests += 1
        filtered_docs = get_documents_for_message(
            db=db,
            message_id=message_id,
            collection_filter=document_collection,
            edge_collection=relationship_collection
        )
        
        if not filtered_docs or not isinstance(filtered_docs, list):
            all_validation_failures.append("Get filtered documents failed: No results returned")
        elif len(filtered_docs) != 2:
            all_validation_failures.append(f"Get filtered documents failed: Expected 2 documents, got {len(filtered_docs)}")
            
        # Test 5: Get messages for document
        total_tests += 1
        related_messages = get_messages_for_document(
            db=db,
            document_id=doc1_id,
            message_collection=message_collection,
            edge_collection=relationship_collection
        )
        
        if not related_messages or not isinstance(related_messages, list):
            all_validation_failures.append("Get messages for document failed: No results returned")
        elif len(related_messages) != 1:
            all_validation_failures.append(f"Get messages for document failed: Expected 1 message, got {len(related_messages)}")
            
        # Test 6: Delete relationship by key
        total_tests += 1
        if relationship_key:
            delete_result = delete_relationship_by_key(
                db=db,
                edge_key=relationship_key,
                edge_collection=relationship_collection
            )
            
            if not delete_result:
                all_validation_failures.append("Delete relationship by key failed: Operation unsuccessful")
                
            # Verify deletion
            remaining_docs = get_documents_for_message(
                db=db,
                message_id=message_id,
                edge_collection=relationship_collection
            )
            
            if len(remaining_docs) != 1:  # Should have 1 relationship left
                all_validation_failures.append(f"Relationship deletion verification failed: Expected 1 relationship, got {len(remaining_docs)}")
        else:
            all_validation_failures.append("Delete relationship by key skipped: No valid relationship key")
            
        # Test 7: Delete relationships between
        total_tests += 1
        delete_count = delete_relationships_between(
            db=db,
            from_id=message_id,
            to_id=doc2_id,
            edge_collection=relationship_collection
        )
        
        if delete_count != 1:  # Should have deleted 1 relationship
            all_validation_failures.append(f"Delete relationships between failed: Expected 1 deleted relationship, got {delete_count}")
            
        # Verify all relationships are gone
        final_relationships = get_documents_for_message(
            db=db,
            message_id=message_id,
            edge_collection=relationship_collection
        )
        
        if final_relationships and len(final_relationships) > 0:
            all_validation_failures.append(f"Relationship deletion verification failed: Expected 0 relationships, got {len(final_relationships)}")
        
        # Cleanup test collections
        db.collection(message_collection).truncate()
        db.collection(document_collection).truncate()
        db.collection(relationship_collection).truncate()
        
    except Exception as e:
        all_validation_failures.append(f"Test setup failed: {str(e)}")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Relationship operations are validated and formal tests can now be written")
        sys.exit(0)  # Exit with success code