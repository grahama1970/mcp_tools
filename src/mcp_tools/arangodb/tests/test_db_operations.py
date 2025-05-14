"""
Test suite for ArangoDB database operations.

This test suite validates the functionality of the refactored database operations
in the three-layer architecture (core, cli, mcp).

Links:
- pytest: https://docs.pytest.org/
- ArangoDB: https://www.arangodb.com/docs/
"""

import os
import sys
import uuid
import pytest
from typing import Dict, Any, List, Optional, Tuple

from arango import ArangoClient
from arango.database import Database
from arango.exceptions import (
    DocumentInsertError, 
    DocumentGetError,
    DocumentUpdateError,
    DocumentDeleteError,
    AQLQueryExecuteError
)

# Import core database functions
from mcp_tools.arangodb.core.db import (
    # CRUD operations
    create_document,
    get_document,
    update_document,
    delete_document,
    query_documents,
    
    # Message operations
    create_message,
    get_message,
    update_message,
    delete_message,
    get_conversation_messages,
    delete_conversation,
    
    # Relationship operations
    create_relationship,
    delete_relationship_by_key,
    delete_relationships_between,
    link_message_to_document,
    get_documents_for_message,
    get_messages_for_document,
    get_related_documents,
)

# Configure test ArangoDB connection
TEST_HOST = os.environ.get("ARANGO_HOST", "http://localhost:8529")
TEST_USER = os.environ.get("ARANGO_USER", "root")
TEST_PASSWORD = os.environ.get("ARANGO_PASSWORD", "")
TEST_DB = os.environ.get("ARANGO_DB", "_system")


@pytest.fixture
def db() -> Database:
    """Fixture for ArangoDB database connection."""
    client = ArangoClient(hosts=TEST_HOST)
    return client.db(TEST_DB, username=TEST_USER, password=TEST_PASSWORD)


@pytest.fixture
def clean_test_collections(db):
    """
    Fixture to create test collections and clean them before/after tests.
    """
    # Define test collection names
    collections = ["test_docs", "test_messages", "test_relates_to"]
    
    # Create collections if they don't exist
    for col in collections:
        if not db.has_collection(col):
            if col == "test_relates_to":
                db.create_edge_collection(col)
            else:
                db.create_collection(col)
    
    # Truncate collections (clean them before tests)
    for col in collections:
        db.collection(col).truncate()
    
    # Return a dictionary of collection names for use in tests
    yield {
        "docs": "test_docs",
        "messages": "test_messages",
        "edges": "test_relates_to"
    }
    
    # Clean up after tests
    for col in collections:
        db.collection(col).truncate()


# === CRUD Operations Tests ===

def test_create_document(db, clean_test_collections):
    """Test creating a document."""
    collection = clean_test_collections["docs"]
    
    # Test data
    doc_data = {
        "name": "Test Document",
        "value": 42,
        "status": "active"
    }
    
    # Create document
    result = create_document(collection, doc_data, db)
    
    # Assertions
    assert result is not None
    assert isinstance(result, dict)
    assert "_id" in result
    assert "_key" in result
    assert "_rev" in result
    assert "name" in result and result["name"] == "Test Document"
    assert "value" in result and result["value"] == 42
    assert "status" in result and result["status"] == "active"
    assert "created_at" in result
    assert "updated_at" in result
    assert "uuid" in result


def test_get_document(db, clean_test_collections):
    """Test retrieving a document."""
    collection = clean_test_collections["docs"]
    
    # Create test document
    doc_data = {
        "name": "Document to Retrieve",
        "value": 123
    }
    created = create_document(collection, doc_data, db)
    doc_key = created["_key"]
    
    # Retrieve document
    retrieved = get_document(collection, doc_key, db)
    
    # Assertions
    assert retrieved is not None
    assert retrieved["_key"] == doc_key
    assert retrieved["name"] == "Document to Retrieve"
    assert retrieved["value"] == 123


def test_update_document(db, clean_test_collections):
    """Test updating a document."""
    collection = clean_test_collections["docs"]
    
    # Create test document
    doc_data = {
        "name": "Original Document",
        "value": 50,
        "status": "inactive"
    }
    created = create_document(collection, doc_data, db)
    doc_key = created["_key"]
    
    # Update document
    update_data = {
        "name": "Updated Document",
        "value": 100,
        "new_field": "new value"
    }
    updated = update_document(collection, doc_key, update_data, db)
    
    # Assertions
    assert updated is not None
    assert updated["_key"] == doc_key
    assert updated["name"] == "Updated Document"
    assert updated["value"] == 100
    assert updated["status"] == "inactive"  # Unchanged field
    assert updated["new_field"] == "new value"  # New field
    assert updated["created_at"] == created["created_at"]  # Timestamps
    assert updated["updated_at"] != created["updated_at"]  # Updated timestamp


def test_delete_document(db, clean_test_collections):
    """Test deleting a document."""
    collection = clean_test_collections["docs"]
    
    # Create test document
    doc_data = {
        "name": "Document to Delete",
        "value": 999
    }
    created = create_document(collection, doc_data, db)
    doc_key = created["_key"]
    
    # Delete document
    delete_result = delete_document(collection, doc_key, db)
    
    # Assertions
    assert delete_result is True
    
    # Verify document is deleted
    with pytest.raises(DocumentGetError):
        get_document(collection, doc_key, db)


def test_query_documents(db, clean_test_collections):
    """Test querying documents with AQL."""
    collection = clean_test_collections["docs"]
    
    # Create test documents
    docs = [
        {"name": "Doc 1", "category": "A", "value": 10},
        {"name": "Doc 2", "category": "B", "value": 20},
        {"name": "Doc 3", "category": "A", "value": 30},
        {"name": "Doc 4", "category": "C", "value": 40},
        {"name": "Doc 5", "category": "B", "value": 50},
    ]
    
    for doc in docs:
        create_document(collection, doc, db)
    
    # Query 1: Get all documents in category A
    query1 = "FOR doc IN @@collection FILTER doc.category == @category RETURN doc"
    bind_vars1 = {"@collection": collection, "category": "A"}
    results1 = query_documents(query1, db, bind_vars1)
    
    # Assertions for query 1
    assert len(results1) == 2
    assert all(doc["category"] == "A" for doc in results1)
    
    # Query 2: Get documents with value > 25
    query2 = "FOR doc IN @@collection FILTER doc.value > @min_value RETURN doc"
    bind_vars2 = {"@collection": collection, "min_value": 25}
    results2 = query_documents(query2, db, bind_vars2)
    
    # Assertions for query 2
    assert len(results2) == 3
    assert all(doc["value"] > 25 for doc in results2)
    
    # Query 3: Get documents sorted by value
    query3 = "FOR doc IN @@collection SORT doc.value DESC LIMIT 3 RETURN doc"
    bind_vars3 = {"@collection": collection}
    results3 = query_documents(query3, db, bind_vars3)
    
    # Assertions for query 3
    assert len(results3) == 3
    assert results3[0]["value"] > results3[1]["value"] > results3[2]["value"]
    
    # Query 4: With profile information
    query4 = "FOR doc IN @@collection RETURN doc"
    bind_vars4 = {"@collection": collection}
    results4, stats = query_documents(query4, db, bind_vars4, profile=True)
    
    # Assertions for query 4
    assert len(results4) == 5
    assert isinstance(stats, dict)


# === Message Operations Tests ===

def test_create_message(db, clean_test_collections):
    """Test creating a message."""
    message_collection = clean_test_collections["messages"]
    
    # Create message
    conversation_id = "test_conversation_1"
    role = "user"
    content = "Hello, this is a test message"
    metadata = {"source": "test", "importance": "high"}
    
    result = create_message(
        db=db,
        conversation_id=conversation_id,
        role=role,
        content=content,
        metadata=metadata,
        message_collection=message_collection
    )
    
    # Assertions
    assert result is not None
    assert isinstance(result, dict)
    assert "_id" in result
    assert "_key" in result
    assert "conversation_id" in result and result["conversation_id"] == conversation_id
    assert "role" in result and result["role"] == role
    assert "content" in result and result["content"] == content
    assert "metadata" in result and result["metadata"] == metadata
    assert "created_at" in result
    assert "updated_at" in result
    assert "uuid" in result


def test_get_message(db, clean_test_collections):
    """Test retrieving a message."""
    message_collection = clean_test_collections["messages"]
    
    # Create test message
    conversation_id = "test_conversation_1"
    message = create_message(
        db=db,
        conversation_id=conversation_id,
        role="assistant",
        content="Message to retrieve",
        message_collection=message_collection
    )
    message_key = message["_key"]
    
    # Retrieve message
    retrieved = get_message(
        db=db,
        message_key=message_key,
        message_collection=message_collection
    )
    
    # Assertions
    assert retrieved is not None
    assert retrieved["_key"] == message_key
    assert retrieved["role"] == "assistant"
    assert retrieved["content"] == "Message to retrieve"
    assert retrieved["conversation_id"] == conversation_id


def test_update_message(db, clean_test_collections):
    """Test updating a message."""
    message_collection = clean_test_collections["messages"]
    
    # Create test message
    message = create_message(
        db=db,
        conversation_id="test_conversation_1",
        role="user",
        content="Original message",
        message_collection=message_collection
    )
    message_key = message["_key"]
    
    # Update message
    update_data = {
        "content": "Updated message",
        "metadata": {"updated": True}
    }
    
    updated = update_message(
        db=db,
        message_key=message_key,
        data=update_data,
        message_collection=message_collection
    )
    
    # Assertions
    assert updated is not None
    assert updated["_key"] == message_key
    assert updated["content"] == "Updated message"
    assert updated["role"] == "user"  # Unchanged
    assert "metadata" in updated and updated["metadata"]["updated"] is True
    assert updated["created_at"] == message["created_at"]  # Unchanged
    assert updated["updated_at"] != message["updated_at"]  # Changed


def test_delete_message(db, clean_test_collections):
    """Test deleting a message."""
    message_collection = clean_test_collections["messages"]
    
    # Create test message
    message = create_message(
        db=db,
        conversation_id="test_conversation_1",
        role="user",
        content="Message to delete",
        message_collection=message_collection
    )
    message_key = message["_key"]
    
    # Delete message
    delete_result = delete_message(
        db=db,
        message_key=message_key,
        message_collection=message_collection
    )
    
    # Assertions
    assert delete_result is True
    
    # Verify message is deleted
    with pytest.raises(DocumentGetError):
        get_message(
            db=db,
            message_key=message_key,
            message_collection=message_collection
        )


def test_get_conversation_messages(db, clean_test_collections):
    """Test retrieving messages in a conversation."""
    message_collection = clean_test_collections["messages"]
    conversation_id = "test_conversation_2"
    
    # Create test messages
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi, how can I help you?"},
        {"role": "user", "content": "I have a question"},
        {"role": "assistant", "content": "Sure, what's your question?"}
    ]
    
    for msg in messages:
        create_message(
            db=db,
            conversation_id=conversation_id,
            role=msg["role"],
            content=msg["content"],
            message_collection=message_collection
        )
    
    # Test with default sorting (ASC by created_at)
    results1 = get_conversation_messages(
        db=db,
        conversation_id=conversation_id,
        message_collection=message_collection
    )
    
    # Assertions for default sorting
    assert len(results1) == 4
    
    # Test with custom sorting (DESC by created_at)
    results2 = get_conversation_messages(
        db=db,
        conversation_id=conversation_id,
        sort_by="created_at",
        sort_direction="DESC",
        message_collection=message_collection
    )
    
    # Assertions for custom sorting
    assert len(results2) == 4
    assert results2[0]["created_at"] >= results2[1]["created_at"]
    
    # Test with limit and offset
    results3 = get_conversation_messages(
        db=db,
        conversation_id=conversation_id,
        limit=2,
        offset=1,
        message_collection=message_collection
    )
    
    # Assertions for limit and offset
    assert len(results3) == 2


def test_delete_conversation(db, clean_test_collections):
    """Test deleting all messages in a conversation."""
    message_collection = clean_test_collections["messages"]
    conversation_id = "test_conversation_to_delete"
    
    # Create test messages
    for i in range(5):
        create_message(
            db=db,
            conversation_id=conversation_id,
            role="user" if i % 2 == 0 else "assistant",
            content=f"Message {i+1}",
            message_collection=message_collection
        )
    
    # Verify messages exist
    assert len(get_conversation_messages(
        db=db,
        conversation_id=conversation_id,
        message_collection=message_collection
    )) == 5
    
    # Delete conversation
    deleted_count = delete_conversation(
        db=db,
        conversation_id=conversation_id,
        message_collection=message_collection
    )
    
    # Assertions
    assert deleted_count == 5
    
    # Verify all messages are deleted
    assert len(get_conversation_messages(
        db=db,
        conversation_id=conversation_id,
        message_collection=message_collection
    )) == 0


# === Relationship Operations Tests ===

def test_create_relationship(db, clean_test_collections):
    """Test creating a relationship between documents."""
    docs_collection = clean_test_collections["docs"]
    edge_collection = clean_test_collections["edges"]
    
    # Create test documents
    doc1 = create_document(docs_collection, {"name": "Source Doc"}, db)
    doc2 = create_document(docs_collection, {"name": "Target Doc"}, db)
    
    doc1_id = doc1["_id"]
    doc2_id = doc2["_id"]
    
    # Create relationship
    properties = {"type": "references", "weight": 0.85}
    
    relationship = create_relationship(
        db=db,
        from_id=doc1_id,
        to_id=doc2_id,
        edge_collection=edge_collection,
        properties=properties
    )
    
    # Assertions
    assert relationship is not None
    assert relationship["_from"] == doc1_id
    assert relationship["_to"] == doc2_id
    assert "type" in relationship and relationship["type"] == "references"
    assert "weight" in relationship and relationship["weight"] == 0.85


def test_delete_relationship(db, clean_test_collections):
    """Test deleting a relationship."""
    docs_collection = clean_test_collections["docs"]
    edge_collection = clean_test_collections["edges"]
    
    # Create test documents
    doc1 = create_document(docs_collection, {"name": "Doc A"}, db)
    doc2 = create_document(docs_collection, {"name": "Doc B"}, db)
    
    # Create relationship
    relationship = create_relationship(
        db=db,
        from_id=doc1["_id"],
        to_id=doc2["_id"],
        edge_collection=edge_collection
    )
    
    relationship_key = relationship["_key"]
    
    # Delete relationship
    delete_result = delete_relationship_by_key(
        db=db,
        edge_key=relationship_key,
        edge_collection=edge_collection
    )
    
    # Assertions
    assert delete_result is True
    
    # Verify relationship is deleted
    with pytest.raises(DocumentGetError):
        db.collection(edge_collection).get(relationship_key)


def test_delete_relationships_between(db, clean_test_collections):
    """Test deleting all relationships between two documents."""
    docs_collection = clean_test_collections["docs"]
    edge_collection = clean_test_collections["edges"]
    
    # Create test documents
    doc1 = create_document(docs_collection, {"name": "Node 1"}, db)
    doc2 = create_document(docs_collection, {"name": "Node 2"}, db)
    
    doc1_id = doc1["_id"]
    doc2_id = doc2["_id"]
    
    # Create multiple relationships
    for i in range(3):
        create_relationship(
            db=db,
            from_id=doc1_id,
            to_id=doc2_id,
            edge_collection=edge_collection,
            properties={"index": i}
        )
    
    # Delete relationships
    deleted_count = delete_relationships_between(
        db=db,
        from_id=doc1_id,
        to_id=doc2_id,
        edge_collection=edge_collection
    )
    
    # Assertions
    assert deleted_count == 3
    
    # Verify relationships are deleted
    query = """
    FOR edge IN @@edge_collection
        FILTER edge._from == @from_id AND edge._to == @to_id
        RETURN edge
    """
    bind_vars = {
        "@edge_collection": edge_collection,
        "from_id": doc1_id,
        "to_id": doc2_id
    }
    
    remaining = db.aql.execute(query, bind_vars=bind_vars)
    assert len(list(remaining)) == 0


def test_link_message_to_document(db, clean_test_collections):
    """Test linking a message to a document."""
    docs_collection = clean_test_collections["docs"]
    message_collection = clean_test_collections["messages"]
    edge_collection = clean_test_collections["edges"]
    
    # Create test message and document
    message = create_message(
        db=db,
        conversation_id="test_conversation",
        role="user",
        content="Message with attachment",
        message_collection=message_collection
    )
    
    document = create_document(
        docs_collection,
        {"name": "Referenced Document", "type": "attachment"},
        db
    )
    
    message_id = message["_id"]
    document_id = document["_id"]
    
    # Link message to document
    properties = {"relevance": 0.95}
    
    link = link_message_to_document(
        db=db,
        message_id=message_id,
        document_id=document_id,
        properties=properties,
        edge_collection=edge_collection
    )
    
    # Assertions
    assert link is not None
    assert link["_from"] == message_id
    assert link["_to"] == document_id
    assert "relevance" in link and link["relevance"] == 0.95


def test_get_documents_for_message(db, clean_test_collections):
    """Test retrieving documents related to a message."""
    docs_collection = clean_test_collections["docs"]
    message_collection = clean_test_collections["messages"]
    edge_collection = clean_test_collections["edges"]
    
    # Create test message
    message = create_message(
        db=db,
        conversation_id="test_conversation",
        role="user",
        content="Message with multiple references",
        message_collection=message_collection
    )
    
    message_id = message["_id"]
    
    # Create test documents and link them to the message
    num_docs = 3
    for i in range(num_docs):
        doc = create_document(
            docs_collection,
            {"name": f"Referenced Doc {i+1}", "category": "A" if i < 2 else "B"},
            db
        )
        
        link_message_to_document(
            db=db,
            message_id=message_id,
            document_id=doc["_id"],
            edge_collection=edge_collection
        )
    
    # Get related documents (all)
    results1 = get_documents_for_message(
        db=db,
        message_id=message_id,
        edge_collection=edge_collection
    )
    
    # Assertions for all documents
    assert len(results1) == num_docs
    
    # Get related documents with filter
    results2 = get_documents_for_message(
        db=db,
        message_id=message_id,
        collection_filter=docs_collection,
        edge_collection=edge_collection
    )
    
    # Assertions for filtered documents
    assert len(results2) == num_docs
    for result in results2:
        assert "document" in result
        assert "edge" in result


def test_get_messages_for_document(db, clean_test_collections):
    """Test retrieving messages related to a document."""
    docs_collection = clean_test_collections["docs"]
    message_collection = clean_test_collections["messages"]
    edge_collection = clean_test_collections["edges"]
    
    # Create test document
    document = create_document(
        docs_collection,
        {"name": "Referenced Document", "type": "shared"},
        db
    )
    
    document_id = document["_id"]
    
    # Create test messages and link them to the document
    num_messages = 3
    for i in range(num_messages):
        msg = create_message(
            db=db,
            conversation_id=f"conv_{i}",
            role="user" if i % 2 == 0 else "assistant",
            content=f"Message {i+1} referencing document",
            message_collection=message_collection
        )
        
        link_message_to_document(
            db=db,
            message_id=msg["_id"],
            document_id=document_id,
            edge_collection=edge_collection
        )
    
    # Get related messages
    results = get_messages_for_document(
        db=db,
        document_id=document_id,
        message_collection=message_collection,
        edge_collection=edge_collection
    )
    
    # Assertions
    assert len(results) == num_messages
    for result in results:
        assert "message" in result
        assert "edge" in result
        assert result["message"]["content"].endswith("referencing document")


def test_get_related_documents(db, clean_test_collections):
    """Test retrieving documents related to a document through graph traversal."""
    docs_collection = clean_test_collections["docs"]
    edge_collection = clean_test_collections["edges"]
    
    # Create a chain of documents with relationships
    doc1 = create_document(docs_collection, {"name": "Root Document"}, db)
    doc2 = create_document(docs_collection, {"name": "Level 1 Document"}, db)
    doc3 = create_document(docs_collection, {"name": "Level 2 Document"}, db)
    
    # Create relationships (doc1 -> doc2 -> doc3)
    create_relationship(
        db=db,
        from_id=doc1["_id"],
        to_id=doc2["_id"],
        edge_collection=edge_collection,
        properties={"level": 1}
    )
    
    create_relationship(
        db=db,
        from_id=doc2["_id"],
        to_id=doc3["_id"],
        edge_collection=edge_collection,
        properties={"level": 2}
    )
    
    # Test with depth 1 (should find only doc2)
    results1 = get_related_documents(
        db=db,
        document_id=doc1["_id"],
        edge_collection=edge_collection,
        direction="outbound",
        max_depth=1
    )
    
    # Assertions for depth 1
    assert len(results1) == 1
    assert results1[0]["document"]["name"] == "Level 1 Document"
    
    # Test with depth 2 (should find doc2 and doc3)
    results2 = get_related_documents(
        db=db,
        document_id=doc1["_id"],
        edge_collection=edge_collection,
        direction="outbound",
        max_depth=2
    )
    
    # Assertions for depth 2
    assert len(results2) == 2
    assert any(r["document"]["name"] == "Level 1 Document" for r in results2)
    assert any(r["document"]["name"] == "Level 2 Document" for r in results2)
    
    # Test with different direction
    create_relationship(
        db=db,
        from_id=doc3["_id"],
        to_id=doc1["_id"],
        edge_collection=edge_collection,
        properties={"type": "circular"}
    )
    
    results3 = get_related_documents(
        db=db,
        document_id=doc2["_id"],
        edge_collection=edge_collection,
        direction="any",
        max_depth=1
    )
    
    # Assertions for any direction
    assert len(results3) == 2  # Should find both doc1 and doc3


if __name__ == "__main__":
    """Main function to run tests directly."""
    # Set up pytest arguments
    args = [
        "-xvs",  # Verbose output, stop on first failure
        __file__,  # This file
    ]
    
    # Run pytest
    sys.exit(pytest.main(args))