"""
ArangoDB Message Operations.

This module provides operations for managing conversation messages in ArangoDB, including:
- Creating, retrieving, updating, and deleting messages
- Managing conversation history
- Linking messages to related documents

These functions build on the core CRUD operations to provide domain-specific
message handling capabilities.

Links:
- ArangoDB Python Driver: https://docs.python-arango.com/
- ArangoDB Graph Features: https://www.arangodb.com/docs/stable/graphs.html

Sample input:
    create_message(db, 'conversation123', 'user', 'Hello, how can I help you?')

Expected output:
    {'_id': 'messages/12345', '_key': '12345', '_rev': '12345', 
     'conversation_id': 'conversation123', 'role': 'user',
     'content': 'Hello, how can I help you?', 'created_at': '2023-05-11T10:30:00.000000', 
     'updated_at': '2023-05-11T10:30:00.000000', 'uuid': '123e4567-e89b-12d3-a456-426614174000'}
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import datetime

from arango.database import Database
from arango.exceptions import (
    DocumentGetError,
    DocumentUpdateError,
    DocumentDeleteError,
    AQLQueryExecuteError
)

from mcp_tools.arangodb.core.db.crud import (
    create_document,
    get_document,
    update_document,
    delete_document,
    query_documents
)


def create_message(
    db: Database,
    conversation_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    message_collection: str = "messages",
) -> Dict[str, Any]:
    """
    Create a message in the specified conversation.

    Args:
        db: ArangoDB database connection.
        conversation_id: The conversation identifier.
        role: The role of the message sender (e.g., 'user', 'assistant').
        content: The message content.
        metadata: Optional metadata for the message.
        message_collection: The name of the messages collection.

    Returns:
        dict: The created message document.
    """
    message_data = {
        "conversation_id": conversation_id,
        "role": role,
        "content": content,
    }

    # Add metadata if provided
    if metadata:
        message_data["metadata"] = metadata

    return create_document(message_collection, message_data, db)


def get_message(
    db: Database,
    message_key: str,
    message_collection: str = "messages",
) -> Dict[str, Any]:
    """
    Retrieve a message by its key.

    Args:
        db: ArangoDB database connection.
        message_key: The message key.
        message_collection: The name of the messages collection.

    Returns:
        dict: The retrieved message document.

    Raises:
        DocumentGetError: If message retrieval fails.
    """
    return get_document(message_collection, message_key, db)


def update_message(
    db: Database,
    message_key: str,
    data: Dict[str, Any],
    message_collection: str = "messages",
) -> Dict[str, Any]:
    """
    Update a message document.

    Args:
        db: ArangoDB database connection.
        message_key: The message key.
        data: The data to update.
        message_collection: The name of the messages collection.

    Returns:
        dict: The updated message document.

    Raises:
        DocumentUpdateError: If message update fails.
    """
    return update_document(message_collection, message_key, data, db)


def delete_message(
    db: Database,
    message_key: str,
    delete_relationships: bool = True,
    message_collection: str = "messages",
    relationship_collection: str = "relates_to",
) -> bool:
    """
    Delete a message and optionally its relationships.

    Args:
        db: ArangoDB database connection.
        message_key: The message key.
        delete_relationships: Whether to delete relationships.
        message_collection: The name of the messages collection.
        relationship_collection: The name of the relationship collection.

    Returns:
        bool: True if the message was deleted.

    Raises:
        DocumentDeleteError: If message deletion fails.
    """
    # Delete relationships if requested
    if delete_relationships:
        message_id = f"{message_collection}/{message_key}"
        query = """
        FOR edge IN @@relationship_collection
            FILTER edge._from == @message_id OR edge._to == @message_id
            REMOVE edge IN @@relationship_collection
        """
        bind_vars = {
            "@relationship_collection": relationship_collection,
            "message_id": message_id,
        }
        query_documents(query, db, bind_vars)

    # Delete the message
    return delete_document(message_collection, message_key, db)


def get_conversation_messages(
    db: Database,
    conversation_id: str,
    sort_by: str = "created_at",
    sort_direction: str = "ASC",
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    message_collection: str = "messages",
) -> List[Dict[str, Any]]:
    """
    Retrieve all messages in a conversation.

    Args:
        db: ArangoDB database connection.
        conversation_id: The conversation identifier.
        sort_by: The field to sort by.
        sort_direction: The sort direction ('ASC' or 'DESC').
        limit: Maximum number of messages to return.
        offset: Number of messages to skip.
        message_collection: The name of the messages collection.

    Returns:
        list: A list of message documents.
    """
    # Validate sort_direction value
    if sort_direction not in ["ASC", "DESC"]:
        sort_direction = "ASC"

    # Construct limit clause if limit is provided
    limit_clause = ""
    if limit is not None:
        limit_clause = f"LIMIT {offset if offset is not None else 0}, {limit}"

    # Query for messages in the conversation
    query = f"""
    FOR message IN @@message_collection
        FILTER message.conversation_id == @conversation_id
        SORT message.{sort_by} {sort_direction}
        {limit_clause}
        RETURN message
    """

    bind_vars = {
        "@message_collection": message_collection,
        "conversation_id": conversation_id,
    }

    return query_documents(query, db, bind_vars)


def delete_conversation(
    db: Database,
    conversation_id: str,
    delete_relationships: bool = True,
    message_collection: str = "messages",
    relationship_collection: str = "relates_to",
) -> int:
    """
    Delete all messages in a conversation.

    Args:
        db: ArangoDB database connection.
        conversation_id: The conversation identifier.
        delete_relationships: Whether to delete relationships.
        message_collection: The name of the messages collection.
        relationship_collection: The name of the relationship collection.

    Returns:
        int: The number of deleted messages.
    """
    # If we need to delete relationships, first get all message IDs
    if delete_relationships:
        # Get all message IDs in the conversation
        message_query = """
        FOR message IN @@message_collection
            FILTER message.conversation_id == @conversation_id
            RETURN message._id
        """
        message_bind_vars = {
            "@message_collection": message_collection,
            "conversation_id": conversation_id,
        }
        
        message_ids = query_documents(message_query, db, message_bind_vars)
        
        if message_ids:
            # Delete all relationships involving these messages
            relationship_query = """
            FOR edge IN @@relationship_collection
                FILTER edge._from IN @message_ids OR edge._to IN @message_ids
                REMOVE edge IN @@relationship_collection
            """
            relationship_bind_vars = {
                "@relationship_collection": relationship_collection,
                "message_ids": message_ids,
            }
            query_documents(relationship_query, db, relationship_bind_vars)

    # Delete all messages in the conversation
    delete_query = """
    FOR message IN @@message_collection
        FILTER message.conversation_id == @conversation_id
        REMOVE message IN @@message_collection
        COLLECT WITH COUNT INTO count
        RETURN count
    """
    delete_bind_vars = {
        "@message_collection": message_collection,
        "conversation_id": conversation_id,
    }
    
    result = query_documents(delete_query, db, delete_bind_vars)
    return result[0] if result else 0


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
            
        if not db.has_collection("test_relates_to"):
            db.create_edge_collection("test_relates_to")
            
        message_collection = "test_messages"
        relationship_collection = "test_relates_to"
        test_conversation_id = "test_conversation_123"
        
        # Test 1: Message creation
        total_tests += 1
        message_data = create_message(
            db=db,
            conversation_id=test_conversation_id,
            role="user",
            content="Hello, this is a test message",
            message_collection=message_collection
        )
        
        if not message_data or not isinstance(message_data, dict):
            all_validation_failures.append("Message creation failed: No result returned")
        elif message_data.get("role") != "user":
            all_validation_failures.append(f"Message creation failed: Expected role 'user', got {message_data.get('role')}")
        elif message_data.get("conversation_id") != test_conversation_id:
            all_validation_failures.append(f"Message creation failed: Incorrect conversation ID")
        
        # Remember message key for further tests
        message_key = message_data.get("_key", "")
        
        # Test 2: Message retrieval
        total_tests += 1
        if message_key:
            retrieved_message = get_message(db, message_key, message_collection)
            if not retrieved_message or retrieved_message.get("content") != "Hello, this is a test message":
                all_validation_failures.append(f"Message retrieval failed: Content mismatch")
        else:
            all_validation_failures.append("Message retrieval skipped: No valid message key")
        
        # Test 3: Message update
        total_tests += 1
        if message_key:
            update_data = {"content": "Updated test message", "metadata": {"updated": True}}
            updated_message = update_message(db, message_key, update_data, message_collection)
            
            if not updated_message or updated_message.get("content") != "Updated test message":
                all_validation_failures.append(f"Message update failed: Content not updated")
            if not updated_message.get("metadata") or not updated_message["metadata"].get("updated"):
                all_validation_failures.append(f"Message update failed: Metadata not updated")
        else:
            all_validation_failures.append("Message update skipped: No valid message key")
        
        # Test 4: Create additional messages for conversation tests
        total_tests += 1
        additional_messages = [
            {"conversation_id": test_conversation_id, "role": "assistant", "content": "Response 1"},
            {"conversation_id": test_conversation_id, "role": "user", "content": "Follow-up question"},
            {"conversation_id": test_conversation_id, "role": "assistant", "content": "Response 2"}
        ]
        
        for msg_data in additional_messages:
            create_message(
                db=db,
                conversation_id=msg_data["conversation_id"],
                role=msg_data["role"],
                content=msg_data["content"],
                message_collection=message_collection
            )
        
        # Test 5: Get conversation messages
        total_tests += 1
        conversation_messages = get_conversation_messages(
            db=db,
            conversation_id=test_conversation_id,
            message_collection=message_collection
        )
        
        if not conversation_messages or not isinstance(conversation_messages, list):
            all_validation_failures.append("Conversation retrieval failed: No results returned")
        elif len(conversation_messages) != 4:  # Original + 3 additional messages
            all_validation_failures.append(f"Conversation retrieval failed: Expected 4 messages, got {len(conversation_messages)}")
            
        # Test 6: Delete a single message
        total_tests += 1
        if message_key:
            delete_result = delete_message(
                db=db,
                message_key=message_key,
                message_collection=message_collection,
                relationship_collection=relationship_collection
            )
            
            if not delete_result:
                all_validation_failures.append("Message deletion failed: Operation unsuccessful")
                
            # Verify message count after deletion
            remaining_messages = get_conversation_messages(
                db=db,
                conversation_id=test_conversation_id,
                message_collection=message_collection
            )
            
            if len(remaining_messages) != 3:  # Should have 3 messages left
                all_validation_failures.append(f"Message deletion verification failed: Expected 3 messages, got {len(remaining_messages)}")
        else:
            all_validation_failures.append("Message deletion skipped: No valid message key")
            
        # Test 7: Delete entire conversation
        total_tests += 1
        delete_count = delete_conversation(
            db=db,
            conversation_id=test_conversation_id,
            message_collection=message_collection,
            relationship_collection=relationship_collection
        )
        
        if delete_count != 3:  # Should have deleted 3 messages
            all_validation_failures.append(f"Conversation deletion failed: Expected 3 deleted messages, got {delete_count}")
            
        # Verify conversation is empty
        final_messages = get_conversation_messages(
            db=db,
            conversation_id=test_conversation_id,
            message_collection=message_collection
        )
        
        if final_messages and len(final_messages) > 0:
            all_validation_failures.append(f"Conversation deletion verification failed: Expected 0 messages, got {len(final_messages)}")
        
        # Cleanup test collections
        db.collection(message_collection).truncate()
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
        print("Message operations are validated and formal tests can now be written")
        sys.exit(0)  # Exit with success code