"""
ArangoDB Core Database Operations.

This package provides core database operations for ArangoDB, structured in layers:
- crud: CRUD operations for document management
- messages: Message history operations for conversations
- relationships: Graph relationship operations

These modules provide pure business logic with no presentation concerns.
"""

from mcp_tools.arangodb.core.db.crud import (
    create_document,
    get_document,
    update_document,
    delete_document,
    query_documents,
)

from mcp_tools.arangodb.core.db.messages import (
    create_message,
    get_message,
    update_message,
    delete_message,
    get_conversation_messages,
    delete_conversation,
)

from mcp_tools.arangodb.core.db.relationships import (
    create_relationship,
    delete_relationship_by_key,
    delete_relationships_between,
    link_message_to_document,
    get_documents_for_message,
    get_messages_for_document,
    get_related_documents,
)

__all__ = [
    # CRUD operations
    "create_document",
    "get_document", 
    "update_document",
    "delete_document",
    "query_documents",
    
    # Message operations
    "create_message",
    "get_message",
    "update_message",
    "delete_message",
    "get_conversation_messages",
    "delete_conversation",
    
    # Relationship operations
    "create_relationship",
    "delete_relationship_by_key",
    "delete_relationships_between",
    "link_message_to_document",
    "get_documents_for_message",
    "get_messages_for_document",
    "get_related_documents",
]