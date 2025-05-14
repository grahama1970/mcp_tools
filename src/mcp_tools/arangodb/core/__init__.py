"""Core functionality for ArangoDB integration.

This module provides pure business logic for database operations, search
functionality, and relationship management.
"""

# Database operations
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

# Export all imported functions
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

# Version information
__version__ = "0.1.0"