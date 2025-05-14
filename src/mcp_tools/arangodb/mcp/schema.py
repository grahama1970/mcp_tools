"""JSON Schema definitions for ArangoDB MCP operations.

This module defines the JSON schemas for various ArangoDB operations
exposed through the MCP interface.
"""

import json
from typing import Dict

# Define base schemas that can be reused
BASE_SCHEMAS = {
    "output_format": {
        "type": "string",
        "enum": ["json", "table"],
        "default": "json",
        "description": "Output format (json or table)"
    },
    "database_config": {
        "type": "object",
        "properties": {
            "host": {"type": "string", "description": "ArangoDB host URL"},
            "username": {"type": "string", "description": "ArangoDB username"},
            "password": {"type": "string", "description": "ArangoDB password"},
            "database": {"type": "string", "description": "ArangoDB database name"}
        },
        "required": ["host", "username", "password", "database"],
        "description": "ArangoDB connection configuration"
    }
}

# ====================================
# Database Operations Schemas
# ====================================

# Create Document Schema
CREATE_DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "collection": {"type": "string", "description": "Collection name to create document in"},
        "data": {"type": "object", "description": "Document data to store"},
        "return_new": {"type": "boolean", "description": "Whether to return the new document"},
        "overwrite": {"type": "boolean", "description": "Whether to overwrite an existing document"},
        "overwrite_mode": {"type": "string", "enum": ["replace", "update", "ignore"], "description": "Overwrite mode"},
        "generate_uuid": {"type": "boolean", "description": "Whether to generate a UUID for the document"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["collection", "data", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Create a document in an ArangoDB collection"
}

# Get Document Schema
GET_DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "collection": {"type": "string", "description": "Collection name to get document from"},
        "document_key": {"type": "string", "description": "Document key"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["collection", "document_key", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Retrieve a document from an ArangoDB collection by its key"
}

# Update Document Schema
UPDATE_DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "collection": {"type": "string", "description": "Collection name to update document in"},
        "document_key": {"type": "string", "description": "Document key"},
        "data": {"type": "object", "description": "Data to update"},
        "return_new": {"type": "boolean", "description": "Whether to return the new document"},
        "return_old": {"type": "boolean", "description": "Whether to return the old document"},
        "keep_none": {"type": "boolean", "description": "Whether to keep None values"},
        "merge": {"type": "boolean", "description": "Whether to merge the update data with the existing document"},
        "check_rev": {"type": "string", "description": "Revision to check before updating"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["collection", "document_key", "data", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Update a document in an ArangoDB collection"
}

# Delete Document Schema
DELETE_DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "collection": {"type": "string", "description": "Collection name to delete document from"},
        "document_key": {"type": "string", "description": "Document key"},
        "return_old": {"type": "boolean", "description": "Whether to return the old document"},
        "check_rev": {"type": "string", "description": "Revision to check before deleting"},
        "ignore_missing": {"type": "boolean", "description": "Whether to ignore missing documents"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["collection", "document_key", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Delete a document from an ArangoDB collection"
}

# Query Documents Schema
QUERY_DOCUMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "AQL query string"},
        "bind_vars": {"type": "object", "description": "Variables to bind to the query"},
        "count": {"type": "boolean", "description": "Whether to count the results"},
        "batch_size": {"type": "integer", "description": "Batch size for retrieving results"},
        "full_count": {"type": "boolean", "description": "Whether to return the full count"},
        "max_runtime": {"type": "integer", "description": "Maximum runtime for the query in seconds"},
        "profile": {"type": "boolean", "description": "Whether to return profiling information"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["query", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Execute an AQL query against ArangoDB"
}

# ====================================
# Message Operations Schemas
# ====================================

# Create Message Schema
CREATE_MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "conversation_id": {"type": "string", "description": "Conversation identifier"},
        "role": {"type": "string", "description": "Message role (e.g., 'user', 'assistant')"},
        "content": {"type": "string", "description": "Message content"},
        "metadata": {"type": "object", "description": "Optional metadata for the message"},
        "message_collection": {"type": "string", "description": "Name of the messages collection"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["conversation_id", "role", "content", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Create a message in a conversation"
}

# Get Message Schema
GET_MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "message_key": {"type": "string", "description": "Message key"},
        "message_collection": {"type": "string", "description": "Name of the messages collection"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["message_key", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Retrieve a message by its key"
}

# Update Message Schema
UPDATE_MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "message_key": {"type": "string", "description": "Message key"},
        "data": {"type": "object", "description": "Data to update"},
        "message_collection": {"type": "string", "description": "Name of the messages collection"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["message_key", "data", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Update a message document"
}

# Delete Message Schema
DELETE_MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "message_key": {"type": "string", "description": "Message key"},
        "delete_relationships": {"type": "boolean", "description": "Whether to delete relationships"},
        "message_collection": {"type": "string", "description": "Name of the messages collection"},
        "relationship_collection": {"type": "string", "description": "Name of the relationship collection"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["message_key", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Delete a message and optionally its relationships"
}

# Get Conversation Messages Schema
GET_CONVERSATION_SCHEMA = {
    "type": "object",
    "properties": {
        "conversation_id": {"type": "string", "description": "Conversation identifier"},
        "sort_by": {"type": "string", "description": "Field to sort by"},
        "sort_direction": {"type": "string", "enum": ["ASC", "DESC"], "description": "Sort direction"},
        "limit": {"type": "integer", "minimum": 1, "description": "Maximum number of messages to return"},
        "offset": {"type": "integer", "minimum": 0, "description": "Number of messages to skip"},
        "message_collection": {"type": "string", "description": "Name of the messages collection"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["conversation_id", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Retrieve all messages in a conversation"
}

# Delete Conversation Schema
DELETE_CONVERSATION_SCHEMA = {
    "type": "object",
    "properties": {
        "conversation_id": {"type": "string", "description": "Conversation identifier"},
        "delete_relationships": {"type": "boolean", "description": "Whether to delete relationships"},
        "message_collection": {"type": "string", "description": "Name of the messages collection"},
        "relationship_collection": {"type": "string", "description": "Name of the relationship collection"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["conversation_id", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Delete all messages in a conversation"
}

# ====================================
# Relationship Operations Schemas
# ====================================

# Create Relationship Schema
CREATE_RELATIONSHIP_SCHEMA = {
    "type": "object",
    "properties": {
        "from_id": {"type": "string", "description": "Source document ID (_id format: collection/key)"},
        "to_id": {"type": "string", "description": "Target document ID (_id format: collection/key)"},
        "edge_collection": {"type": "string", "description": "Name of the edge collection"},
        "properties": {"type": "object", "description": "Optional properties for the relationship"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["from_id", "to_id", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Create a relationship (edge) between two documents"
}

# Delete Relationship Schema
DELETE_RELATIONSHIP_SCHEMA = {
    "type": "object",
    "properties": {
        "edge_key": {"type": "string", "description": "Edge key"},
        "edge_collection": {"type": "string", "description": "Name of the edge collection"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["edge_key", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Delete a relationship by its key"
}

# Delete Relationships Between Schema
DELETE_RELATIONSHIPS_BETWEEN_SCHEMA = {
    "type": "object",
    "properties": {
        "from_id": {"type": "string", "description": "Source document ID"},
        "to_id": {"type": "string", "description": "Target document ID"},
        "edge_collection": {"type": "string", "description": "Name of the edge collection"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["from_id", "to_id", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Delete all relationships between two documents"
}

# Link Message to Document Schema
LINK_MESSAGE_TO_DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "message_id": {"type": "string", "description": "Message ID (_id format: collection/key)"},
        "document_id": {"type": "string", "description": "Document ID (_id format: collection/key)"},
        "properties": {"type": "object", "description": "Optional properties for the relationship"},
        "edge_collection": {"type": "string", "description": "Name of the edge collection"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["message_id", "document_id", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Create a relationship between a message and a document"
}

# Get Documents for Message Schema
GET_DOCUMENTS_FOR_MESSAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "message_id": {"type": "string", "description": "Message ID (_id format: collection/key)"},
        "collection_filter": {"type": "string", "description": "Optional filter for document collections"},
        "edge_collection": {"type": "string", "description": "Name of the edge collection"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["message_id", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Retrieve all documents related to a message"
}

# Get Messages for Document Schema
GET_MESSAGES_FOR_DOCUMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "document_id": {"type": "string", "description": "Document ID (_id format: collection/key)"},
        "message_collection": {"type": "string", "description": "Name of the messages collection"},
        "edge_collection": {"type": "string", "description": "Name of the edge collection"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["document_id", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Retrieve all messages related to a document"
}

# Get Related Documents Schema
GET_RELATED_DOCUMENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "document_id": {"type": "string", "description": "Document ID (_id format: collection/key)"},
        "collection_filter": {"type": "string", "description": "Optional filter for document collections"},
        "edge_collection": {"type": "string", "description": "Name of the edge collection"},
        "direction": {"type": "string", "enum": ["outbound", "inbound", "any"], "description": "Direction to traverse"},
        "max_depth": {"type": "integer", "minimum": 1, "maximum": 10, "description": "Maximum traversal depth"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["document_id", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Retrieve documents related to a document through graph traversal"
}

# ====================================
# Search Operations Schemas
# ====================================

# BM25 Search Schema
BM25_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query text"},
        "collection": {"type": "string", "description": "Collection to search in"},
        "fields": {"type": "array", "items": {"type": "string"}, "description": "Fields to search within"},
        "filter": {"type": "object", "description": "Additional filter conditions"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Maximum number of results to return"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["query", "collection", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Perform a BM25 text search across documents in ArangoDB"
}

# Semantic Search Schema
SEMANTIC_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query text"},
        "collection": {"type": "string", "description": "Collection to search in"},
        "vector_field": {"type": "string", "description": "Field containing vectors"},
        "filter": {"type": "object", "description": "Additional filter conditions"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Maximum number of results to return"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["query", "collection", "vector_field", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Perform a semantic vector search across documents in ArangoDB"
}

# Hybrid Search Schema
HYBRID_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query text"},
        "collection": {"type": "string", "description": "Collection to search in"},
        "fields": {"type": "array", "items": {"type": "string"}, "description": "Fields to search within for BM25"},
        "vector_field": {"type": "string", "description": "Field containing vectors for semantic search"},
        "filter": {"type": "object", "description": "Additional filter conditions"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Maximum number of results to return"},
        "weight_bm25": {"type": "number", "minimum": 0, "maximum": 1, "description": "Weight for BM25 results"},
        "weight_vector": {"type": "number", "minimum": 0, "maximum": 1, "description": "Weight for vector results"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["query", "collection", "fields", "vector_field", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Perform a hybrid search combining BM25 and semantic search in ArangoDB"
}

# Tag Search Schema
TAG_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags to search for"},
        "collection": {"type": "string", "description": "Collection to search in"},
        "tag_field": {"type": "string", "description": "Field containing tags"},
        "operator": {"type": "string", "enum": ["AND", "OR"], "description": "Logical operator for tag combination"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Maximum number of results to return"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["tags", "collection", "tag_field", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Search for documents with specific tags in ArangoDB"
}

# Keyword Search Schema
KEYWORD_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query text"},
        "collection": {"type": "string", "description": "Collection to search in"},
        "fields": {"type": "array", "items": {"type": "string"}, "description": "Fields to search within"},
        "threshold": {"type": "number", "minimum": 0, "maximum": 100, "description": "Fuzzy matching threshold (0-100)"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 1000, "description": "Maximum number of results to return"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["query", "collection", "fields", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Perform a fuzzy keyword search across documents in ArangoDB"
}

# Glossary Search Schema
GLOSSARY_SEARCH_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Text to search for glossary terms in"},
        "collection": {"type": "string", "description": "Glossary collection name"},
        "term_field": {"type": "string", "description": "Field containing glossary terms"},
        "definition_field": {"type": "string", "description": "Field containing term definitions"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["query", "collection", "term_field", "definition_field", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Find and highlight glossary terms in a text"
}

# Glossary Terms Listing Schema
GLOSSARY_TERMS_SCHEMA = {
    "type": "object",
    "properties": {
        "collection": {"type": "string", "description": "Glossary collection name"},
        "term_field": {"type": "string", "description": "Field containing glossary terms"},
        "definition_field": {"type": "string", "description": "Field containing term definitions"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["collection", "term_field", "definition_field", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "List all glossary terms and definitions"
}

# Glossary Term Addition Schema
GLOSSARY_ADD_SCHEMA = {
    "type": "object",
    "properties": {
        "term": {"type": "string", "description": "Glossary term to add"},
        "definition": {"type": "string", "description": "Term definition"},
        "collection": {"type": "string", "description": "Glossary collection name"},
        "term_field": {"type": "string", "description": "Field containing glossary terms"},
        "definition_field": {"type": "string", "description": "Field containing term definitions"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["term", "definition", "collection", "term_field", "definition_field", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Add a term to the glossary"
}

# Highlight Text Schema
HIGHLIGHT_TEXT_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "Text to highlight terms in"},
        "collection": {"type": "string", "description": "Glossary collection name"},
        "term_field": {"type": "string", "description": "Field containing glossary terms"},
        "definition_field": {"type": "string", "description": "Field containing term definitions"},
        "output_format": {"$ref": "#/definitions/output_format"},
        "db_config": {"$ref": "#/definitions/database_config"}
    },
    "required": ["text", "collection", "term_field", "definition_field", "db_config"],
    "definitions": BASE_SCHEMAS,
    "description": "Highlight glossary terms in a text with their definitions"
}

# Collect all schemas
SCHEMAS = {
    # Database Operations
    "create_document": CREATE_DOCUMENT_SCHEMA,
    "get_document": GET_DOCUMENT_SCHEMA,
    "update_document": UPDATE_DOCUMENT_SCHEMA,
    "delete_document": DELETE_DOCUMENT_SCHEMA,
    "query_documents": QUERY_DOCUMENTS_SCHEMA,
    
    # Message Operations
    "create_message": CREATE_MESSAGE_SCHEMA,
    "get_message": GET_MESSAGE_SCHEMA,
    "update_message": UPDATE_MESSAGE_SCHEMA,
    "delete_message": DELETE_MESSAGE_SCHEMA,
    "get_conversation_messages": GET_CONVERSATION_SCHEMA,
    "delete_conversation": DELETE_CONVERSATION_SCHEMA,
    
    # Relationship Operations
    "create_relationship": CREATE_RELATIONSHIP_SCHEMA,
    "delete_relationship": DELETE_RELATIONSHIP_SCHEMA,
    "delete_relationships_between": DELETE_RELATIONSHIPS_BETWEEN_SCHEMA,
    "link_message_to_document": LINK_MESSAGE_TO_DOCUMENT_SCHEMA,
    "get_documents_for_message": GET_DOCUMENTS_FOR_MESSAGE_SCHEMA,
    "get_messages_for_document": GET_MESSAGES_FOR_DOCUMENT_SCHEMA,
    "get_related_documents": GET_RELATED_DOCUMENTS_SCHEMA,
    
    # Search Operations
    "bm25_search": BM25_SEARCH_SCHEMA,
    "semantic_search": SEMANTIC_SEARCH_SCHEMA,
    "hybrid_search": HYBRID_SEARCH_SCHEMA,
    "tag_search": TAG_SEARCH_SCHEMA,
    "keyword_search": KEYWORD_SEARCH_SCHEMA,
    "glossary_search": GLOSSARY_SEARCH_SCHEMA,
    "get_glossary_terms": GLOSSARY_TERMS_SCHEMA,
    "add_glossary_term": GLOSSARY_ADD_SCHEMA,
    "highlight_text": HIGHLIGHT_TEXT_SCHEMA,
}

# Validation code
if __name__ == "__main__":
    import sys
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test each schema
    for schema_name, schema in SCHEMAS.items():
        total_tests += 1
        
        # Check that schema is valid JSON
        try:
            json.dumps(schema)
        except Exception as e:
            all_validation_failures.append(f"Schema '{schema_name}' is not valid JSON: {str(e)}")
        
        # Check for required fields
        for required_field in schema.get("required", []):
            if required_field not in schema.get("properties", {}):
                all_validation_failures.append(
                    f"Schema '{schema_name}' requires '{required_field}' but it is not defined in properties")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} schema definitions are valid")
        sys.exit(0)  # Exit with success code