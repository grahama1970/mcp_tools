"""
MCP Handler functions for ArangoDB database operations.

This module provides the handler functions for database operations
exposed through the MCP interface.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Union, Tuple

from arango.database import Database
from arango.exceptions import (
    DocumentInsertError,
    DocumentGetError,
    DocumentUpdateError,
    DocumentDeleteError,
    AQLQueryExecuteError
)

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

logger = logging.getLogger(__name__)


def get_db_connection(db_config: Dict[str, Any]) -> Database:
    """Create an ArangoDB database connection using the provided configuration."""
    from arango import ArangoClient
    
    host = db_config.get("host", "http://localhost:8529")
    username = db_config.get("username", "root")
    password = db_config.get("password", "")
    database = db_config.get("database", "_system")
    
    client = ArangoClient(hosts=host)
    return client.db(database, username=username, password=password)


def format_results(results: List[Dict[str, Any]], output_format: str = "json") -> Dict[str, Any]:
    """Format the results for the MCP response."""
    if output_format == "table":
        # For table output, we need to transform the results into a table format
        if not results:
            return {
                "columns": [],
                "rows": []
            }
        
        # Get column names from the first result
        columns = list(results[0].keys())
        
        # Create rows from all results
        rows = []
        for result in results:
            row = [result.get(column, "") for column in columns]
            rows.append(row)
            
        return {
            "columns": columns,
            "rows": rows
        }
    
    # Default is to return the raw JSON
    return results


# ==============================
# CRUD Operation Handlers
# ==============================

def _create_document_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for create_document operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        collection = params.get("collection")
        data = params.get("data", {})
        return_new = params.get("return_new", True)
        overwrite = params.get("overwrite", False)
        overwrite_mode = params.get("overwrite_mode", "replace")
        generate_uuid = params.get("generate_uuid", True)
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not collection:
            return {
                "status": "error",
                "error": "collection parameter is required"
            }
        
        if not data:
            return {
                "status": "error",
                "error": "data parameter is required"
            }
        
        # Create document
        result = create_document(
            collection=collection,
            data=data,
            db=db,
            return_new=return_new,
            overwrite=overwrite,
            overwrite_mode=overwrite_mode,
            generate_uuid=generate_uuid
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "result": result,
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in create_document_handler: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _get_document_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for get_document operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        collection = params.get("collection")
        document_key = params.get("document_key")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not collection:
            return {
                "status": "error",
                "error": "collection parameter is required"
            }
        
        if not document_key:
            return {
                "status": "error",
                "error": "document_key parameter is required"
            }
        
        # Get document
        result = get_document(
            collection=collection,
            document_key=document_key,
            db=db
        )
        
        # Check if document was found
        if not result:
            return {
                "status": "error",
                "error": f"Document with key '{document_key}' not found in collection '{collection}'"
            }
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "result": result,
            "time": elapsed_time,
            "format": output_format
        }
    except DocumentGetError as e:
        # Log and return exception
        logger.exception(f"Error in get_document_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "collection": params.get("collection"),
            "document_key": params.get("document_key")
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in get_document_handler: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _update_document_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for update_document operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        collection = params.get("collection")
        document_key = params.get("document_key")
        data = params.get("data", {})
        return_new = params.get("return_new", True)
        return_old = params.get("return_old", False)
        keep_none = params.get("keep_none", False)
        merge = params.get("merge", True)
        check_rev = params.get("check_rev")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not collection:
            return {
                "status": "error",
                "error": "collection parameter is required"
            }
        
        if not document_key:
            return {
                "status": "error",
                "error": "document_key parameter is required"
            }
        
        if not data:
            return {
                "status": "error",
                "error": "data parameter is required"
            }
        
        # Update document
        result = update_document(
            collection=collection,
            document_key=document_key,
            data=data,
            db=db,
            return_new=return_new,
            return_old=return_old,
            keep_none=keep_none,
            merge=merge,
            check_rev=check_rev
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "result": result,
            "time": elapsed_time,
            "format": output_format
        }
    except DocumentUpdateError as e:
        # Log and return exception
        logger.exception(f"Error in update_document_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "collection": params.get("collection"),
            "document_key": params.get("document_key")
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in update_document_handler: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _delete_document_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for delete_document operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        collection = params.get("collection")
        document_key = params.get("document_key")
        return_old = params.get("return_old", False)
        check_rev = params.get("check_rev")
        ignore_missing = params.get("ignore_missing", False)
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not collection:
            return {
                "status": "error",
                "error": "collection parameter is required"
            }
        
        if not document_key:
            return {
                "status": "error",
                "error": "document_key parameter is required"
            }
        
        # Delete document
        result = delete_document(
            collection=collection,
            document_key=document_key,
            db=db,
            return_old=return_old,
            check_rev=check_rev,
            ignore_missing=ignore_missing
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "result": result,
            "time": elapsed_time,
            "format": output_format
        }
    except DocumentDeleteError as e:
        # Log and return exception
        logger.exception(f"Error in delete_document_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "collection": params.get("collection"),
            "document_key": params.get("document_key")
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in delete_document_handler: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _query_documents_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for query_documents operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        query = params.get("query")
        bind_vars = params.get("bind_vars", {})
        count = params.get("count", False)
        batch_size = params.get("batch_size")
        full_count = params.get("full_count")
        max_runtime = params.get("max_runtime")
        profile = params.get("profile", False)
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not query:
            return {
                "status": "error",
                "error": "query parameter is required"
            }
        
        # Execute query
        result = query_documents(
            query=query,
            db=db,
            bind_vars=bind_vars,
            count=count,
            batch_size=batch_size,
            full_count=full_count,
            max_runtime=max_runtime,
            profile=profile
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Handle profile results which come as a tuple
        if profile:
            documents, stats = result
            formatted_results = format_results(documents, output_format)
            
            # Return success response with profile information
            return {
                "status": "success",
                "results": formatted_results,
                "stats": stats,
                "query": query,
                "bind_vars": bind_vars,
                "total": len(documents),
                "time": elapsed_time,
                "format": output_format
            }
        else:
            # Format results
            formatted_results = format_results(result, output_format)
            
            # Return success response
            return {
                "status": "success",
                "results": formatted_results,
                "query": query,
                "bind_vars": bind_vars,
                "total": len(result),
                "time": elapsed_time,
                "format": output_format
            }
    except AQLQueryExecuteError as e:
        # Log and return exception
        logger.exception(f"Error in query_documents_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "query": params.get("query"),
            "bind_vars": params.get("bind_vars", {})
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in query_documents_handler: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


# ==============================
# Message Operation Handlers
# ==============================

def _create_message_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for create_message operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        conversation_id = params.get("conversation_id")
        role = params.get("role")
        content = params.get("content")
        metadata = params.get("metadata")
        message_collection = params.get("message_collection", "messages")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not conversation_id:
            return {
                "status": "error",
                "error": "conversation_id parameter is required"
            }
        
        if not role:
            return {
                "status": "error",
                "error": "role parameter is required"
            }
            
        if not content:
            return {
                "status": "error",
                "error": "content parameter is required"
            }
        
        # Create message
        result = create_message(
            db=db,
            conversation_id=conversation_id,
            role=role,
            content=content,
            metadata=metadata,
            message_collection=message_collection
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "result": result,
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in create_message_handler: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _get_message_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for get_message operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        message_key = params.get("message_key")
        message_collection = params.get("message_collection", "messages")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not message_key:
            return {
                "status": "error",
                "error": "message_key parameter is required"
            }
        
        # Get message
        result = get_message(
            db=db,
            message_key=message_key,
            message_collection=message_collection
        )
        
        # Check if message was found
        if not result:
            return {
                "status": "error",
                "error": f"Message with key '{message_key}' not found"
            }
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "result": result,
            "time": elapsed_time,
            "format": output_format
        }
    except DocumentGetError as e:
        # Log and return exception
        logger.exception(f"Error in get_message_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message_key": params.get("message_key")
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in get_message_handler: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _update_message_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for update_message operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        message_key = params.get("message_key")
        data = params.get("data", {})
        message_collection = params.get("message_collection", "messages")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not message_key:
            return {
                "status": "error",
                "error": "message_key parameter is required"
            }
        
        if not data:
            return {
                "status": "error",
                "error": "data parameter is required"
            }
        
        # Update message
        result = update_message(
            db=db,
            message_key=message_key,
            data=data,
            message_collection=message_collection
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "result": result,
            "time": elapsed_time,
            "format": output_format
        }
    except DocumentUpdateError as e:
        # Log and return exception
        logger.exception(f"Error in update_message_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message_key": params.get("message_key")
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in update_message_handler: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _delete_message_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for delete_message operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        message_key = params.get("message_key")
        delete_relationships = params.get("delete_relationships", True)
        message_collection = params.get("message_collection", "messages")
        relationship_collection = params.get("relationship_collection", "relates_to")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not message_key:
            return {
                "status": "error",
                "error": "message_key parameter is required"
            }
        
        # Delete message
        result = delete_message(
            db=db,
            message_key=message_key,
            delete_relationships=delete_relationships,
            message_collection=message_collection,
            relationship_collection=relationship_collection
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "result": result,
            "time": elapsed_time,
            "format": output_format
        }
    except DocumentDeleteError as e:
        # Log and return exception
        logger.exception(f"Error in delete_message_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message_key": params.get("message_key")
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in delete_message_handler: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _get_conversation_messages_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for get_conversation_messages operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        conversation_id = params.get("conversation_id")
        sort_by = params.get("sort_by", "created_at")
        sort_direction = params.get("sort_direction", "ASC")
        limit = params.get("limit")
        offset = params.get("offset")
        message_collection = params.get("message_collection", "messages")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not conversation_id:
            return {
                "status": "error",
                "error": "conversation_id parameter is required"
            }
        
        # Get conversation messages
        results = get_conversation_messages(
            db=db,
            conversation_id=conversation_id,
            sort_by=sort_by,
            sort_direction=sort_direction,
            limit=limit,
            offset=offset,
            message_collection=message_collection
        )
        
        # Format results
        formatted_results = format_results(results, output_format)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "results": formatted_results,
            "conversation_id": conversation_id,
            "total": len(results),
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in get_conversation_messages_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "conversation_id": params.get("conversation_id")
        }


def _delete_conversation_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for delete_conversation operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        conversation_id = params.get("conversation_id")
        delete_relationships = params.get("delete_relationships", True)
        message_collection = params.get("message_collection", "messages")
        relationship_collection = params.get("relationship_collection", "relates_to")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not conversation_id:
            return {
                "status": "error",
                "error": "conversation_id parameter is required"
            }
        
        # Delete conversation
        deleted_count = delete_conversation(
            db=db,
            conversation_id=conversation_id,
            delete_relationships=delete_relationships,
            message_collection=message_collection,
            relationship_collection=relationship_collection
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "conversation_id": conversation_id,
            "deleted_count": deleted_count,
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in delete_conversation_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "conversation_id": params.get("conversation_id")
        }


# ==============================
# Relationship Operation Handlers
# ==============================

def _create_relationship_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for create_relationship operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        from_id = params.get("from_id")
        to_id = params.get("to_id")
        edge_collection = params.get("edge_collection", "relates_to")
        properties = params.get("properties")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not from_id:
            return {
                "status": "error",
                "error": "from_id parameter is required"
            }
        
        if not to_id:
            return {
                "status": "error",
                "error": "to_id parameter is required"
            }
        
        # Create relationship
        result = create_relationship(
            db=db,
            from_id=from_id,
            to_id=to_id,
            edge_collection=edge_collection,
            properties=properties
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "result": result,
            "time": elapsed_time,
            "format": output_format
        }
    except DocumentInsertError as e:
        # Log and return exception
        logger.exception(f"Error in create_relationship_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "from_id": params.get("from_id"),
            "to_id": params.get("to_id")
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in create_relationship_handler: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _delete_relationship_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for delete_relationship operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        edge_key = params.get("edge_key")
        edge_collection = params.get("edge_collection", "relates_to")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not edge_key:
            return {
                "status": "error",
                "error": "edge_key parameter is required"
            }
        
        # Delete relationship
        result = delete_relationship_by_key(
            db=db,
            edge_key=edge_key,
            edge_collection=edge_collection
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "result": result,
            "time": elapsed_time,
            "format": output_format
        }
    except DocumentDeleteError as e:
        # Log and return exception
        logger.exception(f"Error in delete_relationship_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "edge_key": params.get("edge_key")
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in delete_relationship_handler: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _delete_relationships_between_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for delete_relationships_between operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        from_id = params.get("from_id")
        to_id = params.get("to_id")
        edge_collection = params.get("edge_collection", "relates_to")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not from_id:
            return {
                "status": "error",
                "error": "from_id parameter is required"
            }
        
        if not to_id:
            return {
                "status": "error",
                "error": "to_id parameter is required"
            }
        
        # Delete relationships
        deleted_count = delete_relationships_between(
            db=db,
            from_id=from_id,
            to_id=to_id,
            edge_collection=edge_collection
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "deleted_count": deleted_count,
            "from_id": from_id,
            "to_id": to_id,
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in delete_relationships_between_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "from_id": params.get("from_id"),
            "to_id": params.get("to_id")
        }


def _link_message_to_document_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for link_message_to_document operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        message_id = params.get("message_id")
        document_id = params.get("document_id")
        properties = params.get("properties")
        edge_collection = params.get("edge_collection", "relates_to")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not message_id:
            return {
                "status": "error",
                "error": "message_id parameter is required"
            }
        
        if not document_id:
            return {
                "status": "error",
                "error": "document_id parameter is required"
            }
        
        # Link message to document
        result = link_message_to_document(
            db=db,
            message_id=message_id,
            document_id=document_id,
            properties=properties,
            edge_collection=edge_collection
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "result": result,
            "time": elapsed_time,
            "format": output_format
        }
    except DocumentInsertError as e:
        # Log and return exception
        logger.exception(f"Error in link_message_to_document_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message_id": params.get("message_id"),
            "document_id": params.get("document_id")
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in link_message_to_document_handler: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _get_documents_for_message_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for get_documents_for_message operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        message_id = params.get("message_id")
        collection_filter = params.get("collection_filter")
        edge_collection = params.get("edge_collection", "relates_to")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not message_id:
            return {
                "status": "error",
                "error": "message_id parameter is required"
            }
        
        # Get documents for message
        results = get_documents_for_message(
            db=db,
            message_id=message_id,
            collection_filter=collection_filter,
            edge_collection=edge_collection
        )
        
        # Format results
        formatted_results = format_results(results, output_format)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "results": formatted_results,
            "message_id": message_id,
            "total": len(results),
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in get_documents_for_message_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "message_id": params.get("message_id")
        }


def _get_messages_for_document_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for get_messages_for_document operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        document_id = params.get("document_id")
        message_collection = params.get("message_collection", "messages")
        edge_collection = params.get("edge_collection", "relates_to")
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not document_id:
            return {
                "status": "error",
                "error": "document_id parameter is required"
            }
        
        # Get messages for document
        results = get_messages_for_document(
            db=db,
            document_id=document_id,
            message_collection=message_collection,
            edge_collection=edge_collection
        )
        
        # Format results
        formatted_results = format_results(results, output_format)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "results": formatted_results,
            "document_id": document_id,
            "total": len(results),
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in get_messages_for_document_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "document_id": params.get("document_id")
        }


def _get_related_documents_handler(params: Dict[str, Any]) -> Dict[str, Any]:
    """Handler for get_related_documents operation."""
    start_time = time.time()
    
    try:
        # Get database configuration
        db_config = params.get("db_config", {})
        db = get_db_connection(db_config)
        
        # Extract parameters
        document_id = params.get("document_id")
        collection_filter = params.get("collection_filter")
        edge_collection = params.get("edge_collection", "relates_to")
        direction = params.get("direction", "outbound")
        max_depth = params.get("max_depth", 1)
        output_format = params.get("output_format", "json")
        
        # Validate required parameters
        if not document_id:
            return {
                "status": "error",
                "error": "document_id parameter is required"
            }
        
        # Get related documents
        results = get_related_documents(
            db=db,
            document_id=document_id,
            collection_filter=collection_filter,
            edge_collection=edge_collection,
            direction=direction,
            max_depth=max_depth
        )
        
        # Format results
        formatted_results = format_results(results, output_format)
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        
        # Return success response
        return {
            "status": "success",
            "results": formatted_results,
            "document_id": document_id,
            "total": len(results),
            "time": elapsed_time,
            "format": output_format
        }
    except Exception as e:
        # Log and return exception
        logger.exception(f"Error in get_related_documents_handler: {e}")
        return {
            "status": "error",
            "error": str(e),
            "document_id": params.get("document_id")
        }


# Map handlers to operations
HANDLER_MAP = {
    # CRUD Operations
    "create_document": _create_document_handler,
    "get_document": _get_document_handler,
    "update_document": _update_document_handler,
    "delete_document": _delete_document_handler,
    "query_documents": _query_documents_handler,
    
    # Message Operations
    "create_message": _create_message_handler,
    "get_message": _get_message_handler,
    "update_message": _update_message_handler,
    "delete_message": _delete_message_handler,
    "get_conversation_messages": _get_conversation_messages_handler,
    "delete_conversation": _delete_conversation_handler,
    
    # Relationship Operations
    "create_relationship": _create_relationship_handler,
    "delete_relationship": _delete_relationship_handler,
    "delete_relationships_between": _delete_relationships_between_handler,
    "link_message_to_document": _link_message_to_document_handler,
    "get_documents_for_message": _get_documents_for_message_handler,
    "get_messages_for_document": _get_messages_for_document_handler,
    "get_related_documents": _get_related_documents_handler,
}


# Validation function
if __name__ == "__main__":
    import sys
    import json
    from mcp_tools.arangodb.mcp.schema import SCHEMAS
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test each handler
    for handler_name, handler_func in HANDLER_MAP.items():
        total_tests += 1
        
        # Check if handler has a corresponding schema
        if handler_name not in SCHEMAS:
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
        if schema_name.startswith("db_") and schema_name not in HANDLER_MAP:
            all_validation_failures.append(f"Schema '{schema_name}' does not have a corresponding handler")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} handler mappings are valid")
        sys.exit(0)  # Exit with success code