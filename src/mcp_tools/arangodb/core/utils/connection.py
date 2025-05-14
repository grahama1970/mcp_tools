"""
ArangoDB connection utilities.

This module provides functions for connecting to ArangoDB and ensuring that
required databases, collections, and views exist.
"""

import os
from typing import Dict, List, Any, Optional, Union
from loguru import logger

from arango import ArangoClient
from arango.database import StandardDatabase


def connect_arango(hosts: Optional[str] = None) -> ArangoClient:
    """
    Connect to ArangoDB server.
    
    Args:
        hosts: ArangoDB host URL(s) (default: environment variable or localhost)
        
    Returns:
        ArangoClient instance
        
    Raises:
        ConnectionError: If connection fails
    """
    # Get hosts from environment if not provided
    if hosts is None:
        hosts = os.environ.get("ARANGO_HOSTS", "http://localhost:8529")
    
    try:
        # Connect to ArangoDB
        client = ArangoClient(hosts=hosts)
        logger.info(f"Connected to ArangoDB at {hosts}")
        return client
    except Exception as e:
        error_msg = f"Failed to connect to ArangoDB at {hosts}: {str(e)}"
        logger.error(error_msg)
        raise ConnectionError(error_msg) from e


def ensure_database(
    client: ArangoClient,
    name: str = None,
    username: str = None,
    password: str = None
) -> StandardDatabase:
    """
    Ensure that the database exists, creating it if necessary.
    
    Args:
        client: ArangoClient instance
        name: Database name (default: environment variable or "_system")
        username: Username for authentication (default: environment variable or "root")
        password: Password for authentication (default: environment variable or "")
        
    Returns:
        Database instance
        
    Raises:
        ConnectionError: If database connection fails
        RuntimeError: If database creation fails
    """
    # Get credentials from environment if not provided
    if name is None:
        name = os.environ.get("ARANGO_DB", "_system")
    if username is None:
        username = os.environ.get("ARANGO_USER", "root")
    if password is None:
        password = os.environ.get("ARANGO_PASSWORD", "")
    
    try:
        # Get system database to check if target database exists
        sys_db = client.db("_system", username=username, password=password)
        
        # Create database if it doesn't exist
        if not sys_db.has_database(name):
            sys_db.create_database(name)
            logger.info(f"Created database '{name}'")
        
        # Connect to the database
        db = client.db(name, username=username, password=password)
        logger.info(f"Connected to database '{name}'")
        return db
    except Exception as e:
        error_msg = f"Failed to ensure database '{name}': {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def ensure_collection(
    db: StandardDatabase,
    name: str,
    edge: bool = False,
    index_fields: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Ensure that a collection exists, creating it if necessary.
    
    Args:
        db: Database instance
        name: Collection name
        edge: Whether the collection is an edge collection
        index_fields: List of fields to create indexes for
        
    Returns:
        Collection information
        
    Raises:
        RuntimeError: If collection creation fails
    """
    try:
        # Check if collection exists
        if db.has_collection(name):
            collection = db.collection(name)
            logger.debug(f"Collection '{name}' already exists")
        else:
            # Create collection
            if edge:
                collection = db.create_collection(name, edge=True)
                logger.info(f"Created edge collection '{name}'")
            else:
                collection = db.create_collection(name)
                logger.info(f"Created document collection '{name}'")
        
        # Create indexes if specified
        if index_fields:
            for field in index_fields:
                # Check if index already exists
                indexes = collection.indexes()
                field_indexed = any(
                    index.get("fields", []) == [field] and index.get("type") == "persistent"
                    for index in indexes
                )
                
                if not field_indexed:
                    collection.add_persistent_index([field])
                    logger.info(f"Created persistent index on '{field}' for collection '{name}'")
        
        return collection.properties()
    except Exception as e:
        error_msg = f"Failed to ensure collection '{name}': {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def ensure_memory_agent_collections(db: StandardDatabase) -> Dict[str, Dict[str, Any]]:
    """
    Ensure that all collections required by the Memory Agent exist.
    
    Args:
        db: Database instance
        
    Returns:
        Dictionary of collection information
        
    Raises:
        RuntimeError: If collection creation fails
    """
    try:
        # Create document collections
        message_collection = ensure_collection(
            db, 
            "agent_messages", 
            edge=False,
            index_fields=["conversation_id", "timestamp"]
        )
        
        memory_collection = ensure_collection(
            db, 
            "agent_memories", 
            edge=False,
            index_fields=["conversation_id", "timestamp"]
        )
        
        # Create edge collection
        edge_collection = ensure_collection(
            db,
            "agent_relationships",
            edge=True,
            index_fields=["type", "timestamp"]
        )
        
        # Ensure search view exists
        view = ensure_arangosearch_view(
            db,
            "agent_memory_view",
            ["agent_messages", "agent_memories"],
            ["content", "summary"]
        )
        
        return {
            "agent_messages": message_collection,
            "agent_memories": memory_collection,
            "agent_relationships": edge_collection,
            "agent_memory_view": view
        }
    except Exception as e:
        error_msg = f"Failed to ensure memory agent collections: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def ensure_arangosearch_view(
    db: StandardDatabase,
    name: str,
    collections: List[str],
    fields: List[str]
) -> Dict[str, Any]:
    """
    Ensure that an ArangoSearch view exists, creating it if necessary.
    
    Args:
        db: Database instance
        name: View name
        collections: List of collections to include in the view
        fields: List of fields to index
        
    Returns:
        View information
        
    Raises:
        RuntimeError: If view creation fails
    """
    try:
        # Check if view exists
        if db.has_arangosearch_view(name):
            view = db.arangosearch_view(name)
            logger.debug(f"ArangoSearch view '{name}' already exists")
        else:
            # Create view
            view = db.create_arangosearch_view(name)
            logger.info(f"Created ArangoSearch view '{name}'")
        
        # Configure view
        view_props = {
            "links": {}
        }
        
        # Add collections to view
        for collection in collections:
            view_props["links"][collection] = {
                "includeAllFields": False,
                "fields": {}
            }
            
            # Add fields to index
            for field in fields:
                view_props["links"][collection]["fields"][field] = {
                    "analyzers": ["text_en"],
                    "includeAllFields": False
                }
        
        # Update view properties
        view.update_properties(view_props)
        logger.info(f"Configured ArangoSearch view '{name}'")
        
        return view.properties()
    except Exception as e:
        error_msg = f"Failed to ensure ArangoSearch view '{name}': {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


if __name__ == "__main__":
    """
    Validation function for connection utilities.
    
    Note: This validation requires an actual ArangoDB instance to be running.
    If no ArangoDB is available, validation will fail.
    """
    import sys
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Connect to ArangoDB
    total_tests += 1
    try:
        client = connect_arango()
        if not isinstance(client, ArangoClient):
            all_validation_failures.append(f"connect_arango returned {type(client)} instead of ArangoClient")
    except Exception as e:
        all_validation_failures.append(f"connect_arango raised exception: {e}")
        # Skip remaining tests if connection fails
        print(f"❌ VALIDATION FAILED - Cannot connect to ArangoDB. Aborting further tests.")
        sys.exit(1)
    
    # Test 2: Ensure database
    total_tests += 1
    try:
        db = ensure_database(client)
        if not isinstance(db, StandardDatabase):
            all_validation_failures.append(f"ensure_database returned {type(db)} instead of StandardDatabase")
    except Exception as e:
        all_validation_failures.append(f"ensure_database raised exception: {e}")
        # Skip remaining tests if database connection fails
        print(f"❌ VALIDATION FAILED - Cannot connect to database. Aborting further tests.")
        sys.exit(1)
    
    # Test 3: Ensure collection
    total_tests += 1
    try:
        collection = ensure_collection(db, "test_collection")
        if not isinstance(collection, dict):
            all_validation_failures.append(f"ensure_collection returned {type(collection)} instead of dict")
    except Exception as e:
        all_validation_failures.append(f"ensure_collection raised exception: {e}")
    
    # Test 4: Ensure edge collection
    total_tests += 1
    try:
        edge_collection = ensure_collection(db, "test_edge_collection", edge=True)
        if not isinstance(edge_collection, dict):
            all_validation_failures.append(f"ensure_collection (edge) returned {type(edge_collection)} instead of dict")
    except Exception as e:
        all_validation_failures.append(f"ensure_collection (edge) raised exception: {e}")
    
    # Test 5: Ensure collection with index
    total_tests += 1
    try:
        indexed_collection = ensure_collection(db, "test_indexed_collection", index_fields=["testField"])
        if not isinstance(indexed_collection, dict):
            all_validation_failures.append(f"ensure_collection (indexed) returned {type(indexed_collection)} instead of dict")
    except Exception as e:
        all_validation_failures.append(f"ensure_collection (indexed) raised exception: {e}")
    
    # Test 6: Ensure ArangoSearch view
    total_tests += 1
    try:
        view = ensure_arangosearch_view(db, "test_view", ["test_collection"], ["testField"])
        if not isinstance(view, dict):
            all_validation_failures.append(f"ensure_arangosearch_view returned {type(view)} instead of dict")
    except Exception as e:
        all_validation_failures.append(f"ensure_arangosearch_view raised exception: {e}")
    
    # Test 7: Ensure memory agent collections
    total_tests += 1
    try:
        memory_collections = ensure_memory_agent_collections(db)
        if not isinstance(memory_collections, dict):
            all_validation_failures.append(f"ensure_memory_agent_collections returned {type(memory_collections)} instead of dict")
    except Exception as e:
        all_validation_failures.append(f"ensure_memory_agent_collections raised exception: {e}")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        sys.exit(0)  # Exit with success code