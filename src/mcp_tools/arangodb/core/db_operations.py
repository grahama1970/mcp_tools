"""
Core database operations for ArangoDB integration.

This module provides pure business logic for generic CRUD operations on ArangoDB
collections. These functions are independent of any UI or presentation concerns.

Links to third-party documentation:
- ArangoDB Python Driver: https://docs.python-arango.com/en/main/
- ArangoDB AQL: https://www.arangodb.com/docs/stable/aql/

Sample input:
    db = connect_to_arango()
    document = {
        "name": "Sample Document",
        "content": "This is a test document",
        "timestamp": "2023-01-01T12:00:00Z"
    }
    result = create_document(db, "collection_name", document)

Expected output:
    {
        "_id": "collection_name/12345",
        "_key": "12345",
        "_rev": "12345"
    }
"""

import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple, Union

from loguru import logger
from arango.database import StandardDatabase
from arango.exceptions import (
    DocumentInsertError,
    DocumentGetError,
    DocumentUpdateError,
    DocumentDeleteError,
    AQLQueryExecuteError
)

# =============================================================================
# GENERIC CRUD OPERATIONS
# =============================================================================

def create_document(
    db: StandardDatabase,
    collection_name: str,
    document: Dict[str, Any],
    document_key: Optional[str] = None,
    return_new: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Insert a document into a collection.

    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        document: Document data to insert
        document_key: Optional key for the document (auto-generated if not provided)
        return_new: Whether to return the new document

    Returns:
        Optional[Dict[str, Any]]: The inserted document or metadata if successful, None otherwise
    """
    try:
        # Generate a key if not provided
        if document_key:
            document["_key"] = document_key
        elif "_key" not in document:
            document["_key"] = str(uuid.uuid4())

        # Add timestamp if not present
        if "timestamp" not in document:
            document["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Get the collection and insert document
        collection = db.collection(collection_name)
        result = collection.insert(document, return_new=return_new)

        logger.info(f"Created document in {collection_name}: {result.get('_key', result)}")
        return result["new"] if return_new and "new" in result else result

    except DocumentInsertError as e:
        logger.error(f"Failed to create document in {collection_name}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error creating document in {collection_name}: {e}")
        return None


def get_document(
    db: StandardDatabase,
    collection_name: str,
    document_key: str
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a document by key.

    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        document_key: Key of the document to retrieve

    Returns:
        Optional[Dict[str, Any]]: The document if found, None otherwise
    """
    try:
        collection = db.collection(collection_name)
        document = collection.get(document_key)

        if document:
            logger.debug(f"Retrieved document from {collection_name}: {document_key}")
        else:
            logger.warning(f"Document not found in {collection_name}: {document_key}")

        return document

    except DocumentGetError as e:
        logger.error(f"Failed to get document from {collection_name}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error getting document from {collection_name}: {e}")
        return None


def update_document(
    db: StandardDatabase,
    collection_name: str,
    document_key: str,
    updates: Dict[str, Any],
    return_new: bool = True,
    check_rev: bool = False,
    rev: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Update a document with new values.

    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        document_key: Key of the document to update
        updates: Dictionary of fields to update
        return_new: Whether to return the updated document
        check_rev: Whether to check document revision
        rev: Document revision (required if check_rev is True)

    Returns:
        Optional[Dict[str, Any]]: The updated document if successful, None otherwise
    """
    try:
        collection = db.collection(collection_name)

        # 1. Get the existing document
        existing_doc = collection.get(document_key)
        if not existing_doc:
            logger.error(f"Document {document_key} not found in {collection_name} for update.")
            return None

        # 2. Merge updates into the existing document
        merged_doc = existing_doc.copy()
        merged_doc.update(updates)

        # Add/update timestamp
        merged_doc["updated_at"] = datetime.now(timezone.utc).isoformat()
        # Ensure required fields like _key are present for replace
        merged_doc["_key"] = document_key  # Ensure _key is set

        # 3. Handle revision check if needed
        if check_rev:
            # If check_rev is True, we MUST use the _rev from the fetched doc
            if "_rev" not in existing_doc:
                logger.warning(f"Revision check requested but _rev not found in fetched document {document_key}")
                # Disable check if _rev is missing
                check_rev = False  
            else:
                # Include _rev in the merged document for replacement
                merged_doc["_rev"] = existing_doc["_rev"] if rev is None else rev

        # 4. Replace the document
        result = collection.replace(
            merged_doc,  # Pass the entire merged document
            return_new=return_new
        )

        logger.info(f"Replaced document in {collection_name}: {document_key}")
        return result["new"] if return_new and "new" in result else result

    except DocumentUpdateError as e:  # Replace might still raise DocumentUpdateError on rev mismatch
        logger.error(f"Failed to update document in {collection_name}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error updating document in {collection_name}: {e}")
        return None


def delete_document(
    db: StandardDatabase,
    collection_name: str,
    document_key: str,
    ignore_missing: bool = True,
    return_old: bool = False,
    check_rev: bool = False,
    rev: Optional[str] = None
) -> bool:
    """
    Delete a document from a collection.

    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        document_key: Key of the document to delete
        ignore_missing: Whether to ignore if document doesn't exist
        return_old: Whether to return the old document
        check_rev: Whether to check document revision
        rev: Document revision (required if check_rev is True)

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get the collection and delete document
        collection = db.collection(collection_name)

        # Add revision if needed
        params = {}
        if check_rev and rev:
            params["rev"] = rev

        result = collection.delete(
            document=document_key,
            ignore_missing=ignore_missing,
            return_old=return_old,
            check_rev=check_rev,
            **params
        )

        if result is False and ignore_missing:
            logger.info(f"Document not found for deletion in {collection_name}: {document_key}")
            return True

        logger.info(f"Deleted document from {collection_name}: {document_key}")
        return True

    except DocumentDeleteError as e:
        logger.error(f"Failed to delete document from {collection_name}: {e}")
        return False
    except Exception as e:
        logger.exception(f"Unexpected error deleting document from {collection_name}: {e}")
        return False


def query_documents(
    db: StandardDatabase,
    collection_name: str,
    filter_clause: str = "",
    sort_clause: str = "",
    limit: int = 100,
    offset: int = 0,
    bind_vars: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Query documents from a collection.

    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        filter_clause: AQL filter clause (e.g., "FILTER doc.field == @value")
        sort_clause: AQL sort clause (e.g., "SORT doc.field DESC")
        limit: Maximum number of documents to return
        offset: Number of documents to skip
        bind_vars: Bind variables for the query

    Returns:
        List[Dict[str, Any]]: List of documents matching the query
    """
    try:
        # Build AQL query
        aql = f"""
        FOR doc IN {collection_name}
        {filter_clause}
        {sort_clause}
        LIMIT {offset}, {limit}
        RETURN doc
        """

        # Set default bind variables
        if bind_vars is None:
            bind_vars = {}

        # Execute query
        cursor = db.aql.execute(aql, bind_vars=bind_vars)
        results = list(cursor)

        logger.info(f"Query returned {len(results)} documents from {collection_name}")
        return results

    except Exception as e:
        logger.exception(f"Error querying documents from {collection_name}: {e}")
        return []


def bulk_import_documents(
    db: StandardDatabase,
    collection_name: str,
    documents: List[Dict[str, Any]],
    on_duplicate: str = "error"
) -> Dict[str, Any]:
    """
    Import multiple documents into a collection in a single operation.

    Args:
        db: ArangoDB database handle
        collection_name: Name of the collection
        documents: List of document dictionaries to import
        on_duplicate: Strategy for handling duplicates ("error", "update", "replace", "ignore")

    Returns:
        Dict[str, Any]: Import results with counts of created, errors, etc.
    """
    try:
        collection = db.collection(collection_name)
        
        # Add timestamps and keys if needed
        current_time = datetime.now(timezone.utc).isoformat()
        for doc in documents:
            if "timestamp" not in doc:
                doc["timestamp"] = current_time
            if "_key" not in doc:
                doc["_key"] = str(uuid.uuid4())
        
        # Execute bulk import
        result = collection.import_bulk(
            documents,
            on_duplicate=on_duplicate
        )
        
        logger.info(f"Bulk imported {result['created']} documents into {collection_name}")
        return result
        
    except Exception as e:
        logger.exception(f"Error during bulk import to {collection_name}: {e}")
        return {
            "error": True,
            "message": str(e),
            "created": 0,
            "errors": len(documents)
        }


def execute_aql(
    db: StandardDatabase,
    query: str,
    bind_vars: Optional[Dict[str, Any]] = None,
    batch_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Execute a custom AQL query.

    Args:
        db: ArangoDB database handle
        query: AQL query string
        bind_vars: Bind variables for the query
        batch_size: Number of results to fetch per batch

    Returns:
        List[Dict[str, Any]]: Query results
    """
    try:
        if bind_vars is None:
            bind_vars = {}
            
        cursor = db.aql.execute(
            query,
            bind_vars=bind_vars,
            batch_size=batch_size
        )
        
        results = list(cursor)
        logger.info(f"AQL query returned {len(results)} results")
        return results
        
    except AQLQueryExecuteError as e:
        logger.error(f"AQL query execution failed: {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error executing AQL query: {e}")
        return []


if __name__ == "__main__":
    import sys
    from arango import ArangoClient
    
    # Validation setup
    all_validation_failures = []
    total_tests = 0
    
    # Setup test database connection
    # Note: This is for validation only and uses an ephemeral database
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
        
        # Create a test collection
        test_collection = "test_collection"
        if db.has_collection(test_collection):
            db.delete_collection(test_collection)
        db.create_collection(test_collection)
        
        # Test 1: Create document
        total_tests += 1
        test_doc = {
            "name": "Test Document",
            "value": 42,
            "tags": ["test", "validation"]
        }
        result = create_document(db, test_collection, test_doc)
        if not result or "_key" not in result:
            all_validation_failures.append("Test 1: Create document failed")
        else:
            doc_key = result["_key"]
            
        # Test 2: Get document
        total_tests += 1
        fetched_doc = get_document(db, test_collection, doc_key)
        if not fetched_doc or fetched_doc.get("name") != "Test Document":
            all_validation_failures.append("Test 2: Get document failed")
            
        # Test 3: Update document
        total_tests += 1
        update_result = update_document(
            db, 
            test_collection, 
            doc_key, 
            {"value": 100, "updated": True}
        )
        if not update_result or update_result.get("value") != 100 or not update_result.get("updated"):
            all_validation_failures.append("Test 3: Update document failed")
            
        # Test 4: Query documents
        total_tests += 1
        query_result = query_documents(
            db,
            test_collection,
            filter_clause="FILTER doc.value == 100",
            limit=10
        )
        if not query_result or len(query_result) != 1 or query_result[0].get("_key") != doc_key:
            all_validation_failures.append("Test 4: Query documents failed")
            
        # Test 5: Bulk import
        total_tests += 1
        bulk_docs = [
            {"name": "Bulk 1", "value": 1},
            {"name": "Bulk 2", "value": 2},
            {"name": "Bulk 3", "value": 3}
        ]
        bulk_result = bulk_import_documents(db, test_collection, bulk_docs)
        if not bulk_result or bulk_result.get("created") != 3:
            all_validation_failures.append("Test 5: Bulk import failed")
            
        # Test 6: Execute AQL
        total_tests += 1
        aql_result = execute_aql(
            db,
            "FOR doc IN @@collection FILTER doc.name LIKE 'Bulk%' RETURN doc",
            {"@collection": test_collection}
        )
        if not aql_result or len(aql_result) != 3:
            all_validation_failures.append("Test 6: Execute AQL failed")
            
        # Test 7: Delete document
        total_tests += 1
        delete_result = delete_document(db, test_collection, doc_key)
        if not delete_result:
            all_validation_failures.append("Test 7: Delete document failed")
            
        # Verify deletion
        total_tests += 1
        should_be_none = get_document(db, test_collection, doc_key)
        if should_be_none is not None:
            all_validation_failures.append("Test 8: Document wasn't actually deleted")
            
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