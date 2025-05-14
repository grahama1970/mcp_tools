"""
Core ArangoDB CRUD operations.

This module provides foundational database operations for ArangoDB, including:
- Document creation with automatic ID and timestamp generation
- Document retrieval, update, and deletion
- Generic query execution with pagination support

Links:
- ArangoDB Python Driver: https://docs.python-arango.com/
- ArangoDB AQL: https://www.arangodb.com/docs/stable/aql/

Sample input:
    create_document('collection_name', {'name': 'Test', 'value': 123})

Expected output:
    {'_id': 'collection_name/12345', '_key': '12345', '_rev': '12345', 
     'name': 'Test', 'value': 123, 'created_at': '2023-05-11T10:30:00.000000', 
     'updated_at': '2023-05-11T10:30:00.000000', 'uuid': '123e4567-e89b-12d3-a456-426614174000'}
"""

import uuid
import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

from arango.database import Database
from arango.exceptions import (
    DocumentInsertError,
    DocumentGetError,
    DocumentUpdateError,
    DocumentDeleteError,
    AQLQueryExecuteError
)


def create_document(
    collection: str,
    data: Dict[str, Any],
    db: Database,
    return_new: bool = True,
    overwrite: bool = False,
    overwrite_mode: str = "replace",
    generate_uuid: bool = True,
) -> Dict[str, Any]:
    """
    Create a document in an ArangoDB collection.

    Args:
        collection: The name of the collection.
        data: The document data to store.
        db: ArangoDB database connection.
        return_new: Whether to return the new document.
        overwrite: Whether to overwrite an existing document.
        overwrite_mode: The overwrite mode to use (replace, update, ignore).
        generate_uuid: Whether to generate a UUID for the document.

    Returns:
        dict: The created document including ArangoDB metadata.

    Raises:
        DocumentInsertError: If document insertion fails.
    """
    # Add timestamps
    timestamp = datetime.datetime.now().isoformat()
    document = data.copy()
    document["created_at"] = timestamp
    document["updated_at"] = timestamp

    # Add UUID if requested
    if generate_uuid:
        document["uuid"] = str(uuid.uuid4())

    try:
        result = db.collection(collection).insert(
            document,
            return_new=return_new,
            overwrite=overwrite,
            overwrite_mode=overwrite_mode,
        )
        if return_new and "new" in result:
            return result["new"]
        return result
    except DocumentInsertError as e:
        error_msg = f"Failed to create document in collection '{collection}': {str(e)}"
        raise DocumentInsertError(error_msg) from e


def get_document(
    collection: str,
    document_key: str,
    db: Database,
) -> Dict[str, Any]:
    """
    Retrieve a document from an ArangoDB collection by its key.

    Args:
        collection: The name of the collection.
        document_key: The document key.
        db: ArangoDB database connection.

    Returns:
        dict: The retrieved document.

    Raises:
        DocumentGetError: If document retrieval fails.
    """
    try:
        return db.collection(collection).get(document_key)
    except DocumentGetError as e:
        error_msg = f"Failed to get document with key '{document_key}' from collection '{collection}': {str(e)}"
        raise DocumentGetError(error_msg) from e


def update_document(
    collection: str,
    document_key: str,
    data: Dict[str, Any],
    db: Database,
    return_new: bool = True,
    return_old: bool = False,
    keep_none: bool = False,
    merge: bool = True,
    check_rev: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Update a document in an ArangoDB collection.

    Args:
        collection: The name of the collection.
        document_key: The document key.
        data: The data to update.
        db: ArangoDB database connection.
        return_new: Whether to return the new document.
        return_old: Whether to return the old document.
        keep_none: Whether to keep None values.
        merge: Whether to merge the update data with the existing document.
        check_rev: The revision to check before updating.

    Returns:
        dict: The update result, including the new document if return_new is True.

    Raises:
        DocumentUpdateError: If document update fails.
    """
    # Add updated timestamp
    document = data.copy()
    document["updated_at"] = datetime.datetime.now().isoformat()

    try:
        result = db.collection(collection).update(
            document_key,
            document,
            return_new=return_new,
            return_old=return_old,
            keep_none=keep_none,
            merge=merge,
            check_rev=check_rev,
        )
        if return_new and "new" in result:
            return result["new"]
        return result
    except DocumentUpdateError as e:
        error_msg = f"Failed to update document with key '{document_key}' in collection '{collection}': {str(e)}"
        raise DocumentUpdateError(error_msg) from e


def delete_document(
    collection: str,
    document_key: str,
    db: Database,
    return_old: bool = False,
    check_rev: Optional[str] = None,
    ignore_missing: bool = False,
) -> Union[bool, Dict[str, Any]]:
    """
    Delete a document from an ArangoDB collection.

    Args:
        collection: The name of the collection.
        document_key: The document key.
        db: ArangoDB database connection.
        return_old: Whether to return the old document.
        check_rev: The revision to check before deleting.
        ignore_missing: Whether to ignore missing documents.

    Returns:
        bool|dict: True if the document was deleted, or the old document if return_old is True.

    Raises:
        DocumentDeleteError: If document deletion fails.
    """
    try:
        return db.collection(collection).delete(
            document_key,
            return_old=return_old,
            check_rev=check_rev,
            ignore_missing=ignore_missing,
        )
    except DocumentDeleteError as e:
        error_msg = f"Failed to delete document with key '{document_key}' from collection '{collection}': {str(e)}"
        raise DocumentDeleteError(error_msg) from e


def query_documents(
    query: str,
    db: Database,
    bind_vars: Optional[Dict[str, Any]] = None,
    count: bool = False,
    batch_size: Optional[int] = None,
    ttl: Optional[int] = None,
    optimizer_rules: Optional[List[str]] = None,
    cache: bool = True,
    memory_limit: Optional[int] = None,
    fail_on_warning: bool = False,
    profile: bool = False,
    max_runtime: Optional[int] = None,
    max_warning_count: Optional[int] = None,
    max_transaction_size: Optional[int] = None,
    max_plans: Optional[int] = None,
    full_count: Optional[bool] = None,
    fill_block_cache: Optional[bool] = None,
    max_nodes_size: Optional[int] = None,
    return_cursor: bool = False,
    stream: bool = False,
) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
    """
    Execute an AQL query against ArangoDB.

    Args:
        query: The AQL query string.
        db: ArangoDB database connection.
        bind_vars: Variables to bind to the query.
        count: Whether to count the results.
        batch_size: The batch size for retrieving results.
        ttl: Time-to-live for the cursor (in seconds).
        optimizer_rules: List of optimizer rules.
        cache: Whether to use the query cache.
        memory_limit: Memory limit for the query.
        fail_on_warning: Whether to fail on warnings.
        profile: Whether to return profiling information.
        max_runtime: Maximum runtime for the query.
        max_warning_count: Maximum warning count.
        max_transaction_size: Maximum transaction size.
        max_plans: Maximum number of plans.
        full_count: Whether to return the full count.
        fill_block_cache: Whether to fill the block cache.
        max_nodes_size: Maximum size for the node cache.
        return_cursor: Whether to return the cursor.
        stream: Whether to use streaming.

    Returns:
        list|tuple: Query results as a list of dictionaries or a tuple with (results, stats).

    Raises:
        AQLQueryExecuteError: If query execution fails.
    """
    try:
        cursor = db.aql.execute(
            query,
            bind_vars=bind_vars or {},
            count=count,
            batch_size=batch_size,
            ttl=ttl,
            optimizer_rules=optimizer_rules,
            cache=cache,
            memory_limit=memory_limit,
            fail_on_warning=fail_on_warning,
            profile=profile,
            max_runtime=max_runtime,
            max_warning_count=max_warning_count,
            max_transaction_size=max_transaction_size,
            max_plans=max_plans,
            full_count=full_count,
            fill_block_cache=fill_block_cache,
            max_nodes_size=max_nodes_size,
            stream=stream,
        )

        if return_cursor:
            return cursor

        results = [doc for doc in cursor]
        
        if profile or (hasattr(cursor, 'extra') and cursor.extra):
            return results, cursor.extra
        
        return results
    except AQLQueryExecuteError as e:
        error_msg = f"AQL query execution failed: {str(e)}"
        raise AQLQueryExecuteError(error_msg) from e


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
        
        # Create test collection if it doesn't exist
        if not db.has_collection("test_crud"):
            db.create_collection("test_crud")
            
        collection_name = "test_crud"
        
        # Test 1: Document creation
        total_tests += 1
        test_data = {"name": "Test Document", "value": 42}
        result = create_document(collection_name, test_data, db)
        
        if not result or not isinstance(result, dict):
            all_validation_failures.append("Document creation failed: No result returned")
        elif "name" not in result or result["name"] != "Test Document":
            all_validation_failures.append(f"Document creation failed: Expected 'Test Document', got {result.get('name')}")
        elif "created_at" not in result or "updated_at" not in result:
            all_validation_failures.append("Document creation failed: Missing timestamps")
        elif "uuid" not in result:
            all_validation_failures.append("Document creation failed: Missing UUID")
            
        # Remember document key for further tests
        doc_key = result.get("_key", "")
        
        # Test 2: Document retrieval
        total_tests += 1
        if doc_key:
            retrieved = get_document(collection_name, doc_key, db)
            if not retrieved or retrieved.get("name") != "Test Document":
                all_validation_failures.append(f"Document retrieval failed: Expected 'Test Document', got {retrieved.get('name')}")
        else:
            all_validation_failures.append("Document retrieval skipped: No valid document key")
            
        # Test 3: Document update
        total_tests += 1
        if doc_key:
            update_data = {"name": "Updated Document", "new_field": "new value"}
            updated = update_document(collection_name, doc_key, update_data, db)
            
            if not updated or updated.get("name") != "Updated Document":
                all_validation_failures.append(f"Document update failed: Expected 'Updated Document', got {updated.get('name')}")
            if "new_field" not in updated or updated["new_field"] != "new value":
                all_validation_failures.append(f"Document update failed: Missing or incorrect new field")
            if updated.get("value") != 42:
                all_validation_failures.append(f"Document update failed: Lost existing value")
        else:
            all_validation_failures.append("Document update skipped: No valid document key")
            
        # Test 4: Query execution
        total_tests += 1
        query = "FOR doc IN @@collection FILTER doc.name == @name RETURN doc"
        bind_vars = {"@collection": collection_name, "name": "Updated Document"}
        
        query_result = query_documents(query, db, bind_vars)
        
        if not query_result or not isinstance(query_result, list):
            all_validation_failures.append("Query execution failed: No results returned")
        elif len(query_result) != 1:
            all_validation_failures.append(f"Query execution failed: Expected 1 result, got {len(query_result)}")
        elif query_result[0].get("name") != "Updated Document":
            all_validation_failures.append(f"Query execution failed: Expected 'Updated Document', got {query_result[0].get('name')}")
            
        # Test 5: Document deletion
        total_tests += 1
        if doc_key:
            deletion_result = delete_document(collection_name, doc_key, db)
            
            if not deletion_result:
                all_validation_failures.append("Document deletion failed: Operation unsuccessful")
                
            # Verify deletion by trying to retrieve
            try:
                deleted_doc = get_document(collection_name, doc_key, db)
                if deleted_doc:
                    all_validation_failures.append("Document deletion verification failed: Document still exists")
            except DocumentGetError:
                # This is expected
                pass
        else:
            all_validation_failures.append("Document deletion skipped: No valid document key")
            
        # Cleanup test collection
        db.collection(collection_name).truncate()
        
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
        print("CRUD operations are validated and formal tests can now be written")
        sys.exit(0)  # Exit with success code