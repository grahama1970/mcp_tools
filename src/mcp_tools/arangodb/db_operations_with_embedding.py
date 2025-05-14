"""
Database Operations Module with Embedding Validation

This module extends the basic CRUD operations with embedding validation
to ensure all documents have properly formatted embeddings before being
inserted or updated in the database.

Functions:
    create_document: Create a document with embedding validation
    create_documents_batch: Create multiple documents with embedding validation
    update_document: Update a document with embedding validation
    upsert_document: Upsert a document with embedding validation
    replace_document: Replace a document with embedding validation
"""

import os
import sys
from typing import Dict, Any, List, Optional, Tuple
from arango.database import StandardDatabase
from arango.exceptions import DocumentInsertError, DocumentUpdateError
from loguru import logger

# Import embedding validation
from arangodb.utils.embedding_validator import (
        ensure_document_embedding,
        validate_embedding,
        normalize_embedding,
        EMBEDDING_FIELD,
        EMBEDDING_DIMENSION
)

def create_document(
    db: StandardDatabase,
    collection_name: str,
    document: Dict[str, Any],
    validate_embedding_field: bool = True,
    generate_embedding: bool = False,
    skip_if_invalid: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Create a document with embedding validation.
    
    Args:
        db: ArangoDB database instance
        collection_name: Name of the collection
        document: Document to create
        validate_embedding_field: Whether to validate embedding field
        generate_embedding: Whether to generate embedding if missing
        skip_if_invalid: Whether to skip insertion if embedding is invalid
        
    Returns:
        Created document or None if creation failed
    """
    # Get collection
    collection = db.collection(collection_name)
    
    try:
        # If embedding validation is enabled
        if validate_embedding_field:
            # Ensure document has valid embedding
            modified_doc, is_valid, error = ensure_document_embedding(
                document, 
                generate_if_missing=generate_embedding
            )
            
            # If embedding is invalid and we shouldn't skip
            if not is_valid and not skip_if_invalid:
                logger.warning(f"Document has invalid embedding: {error}")
                raise ValueError(f"Invalid embedding: {error}")
            
            # If embedding is invalid but we should skip
            if not is_valid and skip_if_invalid:
                logger.warning(f"Skipping document with invalid embedding: {error}")
                return None
            
            # Insert the modified document
            result = collection.insert(modified_doc)
            return {**modified_doc, "_id": result["_id"], "_rev": result["_rev"]}
        else:
            # Insert without validation
            result = collection.insert(document)
            return {**document, "_id": result["_id"], "_rev": result["_rev"]}
            
    except DocumentInsertError as e:
        logger.error(f"Failed to insert document: {e}")
        return None
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error creating document: {e}")
        return None

def create_documents_batch(
    db: StandardDatabase,
    collection_name: str,
    documents: List[Dict[str, Any]],
    validate_embedding_field: bool = True,
    generate_embedding: bool = False,
    skip_if_invalid: bool = True
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create multiple documents with embedding validation.
    
    Args:
        db: ArangoDB database instance
        collection_name: Name of the collection
        documents: List of documents to create
        validate_embedding_field: Whether to validate embedding field
        generate_embedding: Whether to generate embedding if missing
        skip_if_invalid: Whether to skip documents with invalid embeddings
        
    Returns:
        Tuple of (created_documents, failed_documents)
    """
    # Get collection
    collection = db.collection(collection_name)
    
    # Initialize result lists
    created_docs = []
    failed_docs = []
    
    # Prepare validated documents for insertion
    valid_docs = []
    
    try:
        # If embedding validation is enabled
        if validate_embedding_field:
            for doc in documents:
                # Ensure document has valid embedding
                modified_doc, is_valid, error = ensure_document_embedding(
                    doc, 
                    generate_if_missing=generate_embedding
                )
                
                # If embedding is valid, add to batch
                if is_valid:
                    valid_docs.append(modified_doc)
                # If embedding is invalid but we should skip
                elif skip_if_invalid:
                    logger.warning(f"Skipping document with invalid embedding: {error}")
                    failed_docs.append({"document": doc, "error": error})
                # If embedding is invalid and we shouldn't skip
                else:
                    logger.error(f"Document has invalid embedding: {error}")
                    failed_docs.append({"document": doc, "error": error})
                    raise ValueError(f"Invalid embedding: {error}")
        else:
            # No validation, use documents as-is
            valid_docs = documents
        
        # Insert documents in batch
        if valid_docs:
            results = collection.insert_many(valid_docs)
            # Combine original documents with result IDs and revisions
            for doc, result in zip(valid_docs, results):
                if "_id" in result and "_rev" in result:
                    created_docs.append({**doc, "_id": result["_id"], "_rev": result["_rev"]})
                    
    except Exception as e:
        logger.exception(f"Error in batch document creation: {e}")
        # Add remaining documents to failed list
        for doc in valid_docs:
            if doc not in [d["document"] for d in failed_docs]:
                failed_docs.append({"document": doc, "error": str(e)})
    
    return created_docs, failed_docs

def update_document(
    db: StandardDatabase,
    collection_name: str,
    document_key: str,
    update_data: Dict[str, Any],
    validate_embedding_field: bool = True,
    keep_existing_embedding: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Update a document with embedding validation.
    
    Args:
        db: ArangoDB database instance
        collection_name: Name of the collection
        document_key: Key of the document to update
        update_data: Data to update
        validate_embedding_field: Whether to validate embedding field
        keep_existing_embedding: Whether to keep existing embedding if present
        
    Returns:
        Updated document or None if update failed
    """
    # Get collection
    collection = db.collection(collection_name)
    
    try:
        # First, get the existing document
        existing_doc = collection.get(document_key)
        if not existing_doc:
            logger.error(f"Document {document_key} not found for update")
            return None
        
        # If embedding validation is enabled
        if validate_embedding_field:
            # If update_data contains embedding field
            if EMBEDDING_FIELD in update_data:
                # Validate the new embedding
                temp_doc = {EMBEDDING_FIELD: update_data[EMBEDDING_FIELD]}
                is_valid, error = validate_embedding(temp_doc)
                
                if not is_valid:
                    logger.error(f"Invalid embedding in update data: {error}")
                    raise ValueError(f"Invalid embedding: {error}")
                
                # Normalize the embedding
                update_data[EMBEDDING_FIELD] = normalize_embedding(update_data[EMBEDDING_FIELD])
            
            # If keep_existing_embedding is True and document has embedding
            elif keep_existing_embedding and EMBEDDING_FIELD in existing_doc:
                # Keep the existing embedding
                update_data[EMBEDDING_FIELD] = existing_doc[EMBEDDING_FIELD]
        
        # Update the document
        result = collection.update(document_key, update_data)
        
        # Get the updated document
        updated_doc = collection.get(document_key)
        return updated_doc
        
    except DocumentUpdateError as e:
        logger.error(f"Failed to update document: {e}")
        return None
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error updating document: {e}")
        return None

def upsert_document(
    db: StandardDatabase,
    collection_name: str,
    search_key: str,
    search_value: Any,
    document: Dict[str, Any],
    validate_embedding_field: bool = True,
    generate_embedding: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Upsert a document with embedding validation.
    
    Args:
        db: ArangoDB database instance
        collection_name: Name of the collection
        search_key: Key to search for existing document
        search_value: Value to search for existing document
        document: Document data
        validate_embedding_field: Whether to validate embedding field
        generate_embedding: Whether to generate embedding if missing
        
    Returns:
        Upserted document or None if operation failed
    """
    # Get collection
    collection = db.collection(collection_name)
    
    try:
        # First, check if document exists
        query = f"""
        FOR doc IN {collection_name}
        FILTER doc.{search_key} == @value
        LIMIT 1
        RETURN doc
        """
        cursor = db.aql.execute(query, bind_vars={"value": search_value})
        existing_docs = list(cursor)
        
        # If document exists, update it
        if existing_docs:
            existing_doc = existing_docs[0]
            doc_key = existing_doc["_key"]
            
            # If validation is enabled, ensure document has valid embedding
            if validate_embedding_field:
                modified_doc, is_valid, error = ensure_document_embedding(
                    document,
                    generate_if_missing=generate_embedding
                )
                
                if not is_valid:
                    logger.error(f"Document has invalid embedding: {error}")
                    raise ValueError(f"Invalid embedding: {error}")
                
                # Update with modified document
                result = collection.update(doc_key, modified_doc)
            else:
                # Update without validation
                result = collection.update(doc_key, document)
            
            # Get the updated document
            updated_doc = collection.get(doc_key)
            return updated_doc
            
        # If document doesn't exist, create it
        else:
            # If validation is enabled, ensure document has valid embedding
            if validate_embedding_field:
                modified_doc, is_valid, error = ensure_document_embedding(
                    document,
                    generate_if_missing=generate_embedding
                )
                
                if not is_valid:
                    logger.error(f"Document has invalid embedding: {error}")
                    raise ValueError(f"Invalid embedding: {error}")
                
                # Insert with modified document
                result = collection.insert(modified_doc)
                return {**modified_doc, "_id": result["_id"], "_rev": result["_rev"]}
            else:
                # Insert without validation
                result = collection.insert(document)
                return {**document, "_id": result["_id"], "_rev": result["_rev"]}
            
    except Exception as e:
        logger.exception(f"Error in upsert operation: {e}")
        return None

def replace_document(
    db: StandardDatabase,
    collection_name: str,
    document_key: str,
    document: Dict[str, Any],
    validate_embedding_field: bool = True,
    generate_embedding: bool = True,
    keep_existing_embedding: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Replace a document with embedding validation.
    
    Args:
        db: ArangoDB database instance
        collection_name: Name of the collection
        document_key: Key of the document to replace
        document: New document data
        validate_embedding_field: Whether to validate embedding field
        generate_embedding: Whether to generate embedding if missing
        keep_existing_embedding: Whether to keep existing embedding
        
    Returns:
        Replaced document or None if operation failed
    """
    # Get collection
    collection = db.collection(collection_name)
    
    try:
        # First, get the existing document for embedding preservation
        existing_doc = collection.get(document_key)
        if not existing_doc:
            logger.error(f"Document {document_key} not found for replacement")
            return None
        
        # If validation is enabled
        if validate_embedding_field:
            # If document has embedding field
            if EMBEDDING_FIELD in document:
                # Validate the embedding
                is_valid, error = validate_embedding(document)
                
                if not is_valid:
                    logger.error(f"Document has invalid embedding: {error}")
                    raise ValueError(f"Invalid embedding: {error}")
                
                # Normalize the embedding
                document[EMBEDDING_FIELD] = normalize_embedding(document[EMBEDDING_FIELD])
                
            # If document doesn't have embedding field but we should keep existing
            elif keep_existing_embedding and EMBEDDING_FIELD in existing_doc:
                # Keep the existing embedding
                document[EMBEDDING_FIELD] = existing_doc[EMBEDDING_FIELD]
                
            # If document doesn't have embedding field and we should generate
            elif generate_embedding and EMBEDDING_FIELD not in document:
                # Generate embedding
                modified_doc, is_valid, error = ensure_document_embedding(
                    document,
                    generate_if_missing=True
                )
                
                if not is_valid:
                    logger.error(f"Failed to generate embedding: {error}")
                    raise ValueError(f"Failed to generate embedding: {error}")
                
                document = modified_doc
        
        # Replace the document
        result = collection.replace(document_key, document)
        
        # Get the replaced document
        replaced_doc = collection.get(document_key)
        return replaced_doc
        
    except Exception as e:
        logger.exception(f"Error in document replacement: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Simple integration test
    from complexity.arangodb.arango_setup import connect_arango, ensure_database
    
    # Connect to database
    client = connect_arango()
    db = ensure_database(client)
    
    # Example document
    test_doc = {
        "title": "Test Document",
        "content": "This is a test document for embedding validation",
        "tags": ["test", "embedding", "validation"]
    }
    
    # Create document with embedding generation
    created_doc = create_document(
        db=db,
        collection_name="test_docs",
        document=test_doc,
        validate_embedding_field=True,
        generate_embedding=True
    )
    
    if created_doc:
        print(f"✅ Document created with key: {created_doc.get('_key')}")
        print(f"✅ Embedding dimension: {len(created_doc.get(EMBEDDING_FIELD, []))}")
    else:
        print("❌ Failed to create document")