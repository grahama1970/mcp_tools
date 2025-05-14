"""
Embedding Validation Module for ArangoDB Operations

This module ensures that all documents inserted or updated in ArangoDB have
properly formatted embeddings with the correct dimensions according to the
system configuration.

Functions:
    validate_embedding: Validates embedding field dimensions
    ensure_document_embedding: Ensures a document has proper embedding before insertion
    normalize_embedding: Normalizes embedding vectors for consistency
"""

import os
import sys
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants for embedding configuration
EMBEDDING_FIELD = os.environ.get("EMBEDDING_FIELD", "embedding")
EMBEDDING_DIMENSION = int(os.environ.get("EMBEDDING_DIMENSION", 1024))
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")

def validate_embedding(document: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """
    Validate embedding field in a document.
    
    Args:
        document: Document dictionary to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if embedding field exists
    if EMBEDDING_FIELD not in document:
        return False, f"Missing embedding field '{EMBEDDING_FIELD}'"
    
    embedding = document[EMBEDDING_FIELD]
    
    # Check if embedding is a list/array
    if not isinstance(embedding, (list, np.ndarray)):
        return False, f"Embedding field '{EMBEDDING_FIELD}' must be a list or array"
    
    # Check embedding dimension
    actual_dimension = len(embedding)
    if actual_dimension != EMBEDDING_DIMENSION:
        return False, f"Embedding dimension mismatch: got {actual_dimension}, expected {EMBEDDING_DIMENSION}"
    
    # Check for NaN or infinite values
    if any(not isinstance(x, (int, float)) or np.isnan(x) or np.isinf(x) for x in embedding):
        return False, f"Embedding contains invalid values (NaN or infinite)"
    
    return True, None

def normalize_embedding(embedding: List[float]) -> List[float]:
    """
    Normalize embedding vector to unit length for cosine similarity.
    
    Args:
        embedding: Embedding vector
        
    Returns:
        Normalized embedding vector
    """
    # Convert to numpy array for efficient operations
    embedding_np = np.array(embedding, dtype=np.float32)
    
    # Calculate norm (vector magnitude)
    norm = np.linalg.norm(embedding_np)
    
    # Normalize if norm is not zero
    if norm > 0:
        normalized = embedding_np / norm
        return normalized.tolist()
    else:
        # If norm is zero, return original (can't normalize)
        logger.warning("Cannot normalize zero-magnitude embedding vector")
        return embedding

def ensure_document_embedding(
    document: Dict[str, Any], 
    generate_if_missing: bool = False
) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """
    Ensure document has valid embedding field before insertion/update.
    
    Args:
        document: Document to validate and potentially modify
        generate_if_missing: Whether to generate embeddings if missing
        
    Returns:
        Tuple of (modified_document, is_valid, error_message)
    """
    # Don't modify the original document
    doc = document.copy()
    
    # Check if document has embedding field
    has_embedding = EMBEDDING_FIELD in doc
    
    # If embedding is missing and should be generated
    if not has_embedding and generate_if_missing:
        try:
            # Import embedding generation function (lazy import)
            from complexity.arangodb.embedding_utils import get_embedding
            
            # Determine text to embed based on document fields
            text_to_embed = None
            for field in ['content', 'text', 'question', 'title', 'summary']:
                if field in doc and doc[field]:
                    text_to_embed = doc[field]
                    break
            
            if text_to_embed:
                # Generate embedding
                doc[EMBEDDING_FIELD] = get_embedding(text_to_embed)
                has_embedding = True
                logger.debug(f"Generated embedding for document with field '{field}'")
            else:
                return doc, False, "No suitable text field found for embedding generation"
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return doc, False, f"Embedding generation failed: {str(e)}"
    
    # If document has embedding field, validate it
    if has_embedding:
        is_valid, error_message = validate_embedding(doc)
        
        # If embedding is valid but not normalized, normalize it
        if is_valid:
            doc[EMBEDDING_FIELD] = normalize_embedding(doc[EMBEDDING_FIELD])
            
        return doc, is_valid, error_message
    
    # If we reach here, document doesn't have embedding and we didn't generate one
    return doc, False, f"Missing embedding field '{EMBEDDING_FIELD}'"

def validate_document_batch(documents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Validate a batch of documents and separate into valid and invalid.
    
    Args:
        documents: List of documents to validate
        
    Returns:
        Tuple of (valid_documents, invalid_documents)
    """
    valid_docs = []
    invalid_docs = []
    
    for doc in documents:
        modified_doc, is_valid, error = ensure_document_embedding(doc)
        if is_valid:
            valid_docs.append(modified_doc)
        else:
            invalid_docs.append({
                "document": doc,
                "error": error
            })
    
    return valid_docs, invalid_docs

# Example usage
if __name__ == "__main__":
    # Example document with valid embedding
    valid_doc = {
        "_key": "test123",
        "content": "This is a test document",
        EMBEDDING_FIELD: [0.1] * EMBEDDING_DIMENSION
    }
    
    # Example document with invalid embedding
    invalid_doc = {
        "_key": "test456",
        "content": "This is a test document with wrong embedding",
        EMBEDDING_FIELD: [0.1] * (EMBEDDING_DIMENSION - 10)  # Wrong dimension
    }
    
    # Example document without embedding
    missing_doc = {
        "_key": "test789",
        "content": "This is a test document without embedding"
    }
    
    # Test validation
    for doc in [valid_doc, invalid_doc, missing_doc]:
        modified_doc, is_valid, error = ensure_document_embedding(doc)
        print(f"Document {doc.get('_key')}: {'Valid' if is_valid else 'Invalid'}")
        if error:
            print(f"  Error: {error}")