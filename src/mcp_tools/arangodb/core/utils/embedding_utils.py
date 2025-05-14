"""
Embedding utilities for ArangoDB operations.

This module provides functions for generating and working with vector embeddings,
including generating embeddings from text and calculating similarity between embeddings.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
from loguru import logger

# Try to import embedding model, with graceful fallback
try:
    from sentence_transformers import SentenceTransformer
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    HAS_EMBEDDING_MODEL = True
except ImportError:
    logger.warning("SentenceTransformer not available - embedding functionality will be limited")
    embedding_model = None
    HAS_EMBEDDING_MODEL = False


def get_embedding(text: str) -> Optional[List[float]]:
    """
    Generate an embedding vector for the given text.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding vector, or None if embedding failed
    """
    if not text or not text.strip():
        return None
        
    if not HAS_EMBEDDING_MODEL:
        logger.warning("Embedding model not available - returning None")
        return None
        
    try:
        # Generate embedding
        embedding = embedding_model.encode(text)
        
        # Convert to list of floats (for JSON serialization)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None


def calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    if not embedding1 or not embedding2:
        return 0.0
        
    try:
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0


if __name__ == "__main__":
    """
    Validation function for embedding_utils.
    """
    import sys
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Get embedding
    total_tests += 1
    try:
        embedding = get_embedding("This is a test text")
        
        if HAS_EMBEDDING_MODEL:
            if embedding is None:
                all_validation_failures.append("get_embedding returned None despite model being available")
            elif not isinstance(embedding, list):
                all_validation_failures.append(f"get_embedding returned {type(embedding)} instead of list")
            elif len(embedding) < 10:  # Embeddings should have reasonable dimension
                all_validation_failures.append(f"get_embedding returned very short embedding: {len(embedding)} dimensions")
        else:
            # If model not available, should return None
            if embedding is not None:
                all_validation_failures.append("get_embedding returned non-None despite model not being available")
    except Exception as e:
        all_validation_failures.append(f"get_embedding raised exception: {e}")
    
    # Test 2: Empty text
    total_tests += 1
    try:
        empty_embedding = get_embedding("")
        if empty_embedding is not None:
            all_validation_failures.append("get_embedding returned non-None for empty text")
    except Exception as e:
        all_validation_failures.append(f"get_embedding with empty text raised exception: {e}")
    
    # Test 3: Calculate cosine similarity
    total_tests += 1
    if HAS_EMBEDDING_MODEL:
        try:
            # Create two similar texts
            text1 = "This is a sentence about machine learning."
            text2 = "This text discusses artificial intelligence and machine learning."
            
            # Generate embeddings
            emb1 = get_embedding(text1)
            emb2 = get_embedding(text2)
            
            # Calculate similarity
            similarity = calculate_cosine_similarity(emb1, emb2)
            
            # Check similarity is between 0 and 1
            if similarity < 0 or similarity > 1:
                all_validation_failures.append(f"calculate_cosine_similarity returned invalid value: {similarity}")
                
            # Similar texts should have reasonable similarity
            if similarity < 0.5:
                all_validation_failures.append(f"calculate_cosine_similarity for similar texts too low: {similarity}")
                
        except Exception as e:
            all_validation_failures.append(f"calculate_cosine_similarity raised exception: {e}")
    
    # Test 4: Cosine similarity with empty embeddings
    total_tests += 1
    try:
        # Calculate similarity with empty embeddings
        similarity = calculate_cosine_similarity([], [])
        
        if similarity != 0.0:
            all_validation_failures.append(f"calculate_cosine_similarity for empty embeddings returned {similarity} instead of 0.0")
    except Exception as e:
        all_validation_failures.append(f"calculate_cosine_similarity with empty embeddings raised exception: {e}")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        sys.exit(0)  # Exit with success code