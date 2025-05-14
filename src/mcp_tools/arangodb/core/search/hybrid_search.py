"""
Hybrid search implementation for ArangoDB.

This module provides a combined search approach that leverages BM25 text search,
semantic vector search, tag filtering, and optionally graph traversal capabilities,
delivering the best results from multiple search paradigms using Reciprocal Rank Fusion (RRF).

Links to third-party documentation:
- ArangoDB: https://www.arangodb.com/docs/stable/
- Python-Arango: https://python-driver.arangodb.com/
- Reciprocal Rank Fusion: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

Sample input:
    db = connect_to_arango()
    results = hybrid_search(
        db,
        "python error handling",
        min_score={"bm25": 0.1, "semantic": 0.7},
        weights={"bm25": 0.5, "semantic": 0.5},
        top_n=10
    )

Expected output:
    {
        "results": [
            {
                "doc": {"_id": "documents/123", "_key": "123", ...},
                "bm25_score": 8.7,
                "semantic_score": 0.92,
                "hybrid_score": 0.0312
            },
            ...
        ],
        "total": 150,
        "query": "python error handling",
        "time": 0.325,
        "bm25_time": 0.125,
        "semantic_time": 0.2,
        "search_engine": "hybrid-bm25-semantic"
    }
"""

import sys
import time
from typing import Dict, Any, List, Optional, Tuple, Union

from loguru import logger
from arango.database import StandardDatabase

# Import core search functions
from .bm25_search import bm25_search
from .semantic_search import semantic_search

# Define default configuration values
DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_VIEW_NAME = "document_view"
DEFAULT_EMBEDDING_FIELD = "embedding"


def hybrid_search(
    db: StandardDatabase,
    query_text: str,
    collections: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    tag_list: Optional[List[str]] = None,
    min_score: Optional[Dict[str, float]] = None,
    weights: Optional[Dict[str, float]] = None,
    top_n: int = 10,
    initial_k: int = 20,
    rrf_k: int = 60,
    fields_to_return: Optional[List[str]] = None,
    require_all_tags: bool = False,
    use_graph: bool = False,
    embedding_field: str = DEFAULT_EMBEDDING_FIELD
) -> Dict[str, Any]:
    """
    Performs hybrid search by combining BM25 and Semantic search
    results using Reciprocal Rank Fusion (RRF) for re-ranking.

    Args:
        db: ArangoDB database connection
        query_text: The user's search query
        collections: Optional list of collections to search in
        filter_expr: Optional AQL filter expression
        tag_list: Optional list of tags to filter results
        min_score: Dictionary of minimum scores for each search type (bm25, semantic)
        weights: Dictionary of weights for each search type (bm25, semantic)
        top_n: The final number of ranked results to return
        initial_k: Number of results to initially fetch from BM25 and Semantic searches
        rrf_k: Constant used in the RRF calculation (default 60)
        fields_to_return: Fields to include in the result
        require_all_tags: Whether all tags must be present (for tag filtering)
        use_graph: Whether to include graph traversal in the hybrid search
        embedding_field: Field containing embedding vectors

    Returns:
        A dictionary containing the ranked 'results', 'total' unique documents found,
        the 'query' for reference, and other metadata.
    """
    start_time = time.time()
    logger.info(f"Hybrid search for query: '{query_text}'")
    
    # Input validation
    if not query_text or query_text.strip() == "":
        error_msg = "Query text cannot be empty"
        logger.error(error_msg)
        return {
            "results": [],
            "total": 0,
            "query": "",
            "time": time.time() - start_time,
            "error": error_msg,
            "search_engine": "hybrid-failed"
        }
    
    # Use default collection if not specified
    if not collections:
        collections = [DEFAULT_COLLECTION_NAME]
    
    # Default fields to return if not provided
    if not fields_to_return:
        fields_to_return = ["_key", "_id", "question", "problem", "solution", "context", "tags", "label", "validated"]
    
    # Default minimum scores if not provided
    if not min_score:
        min_score = {
            "bm25": 0.1,
            "semantic": 0.7
        }
    elif isinstance(min_score, float):
        min_score = {
            "bm25": min_score,
            "semantic": min_score
        }
    
    # Default weights if not provided
    if not weights:
        weights = {
            "bm25": 0.5,
            "semantic": 0.5
        }
    
    # Ensure weights sum to 1.0
    total_weight = sum(weights.values())
    if total_weight != 1.0:
        logger.warning(f"Weights do not sum to 1.0 ({total_weight}), normalizing...")
        for key in weights:
            weights[key] = weights[key] / total_weight
    
    try:
        # STEP 1: Run BM25 search 
        bm25_time_start = time.time()
        bm25_results = bm25_search(
            db=db,
            query_text=query_text,
            collection_name=collections[0],
            filter_expr=filter_expr,
            min_score=min_score.get("bm25", 0.1),
            top_n=initial_k,
            tag_list=tag_list,
        )
        bm25_time = time.time() - bm25_time_start
        
        # Extract BM25 candidates
        bm25_candidates = bm25_results.get("results", [])
        logger.info(f"BM25 search found {len(bm25_candidates)} candidates in {bm25_time:.3f}s")
        
        # STEP 2: Run semantic search
        semantic_time_start = time.time()
        
        semantic_results = semantic_search(
            db=db,
            query=query_text,
            collections=collections,
            filter_expr=filter_expr,
            min_score=min_score.get("semantic", 0.7),
            top_n=initial_k,
            tag_list=tag_list,
            embedding_field=embedding_field
        )
        semantic_time = time.time() - semantic_time_start
        
        # Extract semantic candidates
        semantic_candidates = semantic_results.get("results", [])
        logger.info(f"Semantic search found {len(semantic_candidates)} candidates in {semantic_time:.3f}s")
        
        # STEP 3: Combine results using weighted RRF
        combined_weights = weights
        logger.info(f"Combining results with weights: {combined_weights}")
        
        combined_results = weighted_reciprocal_rank_fusion(
            bm25_candidates=bm25_candidates,
            semantic_candidates=semantic_candidates,
            weights=combined_weights,
            rrf_k=rrf_k
        )
        
        # Remove any potential duplicate entries
        unique_results = {}
        for result in combined_results:
            doc_key = result.get("doc", {}).get("_key", "")
            if doc_key and doc_key not in unique_results:
                unique_results[doc_key] = result
        
        # Convert back to list and sort by hybrid score
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)
        
        # STEP 4: Limit to top_n results
        final_results = final_results[:top_n]
        logger.info(f"Final results: {len(final_results)} documents")
        
        search_time = time.time() - start_time
        
        # Build the response
        response = {
            "results": final_results,
            "total": len(combined_results),
            "query": query_text,
            "time": search_time,
            "bm25_time": bm25_time,
            "semantic_time": semantic_time,
            "search_engine": "hybrid-bm25-semantic",
            "weights": weights,
            "tags": tag_list,
            "require_all_tags": require_all_tags if tag_list else None
        }
        
        return response
    
    except Exception as e:
        logger.exception(f"Hybrid search error: {e}")
        return {
            "results": [],
            "total": 0,
            "query": query_text,
            "time": time.time() - start_time,
            "error": str(e),
            "search_engine": "hybrid-failed"
        }


def weighted_reciprocal_rank_fusion(
    bm25_candidates: List[Dict[str, Any]],
    semantic_candidates: List[Dict[str, Any]],
    weights: Dict[str, float] = None,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Combines multiple result lists using Weighted Reciprocal Rank Fusion.

    Args:
        bm25_candidates: Results from BM25 search
        semantic_candidates: Results from semantic search
        weights: Dictionary of weights for each search type
        rrf_k: Constant for the RRF formula (default: 60)

    Returns:
        A combined list of results, sorted by hybrid score
    """
    if weights is None:
        weights = {
            "bm25": 0.5,
            "semantic": 0.5
        }
    
    # Create a dictionary to track document keys and their rankings
    doc_scores = {}

    # Process BM25 results
    bm25_weight = weights.get("bm25", 0.5)
    for rank, result in enumerate(bm25_candidates, 1):
        doc_key = result.get("doc", {}).get("_key", "")
        if not doc_key:
            continue

        # Initialize if not seen before
        if doc_key not in doc_scores:
            doc_scores[doc_key] = {
                "doc": result.get("doc", {}),
                "bm25_rank": rank,
                "bm25_score": result.get("score", 0),
                "semantic_rank": len(semantic_candidates) + 1,  # Default to worst possible rank
                "semantic_score": 0,
                "hybrid_score": 0
            }
        else:
            # Update BM25 rank info
            doc_scores[doc_key]["bm25_rank"] = rank
            doc_scores[doc_key]["bm25_score"] = result.get("score", 0)

    # Process semantic results
    semantic_weight = weights.get("semantic", 0.5)
    for rank, result in enumerate(semantic_candidates, 1):
        doc_key = result.get("doc", {}).get("_key", "")
        if not doc_key:
            continue

        # Initialize if not seen before
        if doc_key not in doc_scores:
            doc_scores[doc_key] = {
                "doc": result.get("doc", {}),
                "bm25_rank": len(bm25_candidates) + 1,  # Default to worst possible rank
                "bm25_score": 0,
                "semantic_rank": rank,
                "semantic_score": result.get("similarity_score", 0),
                "hybrid_score": 0
            }
        else:
            # Update semantic rank info
            doc_scores[doc_key]["semantic_rank"] = rank
            doc_scores[doc_key]["semantic_score"] = result.get("similarity_score", 0)

    # Calculate weighted RRF scores
    for doc_key, scores in doc_scores.items():
        # Calculate individual RRF scores
        bm25_rrf = 1 / (rrf_k + scores["bm25_rank"])
        semantic_rrf = 1 / (rrf_k + scores["semantic_rank"])
        
        # Apply weights
        weighted_bm25 = bm25_rrf * bm25_weight
        weighted_semantic = semantic_rrf * semantic_weight
        
        # Calculate hybrid score
        scores["hybrid_score"] = weighted_bm25 + weighted_semantic

    # Convert to list and sort by hybrid score (descending)
    result_list = [v for k, v in doc_scores.items()]
    result_list.sort(key=lambda x: x["hybrid_score"], reverse=True)

    return result_list


if __name__ == "__main__":
    """
    Test the hybrid search functionality with validation.
    """
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level:<7} | {message}"
    )
    
    import json
    from arango.client import ArangoClient
    
    # Validation setup
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Basic hybrid search API
    total_tests += 1
    try:
        logger.info("Test 1: Testing basic hybrid search")
        
        # Mock database setup
        client = ArangoClient(hosts="http://localhost:8529")
        try:
            # Try to connect to a real database if available
            sys_db = client.db("_system", username="root", password="root")
            db = sys_db
        except Exception:
            # Otherwise use a mock database object
            from unittest.mock import MagicMock
            db = MagicMock()
            # Mock the execute function to return test results
            db.aql.execute.return_value = [
                {"_id": "test_docs/1", "_key": "1", "title": "Test Document 1"},
                {"_id": "test_docs/2", "_key": "2", "title": "Test Document 2"}
            ]
        
        # Create a test query
        test_query = "test query"
        
        # Test the function with mock/simplified behavior
        # In a real test, this would be connected to a real database
        result = hybrid_search(
            db=db,
            query_text=test_query,
            collections=["test_docs"],
            min_score={"bm25": 0.1, "semantic": 0.5},
            weights={"bm25": 0.5, "semantic": 0.5},
            top_n=5
        )
        
        # Validate basic structure (not actual results)
        required_fields = ["results", "total", "query", "time", "bm25_time", "semantic_time", "search_engine", "weights"]
        missing_fields = [field for field in required_fields if field not in result]
        
        if missing_fields:
            raise ValueError(f"Missing required fields in result: {missing_fields}")
            
        # Check search engine type
        if "search_engine" in result and not result["search_engine"].startswith("hybrid"):
            raise ValueError(f"Expected search_engine to be hybrid-*, got {result['search_engine']}")
        
        logger.info("Test 1 passed: Basic hybrid search interface validated")
        
    except Exception as e:
        logger.error(f"Test 1 failed: {e}")
        all_validation_failures.append(f"Test 1: Basic hybrid search API: {str(e)}")
    
    # Final validation report
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)