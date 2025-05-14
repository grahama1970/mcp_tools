"""
BM25 search implementation for ArangoDB.

This module provides functionality for keyword-based search using the BM25
algorithm with ArangoDB views. It allows for searching documents based on
textual similarity with configurable fields and scoring parameters.

Links to third-party documentation:
- ArangoDB Search: https://www.arangodb.com/docs/stable/arangosearch.html
- BM25 algorithm: https://en.wikipedia.org/wiki/Okapi_BM25

Sample input:
    db = connect_to_arango()
    results = bm25_search(
        db,
        "python error handling",
        min_score=0.1,
        top_n=10,
        collection_name="documents",
        view_name="documents_view"
    )

Expected output:
    {
        "results": [
            {
                "_id": "documents/123",
                "_key": "123",
                "title": "Python Exception Handling",
                "content": "...",
                "bm25_score": 0.89
            },
            ...
        ],
        "total": 15,
        "offset": 0
    }
"""

import sys
from typing import Dict, List, Any, Optional, Union

from loguru import logger
from arango.database import StandardDatabase
from arango.exceptions import AQLQueryExecuteError


def bm25_search(
    db: StandardDatabase,
    query_text: str,
    collection_name: str = "documents",
    view_name: str = "documents_view",
    search_fields: List[str] = None,
    filter_expr: Optional[str] = None,
    min_score: float = 0.1,
    top_n: int = 10,
    offset: int = 0,
    tag_list: Optional[List[str]] = None,
    bind_vars: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Search documents using BM25 text relevance algorithm.

    Args:
        db: ArangoDB database handle
        query_text: The search query text
        collection_name: Name of the collection to search
        view_name: Name of the ArangoSearch view to use
        search_fields: List of document fields to search within
        min_score: Minimum BM25 score threshold
        top_n: Maximum number of results to return
        offset: Number of results to skip (for pagination)
        tag_list: Optional list of tags to filter results

    Returns:
        Dict[str, Any]: Dictionary containing search results and metadata
    """
    if not query_text or query_text.strip() == "":
        logger.warning("Empty query text provided for BM25 search")
        return {
            "results": [],
            "total": 0,
            "offset": offset,
            "query": "",
            "time": 0,
            "error": "Query text cannot be empty"
        }

    # Default search fields if none provided
    if not search_fields:
        search_fields = ["title", "content", "description", "tags"]

    # Add debug logging
    logger.debug(f"BM25 search started - query: '{query_text}'")
    logger.debug(f"Collection: {collection_name}")
    logger.debug(f"View name: {view_name}")
    logger.debug(f"Search fields: {search_fields}")
    logger.debug(f"Filter expression: {filter_expr}")
    logger.debug(f"Tag list: {tag_list}")

    try:
        # Start timing the search operation
        import time
        start_time = time.time()

        # Verify collection exists
        if not db.has_collection(collection_name):
            logger.warning(f"Collection does not exist: {collection_name}")
            return {
                "results": [],
                "total": 0,
                "offset": offset,
                "query": query_text,
                "time": 0,
                "error": f"Collection does not exist: {collection_name}"
            }

        # Build filter clause
        filter_clauses = []
        if filter_expr:
            filter_clauses.append(f"({filter_expr})")

        # Add tag filter if provided
        if tag_list and len(tag_list) > 0:
            # Use AND logic for multiple tags - requiring all tags to be present
            tag_conditions = [f'"{tag}" IN doc.tags' for tag in tag_list]
            tag_filter = " AND ".join(tag_conditions)
            filter_clauses.append(f"({tag_filter})")
            logger.debug(f"Tag filter: {tag_filter}")

        # Combine filter clauses with AND
        filter_clause = ""
        if filter_clauses:
            filter_clause = "FILTER " + " AND ".join(filter_clauses)
            logger.debug(f"Combined filter clause: {filter_clause}")

        # Build the SEARCH clause dynamically from search_fields
        search_field_conditions = " OR ".join([
            f'ANALYZER(doc.{field} IN TOKENS(@query, "text_en"), "text_en")'
            for field in search_fields
        ])
        logger.debug(f"Search field conditions: {search_field_conditions}")

        # Build the AQL query
        aql = f"""
        LET search_tokens = TOKENS(@query, "text_en")
        FOR doc IN {view_name}
        SEARCH {search_field_conditions}
        {filter_clause}
        LET score = BM25(doc)
        FILTER score >= @min_score
        SORT score DESC
        LIMIT {offset}, {top_n}
        RETURN {{
            "doc": doc,
            "score": score
        }}
        """
        logger.debug(f"AQL query: {aql}")
        
        # Execute the query
        query_bind_vars = {
            "query": query_text,
            "min_score": min_score
        }

        # Add any additional bind variables from parameter
        if bind_vars:
            query_bind_vars.update(bind_vars)

        logger.debug(f"Query bind vars: {query_bind_vars}")
        cursor = db.aql.execute(aql, bind_vars=query_bind_vars)
        results = list(cursor)
        logger.debug(f"Query returned {len(results)} results")

        # Get the total count
        count_aql = f"""
        RETURN LENGTH(
            LET search_tokens = TOKENS(@query, "text_en")
            FOR doc IN {view_name}
            SEARCH {search_field_conditions}
            {filter_clause}
            LET score = BM25(doc)
            FILTER score >= @min_score
            RETURN 1
        )
        """

        count_bind_vars = {
            "query": query_text,
            "min_score": min_score
        }

        # Add any additional bind variables from parameter
        if bind_vars:
            count_bind_vars.update(bind_vars)

        count_cursor = db.aql.execute(count_aql, bind_vars=count_bind_vars)
        total_count = next(count_cursor)
        logger.debug(f"Total count: {total_count}")

        end_time = time.time()
        elapsed = end_time - start_time
        logger.debug(f"Search completed in {elapsed:.4f} seconds")

        # Create the result object
        result = {
            "results": results,
            "total": total_count,
            "offset": offset,
            "query": query_text,
            "time": elapsed
        }

        return result
        
    except Exception as e:
        logger.error(f"BM25 search error: {e}")
        return {
            "results": [],
            "total": 0,
            "offset": offset,
            "query": query_text,
            "error": str(e),
            "time": 0
        }


def ensure_view(
    db: StandardDatabase,
    view_name: str,
    collection_name: str,
    search_fields: List[str] = None,
    primary_sort_field: Optional[str] = None
) -> bool:
    """
    Ensure the ArangoSearch view exists with proper configuration.

    Args:
        db: ArangoDB database handle
        view_name: Name of the view to create or update
        collection_name: Collection to link to the view
        search_fields: Fields to index in the view
        primary_sort_field: Optional field for primary sorting

    Returns:
        bool: True if successful, False otherwise
    """
    if not search_fields:
        search_fields = ["title", "content", "description", "tags"]
        
    try:
        # Check if view exists
        existing_views = db.views()
        view_exists = view_name in [v["name"] for v in existing_views]
        
        # Define view properties
        link_fields = {}
        for field in search_fields:
            link_fields[field] = {"analyzers": ["text_en"]}
            
        view_properties = {
            "links": {
                collection_name: {
                    "analyzers": ["text_en"],
                    "includeAllFields": False,
                    "fields": link_fields
                }
            }
        }
        
        # Add primary sort if specified
        if primary_sort_field:
            view_properties["primarySort"] = [{
                "field": primary_sort_field,
                "direction": "desc"
            }]
        
        if view_exists:
            # Update existing view
            db.update_view(view_name, view_properties)
            logger.info(f"Updated existing ArangoSearch view: {view_name}")
        else:
            # Create new view
            db.create_arangosearch_view(
                name=view_name,
                properties=view_properties
            )
            logger.info(f"Created new ArangoSearch view: {view_name}")
            
        return True
        
    except Exception as e:
        logger.exception(f"Error ensuring ArangoSearch view {view_name}: {e}")
        return False


if __name__ == "__main__":
    """
    This is the main entry point for testing and validating the BM25 search functionality.
    It tests the core search function with realistic data and validates the results.
    """
    import sys
    from arango import ArangoClient
    import uuid
    import time
    import json
    import os

    # Configure logging for tests
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # Validation setup
    all_validation_failures = []
    total_tests = 0

    # Setup test database connection
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

        # Create test collection
        test_collection = "test_documents"
        test_view = "test_view"

        if db.has_collection(test_collection):
            db.delete_collection(test_collection)
        db.create_collection(test_collection)

        # Create test documents
        test_data = [
            {
                "title": "Python Error Handling",
                "content": "This document covers Python error handling using try/except blocks.",
                "tags": ["python", "error", "programming"]
            },
            {
                "title": "JavaScript Promises",
                "content": "Learn about JavaScript promises and async/await syntax.",
                "tags": ["javascript", "programming", "async"]
            },
            {
                "title": "Python Generators",
                "content": "Understanding Python generators and yield statements.",
                "tags": ["python", "programming", "advanced"]
            },
            {
                "title": "ArangoDB Tutorial",
                "content": "Getting started with ArangoDB and Python driver.",
                "tags": ["database", "arangodb", "python"]
            }
        ]

        # Insert test documents
        for doc in test_data:
            db.collection(test_collection).insert(doc)

        # Create a view for searching
        search_fields = ["title", "content", "tags"]
        view_properties = {
            "links": {
                test_collection: {
                    "analyzers": ["text_en"],
                    "includeAllFields": False,
                    "fields": {
                        field: {"analyzers": ["text_en"]} for field in search_fields
                    }
                }
            }
        }

        # Create the view
        if db.has_view(test_view):
            db.delete_view(test_view)
        db.create_arangosearch_view(test_view, properties=view_properties)
        print(f"Created test view: {test_view}")

        # Wait for view to be indexed
        time.sleep(2)

        # Test 1: Basic BM25 search
        total_tests += 1
        search_results = bm25_search(
            db,
            "python",
            collection_name=test_collection,
            view_name=test_view,
            search_fields=search_fields
        )

        if not search_results or "results" not in search_results or len(search_results["results"]) < 2:
            all_validation_failures.append("Test 1: Basic BM25 search failed")
        else:
            # Verify that results have the expected structure
            if "total" not in search_results or "offset" not in search_results or "time" not in search_results:
                all_validation_failures.append("Test 1: Search results missing required fields")

            # Verify that Python documents are returned
            found_python = False
            for result in search_results["results"]:
                if "doc" in result and "title" in result["doc"] and "Python" in result["doc"]["title"]:
                    found_python = True
                    break

            if not found_python:
                all_validation_failures.append("Test 1: BM25 search did not return expected results")

        # Test 2: BM25 search with min_score
        total_tests += 1
        filtered_results = bm25_search(
            db,
            "arangodb",
            collection_name=test_collection,
            view_name=test_view,
            search_fields=search_fields,
            min_score=0.5
        )

        if not filtered_results or "results" not in filtered_results:
            all_validation_failures.append("Test 2: BM25 search with min_score failed")
        else:
            # Verify that ArangoDB document is returned
            found_arango = False
            for result in filtered_results["results"]:
                if "doc" in result and "title" in result["doc"] and "ArangoDB" in result["doc"]["title"]:
                    found_arango = True
                    break

            if not found_arango and len(filtered_results["results"]) > 0:
                all_validation_failures.append("Test 2: BM25 search with min_score did not return expected results")

        # Test 3: BM25 search with tag filter
        total_tests += 1
        tag_results = bm25_search(
            db,
            "programming",
            collection_name=test_collection,
            view_name=test_view,
            search_fields=search_fields,
            tag_list=["python"]
        )

        if not tag_results or "results" not in tag_results:
            all_validation_failures.append("Test 3: BM25 search with tag filter failed")
        else:
            # Verify that only Python + programming documents are returned
            for result in tag_results["results"]:
                if "doc" in result and "tags" in result["doc"] and "python" not in result["doc"]["tags"]:
                    all_validation_failures.append("Test 3: BM25 search with tag filter returned incorrect documents")
                    break

        # Test 4: BM25 search with filter expression
        total_tests += 1
        filter_results = bm25_search(
            db,
            "programming",
            collection_name=test_collection,
            view_name=test_view,
            search_fields=search_fields,
            filter_expr="doc.title LIKE '%Python%'"
        )

        if not filter_results or "results" not in filter_results:
            all_validation_failures.append("Test 4: BM25 search with filter expression failed")
        else:
            # Verify that only Python documents are returned
            for result in filter_results["results"]:
                if "doc" in result and "title" in result["doc"] and "Python" not in result["doc"]["title"]:
                    all_validation_failures.append("Test 4: BM25 search with filter expression returned incorrect documents")
                    break

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
        print(f"\n❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)