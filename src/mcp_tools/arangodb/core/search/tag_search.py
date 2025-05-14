"""
# Tag-Based Search Module Core Implementation

This module provides core functionality for searching documents by tags in ArangoDB.
It allows filtering by one or more tags with options for requiring all tags or any tag.

## Features:
- Flexible tag-based filtering (ANY or ALL tags)
- Dynamic field selection in results
- Configurable result limit and offset for pagination
- Optional additional filter expressions

## Third-Party Packages:
- python-arango: https://python-driver.arangodb.com/ (v3.10.0)
- loguru: https://github.com/Delgan/loguru (v0.7.2)

## Sample Input:
```python
db = connect_to_arango()
tags = ["python", "error-handling"]
require_all_tags = True
limit = 10
fields_to_return = ["problem", "solution", "context"]
results = tag_search(db, tags, require_all_tags=require_all_tags, limit=limit, 
                    fields_to_return=fields_to_return)
```

## Expected Output:
```python
{
  "results": [
    {
      "doc": {
        "_key": "doc1",
        "problem": "Python error when processing JSON data",
        "solution": "Use try/except blocks to handle JSON parsing exceptions",
        "context": "Error handling in data processing",
        "tags": ["python", "json", "error-handling"]
      },
      "collection": "complexity",
      "tag_match_score": 1.0
    }
  ],
  "total": 1,
  "offset": 0,
  "limit": 10,
  "tags": ["python", "error-handling"],
  "require_all_tags": true,
  "time": 0.023,
  "search_engine": "tag-search",
  "search_type": "tag"
}
```
"""
import time
from typing import Dict, Any, List, Optional, Union, Set
from loguru import logger
from arango.database import StandardDatabase
from arango.exceptions import AQLQueryExecuteError, ArangoServerError


def tag_search(
    db: StandardDatabase,
    tags: List[str],
    collections: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    require_all_tags: bool = False,
    limit: int = 10,
    offset: int = 0,
    fields_to_return: Optional[List[str]] = None,
    bind_vars: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Search for documents by tags, returning only pure data without any presentation formatting.
    
    Args:
        db: ArangoDB database
        tags: List of tags to search for
        collections: Optional list of collections to search, defaults to first collection if not specified
        filter_expr: Optional AQL filter expression
        require_all_tags: Whether all tags must be present (True for AND logic, False for OR logic)
        limit: Maximum number of results to return
        offset: Result offset for pagination
        fields_to_return: List of fields to return in results (defaults to basic fields)
        bind_vars: Optional additional bind variables for the query
        
    Returns:
        Dictionary with search results, including tag_match_score for each result
    """
    start_time = time.time()
    logger.info(f"Searching for documents with tags: {tags}")
    
    try:
        # Validate input
        if not tags:
            return {
                "results": [],
                "total": 0,
                "offset": offset,
                "limit": limit,
                "tags": [],
                "require_all_tags": require_all_tags,
                "time": 0,
                "search_engine": "tag-search",
                "search_type": "tag",
                "error": "No tags provided for search"
            }
        
        # Use default collection if not specified
        if not collections or not collections[0]:
            logger.warning("No collection specified for tag search, using fallback 'complexity'")
            collections = ["complexity"]
            
        # Default fields to return if not provided
        if not fields_to_return:
            fields_to_return = ["problem", "solution", "context", "question"]
        
        # Build filter clause based on tags
        tag_conditions = []
        
        for i, tag in enumerate(tags):
            tag_conditions.append(f'@tag_{i} IN doc.tags')
        
        # Create tag filter
        tag_filter = f"FILTER {(' AND ' if require_all_tags else ' OR ').join(tag_conditions)}"
        
        # Add additional filter if provided
        if filter_expr:
            tag_filter += f" AND ({filter_expr})"
            
        # Create a list of all fields to keep
        fields_to_keep = ["_key", "_id", "tags"] + fields_to_return
        fields_to_keep = list(set(fields_to_keep))  # Remove duplicates
        
        # Convert to comma-separated string for KEEP
        fields_to_keep_str = '", "'.join(fields_to_keep)
        fields_to_keep_str = f'"{fields_to_keep_str}"'
        
        # Build the AQL query
        aql = f"""
        FOR doc IN {collections[0]}
        {tag_filter}
        SORT doc._key
        LIMIT {offset}, {limit}
        RETURN {{
            "doc": KEEP(doc, {fields_to_keep_str}),
            "collection": "{collections[0]}"
        }}
        """
        
        logger.debug(f"Executing AQL query: {aql}")
        
        # Create bind variables for tags
        tag_vars = {f"tag_{i}": tag for i, tag in enumerate(tags)}
        
        # Add any custom bind variables provided
        if bind_vars:
            tag_vars.update(bind_vars)
        
        logger.debug(f"With bind variables: {tag_vars}")
        
        # Execute the query
        cursor = db.aql.execute(aql, bind_vars=tag_vars)
        raw_results = list(cursor)
        
        # Compute tag_match_score for each result
        results = []
        for result in raw_results:
            doc = result.get("doc", {})
            doc_tags = doc.get("tags", [])
            # Count matching tags (case-insensitive comparison)
            matched_tags = sum(1 for tag in tags if tag.lower() in [t.lower() for t in doc_tags])
            total_tags = len(tags) if tags else 1  # Avoid division by zero
            tag_match_score = matched_tags / total_tags if total_tags > 0 else 0.0
            results.append({
                "doc": doc,
                "collection": result.get("collection", collections[0]),
                "tag_match_score": tag_match_score
            })
        
        logger.info(f"Found {len(results)} documents matching the tag criteria")
        
        # Determine total count
        if offset == 0 and len(results) < limit:
            total_count = len(results)
            logger.debug(f"Using result length as total count: {total_count}")
        else:
            logger.debug("Executing count query to determine total matches")
            count_aql = f"""
            RETURN LENGTH(
                FOR doc IN {collections[0]}
                {tag_filter}
                RETURN 1
            )
            """
            count_cursor = db.aql.execute(count_aql, bind_vars=tag_vars)
            total_count = next(count_cursor)
            logger.debug(f"Count query returned: {total_count}")
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        return {
            "results": results,
            "total": total_count,
            "offset": offset,
            "limit": limit,
            "tags": tags,
            "require_all_tags": require_all_tags,
            "time": elapsed,
            "search_engine": "tag-search",
            "search_type": "tag",
            "fields_to_return": fields_to_return
        }
    
    except Exception as e:
        logger.error(f"Tag search error: {e}")
        return {
            "results": [],
            "total": 0,
            "offset": offset,
            "limit": limit,
            "tags": tags,
            "require_all_tags": require_all_tags,
            "error": str(e),
            "time": time.time() - start_time,
            "search_engine": "tag-search-failed",
            "search_type": "tag",
            "fields_to_return": fields_to_return or ["problem", "solution", "context"]
        }


def filter_by_tags(
    db: StandardDatabase,
    tags: List[str],
    collections: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    require_all_tags: bool = False,
    limit: Optional[int] = None,
    fields_to_return: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Helper function to get document IDs that match tag criteria.
    Used for pre-filtering in hybrid search.
    
    Args:
        db: ArangoDB database
        tags: List of tags to filter by
        collections: Collections to search in
        filter_expr: Additional filter expression
        require_all_tags: Whether all tags must match
        limit: Optional limit on results
        fields_to_return: Fields to include (minimal for efficiency)
        
    Returns:
        Dictionary with tag filtered results and document IDs
    """
    logger.info(f"Pre-filtering by tags: {tags} (require_all_tags={require_all_tags})")
    
    # Use a higher limit for pre-filtering to ensure enough documents after filtering
    actual_limit = limit * 5 if limit else 1000
    
    # Use minimal fields for efficiency in pre-filtering
    minimal_fields = fields_to_return or ["_id", "_key"]
    
    results = tag_search(
        db=db,
        tags=tags,
        collections=collections,
        filter_expr=filter_expr,
        require_all_tags=require_all_tags,
        limit=actual_limit,
        fields_to_return=minimal_fields
    )
    
    # Extract document IDs for filtering
    doc_ids = {r["doc"]["_id"] for r in results.get("results", [])}
    
    return {
        "results": results.get("results", []),
        "doc_ids": doc_ids,
        "total": results.get("total", 0)
    }


def create_tag_filter_expression(
    doc_ids: Set[str], 
    existing_filter: Optional[str] = None
) -> str:
    """
    Create an AQL filter expression to filter by document IDs from tag search.
    
    Args:
        doc_ids: Set of document IDs to include
        existing_filter: Optional existing filter expression to combine with
        
    Returns:
        AQL filter expression
    """
    if not doc_ids:
        return existing_filter or ""
    
    # Create a filter that only includes documents in our tag-filtered set
    id_list_str = ", ".join([f"'{doc_id}'" for doc_id in doc_ids])
    tag_filter = f"doc._id IN [{id_list_str}]"
    
    # Combine with existing filter expression if needed
    if existing_filter:
        return f"({existing_filter}) AND {tag_filter}"
    else:
        return tag_filter


def validate_tag_search_result(result: Dict[str, Any]) -> bool:
    """
    Validate the structure of a tag search result.
    
    Args:
        result: Tag search result to validate
        
    Returns:
        True if the result has the expected structure, False otherwise
    """
    required_fields = ["results", "total", "offset", "limit", "tags", "search_engine", "search_type"]
    
    # Check for required fields
    for field in required_fields:
        if field not in result:
            logger.error(f"Tag search result missing required field: {field}")
            return False
    
    # Check result structure
    if not isinstance(result["results"], list):
        logger.error("Tag search results should be a list")
        return False
    
    # Check for error field
    if "error" in result and result["results"]:
        logger.warning(f"Tag search result has error field but also has results: {result['error']}")
    
    return True


if __name__ == "__main__":
    import sys
    from arango import ArangoClient
    
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {level:<7} | {message}"
    )
    
    # Track validation failures
    all_validation_failures = []
    total_tests = 0
    
    try:
        # Test 1: Basic functionality - connection and setup
        total_tests += 1
        logger.info("TEST 1: Setting up database connection")
        
        try:
            # Connect to ArangoDB
            client = ArangoClient(hosts="http://localhost:8529")
            sys_db = client.db("_system")
            
            # Check if test database exists
            db_name = "tag_search_test"
            if not sys_db.has_database(db_name):
                sys_db.create_database(db_name)
                
            # Connect to test database
            db = client.db(db_name)
            
            # Check if test collection exists
            collection_name = "test_docs"
            if db.has_collection(collection_name):
                collection = db.collection(collection_name)
            else:
                collection = db.create_collection(collection_name)
                
            # Insert test documents if collection is empty
            if collection.count() == 0:
                test_docs = [
                    {
                        "_key": "doc1",
                        "title": "Python Error Handling",
                        "content": "How to handle errors in Python",
                        "tags": ["python", "error-handling", "exceptions"]
                    },
                    {
                        "_key": "doc2",
                        "title": "JavaScript Basics",
                        "content": "Introduction to JavaScript",
                        "tags": ["javascript", "web", "programming"]
                    },
                    {
                        "_key": "doc3",
                        "title": "Python Web Frameworks",
                        "content": "Comparison of Python web frameworks",
                        "tags": ["python", "web", "frameworks"]
                    }
                ]
                collection.import_bulk(test_docs)
            
            logger.info("Database setup completed successfully")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            all_validation_failures.append(f"Test 1 (Database setup): {e}")
            # Continue with other tests without the database
        
        # Test 2: Basic tag search functionality with a mock database
        total_tests += 1
        logger.info("TEST 2: Mock tag search test")
        
        class MockCursor:
            def __init__(self, data):
                self.data = data
                
            def __iter__(self):
                return iter(self.data)
                
            def __next__(self):
                if not self.data:
                    raise StopIteration
                return self.data.pop(0)
        
        class MockDatabase:
            class AQL:
                def execute(self, query, bind_vars=None):
                    if "RETURN LENGTH" in query:
                        return MockCursor([3])
                    else:
                        return MockCursor([
                            {"doc": {"_key": "doc1", "tags": ["python", "error-handling"]}, "collection": "test_docs"},
                            {"doc": {"_key": "doc2", "tags": ["javascript", "web"]}, "collection": "test_docs"},
                            {"doc": {"_key": "doc3", "tags": ["python", "web"]}, "collection": "test_docs"}
                        ])
            
            def __init__(self):
                self.aql = self.AQL()
        
        mock_db = MockDatabase()
        result = tag_search(mock_db, ["python", "web"], require_all_tags=False)
        
        expected_result_keys = set(["doc1", "doc2", "doc3"])
        actual_result_keys = set(r["doc"]["_key"] for r in result["results"])
        
        if not expected_result_keys.issubset(actual_result_keys):
            missing_keys = expected_result_keys - actual_result_keys
            all_validation_failures.append(f"Test 2 (Mock tag search): Missing expected keys: {missing_keys}")
            
        # Validate result structure
        if not validate_tag_search_result(result):
            all_validation_failures.append("Test 2 (Mock tag search): Invalid result structure")
            
        # Check tag match scores
        for r in result["results"]:
            if "tag_match_score" not in r:
                all_validation_failures.append(f"Test 2 (Mock tag search): Missing tag_match_score in result {r['doc']['_key']}")
        
        logger.info(f"Mock tag search found {len(result['results'])} results")
        
        # Test 3: Tag filter expression creation
        total_tests += 1
        logger.info("TEST 3: Tag filter expression creation")
        
        doc_ids = {"collection/doc1", "collection/doc2"}
        filter_expr = create_tag_filter_expression(doc_ids)
        expected_expr = "doc._id IN ['collection/doc1', 'collection/doc2']"
        
        if filter_expr != expected_expr:
            all_validation_failures.append(f"Test 3 (Filter expression): Expected {expected_expr}, got {filter_expr}")
            
        # With existing filter
        filter_expr = create_tag_filter_expression(doc_ids, "doc.active == true")
        expected_expr = "(doc.active == true) AND doc._id IN ['collection/doc1', 'collection/doc2']"
        
        if filter_expr != expected_expr:
            all_validation_failures.append(f"Test 3 (Filter expression with existing): Expected {expected_expr}, got {filter_expr}")
        
        logger.info("Filter expression tests completed")
        
        # Test 4: Empty tags handling
        total_tests += 1
        logger.info("TEST 4: Empty tags handling")
        
        result = tag_search(mock_db, [])
        
        if result.get("results"):
            all_validation_failures.append(f"Test 4 (Empty tags): Expected empty results, got {len(result['results'])} results")
            
        if "error" not in result:
            all_validation_failures.append("Test 4 (Empty tags): Expected error field for empty tags")
            
        logger.info("Empty tags test completed")
        
        # Test 5: Validation function
        total_tests += 1
        logger.info("TEST 5: Result validation function")
        
        valid_result = {
            "results": [],
            "total": 0,
            "offset": 0,
            "limit": 10,
            "tags": ["python"],
            "require_all_tags": False,
            "time": 0.1,
            "search_engine": "tag-search",
            "search_type": "tag"
        }
        
        invalid_result = {
            "results": [],
            # Missing required fields
            "tags": ["python"],
            "search_engine": "tag-search"
        }
        
        if not validate_tag_search_result(valid_result):
            all_validation_failures.append("Test 5 (Validation): Failed to validate valid result")
            
        if validate_tag_search_result(invalid_result):
            all_validation_failures.append("Test 5 (Validation): Incorrectly validated invalid result")
            
        logger.info("Validation function tests completed")
        
        # Test 6: Real database test (if available)
        if 'db' in locals() and isinstance(db, StandardDatabase):
            total_tests += 1
            logger.info("TEST 6: Real database tag search")
            
            try:
                result = tag_search(
                    db=db,
                    tags=["python"],
                    collections=[collection_name],
                    limit=10
                )
                
                if result.get("error"):
                    all_validation_failures.append(f"Test 6 (Real DB): Error in search: {result['error']}")
                    
                # Should find at least 2 documents with tag "python"
                if len(result["results"]) < 2:
                    all_validation_failures.append(f"Test 6 (Real DB): Expected at least 2 results for 'python' tag, got {len(result['results'])}")
                    
                logger.info(f"Real DB search found {len(result['results'])} results")
                
                # Test with require_all_tags=True
                result = tag_search(
                    db=db,
                    tags=["python", "error-handling"],
                    collections=[collection_name],
                    require_all_tags=True,
                    limit=10
                )
                
                # Should find exactly 1 document with both tags
                if len(result["results"]) != 1:
                    all_validation_failures.append(f"Test 6 (Real DB): Expected 1 result for 'python' AND 'error-handling' tags, got {len(result['results'])}")
                
                logger.info(f"Real DB search with require_all_tags=True found {len(result['results'])} results")
                
            except Exception as e:
                logger.error(f"Real database test failed: {e}")
                all_validation_failures.append(f"Test 6 (Real DB): {e}")
        
        # Final validation result
        if all_validation_failures:
            print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
            for failure in all_validation_failures:
                print(f"  - {failure}")
            sys.exit(1)
        else:
            print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
            print("Tag search core module is validated and ready for use")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Validation failed with unexpected error: {e}")
        print(f"❌ VALIDATION FAILED - Unexpected error: {e}")
        sys.exit(1)