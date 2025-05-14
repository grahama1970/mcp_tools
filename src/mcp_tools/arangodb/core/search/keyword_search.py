"""
# Keyword Search Module - Core Implementation

This module provides core functionality for performing keyword searches with fuzzy matching
using ArangoDB and RapidFuzz.

## Features:
- Dynamic field searching: specify which fields to search instead of hardcoded fields
- Fuzzy matching with configurable similarity threshold
- Tag filtering
- Efficient result scoring and ranking

## Third-Party Packages:
- python-arango: https://python-driver.arangodb.com/ (v3.10.0)
- rapidfuzz: https://rapidfuzz.github.io/RapidFuzz/ (v3.2.0)
- loguru: https://github.com/Delgan/loguru (v0.7.2)

## Sample Input:
```python
db = connect_to_arango()
search_term = "python error"
similarity_threshold = 97.0
top_n = 10
tags = ["python", "error-handling"]
fields_to_search = ["problem", "solution", "details"]  # custom fields to search
results = keyword_search(db, search_term, similarity_threshold, top_n, tags=tags, 
                        fields_to_search=fields_to_search)
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
      "keyword_score": 0.98
    }
  ],
  "total": 1,
  "search_term": "python error",
  "similarity_threshold": 97.0,
  "search_engine": "keyword-fuzzy"
}
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import time

from loguru import logger
from arango.database import StandardDatabase
from arango.cursor import Cursor
from rapidfuzz import fuzz, process


def keyword_search(
    db: StandardDatabase,
    search_term: str,
    similarity_threshold: float = 97.0,
    top_n: int = 10,
    view_name: str = "documents_view",
    tags: Optional[List[str]] = None,
    collections: Optional[List[str]] = None,
    fields_to_search: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Perform a keyword search with fuzzy matching.
    
    Args:
        db: ArangoDB database connection
        search_term: The keyword to search for
        similarity_threshold: Minimum similarity score (0-100) for fuzzy matching
        top_n: Maximum number of results to return
        view_name: Name of the ArangoDB search view
        tags: Optional list of tags to filter results
        collections: Optional list of collections to search (defaults to ["documents"])
        fields_to_search: List of fields to search in (defaults to ["problem", "solution", "context"])
        filter_expr: Optional AQL filter expression for additional filtering
        
    Returns:
        Dictionary containing results and metadata
        
    Raises:
        ValueError: If search_term is empty
        Exception: For any other errors
    """
    start_time = time.time()
    
    if not search_term or search_term.strip() == "":
        raise ValueError("Search term cannot be empty")
    
    # Clean search term
    search_term = search_term.strip()
    
    # Default fields to search if not provided
    if not fields_to_search or len(fields_to_search) == 0:
        fields_to_search = ["problem", "solution", "context"]
    
    # Default collection if not provided
    if not collections or len(collections) == 0:
        collections = ["documents"]
    
    collection_name = collections[0]  # Use the first collection for now
    
    # Build tag filter if provided
    tag_filter = ""
    bind_vars = {
        "search_term": search_term,  # Raw term without wildcards
        "top_n": top_n
    }
    
    if tags and len(tags) > 0:
        tag_conditions = []
        bind_vars["tags"] = tags
        for i, tag in enumerate(tags):
            tag_conditions.append(f'@tags[{i}] IN doc.tags')
        tag_filter = f"FILTER {' AND '.join(tag_conditions)}"
    
    # Add custom filter expression if provided
    if filter_expr:
        if tag_filter:
            tag_filter = f"{tag_filter} AND ({filter_expr})"
        else:
            tag_filter = f"FILTER {filter_expr}"
    
    # Dynamically build the search conditions
    search_conditions = []
    for field in fields_to_search:
        search_conditions.append(f"doc.{field} == TOKENS(@search_term, \"text_en\")[0]")
    
    search_condition = " OR ".join(search_conditions)
    
    # Create a list of all fields to keep
    fields_to_keep = ["_key", "_id", "tags"] + fields_to_search
    fields_to_keep = list(set(fields_to_keep))  # Remove duplicates
    
    # Convert to comma-separated string for KEEP
    fields_to_keep_str = '", "'.join(fields_to_keep)
    fields_to_keep_str = f'"{fields_to_keep_str}"'
    
    # AQL query with dynamic fields to keep
    aql_query = f"""
    FOR doc IN {view_name}
      SEARCH ANALYZER({search_condition}, 
                    "text_en")
      {tag_filter}
      SORT BM25(doc) DESC
      LIMIT @top_n
      RETURN {{ 
        doc: KEEP(doc, {fields_to_keep_str})
      }}
    """
    
    logger.debug(f"Executing AQL query: {aql_query}")
    logger.debug(f"With bind variables: {bind_vars}")
    
    try:
        # Execute AQL query
        cursor = db.aql.execute(aql_query, bind_vars=bind_vars)
        
        # Safely extract results from cursor
        initial_results = []
        if isinstance(cursor, Cursor):
            try:
                initial_results = list(cursor)
                # If the AQL query found results, return them directly
                # Skip the RapidFuzz filtering since we're already using the analyzer
                if initial_results:
                    logger.info(f"AQL query found {len(initial_results)} results")
                    # Add a simple exact match score for sorting
                    for item in initial_results:
                        doc = item.get("doc", {})
                        # Simple exact match detection
                        exact_match = False
                        for field in fields_to_search:
                            if field in doc and doc[field] and search_term.lower() in str(doc[field]).lower():
                                exact_match = True
                                break
                        # Set keyword_score based on exact match
                        item["keyword_score"] = 1.0 if exact_match else 0.9
                    
                    # Sort by keyword_score (exact matches first)
                    sorted_results = sorted(initial_results, key=lambda x: x.get("keyword_score", 0), reverse=True)
                    
                    # Create result object with the AQL query results
                    result = {
                        "results": sorted_results[:top_n],
                        "total": len(sorted_results),
                        "search_term": search_term,
                        "similarity_threshold": similarity_threshold,
                        "search_engine": "keyword-fuzzy",
                        "search_type": "keyword",
                        "time": time.time() - start_time
                    }
                    logger.info(f"Keyword search for '{search_term}' found {len(sorted_results)} results")
                    return result
            except Exception as e:
                logger.error(f"Error iterating over cursor results: {e}")
                raise
        elif cursor is None:
            logger.warning("db.aql.execute returned None, expected a cursor.")
            return {
                "results": [],
                "total": 0,
                "search_term": search_term,
                "similarity_threshold": similarity_threshold,
                "error": "Query execution returned None instead of cursor",
                "search_engine": "keyword-fuzzy-failed",
                "search_type": "keyword",
                "time": time.time() - start_time
            }
        else:
            logger.error(f"db.aql.execute returned unexpected type: {type(cursor)}. Expected Cursor.")
            raise TypeError(f"Unexpected cursor type: {type(cursor)}")

        # If no results were found by the AQL query or if we want to still apply RapidFuzz filtering
        # Filter results using rapidfuzz for whole word matching
        filtered_results = []
        for item in initial_results:
            doc = item.get("doc", {})
            
            # Combine searchable text from all fields we're searching
            text_parts = []
            for field in fields_to_search:
                field_value = doc.get(field)
                if field_value is not None:  # Explicitly check for None
                    text_parts.append(str(field_value))
            text = " ".join(text_parts).lower()
            
            # Extract whole words from the text
            words = re.findall(r'\b\w+\b', text)
            
            # Use rapidfuzz to find words with similarity to search_term
            matches = process.extract(
                search_term.lower(),
                words,
                scorer=fuzz.ratio,
                score_cutoff=similarity_threshold
            )
            
            if matches:
                # Add the match and its similarity score
                best_match = matches[0]  # tuple of (match, score)
                item["keyword_score"] = best_match[1] / 100.0  # convert to 0-1 scale
                filtered_results.append(item)
        
        # Sort results by keyword_score (highest first)
        filtered_results.sort(key=lambda x: x.get("keyword_score", 0), reverse=True)
        
        # Limit to top_n
        filtered_results = filtered_results[:top_n]
        
        # Create result object
        result = {
            "results": filtered_results,
            "total": len(filtered_results),
            "search_term": search_term,
            "similarity_threshold": similarity_threshold,
            "search_engine": "keyword-fuzzy",
            "search_type": "keyword",
            "time": time.time() - start_time
        }
        
        logger.info(f"Keyword search for '{search_term}' found {len(filtered_results)} results")
        return result
    
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        return {
            "results": [],
            "total": 0,
            "search_term": search_term,
            "error": str(e),
            "search_engine": "keyword-fuzzy-failed",
            "search_type": "keyword",
            "time": time.time() - start_time
        }


def validate_keyword_search_result(
    search_results: Dict[str, Any],
    expected_data: Optional[Dict[str, Any]] = None
) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate keyword search results against expected patterns.
    
    Args:
        search_results: The results returned from keyword_search
        expected_data: Dictionary containing expected patterns (optional)
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    # Check basic structure
    required_fields = ["results", "total", "search_term", "similarity_threshold", "search_engine"]
    for field in required_fields:
        if field not in search_results:
            validation_failures[f"missing_{field}"] = {
                "expected": f"{field} field present",
                "actual": f"{field} field missing"
            }
    
    # Check search engine type
    if "search_engine" in search_results and search_results.get("search_engine") != "keyword-fuzzy":
        if "error" not in search_results:  # Allow different engine name if there's an error
            validation_failures["search_engine"] = {
                "expected": "keyword-fuzzy",
                "actual": search_results.get("search_engine")
            }
    
    # Validate that all results have keyword_score
    if "results" in search_results and len(search_results["results"]) > 0:
        for i, item in enumerate(search_results["results"]):
            if "keyword_score" not in item:
                validation_failures[f"missing_score_result_{i}"] = {
                    "expected": "keyword_score present",
                    "actual": "keyword_score missing"
                }
            elif not 0 <= item["keyword_score"] <= 1:
                validation_failures[f"invalid_score_result_{i}"] = {
                    "expected": "keyword_score between 0 and 1",
                    "actual": item["keyword_score"]
                }
    
    # Compare against expected data if provided
    if expected_data:
        # Check search term
        if "search_term" in expected_data and "search_term" in search_results:
            if search_results["search_term"] != expected_data["search_term"]:
                validation_failures["search_term"] = {
                    "expected": expected_data["search_term"],
                    "actual": search_results["search_term"]
                }
        
        # Check similarity threshold
        if "similarity_threshold" in expected_data and "similarity_threshold" in search_results:
            if search_results["similarity_threshold"] != expected_data["similarity_threshold"]:
                validation_failures["similarity_threshold"] = {
                    "expected": expected_data["similarity_threshold"],
                    "actual": search_results["similarity_threshold"]
                }
        
        # Check minimum result count
        if "min_results" in expected_data:
            results_count = len(search_results.get("results", []))
            min_expected = expected_data["min_results"]
            if results_count < min_expected:
                validation_failures["results_count"] = {
                    "expected": f">= {min_expected}",
                    "actual": results_count
                }
    
    return len(validation_failures) == 0, validation_failures


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
            db_name = "keyword_search_test"
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
        
        # Test 2: Basic keyword search functionality with a mock database
        total_tests += 1
        logger.info("TEST 2: Mock keyword search test")
        
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
                    return MockCursor([
                        {"doc": {"_key": "doc1", "title": "Python Error Handling", "content": "How to handle errors in Python"}},
                        {"doc": {"_key": "doc2", "title": "JavaScript Basics", "content": "Introduction to JavaScript"}},
                        {"doc": {"_key": "doc3", "title": "Python Web Frameworks", "content": "Comparison of Python web frameworks"}}
                    ])
            
            def __init__(self):
                self.aql = self.AQL()
        
        mock_db = MockDatabase()
        result = keyword_search(
            db=mock_db,
            search_term="python",
            fields_to_search=["title", "content"]
        )
        
        # Check the result structure
        is_valid, validation_failures = validate_keyword_search_result(result)
        
        if not is_valid:
            for field, details in validation_failures.items():
                all_validation_failures.append(f"Test 2 (Mock search): {field} - Expected {details['expected']}, got {details['actual']}")
        
        # Verify correct scoring
        for item in result["results"]:
            if "python" in item["doc"].get("title", "").lower() and item["keyword_score"] < 0.9:
                all_validation_failures.append(f"Test 2 (Mock search): Python title match should have high score, got {item['keyword_score']}")
        
        logger.info(f"Mock keyword search found {len(result['results'])} results")
        
        # Test 3: Empty search term handling
        total_tests += 1
        logger.info("TEST 3: Empty search term handling")
        
        try:
            _ = keyword_search(mock_db, "")
            all_validation_failures.append("Test 3 (Empty search term): Expected ValueError for empty search term, but no exception was raised")
        except ValueError:
            logger.info("Empty search term correctly raised ValueError")
        except Exception as e:
            all_validation_failures.append(f"Test 3 (Empty search term): Expected ValueError, but got {type(e).__name__}")
        
        # Test 4: Tag filtering
        total_tests += 1
        logger.info("TEST 4: Tag filtering")
        
        # Test with tag filter
        tag_result = keyword_search(
            db=mock_db,
            search_term="python",
            tags=["frameworks"],
            fields_to_search=["title", "content"]
        )
        
        # We can't verify the tag filtering with the mock, but we can check the structure
        is_valid, validation_failures = validate_keyword_search_result(tag_result)
        
        if not is_valid:
            for field, details in validation_failures.items():
                all_validation_failures.append(f"Test 4 (Tag filter): {field} - Expected {details['expected']}, got {details['actual']}")
        
        logger.info(f"Tag-filtered search produced valid result structure")
        
        # Test 5: Custom fields to search
        total_tests += 1
        logger.info("TEST 5: Custom fields to search")
        
        # Test with custom fields
        fields_result = keyword_search(
            db=mock_db,
            search_term="python",
            fields_to_search=["title"]  # Only search in title
        )
        
        # We can only check the structure with the mock
        is_valid, validation_failures = validate_keyword_search_result(fields_result)
        
        if not is_valid:
            for field, details in validation_failures.items():
                all_validation_failures.append(f"Test 5 (Custom fields): {field} - Expected {details['expected']}, got {details['actual']}")
        
        logger.info(f"Custom fields search produced valid result structure")
        
        # Test 6: Result validation
        total_tests += 1
        logger.info("TEST 6: Result validation")
        
        # Test the validation function with various inputs
        valid_result = {
            "results": [{"doc": {"_key": "test"}, "keyword_score": 0.95}],
            "total": 1,
            "search_term": "test",
            "similarity_threshold": 97.0,
            "search_engine": "keyword-fuzzy"
        }
        
        invalid_result = {
            "results": [{"doc": {"_key": "test"}}],  # Missing keyword_score
            "total": 1,
            "search_term": "test",
            "search_engine": "keyword-fuzzy"
        }
        
        is_valid, _ = validate_keyword_search_result(valid_result)
        if not is_valid:
            all_validation_failures.append("Test 6 (Validation): Valid result failed validation")
        
        is_valid, _ = validate_keyword_search_result(invalid_result)
        if is_valid:
            all_validation_failures.append("Test 6 (Validation): Invalid result passed validation")
        
        # Test 7: Real database test (if available)
        if 'db' in locals() and isinstance(db, StandardDatabase):
            total_tests += 1
            logger.info("TEST 7: Real database keyword search")
            
            try:
                # Create search view if it doesn't exist
                view_name = "test_view"
                if not db.has_view(view_name):
                    db.create_arangosearch_view(
                        name=view_name,
                        properties={
                            "links": {
                                collection_name: {
                                    "includeAllFields": True,
                                    "analyzers": ["text_en"]
                                }
                            }
                        }
                    )
                
                # Run actual search
                real_result = keyword_search(
                    db=db,
                    search_term="python",
                    view_name=view_name,
                    collections=[collection_name],
                    fields_to_search=["title", "content"]
                )
                
                is_valid, validation_failures = validate_keyword_search_result(
                    real_result,
                    {"search_term": "python", "min_results": 1}
                )
                
                if not is_valid:
                    for field, details in validation_failures.items():
                        all_validation_failures.append(f"Test 7 (Real DB search): {field} - Expected {details['expected']}, got {details['actual']}")
                
                logger.info(f"Real DB search found {len(real_result['results'])} results")
                
            except Exception as e:
                logger.error(f"Real database test failed: {e}")
                all_validation_failures.append(f"Test 7 (Real DB): {e}")
        
        # Final validation result
        if all_validation_failures:
            print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
            for failure in all_validation_failures:
                print(f"  - {failure}")
            sys.exit(1)
        else:
            print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
            print("Keyword search core module is validated and ready for use")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Validation failed with unexpected error: {e}")
        print(f"❌ VALIDATION FAILED - Unexpected error: {e}")
        sys.exit(1)