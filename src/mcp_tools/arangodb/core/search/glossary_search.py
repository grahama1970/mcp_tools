"""
# Glossary Search Module - Core Implementation

This module provides core functionality for managing and searching glossary terms in ArangoDB.
It supports adding, updating, and retrieving glossary terms, as well as finding terms within text.

## Features:
- Term management (add, update, get)
- Fast term lookup using in-memory cache
- Term matching in text with overlap prevention
- Term highlighting capabilities
- Bulk operations for improved performance

## Third-Party Packages:
- python-arango: https://python-driver.arangodb.com/ (v3.10.0)
- loguru: https://github.com/Delgan/loguru (v0.7.2)

## Sample Input:
```python
# Initialize and add terms
glossary = GlossaryManager(db, "glossary")
glossary.add_term("primary color", "One of the three colors (red, blue, yellow)")

# Find terms in text
matches = glossary.find_terms_in_text("What are primary colors and RGB?")
```

## Expected Output:
```python
# Match results
[
    {
        "term": "primary color",
        "definition": "One of the three colors (red, blue, yellow)",
        "positions": [(9, 22)]
    },
    {
        "term": "RGB",
        "definition": "Color model that uses red, green, and blue light",
        "positions": [(27, 30)]
    }
]
```
"""

import re
import time
from typing import Dict, List, Any, Optional, Set, Tuple, Union
import json

from loguru import logger
from arango.database import StandardDatabase
from arango.collection import StandardCollection
from arango.exceptions import CollectionCreateError, DocumentInsertError


class GlossaryManager:
    """Core manager for glossary terms in ArangoDB."""
    
    def __init__(self, db: StandardDatabase, collection_name: str = "glossary"):
        """
        Initialize the glossary manager.
        
        Args:
            db: ArangoDB database connection
            collection_name: Name of the glossary collection
        """
        self.db = db
        self.collection_name = collection_name
        self.collection = None
        self._terms_cache = {}  # Cache of terms for faster lookups
        self._terms_by_length = []  # Terms sorted by length (descending)
    
    def initialize_collection(self, truncate: bool = False) -> StandardCollection:
        """
        Initialize the glossary collection in ArangoDB.
        Creates the collection if it doesn't exist.
        
        Args:
            truncate: If True, truncate the collection if it exists
        
        Returns:
            ArangoDB collection object
        """
        collection_exists = any(c['name'] == self.collection_name for c in self.db.collections())
        
        if collection_exists:
            self.collection = self.db.collection(self.collection_name)
            logger.info(f"Using existing glossary collection: {self.collection_name}")
            
            if truncate:
                try:
                    self.collection.truncate()
                    logger.info(f"Truncated glossary collection: {self.collection_name}")
                except Exception as e:
                    logger.error(f"Error truncating collection: {e}")
                    raise
        else:
            try:
                self.collection = self.db.create_collection(self.collection_name)
                logger.info(f"Created glossary collection: {self.collection_name}")
                
                # Create index for fast lookups by term
                self.collection.add_hash_index(["term"], unique=True)
                logger.info("Created index on term field")
            except CollectionCreateError as e:
                logger.error(f"Error creating glossary collection: {e}")
                raise
        
        self._refresh_cache()
        
        return self.collection
    
    def add_term(self, term: str, definition: str) -> bool:
        """
        Add a term to the glossary collection.
        
        Args:
            term: The glossary term to add
            definition: The definition of the term
            
        Returns:
            True if added successfully, False otherwise
        """
        if not self.collection:
            try:
                self.initialize_collection()
            except Exception as e:
                logger.error(f"Failed to initialize collection: {e}")
                return False
        
        try:
            doc = {
                "term": term,
                "term_lower": term.lower(),
                "definition": definition,
                "length": len(term)
            }
            
            aql = f"""
            FOR doc IN {self.collection_name}
            FILTER doc.term_lower == @term_lower
            RETURN doc
            """
            
            cursor = self.db.aql.execute(
                aql,
                bind_vars={
                    "term_lower": term.lower()
                }
            )
            
            existing_term = next(cursor, None)
            
            if existing_term:
                if existing_term["definition"] != definition:
                    self.collection.update(
                        existing_term["_key"],
                        {"definition": definition}
                    )
                    logger.info(f"Updated term: {term}")
                    
                    self._terms_cache[term.lower()] = {
                        "term": term,
                        "definition": definition
                    }
                    self._rebuild_sorted_terms()
                    
                    return True
                else:
                    return True
            else:
                self.collection.insert(doc)
                logger.info(f"Added term: {term}")
                
                self._terms_cache[term.lower()] = {
                    "term": term,
                    "definition": definition
                }
                self._rebuild_sorted_terms()
                
                return True
        except Exception as e:
            logger.error(f"Error adding term '{term}': {e}")
            return False
    
    def add_terms(self, terms_dict: Dict[str, str]) -> int:
        """
        Add multiple terms to the glossary collection.
        
        Args:
            terms_dict: Dictionary mapping terms to definitions
            
        Returns:
            Number of terms added successfully
        """
        if not terms_dict:
            return 0
            
        count = 0
        for term, definition in terms_dict.items():
            if self.add_term(term, definition):
                count += 1
        
        return count
    
    def add_terms_bulk(self, terms_dict: Dict[str, str]) -> int:
        """
        Add multiple terms to the glossary collection using bulk insert.
        
        Args:
            terms_dict: Dictionary mapping terms to definitions
            
        Returns:
            Number of terms added or updated successfully
        """
        if not terms_dict:
            return 0
            
        if not self.collection:
            try:
                self.initialize_collection()
            except Exception as e:
                logger.error(f"Failed to initialize collection: {e}")
                return 0
        
        try:
            documents = [
                {
                    "term": term,
                    "term_lower": term.lower(),
                    "definition": definition,
                    "length": len(term)
                }
                for term, definition in terms_dict.items()
            ]
            
            result = self.collection.insert_many(
                documents,
                overwrite_mode="update",
                return_new=True,
                return_old=True
            )
            
            for doc in result:
                if "new" in doc:
                    term_lower = doc["new"]["term_lower"]
                    self._terms_cache[term_lower] = {
                        "term": doc["new"]["term"],
                        "definition": doc["new"]["definition"]
                    }
                elif "old" in doc:
                    term_lower = doc["old"]["term_lower"]
                    self._terms_cache[term_lower] = {
                        "term": doc["old"]["term"],
                        "definition": doc["old"]["definition"]
                    }
            
            self._rebuild_sorted_terms()
            
            logger.info(f"Bulk inserted/updated {len(result)} terms")
            return len(result)
            
        except DocumentInsertError as e:
            logger.error(f"Error during bulk insert: {e}")
            return 0
        except Exception as e:
            logger.error(f"Unexpected error during bulk insert: {e}")
            return 0
    
    def add_default_terms(self) -> int:
        """
        Add default color-related glossary terms to the collection.
        
        Returns:
            Number of terms added
        """
        DEFAULT_GLOSSARY = {
            "primary color": "One of the three colors (red, blue, yellow) that cannot be created by mixing other colors",
            "secondary color": "A color made by mixing two primary colors",
            "tertiary color": "A color made by mixing a primary color with a secondary color",
            "cool color": "Colors that evoke a sense of calm such as blue, green, and purple",
            "warm color": "Colors that evoke a sense of warmth such as red, orange, and yellow",
            "RGB": "Color model that uses red, green, and blue light to create colors on electronic displays",
            "hue": "The pure color attribute; what we normally think of as 'color'",
            "saturation": "The intensity or purity of a color",
            "value": "The lightness or darkness of a color",
            "azure": "A bright, cyan-blue color named after the blue mineral azurite",
            "ruby": "A deep red color named after the ruby gemstone",
            "color": "The visual perceptual property corresponding to the categories called red, green, blue, etc.",
            "hexadecimal color": "A color expressed as a six-digit combination of numbers and letters, used in HTML and CSS"
        }
        
        return self.add_terms_bulk(DEFAULT_GLOSSARY)
    
    def get_all_terms(self) -> List[Dict[str, str]]:
        """
        Get all terms from the glossary collection.
        
        Returns:
            List of dictionaries containing term and definition
        """
        if not self.collection:
            try:
                self.initialize_collection()
            except Exception as e:
                logger.error(f"Failed to initialize collection: {e}")
                return []
        
        try:
            aql = """
            FOR doc IN @@collection
            SORT doc.term_lower
            RETURN {
                "term": doc.term,
                "definition": doc.definition
            }
            """
            
            cursor = self.db.aql.execute(
                aql,
                bind_vars={
                    "@collection": self.collection_name
                }
            )
            
            terms = [doc for doc in cursor]
            return terms
            
        except Exception as e:
            logger.error(f"Error getting all terms: {e}")
            return []
    
    def find_terms_in_text(self, text: str, include_positions: bool = False) -> List[Dict[str, Any]]:
        """
        Find glossary terms in the provided text.
        
        Args:
            text: The text to search for glossary terms
            include_positions: If True, include position information for each match
            
        Returns:
            List of dictionaries containing matched terms and definitions
        """
        if not text:
            return []
            
        if not self._terms_cache:
            self._refresh_cache()
            
        if not self._terms_cache:
            return []
        
        try:
            # Prevent memory issues with large text
            if len(text) > 10000:
                text = text[:10000]
            
            text_lower = text.lower()
            matched_terms = []
            matched_positions = set()
            position_info = {}
            
            # Process terms longest to shortest to prioritize multi-word terms
            for term_lower, term_info in self._terms_by_length:
                pattern = r'\b' + re.escape(term_lower) + r'\b'
                for match in re.finditer(pattern, text_lower):
                    start, end = match.span()
                    overlap = False
                    
                    # Check for overlap with already matched terms
                    for pos in range(start, end):
                        if pos in matched_positions:
                            overlap = True
                            break
                    
                    if not overlap:
                        # Add term to results
                        term_data = {
                            "term": term_info["term"],
                            "definition": term_info["definition"]
                        }
                        
                        # Track positions if requested
                        if include_positions:
                            if term_info["term"] not in position_info:
                                position_info[term_info["term"]] = []
                            position_info[term_info["term"]].append((start, end))
                        
                        # If this term isn't already in results, add it
                        if not any(term["term"] == term_info["term"] for term in matched_terms):
                            matched_terms.append(term_data)
                        
                        # Mark all positions as matched to prevent overlap
                        for pos in range(start, end):
                            matched_positions.add(pos)
            
            # Sort results alphabetically
            matched_terms.sort(key=lambda x: x["term"].lower())
            
            # Add position information if requested
            if include_positions:
                for term in matched_terms:
                    term["positions"] = position_info.get(term["term"], [])
            
            return matched_terms
            
        except Exception as e:
            logger.error(f"Error finding terms in text: {e}")
            return []
    
    def highlight_terms(self, text: str, marker_start: str = "**", marker_end: str = "**") -> str:
        """
        Highlight glossary terms in the provided text.
        
        Args:
            text: The text to process
            marker_start: String to insert before matched terms
            marker_end: String to insert after matched terms
            
        Returns:
            Text with glossary terms highlighted
        """
        if not text:
            return text
            
        # Prevent memory issues with large text
        if len(text) > 10000:
            text = text[:10000]
        
        # Get terms with positions
        matched_terms = self.find_terms_in_text(text, include_positions=True)
        
        if not matched_terms:
            return text
        
        # Collect all terms and their positions
        term_positions = []
        for term_info in matched_terms:
            term = term_info["term"]
            for start, end in term_info.get("positions", []):
                term_positions.append((start, end, term))
        
        # Sort positions by start position (descending) to avoid offset issues
        term_positions.sort(key=lambda x: x[0], reverse=True)
        
        # Insert markers from end to beginning to maintain indices
        result = text
        for start, end, term in term_positions:
            actual_term = text[start:end]  # Get the actual term from the text with correct case
            result = result[:start] + marker_start + actual_term + marker_end + result[end:]
        
        return result
    
    def _refresh_cache(self) -> None:
        """Refresh the in-memory cache of terms."""
        if not self.collection:
            return
            
        try:
            self._terms_cache = {}
            
            aql = """
            FOR doc IN @@collection
            RETURN {
                "term_lower": doc.term_lower,
                "term": doc.term,
                "definition": doc.definition,
                "length": doc.length
            }
            """
            
            cursor = self.db.aql.execute(
                aql,
                bind_vars={
                    "@collection": self.collection_name
                }
            )
            
            for doc in cursor:
                self._terms_cache[doc["term_lower"]] = {
                    "term": doc["term"],
                    "definition": doc["definition"]
                }
            
            self._rebuild_sorted_terms()
            
            logger.info(f"Refreshed cache with {len(self._terms_cache)} terms")
        except Exception as e:
            logger.error(f"Error refreshing cache: {e}")
    
    def _rebuild_sorted_terms(self) -> None:
        """Rebuild the sorted list of terms by length (descending)."""
        self._terms_by_length = sorted(
            self._terms_cache.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )


# Single-function entry points for compatibility with other search types

def glossary_search(
    db: StandardDatabase,
    text: str,
    collection_name: str = "glossary",
    include_positions: bool = False,
    top_n: int = 100
) -> Dict[str, Any]:
    """
    Find glossary terms in the provided text.
    
    Args:
        db: ArangoDB database connection
        text: The text to search for glossary terms
        collection_name: Name of the glossary collection
        include_positions: If True, include position information for each match
        top_n: Maximum number of results to return
        
    Returns:
        Dictionary containing search results and metadata
    """
    start_time = time.time()
    
    try:
        # Initialize glossary manager
        glossary = GlossaryManager(db, collection_name)
        
        # Find terms in text
        results = glossary.find_terms_in_text(text, include_positions)
        
        # Limit results if needed
        if top_n > 0 and len(results) > top_n:
            results = results[:top_n]
        
        # Return results in standard format
        return {
            "results": [{"doc": result, "glossary_score": 1.0} for result in results],
            "total": len(results),
            "text": text,
            "collection": collection_name,
            "time": time.time() - start_time,
            "search_engine": "glossary",
            "search_type": "glossary"
        }
        
    except Exception as e:
        logger.error(f"Error in glossary search: {e}")
        return {
            "results": [],
            "total": 0,
            "text": text,
            "collection": collection_name,
            "error": str(e),
            "time": time.time() - start_time,
            "search_engine": "glossary-failed",
            "search_type": "glossary"
        }


def get_glossary_terms(
    db: StandardDatabase,
    collection_name: str = "glossary"
) -> Dict[str, Any]:
    """
    Get all glossary terms.
    
    Args:
        db: ArangoDB database connection
        collection_name: Name of the glossary collection
        
    Returns:
        Dictionary containing all terms and metadata
    """
    start_time = time.time()
    
    try:
        # Initialize glossary manager
        glossary = GlossaryManager(db, collection_name)
        
        # Get all terms
        terms = glossary.get_all_terms()
        
        # Return results in standard format
        return {
            "results": [{"doc": term, "glossary_score": 1.0} for term in terms],
            "total": len(terms),
            "collection": collection_name,
            "time": time.time() - start_time,
            "search_engine": "glossary",
            "search_type": "glossary-terms"
        }
        
    except Exception as e:
        logger.error(f"Error getting glossary terms: {e}")
        return {
            "results": [],
            "total": 0,
            "collection": collection_name,
            "error": str(e),
            "time": time.time() - start_time,
            "search_engine": "glossary-failed",
            "search_type": "glossary-terms"
        }


def add_glossary_terms(
    db: StandardDatabase,
    terms_dict: Dict[str, str],
    collection_name: str = "glossary"
) -> Dict[str, Any]:
    """
    Add terms to the glossary.
    
    Args:
        db: ArangoDB database connection
        terms_dict: Dictionary mapping terms to definitions
        collection_name: Name of the glossary collection
        
    Returns:
        Dictionary containing operation result and metadata
    """
    start_time = time.time()
    
    try:
        # Initialize glossary manager
        glossary = GlossaryManager(db, collection_name)
        
        # Add terms
        added_count = glossary.add_terms_bulk(terms_dict)
        
        # Return results in standard format
        return {
            "added_count": added_count,
            "total_terms": len(glossary._terms_cache),
            "collection": collection_name,
            "time": time.time() - start_time,
            "success": added_count > 0,
            "search_engine": "glossary",
            "search_type": "glossary-add"
        }
        
    except Exception as e:
        logger.error(f"Error adding glossary terms: {e}")
        return {
            "added_count": 0,
            "total_terms": 0,
            "collection": collection_name,
            "error": str(e),
            "time": time.time() - start_time,
            "success": False,
            "search_engine": "glossary-failed",
            "search_type": "glossary-add"
        }


def highlight_text_with_glossary(
    db: StandardDatabase,
    text: str,
    collection_name: str = "glossary",
    marker_start: str = "**",
    marker_end: str = "**"
) -> Dict[str, Any]:
    """
    Highlight glossary terms in text.
    
    Args:
        db: ArangoDB database connection
        text: Text to highlight
        collection_name: Name of the glossary collection
        marker_start: String to insert before matched terms
        marker_end: String to insert after matched terms
        
    Returns:
        Dictionary containing highlighted text and metadata
    """
    start_time = time.time()
    
    try:
        # Initialize glossary manager
        glossary = GlossaryManager(db, collection_name)
        
        # Highlight terms
        highlighted_text = glossary.highlight_terms(text, marker_start, marker_end)
        
        # Find terms to get count
        terms = glossary.find_terms_in_text(text)
        
        # Return results in standard format
        return {
            "highlighted_text": highlighted_text,
            "original_text": text,
            "term_count": len(terms),
            "terms": terms,
            "collection": collection_name,
            "time": time.time() - start_time,
            "search_engine": "glossary",
            "search_type": "glossary-highlight"
        }
        
    except Exception as e:
        logger.error(f"Error highlighting text with glossary terms: {e}")
        return {
            "highlighted_text": text,
            "original_text": text,
            "term_count": 0,
            "terms": [],
            "collection": collection_name,
            "error": str(e),
            "time": time.time() - start_time,
            "search_engine": "glossary-failed",
            "search_type": "glossary-highlight"
        }


def validate_glossary_search(result: Dict[str, Any]) -> Tuple[bool, Dict[str, Dict[str, Any]]]:
    """
    Validate glossary search results against expected structure.
    
    Args:
        result: The results returned from glossary_search
        
    Returns:
        Tuple of (validation_passed, validation_failures)
    """
    validation_failures = {}
    
    # Required fields in result
    required_fields = ["results", "total", "search_engine", "search_type", "time"]
    for field in required_fields:
        if field not in result:
            validation_failures[f"missing_{field}"] = {
                "expected": f"{field} field present",
                "actual": f"{field} field missing"
            }
    
    # Search engine type
    if "search_engine" in result and not result["search_engine"].startswith("glossary"):
        if "error" not in result:  # Allow different engine name if there's an error
            validation_failures["search_engine"] = {
                "expected": "glossary or glossary-*",
                "actual": result.get("search_engine")
            }
    
    # Search type
    if "search_type" in result and not result["search_type"].startswith("glossary"):
        if "error" not in result:
            validation_failures["search_type"] = {
                "expected": "glossary or glossary-*",
                "actual": result.get("search_type")
            }
    
    # Check results structure
    if "results" in result and result["results"]:
        for i, item in enumerate(result["results"]):
            # Check for required fields in results
            if "doc" not in item:
                validation_failures[f"result_{i}_missing_doc"] = {
                    "expected": "doc field present",
                    "actual": "doc field missing"
                }
            elif not isinstance(item["doc"], dict):
                validation_failures[f"result_{i}_doc_type"] = {
                    "expected": "dictionary",
                    "actual": type(item["doc"]).__name__
                }
            else:
                # Check structure of doc
                doc = item["doc"]
                if "term" not in doc:
                    validation_failures[f"result_{i}_doc_missing_term"] = {
                        "expected": "term field present",
                        "actual": "term field missing"
                    }
                if "definition" not in doc:
                    validation_failures[f"result_{i}_doc_missing_definition"] = {
                        "expected": "definition field present",
                        "actual": "definition field missing"
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
            db_name = "glossary_search_test"
            if not sys_db.has_database(db_name):
                sys_db.create_database(db_name)
                
            # Connect to test database
            db = client.db(db_name)
            
            # Create a GlossaryManager instance
            glossary = GlossaryManager(db, "test_glossary")
            glossary.initialize_collection(truncate=True)
            
            logger.info("Database setup completed successfully")
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            all_validation_failures.append(f"Test 1 (Database setup): {e}")
            # Continue with other tests using mock database
        
        # Test 2: Adding terms
        total_tests += 1
        logger.info("TEST 2: Adding terms")
        
        test_terms = {
            "test": "A procedure intended to establish quality or performance",
            "glossary": "An alphabetical list of terms with definitions",
            "search": "An attempt to find something"
        }
        
        if 'glossary' in locals():
            try:
                added_count = glossary.add_terms_bulk(test_terms)
                if added_count != len(test_terms):
                    all_validation_failures.append(f"Test 2 (Adding terms): Expected {len(test_terms)} terms added, got {added_count}")
                logger.info(f"Added {added_count} terms")
            except Exception as e:
                all_validation_failures.append(f"Test 2 (Adding terms): {e}")
        
        # Test 3: Getting all terms
        total_tests += 1
        logger.info("TEST 3: Getting all terms")
        
        if 'glossary' in locals():
            try:
                terms = glossary.get_all_terms()
                if len(terms) != len(test_terms):
                    all_validation_failures.append(f"Test 3 (Getting terms): Expected {len(test_terms)} terms, got {len(terms)}")
                
                # Check term structure
                if terms and ("term" not in terms[0] or "definition" not in terms[0]):
                    all_validation_failures.append(f"Test 3 (Getting terms): Invalid term structure: {terms[0]}")
                
                logger.info(f"Got {len(terms)} terms")
            except Exception as e:
                all_validation_failures.append(f"Test 3 (Getting terms): {e}")
        
        # Test 4: Finding terms in text
        total_tests += 1
        logger.info("TEST 4: Finding terms in text")
        
        test_text = "This is a test of the glossary search functionality"
        
        if 'glossary' in locals():
            try:
                matches = glossary.find_terms_in_text(test_text)
                expected_matches = ["test", "glossary", "search"]
                found_terms = [m["term"] for m in matches]
                
                missing_terms = [t for t in expected_matches if t not in found_terms]
                if missing_terms:
                    all_validation_failures.append(f"Test 4 (Finding terms): Missing terms: {missing_terms}")
                
                logger.info(f"Found {len(matches)} terms: {found_terms}")
            except Exception as e:
                all_validation_failures.append(f"Test 4 (Finding terms): {e}")
        
        # Test 5: Highlighting terms
        total_tests += 1
        logger.info("TEST 5: Highlighting terms")
        
        if 'glossary' in locals():
            try:
                highlighted = glossary.highlight_terms(test_text)
                for term in test_terms.keys():
                    if term in test_text and f"**{term}**" not in highlighted:
                        all_validation_failures.append(f"Test 5 (Highlighting): Term '{term}' not highlighted")
                
                logger.info(f"Highlighted text: {highlighted}")
            except Exception as e:
                all_validation_failures.append(f"Test 5 (Highlighting): {e}")
        
        # Test 6: Using function API
        total_tests += 1
        logger.info("TEST 6: Using function API")
        
        try:
            # Create a mock database for testing
            class MockCursor:
                def __init__(self, data):
                    self.data = data
                
                def __iter__(self):
                    return iter(self.data)
                
                def __next__(self):
                    if not self.data:
                        raise StopIteration
                    return self.data.pop(0)
            
            class MockCollection:
                def __init__(self):
                    pass
                
                def truncate(self):
                    pass
                
                def add_hash_index(self, fields, unique=False):
                    pass
                
                def insert(self, doc):
                    return {"_key": "test_key"}
                
                def insert_many(self, docs, **kwargs):
                    return [{"new": doc} for doc in docs]
            
            class MockDatabase:
                class AQL:
                    def execute(self, query, bind_vars=None):
                        if "RETURN {" in query:
                            # For get_all_terms
                            return MockCursor([
                                {"term": "test", "definition": "A procedure"},
                                {"term": "glossary", "definition": "A list of terms"}
                            ])
                        elif "@collection" in query:
                            # For _refresh_cache
                            return MockCursor([
                                {"term_lower": "test", "term": "Test", "definition": "A procedure", "length": 4},
                                {"term_lower": "glossary", "term": "Glossary", "definition": "A list", "length": 8}
                            ])
                        else:
                            return MockCursor([])
                
                def __init__(self):
                    self.aql = self.AQL()
                
                def collection(self, name):
                    return MockCollection()
                
                def create_collection(self, name):
                    return MockCollection()
                
                def collections(self):
                    return [{"name": "mock_collection"}]
            
            mock_db = MockDatabase()
            
            # Test glossary_search function
            result = glossary_search(mock_db, "This is a test")
            
            # Validate result
            is_valid, failures = validate_glossary_search(result)
            if not is_valid:
                for field, details in failures.items():
                    all_validation_failures.append(f"Test 6 (Function API): {field} - Expected {details['expected']}, got {details['actual']}")
            
            # Check for expected terms in the mock results
            if not any("test" in str(r).lower() for r in result.get("results", [])):
                all_validation_failures.append("Test 6 (Function API): Expected 'test' term in results")
            
            logger.info("Function API test completed")
        except Exception as e:
            all_validation_failures.append(f"Test 6 (Function API): {e}")
        
        # Final validation result
        if all_validation_failures:
            print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
            for failure in all_validation_failures:
                print(f"  - {failure}")
            sys.exit(1)
        else:
            print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
            print("Glossary search core module is validated and ready for use")
            sys.exit(0)
            
    except Exception as e:
        logger.error(f"Validation failed with unexpected error: {e}")
        print(f"❌ VALIDATION FAILED - Unexpected error: {e}")
        sys.exit(1)