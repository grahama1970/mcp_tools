"""Pydantic models for ArangoDB CLI data validation.

This module defines Pydantic models for validating input and output data
in the ArangoDB CLI. These models provide strict typing, validation rules,
and documentation for data structures used throughout the application.

Links to third-party documentation:
- Pydantic: https://docs.pydantic.dev/
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field, validator, root_validator


# Define enums for constrained choices
class RelationshipType(str, Enum):
    """Valid relationship types between documents."""
    RELATED = "RELATED"
    DUPLICATE = "DUPLICATE"
    PREREQUISITE = "PREREQUISITE"
    CAUSAL = "CAUSAL"
    CONTEXT = "CONTEXT"
    EXAMPLE = "EXAMPLE"
    NEXT = "NEXT"


class SearchType(str, Enum):
    """Valid search algorithm types."""
    BM25 = "bm25"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    KEYWORD = "keyword"
    TAG = "tag"


class TraversalDirection(str, Enum):
    """Valid graph traversal directions."""
    OUTBOUND = "OUTBOUND"
    INBOUND = "INBOUND"
    ANY = "ANY"


class MessageType(str, Enum):
    """Valid message types for conversation history."""
    USER = "USER"
    AGENT = "AGENT"
    SYSTEM = "SYSTEM"


# Base schema for common fields
class BaseDocumentSchema(BaseModel):
    """Base schema for document data with common fields."""
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = Field(default_factory=list)
    
    @validator('tags', pre=True)
    def ensure_tags_list(cls, v):
        """Ensure tags is a list."""
        if v is None:
            return []
        return v


class DocumentSchema(BaseDocumentSchema):
    """Schema for a complete document."""
    problem: Optional[str] = None
    solution: Optional[str] = None
    context: Optional[str] = None
    severity: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    class Config:
        """Configuration for the schema."""
        extra = "allow"  # Allow extra fields that aren't in the model


class DocumentUpdateSchema(BaseModel):
    """Schema for document update operations."""
    title: Optional[str] = None
    content: Optional[str] = None
    problem: Optional[str] = None
    solution: Optional[str] = None
    context: Optional[str] = None
    tags: Optional[List[str]] = None
    severity: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @root_validator
    def check_not_empty(cls, values):
        """Validate that at least one field is provided for update."""
        if not any(values.values()):
            raise ValueError("At least one field must be provided for update")
        return values
    
    class Config:
        """Configuration for the schema."""
        extra = "allow"  # Allow extra fields that aren't in the model


class RelationshipSchema(BaseModel):
    """Schema for relationship data."""
    from_key: str = Field(..., description="Source document key")
    to_key: str = Field(..., description="Target document key")
    type: RelationshipType = Field(..., description="Type of relationship")
    rationale: str = Field(..., description="Reason for the relationship")
    attributes: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional attributes")


class SearchParamsSchema(BaseModel):
    """Schema for search parameters."""
    query: str = Field(..., description="Search query")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Search algorithm to use")
    top_n: int = Field(default=10, ge=1, description="Number of results to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum relevance score")
    tag_list: Optional[List[str]] = Field(default=None, description="Tags to filter results")
    collection_name: Optional[str] = Field(default=None, description="Collection to search")
    
    @validator('query')
    def check_query_not_empty(cls, v):
        """Validate that query is not empty."""
        if not v or not v.strip():
            raise ValueError("Search query cannot be empty")
        return v


class TraversalParamsSchema(BaseModel):
    """Schema for graph traversal parameters."""
    start_key: str = Field(..., description="Starting vertex key")
    graph_name: str = Field(default="default", description="Name of the graph")
    min_depth: int = Field(default=1, ge=0, description="Minimum traversal depth")
    max_depth: int = Field(default=1, ge=1, description="Maximum traversal depth")
    direction: TraversalDirection = Field(default=TraversalDirection.ANY, description="Traversal direction")
    limit: Optional[int] = Field(default=None, ge=1, description="Result limit")
    
    @validator('max_depth')
    def check_max_depth(cls, v, values):
        """Validate that max_depth is greater than or equal to min_depth."""
        if 'min_depth' in values and v < values['min_depth']:
            raise ValueError("max_depth must be greater than or equal to min_depth")
        return v


class MessageSchema(BaseModel):
    """Schema for a message in conversation history."""
    conversation_id: str = Field(..., description="Conversation ID")
    message_type: MessageType = Field(..., description="Type of message")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata")
    timestamp: Optional[str] = None
    
    @validator('content')
    def check_content_not_empty(cls, v):
        """Validate that message content is not empty."""
        if not v or not v.strip():
            raise ValueError("Message content cannot be empty")
        return v


if __name__ == "__main__":
    import sys
    import json
    
    # Validation setup
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: DocumentSchema validation
    total_tests += 1
    try:
        # Valid document
        valid_doc = {
            "title": "Test Document",
            "content": "This is test content.",
            "tags": ["test", "document"],
            "problem": "Test problem",
            "solution": "Test solution"
        }
        
        model = DocumentSchema(**valid_doc)
        validated = model.dict()
        
        # Check that all fields are present
        for key, value in valid_doc.items():
            if key not in validated or validated[key] != value:
                raise ValueError(f"Field {key} was not properly validated")
                
        # Check default fields
        if "metadata" not in validated:
            raise ValueError("Default field 'metadata' was not added")
            
        print("Document schema validation passed")
    except Exception as e:
        all_validation_failures.append(f"Test 1: DocumentSchema validation failed: {str(e)}")
    
    # Test 2: DocumentUpdateSchema validation
    total_tests += 1
    try:
        # Valid update
        valid_update = {
            "title": "Updated Title",
            "tags": ["updated"]
        }
        
        model = DocumentUpdateSchema(**valid_update)
        
        # Test empty update (should fail)
        try:
            empty_update = {}
            model = DocumentUpdateSchema(**empty_update)
            all_validation_failures.append("Test 2: Empty update validation didn't fail as expected")
        except ValueError:
            # This is expected
            pass
            
        print("Document update schema validation passed")
    except Exception as e:
        all_validation_failures.append(f"Test 2: DocumentUpdateSchema validation failed: {str(e)}")
    
    # Test 3: SearchParamsSchema validation
    total_tests += 1
    try:
        # Valid search params
        valid_search = {
            "query": "test query",
            "search_type": "hybrid",
            "top_n": 10
        }
        
        model = SearchParamsSchema(**valid_search)
        
        # Test empty query (should fail)
        try:
            invalid_search = valid_search.copy()
            invalid_search["query"] = ""
            model = SearchParamsSchema(**invalid_search)
            all_validation_failures.append("Test 3: Empty query validation didn't fail as expected")
        except ValueError:
            # This is expected
            pass
            
        print("Search params schema validation passed")
    except Exception as e:
        all_validation_failures.append(f"Test 3: SearchParamsSchema validation failed: {str(e)}")
    
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
