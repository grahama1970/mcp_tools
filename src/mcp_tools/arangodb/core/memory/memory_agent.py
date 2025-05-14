#!/usr/bin/env python3
"""
Memory Agent for ArangoDB

This module provides functionality to store and retrieve LLM agent conversations
using ArangoDB. It embeds each message exchange to enable semantic search and
creates relationships between messages to facilitate graph traversal.

IMPORTANT: This implementation requires a direct ArangoDB connection and will not
work without a properly configured database. There are no fallbacks or mocks.

Usage:
    from mcp_tools.arangodb.core.memory import MemoryAgent
    from mcp_tools.arangodb.core.db.connection import connect_arango, ensure_database
    
    # Connect to ArangoDB (required)
    client = connect_arango()
    db = ensure_database(client)
    
    # Initialize the agent with the database connection
    memory_agent = MemoryAgent(db=db)
    
    # Store a conversation
    memory_agent.store_conversation(conversation_id, user_message, agent_response)
    
    # Search for relevant conversations
    results = memory_agent.search_memory(query, top_n=5)
    
    # Get related memories through graph traversal
    related = memory_agent.get_related_memories(memory_key)
    
    # Get conversation context
    context = memory_agent.get_conversation_context(conversation_id)
"""

import os
import sys
import uuid
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union

from loguru import logger
from arango.database import StandardDatabase

# Import ArangoDB operations
from mcp_tools.arangodb.core.db import (
    create_document,
    get_conversation_messages,
    create_relationship,
    link_message_to_document
)

# Import required modules - these are core dependencies that should not be mocked
from mcp_tools.arangodb.core.search.hybrid_search import hybrid_search
from mcp_tools.arangodb.core.utils.embedding_utils import get_embedding, calculate_cosine_similarity
from mcp_tools.arangodb.core.utils.connection import (
    connect_arango,
    ensure_database,
    ensure_collection,
    ensure_memory_agent_collections,
    ensure_arangosearch_view
)

# Constants for message types
MESSAGE_TYPE_USER = "user"
MESSAGE_TYPE_AGENT = "assistant"


class MemoryAgent:
    """
    Memory Agent for storing and retrieving LLM conversations.
    
    This class handles:
    1. Storing user-agent message exchanges in ArangoDB
    2. Embedding messages for semantic search
    3. Creating relationships between messages
    4. Searching for relevant past conversations
    
    DEPENDENCIES (all required, no fallbacks):
    - An active ArangoDB connection (passed to constructor)
    - Required collections and views (created at initialization)
    - The hybrid_search module for semantic search
    - The embedding_utils module for generating embeddings
    
    Raises:
        ValueError: If required parameters are missing or invalid
        ConnectionError: If database connection fails
        RuntimeError: If required collections cannot be created
        Exception: For any other database operation failures
    """
    
    def __init__(self, 
                 db: StandardDatabase,
                 message_collection: str = "agent_messages",
                 memory_collection: str = "agent_memories",
                 edge_collection: str = "agent_relationships",
                 view_name: str = "agent_memory_view",
                 embedding_field: str = "embedding"):
        """
        Initialize the MemoryAgent.
        
        Args:
            db: ArangoDB connection (required)
            message_collection: Collection name for messages
            memory_collection: Collection name for memory documents
            edge_collection: Collection name for relationships
            view_name: View name for search
            embedding_field: Field name for embeddings
            
        Raises:
            ValueError: If db is None
        """
        if db is None:
            raise ValueError("Database connection is required for MemoryAgent")
            
        self.message_collection = message_collection
        self.memory_collection = memory_collection
        self.edge_collection = edge_collection
        self.view_name = view_name
        self.embedding_field = embedding_field
        self.db = db
        
        # Ensure required collections and views exist
        ensure_memory_agent_collections(self.db)
        
        logger.info(f"MemoryAgent initialized with database '{db.name}'")
    
    
    def store_conversation(self, 
                          conversation_id: Optional[str] = None,
                          user_message: str = "",
                          agent_response: str = "",
                          metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """
        Store a user-agent message exchange in the database.
        
        Args:
            conversation_id: ID of the conversation (generated if not provided)
            user_message: User's message
            agent_response: Agent's response
            metadata: Additional metadata for the messages
        
        Returns:
            Dictionary with conversation_id, user_key, agent_key, and memory_key
            
        Raises:
            ValueError: If the messages are empty or invalid
            RuntimeError: If database operations fail
        """
        try:
            # Validate inputs
            if not user_message.strip() and not agent_response.strip():
                raise ValueError("Either user_message or agent_response must contain content")
            
            # Generate conversation ID if not provided
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
            
            # Initialize metadata if not provided
            if metadata is None:
                metadata = {}
            
            # Add timestamp to metadata
            metadata["timestamp"] = datetime.now(timezone.utc).isoformat()
            
            # Store user message
            user_key = str(uuid.uuid4())
            user_doc = {
                "_key": user_key,
                "conversation_id": conversation_id,
                "message_type": MESSAGE_TYPE_USER,
                "content": user_message,
                "timestamp": metadata["timestamp"],
                "metadata": metadata
            }
            
            # Generate embedding for user message
            user_embedding = get_embedding(user_message)
            if user_embedding:
                user_doc[self.embedding_field] = user_embedding
            
            # Store the user message
            user_result = create_document(
                collection=self.message_collection,
                data=user_doc,
                db=self.db
            )
            
            # Store agent response
            agent_key = str(uuid.uuid4())
            agent_doc = {
                "_key": agent_key,
                "conversation_id": conversation_id,
                "message_type": MESSAGE_TYPE_AGENT,
                "content": agent_response,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata,
                "previous_message_key": user_key
            }
            
            # Generate embedding for agent response
            agent_embedding = get_embedding(agent_response)
            if agent_embedding:
                agent_doc[self.embedding_field] = agent_embedding
            
            # Store the agent response
            agent_result = create_document(
                collection=self.message_collection,
                data=agent_doc,
                db=self.db
            )
            
            # Create relationship between user message and agent response
            edge = {
                "_from": f"{self.message_collection}/{user_key}",
                "_to": f"{self.message_collection}/{agent_key}",
                "type": "RESPONSE_TO",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Collections are guaranteed to exist from initialization
            self.db.collection(self.edge_collection).insert(edge)
            
            # Create a memory document that combines both messages
            memory_key = str(uuid.uuid4())
            memory_doc = {
                "_key": memory_key,
                "conversation_id": conversation_id,
                "content": f"User: {user_message}\nAgent: {agent_response}",
                "summary": self._generate_summary(user_message, agent_response),
                "timestamp": metadata["timestamp"],
                "metadata": metadata
            }
            
            # Generate embedding for combined content
            memory_embedding = get_embedding(memory_doc["content"])
            if memory_embedding:
                memory_doc[self.embedding_field] = memory_embedding
            
            # Store the memory document
            memory_result = create_document(
                collection=self.memory_collection,
                data=memory_doc,
                db=self.db
            )
            
            # Link messages to memory document
            link_message_to_document(
                db=self.db,
                message_id=f"{self.message_collection}/{user_key}",
                document_id=f"{self.memory_collection}/{memory_key}",
                properties={"type": "PART_OF"},
                edge_collection=self.edge_collection
            )
            link_message_to_document(
                db=self.db,
                message_id=f"{self.message_collection}/{agent_key}",
                document_id=f"{self.memory_collection}/{memory_key}",
                properties={"type": "PART_OF"},
                edge_collection=self.edge_collection
            )
            
            # Generate relationships with other memories (catches and propagates errors)
            self._generate_relationships(memory_key)
            
            logger.info(f"Stored conversation {conversation_id}: user={user_key}, agent={agent_key}, memory={memory_key}")
            
            return {
                "conversation_id": conversation_id,
                "user_key": user_key,
                "agent_key": agent_key,
                "memory_key": memory_key
            }
            
        except Exception as e:
            # Wrap any unexpected errors in a RuntimeError with clear message
            error_msg = f"Failed to store conversation: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _generate_summary(self, user_message: str, agent_response: str) -> str:
        """
        Generate a summary of the conversation exchange.
        
        Args:
            user_message: User's message
            agent_response: Agent's response
        
        Returns:
            String summary
        """
        # Simple summary for now, could use an LLM-based summarizer later
        max_length = 100
        combined = f"{user_message} {agent_response}"
        
        if len(combined) <= max_length:
            return combined
        
        # Basic truncation summary - ensure exactly max_length with ellipsis
        ellipsis = "..."
        # Reserve space for ellipsis
        truncated_length = max_length - len(ellipsis)
        return combined[:truncated_length] + ellipsis
    
    def _generate_relationships(self, memory_key: str) -> int:
        """
        Generate relationships between the new memory and existing memories.
        Uses embeddings for initial similarity and LLM for relationship rationale.
        
        Args:
            memory_key: Key of the memory document
        
        Returns:
            Number of relationships created
            
        Raises:
            ValueError: If memory document not found
            RuntimeError: If database operations fail
            Exception: Any other errors during operation
        """
        try:
            # Get the memory document
            memory_doc = self.db.collection(self.memory_collection).get(memory_key)
            if not memory_doc:
                error_msg = f"Memory document {memory_key} not found"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Skip if no embedding exists
            if self.embedding_field not in memory_doc:
                logger.warning(f"Memory document {memory_key} has no embedding, skipping relationship generation")
                return 0
                
            # Find other memories with embeddings to compare similarity
            aql = f"""
            FOR doc IN {self.memory_collection}
            FILTER doc._key != @memory_key
            FILTER doc.{self.embedding_field} != null
            SORT RAND()
            LIMIT 20  
            RETURN doc
            """
            
            cursor = self.db.aql.execute(aql, bind_vars={"memory_key": memory_key})
            other_memories = list(cursor)
            
            # Calculate embedding similarity for each potential match
            memory_embedding = memory_doc[self.embedding_field]
            
            # Store potential relationships with their similarity scores
            potential_relationships = []
            for other_memory in other_memories:
                if self.embedding_field in other_memory:
                    other_embedding = other_memory[self.embedding_field]
                    similarity = calculate_cosine_similarity(memory_embedding, other_embedding)
                    
                    # Only consider significant similarities
                    if similarity > 0.7:  # Threshold for meaningful similarity
                        potential_relationships.append({
                            "memory": other_memory,
                            "similarity": similarity
                        })
            
            # Sort by similarity and take top matches
            potential_relationships.sort(key=lambda x: x["similarity"], reverse=True)
            top_matches = potential_relationships[:5]  # Limit to 5 most similar
            
            # Create relationships for top matches
            count = 0
            for match in top_matches:
                other_memory = match["memory"]
                similarity = match["similarity"]
                
                # Determine relationship type based on similarity
                rel_type = "strong_semantic_similarity" if similarity > 0.85 else "semantic_similarity"
                
                # Generate rationale using LLM for highest quality
                try:
                    from litellm import completion
                    
                    # Get content from both memories
                    memory_content = memory_doc.get("content", "")
                    other_content = other_memory.get("content", "")
                    
                    # Limit content length for LLM input
                    from_snippet = memory_content[:300] + "..." if len(memory_content) > 300 else memory_content
                    to_snippet = other_content[:300] + "..." if len(other_content) > 300 else other_content
                    
                    # Generate meaningful rationale using LLM
                    prompt = f"""
                    Analyze these two related conversation snippets:
                    
                    Snippet 1: {from_snippet}
                    Snippet 2: {to_snippet}
                    
                    In one brief sentence (15 words or less), explain why these conversations are related or similar.
                    """
                    
                    response = completion(
                        model="gpt-3.5-turbo", 
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.3,
                        max_tokens=50
                    )
                    
                    # Extract text from the response
                    choice = response.choices[0] if response.choices else None
                    message = choice.message if choice else None
                    content = message.content if message else None
                    
                    # Use LLM-generated rationale if available
                    if content and content.strip():
                        rationale = content.strip()
                    else:
                        # Fallback to simple explanation
                        rationale = f"Semantically similar content (similarity: {similarity:.2f})"
                        
                except Exception as llm_error:
                    # Fallback if LLM fails - use metadata or similarity score
                    logger.warning(f"Failed to generate LLM rationale: {llm_error}")
                    
                    # Use tags if available
                    tags1 = memory_doc.get("metadata", {}).get("tags", [])
                    tags2 = other_memory.get("metadata", {}).get("tags", [])
                    common_tags = set(tags1) & set(tags2)
                    
                    if common_tags:
                        tag_list = ", ".join(list(common_tags)[:3])  # Limit to 3 tags
                        rationale = f"Both memories share topics: {tag_list}"
                    else:
                        # Simple content-based rationale using semantic similarity
                        rationale = f"Similar conversation content (similarity: {similarity:.2f})"
                
                # Create a relationship edge
                edge = {
                    "_from": f"{self.memory_collection}/{memory_key}",
                    "_to": f"{self.memory_collection}/{other_memory['_key']}",
                    "type": rel_type,
                    "strength": similarity,
                    "rationale": rationale,
                    "auto_generated": True,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Insert the edge
                self.db.collection(self.edge_collection).insert(edge)
                count += 1
            
            logger.info(f"Created {count} semantic relationships for memory {memory_key}")
            return count
            
        except Exception as e:
            logger.error(f"Error generating relationships for memory {memory_key}: {e}")
            # Wrap database errors in RuntimeError with clear message
            if "arango" in str(e).lower():
                error_msg = f"Database error generating relationships: {str(e)}"
                raise RuntimeError(error_msg) from e
            # Propagate other errors
            raise
    
    def search_memory(self, 
                     query: str,
                     top_n: int = 5,
                     collections: Optional[List[str]] = None,
                     filter_expr: Optional[str] = None,
                     tag_filters: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant memories using hybrid search.
        
        Args:
            query: Search query text
            top_n: Number of results to return
            collections: Collections to search in (defaults to memory_collection)
            filter_expr: Additional filter expression
            tag_filters: List of tags to filter by
        
        Returns:
            List of matching documents with similarity scores
            
        Raises:
            ValueError: If query is empty or invalid
            RuntimeError: If search operation fails
        """
        try:
            # Validate inputs
            if not query or not query.strip():
                raise ValueError("Search query cannot be empty")
                
            if top_n < 1:
                raise ValueError("top_n must be at least 1")
                
            # Set default collections if not provided
            if collections is None:
                collections = [self.memory_collection]
                
            # Execute hybrid search
            start_time = time.time()
            # Create min_score dictionary for thresholds
            min_score = {
                "bm25": 0.01,     # BM25 threshold
                "semantic": 0.65  # Semantic similarity threshold
            }
            
            results = hybrid_search(
                db=self.db,
                query=query,
                collection=collections[0],
                fields=["content", "summary"],
                vector_field=self.embedding_field,
                filter_conditions=filter_expr,
                limit=top_n,
                weight_bm25=0.4,
                weight_vector=0.6
            )
            elapsed_time = time.time() - start_time
            
            # Extract and format the results
            memory_results = results if isinstance(results, list) else []
            total_count = len(memory_results)
            
            logger.info(f"Found {len(memory_results)} memories (from {total_count} total) in {elapsed_time:.3f}s for query: '{query}'")
            
            return memory_results
            
        except ValueError as e:
            # Propagate validation errors with clear context
            logger.error(f"Invalid search parameters: {e}")
            raise
            
        except Exception as e:
            # Wrap other errors in RuntimeError with clear message
            error_msg = f"Memory search failed: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_related_memories(self, 
                            memory_key: str,
                            relationship_type: Optional[str] = None,
                            max_depth: int = 1,
                            limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get related memories using graph traversal.
        
        Args:
            memory_key: Key of the memory document
            relationship_type: Type of relationship to traverse
            max_depth: Maximum traversal depth
            limit: Maximum number of results
        
        Returns:
            List of related memories
        
        Raises:
            ValueError: If parameters are invalid or memory doesn't exist
            RuntimeError: If there's an error during graph traversal
        """
        try:
            # Validate inputs
            if not memory_key or not memory_key.strip():
                raise ValueError("Memory key cannot be empty")
                
            if max_depth < 1:
                raise ValueError("max_depth must be at least 1")
                
            if limit < 1:
                raise ValueError("limit must be at least 1")
                
            # Verify memory exists first
            memory_exists = self.db.collection(self.memory_collection).has(memory_key)
            if not memory_exists:
                raise ValueError(f"Memory with key '{memory_key}' does not exist")
            
            # Build AQL query for graph traversal
            aql = f"""
            FOR v, e, p IN 1..{max_depth} ANY @start_vertex {self.edge_collection}
                FILTER IS_SAME_COLLECTION({self.memory_collection}, v)
                {f"FILTER e.type == @rel_type" if relationship_type else ""}
                SORT LENGTH(p.edges) ASC, e.timestamp DESC
                LIMIT @limit
                RETURN {{
                    memory: v,
                    relationship: e,
                    path_length: LENGTH(p.edges),
                    last_edge_type: p.edges[-1].type
                }}
            """
            
            # Prepare bind variables
            bind_vars = {
                "start_vertex": f"{self.memory_collection}/{memory_key}",
                "limit": limit
            }
            
            if relationship_type:
                bind_vars["rel_type"] = relationship_type
            
            # Execute query
            cursor = self.db.aql.execute(aql, bind_vars=bind_vars)
            results = list(cursor)
            
            logger.info(f"Found {len(results)} related memories for {memory_key}")
            return results
            
        except ValueError as e:
            # Propagate validation errors with clear context
            logger.error(f"Invalid parameters for related memories: {e}")
            raise
            
        except Exception as e:
            # Wrap other errors in RuntimeError with clear message
            error_msg = f"Failed to retrieve related memories: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_conversation_context(self, 
                               conversation_id: str,
                               limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get the context of a conversation by retrieving previous messages.
        
        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of messages to retrieve
        
        Returns:
            List of messages in the conversation
            
        Raises:
            ValueError: If parameters are invalid
            RuntimeError: If there's an error during database query
        """
        try:
            # Validate inputs
            if not conversation_id or not conversation_id.strip():
                raise ValueError("Conversation ID cannot be empty")
                
            if limit < 1:
                raise ValueError("limit must be at least 1")
            
            # Use existing core function to get conversation messages
            messages = get_conversation_messages(
                db=self.db,
                conversation_id=conversation_id,
                sort_by="timestamp",
                sort_direction="ASC",
                limit=limit,
                message_collection=self.message_collection
            )
            
            logger.info(f"Retrieved {len(messages)} messages for conversation {conversation_id}")
            return messages
            
        except ValueError as e:
            # Propagate validation errors with clear context
            logger.error(f"Invalid parameters for conversation context: {e}")
            raise
            
        except Exception as e:
            # Wrap other errors in RuntimeError with clear message
            error_msg = f"Failed to retrieve conversation context: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e