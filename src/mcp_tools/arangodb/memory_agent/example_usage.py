#!/usr/bin/env python3
"""
Example usage of the Memory Agent.

This file provides examples of how to use the Memory Agent in various scenarios.
It demonstrates the core functionality including:
1. Storing conversations
2. Searching for relevant memories
3. Finding related memories through graph traversal
4. Retrieving conversation context

IMPORTANT: This example requires a working ArangoDB connection.
"""

import os
import sys
import uuid
from loguru import logger

# Import the Memory Agent and required database setup
from complexity.arangodb.memory_agent import MemoryAgent
from complexity.arangodb.arango_setup import connect_arango, ensure_database


def run_memory_agent_example():
    """Run a comprehensive example of the Memory Agent functionality."""
    try:
        # Connect to the database
        logger.info("Connecting to ArangoDB...")
        client = connect_arango()
        db = ensure_database(client)
        
        # Create memory agent with the database connection
        logger.info("Initializing Memory Agent...")
        memory_agent = MemoryAgent(db=db)
        
        # Example conversation
        conversation_id = str(uuid.uuid4())
        user_message = "How can I implement a binary search algorithm in Python?"
        agent_response = """
        Here's how to implement a binary search algorithm in Python:

        ```python
        def binary_search(arr, target):
            left, right = 0, len(arr) - 1
            
            while left <= right:
                mid = (left + right) // 2
                
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
                    
            return -1  # Target not found
        ```

        This implementation assumes the array is sorted. Binary search has O(log n) time complexity, which is much faster than linear search O(n) for large arrays.
        """
        
        # Store conversation
        logger.info("Storing example conversation...")
        result = memory_agent.store_conversation(
            conversation_id=conversation_id,
            user_message=user_message,
            agent_response=agent_response,
            metadata={"topic": "algorithms", "tags": ["python", "algorithms", "search"]}
        )
        
        # Search for related memories
        logger.info("Searching for related memories...")
        search_result = memory_agent.search_memory(
            query="How do I write a binary search?",
            top_n=3
        )
        
        # Print results
        print("\nSearch Results:")
        for i, res in enumerate(search_result, 1):
            doc = res.get("doc", {})
            print(f"{i}. {doc.get('summary', 'No summary')} (Score: {res.get('rrf_score', 0):.4f})")
            
        # Get related memories
        memory_key = result.get("memory_key")
        if memory_key:
            logger.info(f"Getting memories related to {memory_key}...")
            related_memories = memory_agent.get_related_memories(memory_key, max_depth=2)
            print(f"\nFound {len(related_memories)} related memories")
        
        # Get conversation context
        logger.info(f"Getting conversation context for {conversation_id}...")
        context = memory_agent.get_conversation_context(conversation_id)
        print(f"\nFound {len(context)} messages in conversation")
        
        print("\nMemory Agent example completed successfully!")
        return 0
    
    except Exception as e:
        logger.error(f"Error in example usage: {e}")
        print(f"Error: {e}")
        return 1


def demonstrate_retrieval():
    """Example focusing on memory retrieval."""
    try:
        # Connect to the database
        client = connect_arango()
        db = ensure_database(client)
        
        # Create memory agent
        memory_agent = MemoryAgent(db=db)
        
        # Search with different parameters
        print("\nExample 1: Basic search")
        results = memory_agent.search_memory(
            query="machine learning algorithms",
            top_n=3
        )
        display_results(results)
        
        print("\nExample 2: Search with tag filters")
        results = memory_agent.search_memory(
            query="python code",
            top_n=3,
            tag_filters=["algorithms", "python"]
        )
        display_results(results)
        
        print("\nExample 3: Search with custom filter expression")
        results = memory_agent.search_memory(
            query="database",
            top_n=3,
            filter_expr="doc.metadata.importance == 'high'"
        )
        display_results(results)
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in retrieval example: {e}")
        print(f"Error: {e}")
        return 1


def display_results(results):
    """Helper function to display search results."""
    if not results:
        print("No results found")
        return
        
    for i, res in enumerate(results, 1):
        doc = res.get("doc", {})
        score = res.get("rrf_score", 0)
        summary = doc.get("summary", "No summary available")
        tags = doc.get("metadata", {}).get("tags", [])
        
        print(f"{i}. {summary[:100]}... (Score: {score:.4f})")
        if tags:
            print(f"   Tags: {', '.join(tags)}")
        print()


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Run the main example
    sys.exit(run_memory_agent_example())