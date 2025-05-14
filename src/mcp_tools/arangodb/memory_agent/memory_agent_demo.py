#!/usr/bin/env python3
"""
Demo script for the Memory Agent functionality.

IMPORTANT: This is a simplified mock implementation for demonstration and educational purposes only.
It shows how the Memory Agent interface works, but does NOT connect to a real database.

For actual usage in production, use the real MemoryAgent class from memory_agent.py,
which properly connects to ArangoDB and provides full functionality.

This demo version is useful for:
1. Understanding the Memory Agent API without needing a database
2. Testing or demoing in environments where ArangoDB is not available
3. Educational purposes to see how the API works

For real applications, use:
   from complexity.arangodb.memory_agent.memory_agent import MemoryAgent
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from loguru import logger

# Mock classes and functions for demonstration purposes
@dataclass
class MockMessage:
    """Mock message for demonstration."""
    content: str
    message_type: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MockMemory:
    """Mock memory for demonstration."""
    content: str
    summary: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    similarity_score: float = 0.0
    
class MockMemoryAgent:
    """
    Mock Memory Agent for demonstration purposes.
    
    This class shows the intended usage of the Memory Agent without requiring
    an actual ArangoDB instance. It provides the same API as the real MemoryAgent.
    """
    
    def __init__(self):
        """Initialize the mock MemoryAgent."""
        self.conversations = {}
        self.memories = []
        logger.info("MockMemoryAgent initialized")
    
    def store_conversation(self, 
                           conversation_id: Optional[str] = None,
                           user_message: str = "",
                           agent_response: str = "",
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, str]:
        """Store a user-agent message exchange."""
        if conversation_id is None:
            conversation_id = f"conv_{len(self.conversations) + 1}"
            
        if metadata is None:
            metadata = {}
            
        # Create mock messages
        user_msg = MockMessage(user_message, "USER", metadata=metadata)
        agent_msg = MockMessage(agent_response, "AGENT", metadata=metadata)
        
        # Store in the conversations dict
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        self.conversations[conversation_id].extend([user_msg, agent_msg])
        
        # Create a memory from the exchange
        summary = self._generate_summary(user_message, agent_response)
        memory = MockMemory(
            content=f"User: {user_message}\nAgent: {agent_response}",
            summary=summary,
            metadata=metadata
        )
        memory_key = f"mem_{len(self.memories) + 1}"
        self.memories.append(memory)
        
        logger.info(f"Stored conversation {conversation_id} with memory {memory_key}")
        return {
            "conversation_id": conversation_id,
            "user_key": f"u_{len(self.conversations[conversation_id]) - 1}",
            "agent_key": f"a_{len(self.conversations[conversation_id])}",
            "memory_key": memory_key
        }
    
    def _generate_summary(self, user_message: str, agent_response: str) -> str:
        """Generate a simple summary of the exchange."""
        max_length = 100
        combined = f"{user_message} {agent_response}"
        
        if len(combined) <= max_length:
            return combined
        
        # Simple truncation summary
        return combined[:max_length] + "..."
    
    def search_memory(self, 
                     query: str,
                     top_n: int = 5,
                     **kwargs) -> List[Dict[str, Any]]:
        """Search for relevant memories."""
        # Mock implementation that does basic substring matching
        results = []
        
        for i, memory in enumerate(self.memories):
            # Simple string matching for demonstration
            if (query.lower() in memory.content.lower() or 
                any(tag.lower() in query.lower() for tag in memory.metadata.get("tags", []))):
                
                # Calculate a mock similarity score (higher for longer match)
                memory.similarity_score = min(0.95, 0.5 + 0.1 * len(query) / len(memory.content))
                
                results.append({
                    "doc": {
                        "content": memory.content,
                        "summary": memory.summary,
                        "metadata": memory.metadata
                    },
                    "rrf_score": memory.similarity_score
                })
        
        # Sort by score and limit
        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        results = results[:top_n]
        
        logger.info(f"Found {len(results)} memories for query: '{query}'")
        return results
    
    def get_related_memories(self, 
                            memory_key: str,
                            **kwargs) -> List[Dict[str, Any]]:
        """Get related memories."""
        # Mock implementation that returns a random subset of memories
        try:
            memory_idx = int(memory_key.split('_')[1]) - 1
            target_memory = self.memories[memory_idx]
            
            results = []
            for i, memory in enumerate(self.memories):
                if i != memory_idx:  # Skip self
                    results.append({
                        "memory": {
                            "content": memory.content,
                            "summary": memory.summary,
                            "metadata": memory.metadata
                        },
                        "relationship": {
                            "type": "RELATED_TO",
                            "strength": 0.75
                        }
                    })
            
            return results[:3]  # Return up to 3 related memories
        except (IndexError, ValueError):
            logger.warning(f"Memory key {memory_key} not found")
            return []
    
    def get_conversation_context(self, 
                               conversation_id: str,
                               limit: int = 10) -> List[Dict[str, Any]]:
        """Get the context of a conversation."""
        if conversation_id not in self.conversations:
            logger.warning(f"Conversation {conversation_id} not found")
            return []
        
        messages = self.conversations[conversation_id]
        result = []
        
        for i, msg in enumerate(messages[:limit]):
            result.append({
                "content": msg.content,
                "message_type": msg.message_type,
                "timestamp": msg.timestamp,
                "metadata": msg.metadata
            })
        
        return result


# Main function for demonstration
def main():
    """Demo of the Memory Agent functionality."""
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    logger.info("Memory Agent Demo")
    logger.info("================")
    
    # Create a mock memory agent
    agent = MockMemoryAgent()
    
    # Demonstrate storing conversations
    logger.info("\nStoring conversations...")
    
    # First conversation
    conv1 = agent.store_conversation(
        user_message="What's the time complexity of quicksort?",
        agent_response="The average time complexity of quicksort is O(n log n), but the worst case is O(n²).",
        metadata={"topic": "algorithms", "tags": ["sorting", "complexity"]}
    )
    
    # Second conversation
    conv2 = agent.store_conversation(
        conversation_id=conv1["conversation_id"],  # Continue the conversation
        user_message="When does the worst case occur?",
        agent_response="The worst case occurs when the pivot selection is poor, such as always picking the smallest or largest element as the pivot.",
        metadata={"topic": "algorithms", "tags": ["sorting", "complexity"]}
    )
    
    # Third conversation (different topic)
    conv3 = agent.store_conversation(
        user_message="What's the difference between Python 2 and Python 3?",
        agent_response="Python 3 introduced many changes: print is a function, division returns float by default, text is Unicode, etc.",
        metadata={"topic": "programming", "tags": ["python", "language"]}
    )
    
    # Demonstrate searching
    logger.info("\nSearching for memories...")
    
    search_results = agent.search_memory(
        query="quicksort worst case",
        top_n=2
    )
    
    logger.info(f"Found {len(search_results)} results for 'quicksort worst case'")
    for i, result in enumerate(search_results, 1):
        logger.info(f"Result {i}: {result['doc']['summary'][:50]}... (Score: {result['rrf_score']:.2f})")
    
    # Demonstrate getting related memories
    logger.info("\nGetting related memories...")
    
    related = agent.get_related_memories(conv1["memory_key"])
    logger.info(f"Found {len(related)} memories related to the first conversation")
    
    # Demonstrate getting conversation context
    logger.info("\nGetting conversation context...")
    
    context = agent.get_conversation_context(conv1["conversation_id"])
    logger.info(f"Found {len(context)} messages in the conversation context")
    for i, msg in enumerate(context, 1):
        logger.info(f"Message {i} ({msg['message_type']}): {msg['content'][:30]}...")
    
    logger.info("\nMemory Agent Demo completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())