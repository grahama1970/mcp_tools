"""
Memory Agent module for storing and retrieving conversation history.

This module provides functionality to store and search LLM agent conversations
using ArangoDB. It embeds each message exchange and creates relationships 
between related conversations.
"""

try:
    from .memory_agent import MemoryAgent
except ImportError:
    # Skip import if it fails when running individually
    pass

__all__ = ["MemoryAgent"]