"""
Utilities for Handling Multimodal Messages in LiteLLM Calls.

This module provides helper functions specifically designed to process message
lists that may contain multimodal content (like images alongside text) before
sending them to a language model via LiteLLM.

Functions:
- `is_multimodal`: Checks if a list of messages contains image URLs.
- `format_multimodal_messages`: Processes messages, potentially handling image
  URLs or other multimodal elements as needed for the target LLM API.
  (Currently, it primarily formats the structure but doesn't perform
  compression or complex processing).

Relevant Documentation:
- LiteLLM Multimodal Support: https://docs.litellm.ai/docs/providers/openai#openai-multimodal-support
- Project Multimodal Handling Notes: ../../repo_docs/multimodal_handling.md (Placeholder)

Input/Output:
- `is_multimodal`: Input is a list of message dictionaries, output is boolean.
- `format_multimodal_messages`: Input is a list of message dictionaries,
  optional image directory, and max size; output is a processed list of
  message dictionaries formatted for API calls.
"""
from typing import List, Dict, Any
from loguru import logger


###
# Helper Functions
###
def is_multimodal(messages: List[Dict[str, Any]]) -> bool:
    """
    Determine if the messages list contains multimodal content (e.g., images).

    Args:
        messages (List[Dict[str, Any]]): List of message dictionaries.

    Returns:
        bool: True if any message contains multimodal content, False otherwise.
    """
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            item.get("type") == "image_url" for item in content
        ):
            return True
    return False


def format_multimodal_messages(
    messages: List[Dict[str, Any]], image_directory: str, max_size_kb: int = 500
) -> List[Dict[str, Any]]:
    """
    Processes a messages list to extract and format content for LLM input.

    Args:
        messages (List[Dict[str, Any]]): List of messages, each containing multimodal content.
        image_directory (str): Directory to store compressed images.
        max_size_kb (int): Maximum size for compressed images in KB.

    Returns:
        List[Dict[str, Any]]: Processed list of content dictionaries.
    """
    if not messages:
        logger.warning("Received empty messages list. Returning an empty content list.")
        return []

    processed_messages = []
    for message in messages:
        if "content" in message and isinstance(message["content"], list):
            processed_content = []
            for item in message["content"]:
                if item.get("type") == "text":
                    processed_content.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image_url":
                    try:
                        # Extract the URL directly from the nested structure
                        image_url = item["image_url"]["url"]
                        # Format it correctly for the API
                        processed_content.append(
                            {"type": "image_url", "url": image_url}
                        )
                    except ValueError as e:
                        logger.error(f"Error processing image: {image_url} - {e}")
                        continue
            processed_messages.append(
                {"role": message.get("role", "user"), "content": processed_content}
            )
        else:
            # If not multimodal content, pass through unchanged
            processed_messages.append(message)
    return processed_messages
