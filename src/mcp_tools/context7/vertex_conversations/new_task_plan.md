Error Handling Granularity: For error handling (especially in phases 2 and 7), how granular should the error logging be? Should we log the specific file that caused the error, the exact error message, and a timestamp?
> yes. use Loguru. perhaps you can use the decorater when necessary

Should we try to use the same tokenizer as the embedding model (likely OpenAI's tiktoken) for accuracy? Or is an approximate word count sufficient for initial development?
> use tiktoken

JSON Validation: Could you provide a brief outline of the expected functions in json_utils.py? Specifically, what kind of validation should we be performing (e.g., schema validation, type checking)? Ar
> use the below json_utls.py as a starting poiint
### `src/mcp_doc_retriever/arangodb/json_utils.py`

```python
import json
import os
from pathlib import Path
import re
from typing import Union

from json_repair import repair_json
from loguru import logger


class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def json_serialize(data, handle_paths=False, **kwargs):
    """Serialize data to JSON, optionally handling Path objects.

    Args:
        data: The data to serialize.
        handle_paths (bool): Whether to handle Path objects explicitly.
        **kwargs: Additional arguments for json.dumps().

    Returns:
        str: JSON-serialized string.
    """
    if handle_paths:
        return json.dumps(data, cls=PathEncoder, **kwargs)
    return json.dumps(data, **kwargs)


def load_json_file(file_path):
    """Load JSON data from a file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        The loaded JSON data, or None if file does not exist.
    """
    if not os.path.exists(file_path):
        logger.warning(f"File does not exist: {file_path}")
        return None

    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        logger.info("JSON file loaded successfully")
        return data
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decoding error: {e}, trying utf-8-sig encoding")
        try:
            with open(file_path, "r", encoding="utf-8-sig") as file:
                data = json.load(file)
            logger.info("JSON file loaded successfully with utf-8-sig encoding")
            return data
        except json.JSONDecodeError:
            logger.error("JSON decoding error persists with utf-8-sig encoding")
            raise
    except IOError as e:
        logger.error(f"I/O error: {e}")
        raise


def save_json_to_file(data, file_path):
    """Save data to a JSON file.

    Args:
        data: The data to save.
        file_path (Union[str, Path]): A string or Path object representing the file path.
    """
    # Convert file_path to a Path object if it isn't one already
    if not isinstance(file_path, Path):
        file_path = Path(file_path)

    directory = file_path.parent

    try:
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {directory}")
    except OSError as e:
        logger.error(f"Failed to create directory {directory}: {e}")
        raise

    try:
        with file_path.open("w") as f:
            json.dump(data, f, indent=4)
            logger.info(f"Saved JSON to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save to {file_path}: {e}")
        raise


def parse_json(content: str) -> Union[dict, list, str]:
    """Parse a JSON string, attempting repairs if needed.

    Args:
        content (str): JSON string to parse.

    Returns:
        Union[dict, list, str]: Parsed JSON data or original string if parsing fails.
    """
    try:
        parsed_content = json.loads(content)
        logger.debug("Successfully parsed JSON")
        return parsed_content
    except json.JSONDecodeError as e:
        logger.warning(f"Direct JSON parsing failed: {e}")

    try:
        # Try to extract JSON from mixed content
        json_match = re.search(r"(\[.*\]|\{.*\})", content, re.DOTALL)
        if json_match:
            content = json_match.group(1)

        # Attempt repair
        repaired_json = repair_json(content, return_objects=True)
        if isinstance(repaired_json, (dict, list)):
            logger.info("Successfully repaired JSON")
            return repaired_json

        parsed_content = json.loads(repaired_json)
        logger.debug("Successfully parsed repaired JSON")
        return parsed_content

    except json.JSONDecodeError as e:
        logger.error(f"JSON repair failed: {e}")
    except Exception as e:
        logger.error(f"JSON parsing failed: {e}")

    logger.debug("Returning original content")
    return content


def clean_json_string(
    content: Union[str, dict, list], return_dict: bool = False
) -> Union[str, dict, list]:
    """Clean and parse JSON content.

    Args:
        content: JSON string, dict, or list to clean.
        return_dict: If True, return Python dict/list; if False, return JSON string.

    Returns:
        Cleaned JSON as string, dict, or list based on return_dict parameter.
    """
    # Handle dict/list input
    if isinstance(content, (dict, list)):
        return content if return_dict else json.dumps(content)

    # Handle string input
    if isinstance(content, str):
        if not return_dict:
            return content

        parsed_content = parse_json(content)
        if return_dict and isinstance(parsed_content, str):
            try:
                return json.loads(parsed_content)
            except Exception as e:
                logger.error(f"Failed to convert to dict/list: {e}")
                return parsed_content
        return parsed_content

    logger.info("Returning original content")
    return content


def usage_example():
    """Example usage of the clean_json_string function."""
    examples = {
        "valid_json": '{"name": "John", "age": 30, "city": "New York"}',
        "invalid_json": '{"name": "John", "age": 30, "city": "New York" some invalid text}',
        "dict": {"name": "John", "age": 30, "city": "New York"},
        "list_of_dicts": """[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string", "default": "celsius"}
                        },
                        "required": ["location"]
                    }
                }
            }
        ]""",
        "mixed_content": 'Text {"name": "John"} more text',
        "nested_json": '{"person": {"name": "John", "details": {"age": 30}}}',
        "partial_json": '{"name": "John", "age": 30, "city":',
    }

    for name, example in examples.items():
        print(f"\n{name}:")
        print(clean_json_string(example, return_dict=True))

if __name__ == "__main__":
    # Run the example usage
    usage_example()

```



Embedding Batch Size: In Phase 5, what is a reasonable starting batch size for the asyncio-based embedding? Should it be a configurable parameter? 
> would a batch size of 50 be reasonable for openai and concurrent processing? 

Sparse Checkout Failure: If sparse checkout fails entirely for a repository (e.g., due to permission issues or an invalid repo URL), should the script exit immediately, or should it attempt to continue with other repositories (assuming this is designed to be part of a larger pipeline)?
> we should be able to pass in list of repositories. If a repolink is bad. You should use the ask-perplexity MCP tool to see if you can find the actual link. If that fails log the error

LiteLLM Fallback: If LiteLLM is used instead of the OpenAI API directly, should we automatically detect the available models or require the user to specify the model in the environment variables?
> Becuase we are using LiteLLM, we should be able to specifify whatver embedding model we want
https://docs.litellm.ai/docs/embedding/supported_embedding
the actual api keys will be stored in an .env file

File size limit: What should be done with files that are individually larger than, say, 5MB? Should we ignore them? Should we try to split them? Should we log a warning and truncate them?
> we are only downloading text files. We are not downloading images or blobs of any kind. If you enounter a log file or a file over 50K..Do not download it as this is liekly NOT a text file with code examples

> also we will be using the redis container for caching and ArangoDB as our main database. 

> we will need a working Dockerfile and docker-compose.yml for this project

Do you have further clarifying questions to ask
