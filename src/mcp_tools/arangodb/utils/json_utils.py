import json
import os
from pathlib import Path
import re
import sys
import logging
from typing import Union, Dict, List, Any, Optional

# Define a simple fallback for repair_json when json_repair is not available
def repair_json(json_string, return_objects=False):
    """Fallback repair function."""
    try:
        # Try to clean up common issues
        json_string = json_string.strip()
        # Remove trailing comma before closing brackets
        json_string = re.sub(r',\s*}', '}', json_string)
        json_string = re.sub(r',\s*]', ']', json_string)
        # Parse the JSON
        if return_objects:
            return json.loads(json_string)
        return json_string
    except:
        return json_string

# Create a fallback logger that mimics loguru's interface
class LoguruFallback:
    def __init__(self):
        self.logger = logging.getLogger("json_utils")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def debug(self, message):
        self.logger.debug(message)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def add(self, *args, **kwargs):
        pass  # Stub for loguru.add functionality

# Create logger fallback
logger = LoguruFallback()

# Import ValidationTracker with error handling
try:
    from arangodb.utils.validation_tracker import ValidationTracker
except ImportError:
    # Define a minimal ValidationTracker if we can't import it
    class ValidationTracker:
        def __init__(self, module_name):
            self.module_name = module_name
            self.test_results = []
            self.total_tests = 0
            self.failed_tests = 0
            print(f"Validation for {module_name}")
            
        def check(self, test_name, expected, actual, description=None):
            self.total_tests += 1
            if expected == actual:
                print(f"✅ PASS: {test_name}")
                return True
            else:
                self.failed_tests += 1
                print(f"❌ FAIL: {test_name}")
                print(f"  Expected: {expected}")
                print(f"  Actual: {actual}")
                if description:
                    print(f"  Description: {description}")
                return False
                
        def pass_(self, test_name, description=None):
            self.total_tests += 1
            print(f"✅ PASS: {test_name}")
            if description:
                print(f"  Description: {description}")
        
        def fail(self, test_name, description=None):
            self.total_tests += 1
            self.failed_tests += 1
            print(f"❌ FAIL: {test_name}")
            if description:
                print(f"  Description: {description}")
        
        def report_and_exit(self):
            print(f"\nResults: {self.total_tests - self.failed_tests} passed, {self.failed_tests} failed")
            if self.failed_tests > 0:
                print("❌ VALIDATION FAILED")
                sys.exit(1)
            else:
                print("✅ VALIDATION PASSED - All tests produced expected results")
                sys.exit(0)

class PathEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)

def json_serialize(data: Any, handle_paths: bool = False, **kwargs) -> str:
    """
    Serialize data to JSON, optionally handling Path objects.
    
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

def load_json_file(file_path: str) -> Optional[Any]:
    """
    Load the extracted tables cache from a JSON file.

    Args:
        file_path (str): The path to the file from where the cache will be loaded.

    Returns:
        Any: The data loaded from the JSON file, or None if the file does not exist.
    """
    if not os.path.exists(file_path):
        logger.warning(f'File does not exist: {file_path}')
        return None
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        logger.info('JSON file loaded successfully')
        return data
    except json.JSONDecodeError as e:
        logger.warning(f'JSON decoding error: {e}, trying utf-8-sig encoding')
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as file:
                data = json.load(file)
            logger.info('JSON file loaded successfully with utf-8-sig encoding')
            return data
        except json.JSONDecodeError:
            logger.error('JSON decoding error persists with utf-8-sig encoding')
            raise
    except IOError as e:
        logger.error(f'I/O error: {e}')
        raise

def save_json_to_file(data: Any, file_path: str) -> None:
    """
    Save the extracted tables cache to a JSON file.

    Args:
        data (Any): The data to be saved.
        file_path (str): The path to the file where cache will be saved.
    """
    directory = os.path.dirname(file_path)
    try:
        if directory:
            os.makedirs(directory, exist_ok=True)
            logger.info(f'Ensured the directory exists: {directory}')
    except OSError as e:
        logger.error(f'Failed to create directory {directory}: {e}')
        raise
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
            logger.info(f'Saved extracted tables to JSON cache at: {file_path}')
    except Exception as e:
        logger.error(f'Failed to save cache to {file_path}: {e}')
        raise

def parse_json(content: str, logger_instance=None) -> Union[Dict, List, str]:
    """
    Attempt to parse a JSON string directly, and if that fails, try repairing it.

    Args:
        content (str): The input JSON string to parse.
        logger_instance: Optional logger instance

    Returns:
        Union[dict, list, str]: Parsed JSON as a dict or list, or the original string if parsing fails.
    """
    log = logger_instance if logger_instance is not None else logger
        
    try:
        parsed_content = json.loads(content)
        log.debug('Successfully parsed JSON response directly')
        return parsed_content
    except json.JSONDecodeError as e:
        log.warning(f'Direct JSON parsing failed: {e}')
    try:
        json_match = re.search('(\\[.*\\]|\\{.*\\})', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        repaired_json = repair_json(content, return_objects=True)
        if isinstance(repaired_json, (dict, list)):
            log.info('Successfully repaired and validated JSON response')
            return repaired_json
        parsed_content = json.loads(repaired_json)
        log.debug('Successfully validated JSON response')
        return parsed_content
    except json.JSONDecodeError as e:
        log.error(f'JSON decode error after repair attempt: {e}')
    except Exception as e:
        log.error(f'Failed to parse JSON response: {e}')
    log.debug(f'Returning original content as string: {content}')
    return content

def clean_json_string(content: Union[str, Dict, List], return_dict: bool = False, logger_instance=None) -> Union[str, Dict, List]:
    """
    Clean and parse a JSON string, dict, or list, returning either a valid JSON string or a Python dict/list.

    Args:
        content (Union[str, dict, list]): The input JSON string, dict, or list to clean.
        return_dict (bool): If True, return a Python dict/list; if False, return a JSON string.
        logger_instance: Optional logger instance

    Returns:
        Union[str, dict, list]: Cleaned JSON as a string, dict, or list, depending on return_dict parameter.
    """
    log = logger_instance if logger_instance is not None else logger
        
    if isinstance(content, (dict, list)):
        return content if return_dict else json.dumps(content)
    elif isinstance(content, str) and return_dict == False:
        return content
    elif isinstance(content, str) and return_dict == True:
        parsed_content = parse_json(content, log)
        if return_dict and isinstance(parsed_content, str):
            try:
                return json.loads(parsed_content)
            except Exception as e:
                log.error(f'Failed to convert parsed content to dict/list: {e}\nFailed content: {type(parsed_content)}: {parsed_content}')
                return parsed_content
        return parsed_content
    log.info(f'Returning original content: {content}')
    return content

def usage_example():
    """
    Example usage of the clean_json_string function.
    """
    example_json_str = '{"name": "John", "age": 30, "city": "New York"}'
    example_invalid_json_str = '{"name": "John", "age": 30, "city": "New York" some invalid text}'
    example_dict = {'name': 'John', 'age': 30, 'city': 'New York'}
    example_list_of_dicts = '[\n        {\n            "type": "function",\n            "function": {\n                "name": "get_current_weather",\n                "description": "Get the current weather in a given location as a plain text string.",\n                "parameters": {\n                    "type": "object",\n                    "properties": {\n                        "location": {"type": "string"},\n                        "unit": {"type": "string", "default": "celsius"}\n                    },\n                    "required": ["location"]\n                },\n                "dependencies": []\n            }\n        },\n        {\n            "type": "function",\n            "function": {\n                "name": "get_clothes",\n                "description": "Function to recommend clothing based on temperature and weather condition.",\n                "parameters": {\n                    "type": "object",\n                    "properties": {\n                        "temperature": {"type": "string"},\n                        "condition": {"type": "string"}\n                    },\n                    "required": ["temperature", "condition"]\n                },\n                "dependencies": ["get_weather"]\n            }\n        }\n    ]'
    schema_v2 = '{authors: list[str], title: str, abstract: str, keywords: list[str]}'
    print(clean_json_string(schema_v2, return_dict=True))
    example_mixed_content = 'Here is some text {"name": "John", "age": 30, "city": "New York"} and more text.'
    example_nested_json = '{"person": {"name": "John", "details": {"age": 30, "city": "New York"}}}'
    example_escaped_characters = '{"text": "He said, \\"Hello, World!\\""}'
    example_large_json = json.dumps([{'index': i, 'value': i * 2} for i in range(1000)])
    example_partial_json = '{"name": "John", "age": 30, "city":'
    print('Valid JSON String (return dict):')
    print(clean_json_string(example_json_str, return_dict=True))
    print('\nInvalid JSON String (return dict):')
    print(clean_json_string(example_invalid_json_str, return_dict=True))
    print('\nDict input (return dict):')
    print(clean_json_string(example_dict, return_dict=True))
    print('\nList of dicts (return dict):')
    print(clean_json_string(example_list_of_dicts, return_dict=True))
    print('\nMixed content (return dict):')
    print(clean_json_string(example_mixed_content, return_dict=True))
    print('\nNested JSON (return dict):')
    print(clean_json_string(example_nested_json, return_dict=True))
    print('\nEscaped characters (return dict):')
    print(clean_json_string(example_escaped_characters, return_dict=True))
    print('\nPartial JSON (return dict):')
    print(clean_json_string(example_partial_json, return_dict=True))

def validate_json_utils():
    """
    Validate the json_utils functions using the ValidationTracker.
    """
    validator = ValidationTracker("JSON Utils Module")
    import tempfile
    
    # Test 1: json_serialize with regular data
    test_data = {"name": "John", "age": 30}
    expected_json = '{"name": "John", "age": 30}'
    actual_json = json_serialize(test_data)
    # JSON strings might have different spacing, so we parse them back to compare
    validator.check("json_serialize - regular data", 
                   expected=json.loads(expected_json), 
                   actual=json.loads(actual_json))
    
    # Test 2: json_serialize with Path objects
    test_path_data = {"name": "John", "file_path": Path("/tmp/test.json")}
    serialized_path_data = json_serialize(test_path_data, handle_paths=True)
    expected_path_result = '{"name": "John", "file_path": "/tmp/test.json"}'
    validator.check("json_serialize - Path objects", 
                   expected=json.loads(expected_path_result),
                   actual=json.loads(serialized_path_data))
    
    # Test 3: save_json_to_file and load_json_file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as temp_file:
        temp_path = temp_file.name
    
    try:
        test_data = {"name": "Alice", "age": 25}
        save_json_to_file(test_data, temp_path)
        loaded_data = load_json_file(temp_path)
        validator.check("save_json_to_file and load_json_file", 
                       expected=test_data, 
                       actual=loaded_data)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    # Test 4: load_json_file with non-existent file
    non_existent_path = "/tmp/non_existent_file_12345.json"
    if os.path.exists(non_existent_path):
        os.remove(non_existent_path)
    
    loaded_data = load_json_file(non_existent_path)
    validator.check("load_json_file - non-existent file", 
                   expected=None, 
                   actual=loaded_data)
    
    # Test 5: parse_json with valid JSON
    valid_json_str = '{"name": "Bob", "age": 40}'
    parsed_valid = parse_json(valid_json_str)
    validator.check("parse_json - valid JSON", 
                   expected={"name": "Bob", "age": 40}, 
                   actual=parsed_valid)
    
    # Test 6: parse_json with invalid but repairable JSON
    invalid_json = '{"name": "Charlie", "age": 35,}'  # Extra comma
    parsed_invalid = parse_json(invalid_json)
    validator.check("parse_json - repairable JSON", 
                   expected={"name": "Charlie", "age": 35}, 
                   actual=parsed_invalid)
    
    # Test 7: clean_json_string with dict input, return_dict=True
    dict_input = {"name": "David", "age": 45}
    cleaned_dict = clean_json_string(dict_input, return_dict=True)
    validator.check("clean_json_string - dict input, return_dict=True", 
                   expected=dict_input, 
                   actual=cleaned_dict)
    
    # Test 8: clean_json_string with string input, return_dict=True
    str_input = '{"name": "Eve", "age": 50}'
    cleaned_str = clean_json_string(str_input, return_dict=True)
    validator.check("clean_json_string - string input, return_dict=True", 
                   expected={"name": "Eve", "age": 50}, 
                   actual=cleaned_str)
    
    # Test 9: clean_json_string with mixed content, return_dict=True
    mixed_input = 'Text before {"name": "Frank", "age": 55} text after'
    cleaned_mixed = clean_json_string(mixed_input, return_dict=True)
    validator.check("clean_json_string - mixed content, return_dict=True", 
                   expected={"name": "Frank", "age": 55}, 
                   actual=cleaned_mixed)
    
    # Test 10: clean_json_string with dict input, return_dict=False
    dict_input = {"name": "Grace", "age": 60}
    cleaned_dict_str = clean_json_string(dict_input, return_dict=False)
    # Compare parsed JSON objects since string formatting may differ
    validator.check("clean_json_string - dict input, return_dict=False", 
                   expected=dict_input, 
                   actual=json.loads(cleaned_dict_str))
    
    # Report results and exit with appropriate code
    validator.report_and_exit()

if __name__ == '__main__':
    # Use the validation function instead of just running the example
    validate_json_utils()