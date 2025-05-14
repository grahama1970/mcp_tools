"""
Logging utilities for ArangoDB operations.

This module provides utilities for formatting and truncating log output,
especially for large data objects like embeddings and base64 images.
"""

import re
import sys
from typing import List, Any, Dict, Optional

# Attempt to import ValidationTracker, with fallback if not available
try:
    from mcp_tools.arangodb.core.utils.validation_tracker import ValidationTracker
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
            print(
                f"\nResults: {self.total_tests - self.failed_tests} passed, {self.failed_tests} failed"
            )
            if self.failed_tests > 0:
                print("❌ VALIDATION FAILED")
                sys.exit(1)
            else:
                print("✅ VALIDATION PASSED - All tests produced expected results")
                sys.exit(0)


# Regex to identify common data URI patterns for images
BASE64_IMAGE_PATTERN = re.compile(r"^(data:image/[a-zA-Z+.-]+;base64,)")


def truncate_large_value(
    value: Any,
    max_str_len: int = 100,
    max_list_elements_shown: int = 10,  # Threshold above which list is summarized
) -> Any:
    """
    Truncate large strings or arrays to make them log-friendly.

    Handles base64 image strings by preserving the header and truncating the data.
    Summarizes lists/arrays longer than `max_list_elements_shown`.

    Args:
        value: The value to potentially truncate
        max_str_len: Maximum length for the data part of strings before truncation
        max_list_elements_shown: Maximum number of elements to show in arrays
                                 before summarizing the array instead.

    Returns:
        Truncated or original value
    """
    if isinstance(value, str):
        # Check if it's a base64 image data URI
        match = BASE64_IMAGE_PATTERN.match(value)
        if match:
            header = match.group(1)
            data = value[len(header) :]
            if len(data) > max_str_len:
                half_len = max_str_len // 2
                if half_len == 0 and max_str_len > 0:
                    half_len = 1
                truncated_data = (
                    f"{data[:half_len]}...{data[-half_len:]}" if half_len > 0 else "..."
                )
                return header + truncated_data
            else:
                return value
        # --- It's not a base64 image string, apply generic string truncation ---
        elif len(value) > max_str_len:
            half_len = max_str_len // 2
            if half_len == 0 and max_str_len > 0:
                half_len = 1
            return (
                f"{value[:half_len]}...{value[-half_len:]}" if half_len > 0 else "..."
            )
        else:
            return value

    elif isinstance(value, list):
        # --- Handle large lists (like embeddings) by summarizing ---
        if len(value) > max_list_elements_shown:
            if value:
                element_type = type(value[0]).__name__
                return f"[<{len(value)} {element_type} elements>]"
            else:
                return "[<0 elements>]"
        else:
            # If list elements are dicts, truncate them recursively
            return [
                truncate_large_value(item, max_str_len, max_list_elements_shown)
                if isinstance(item, dict)
                else item
                for item in value
            ]
    elif isinstance(value, dict):  # Add explicit check for dict
        # Recursively truncate values within dictionaries
        return {
            k: truncate_large_value(v, max_str_len, max_list_elements_shown)
            for k, v in value.items()
        }
    else:
        # Handle other types (int, float, bool, None, etc.) - return as is
        return value


def log_safe_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Create a log-safe version of the results list by truncating large fields
    within each dictionary.

    Args:
        results (list): List of documents (dictionaries) that may contain large fields.

    Returns:
        list: Log-safe version of the input list where large fields are truncated.

    Raises:
        TypeError: If the input `results` is not a list, or if any element
                   within the list is not a dictionary.
    """
    # --- Input Validation ---
    if not isinstance(results, list):
        raise TypeError(
            f"Expected input to be a List[Dict[str, Any]], but got {type(results).__name__}."
        )

    for index, item in enumerate(results):
        if not isinstance(item, dict):
            raise TypeError(
                f"Expected all elements in the input list to be dictionaries (dict), "
                f"but found element of type {type(item).__name__} at index {index}."
            )
    # --- End Input Validation ---

    log_safe_output = []
    for doc in results:  # We now know 'doc' is a dictionary
        doc_copy = {}
        for key, value in doc.items():
            doc_copy[key] = truncate_large_value(value)
        log_safe_output.append(doc_copy)
    return log_safe_output


def validate_log_utils():
    """
    Validate the log_utils functions using validation tracking.
    """
    validator = ValidationTracker("Log Utils Module")

    # --- Test 1: truncate_large_value with a short string ---
    short_string = "This is a short string"
    truncated_short = truncate_large_value(short_string)
    validator.check(
        "truncate_large_value - short string",
        expected=short_string,
        actual=truncated_short,
    )

    # --- Test 2: truncate_large_value with a long string ---
    long_string = "This is a very long string that should be truncated" * 5
    truncated_long = truncate_large_value(long_string)
    # Check that it's shorter than the original and contains "..."
    validator.check(
        "truncate_large_value - long string truncation",
        expected=True,
        actual=len(truncated_long) < len(long_string) and "..." in truncated_long,
    )

    # --- Test 3: truncate_large_value with a base64 image string ---
    base64_image = "data:image/png;base64," + ("A" * 200)
    truncated_image = truncate_large_value(base64_image)
    validator.check(
        "truncate_large_value - base64 image",
        expected=True,
        actual=len(truncated_image) < len(base64_image)
        and truncated_image.startswith("data:image/png;base64,")
        and "..." in truncated_image,
    )

    # --- Test 4: truncate_large_value with a small list ---
    small_list = [1, 2, 3, 4, 5]
    truncated_small_list = truncate_large_value(small_list)
    validator.check(
        "truncate_large_value - small list",
        expected=small_list,
        actual=truncated_small_list,
    )

    # --- Test 5: truncate_large_value with a large list ---
    large_list = [i for i in range(50)]
    truncated_large_list = truncate_large_value(large_list)
    validator.check(
        "truncate_large_value - large list summary",
        expected=True,
        actual=isinstance(truncated_large_list, str)
        and f"<{len(large_list)}" in truncated_large_list
        and "elements>" in truncated_large_list,
    )

    # --- Test 6: truncate_large_value with a dictionary ---
    dict_value = {"short": "value", "long": "x" * 150}
    truncated_dict = truncate_large_value(dict_value)
    validator.check(
        "truncate_large_value - dictionary",
        expected=True,
        actual=isinstance(truncated_dict, dict)
        and truncated_dict["short"] == "value"
        and len(truncated_dict["long"]) < 150
        and "..." in truncated_dict["long"],
    )

    # --- Test 7: truncate_large_value with a list of dictionaries ---
    list_of_dicts = [{"id": 1, "data": "x" * 150}, {"id": 2, "data": "y" * 150}]
    truncated_list_of_dicts = truncate_large_value(list_of_dicts)
    validator.check(
        "truncate_large_value - list of dictionaries",
        expected=True,
        actual=isinstance(truncated_list_of_dicts, list)
        and len(truncated_list_of_dicts) == 2
        and truncated_list_of_dicts[0]["id"] == 1
        and "..." in truncated_list_of_dicts[0]["data"],
    )

    # --- Test 8: log_safe_results with valid data ---
    valid_test_data = [
        {
            "id": 1,
            "description": "A short description.",
            "embedding": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "image_small": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
            "tags": ["short", "list"],
        },
        {
            "id": 2,
            "description": "This description is quite long, much longer than the default one hundred characters allowed, so it should definitely be truncated according to the rules specified in the function."
            * 2,
            "embedding": [float(i) / 100 for i in range(150)],
            "image_large": "data:image/jpeg;base64," + ("B" * 500),
            "tags": ["tag" + str(i) for i in range(20)],
        },
    ]
    safe_results = log_safe_results(valid_test_data)

    validator.check(
        "log_safe_results - valid data processing",
        expected=True,
        actual=isinstance(safe_results, list)
        and len(safe_results) == len(valid_test_data)
        and safe_results[0]["id"] == 1
        and safe_results[1]["id"] == 2
        and isinstance(safe_results[1]["embedding"], str)
        and "<150" in safe_results[1]["embedding"]
        and len(safe_results[1]["description"])
        < len(valid_test_data[1]["description"]),
    )

    # --- Test 9: log_safe_results with invalid input (not a list) ---
    try:
        log_safe_results({"not": "a list"})
        validator.fail(
            "log_safe_results - invalid input (not a list)",
            "Expected TypeError but no exception was raised",
        )
    except TypeError:
        validator.pass_(
            "log_safe_results - invalid input (not a list)",
            "Correctly raised TypeError for non-list input",
        )
    except Exception as e:
        validator.fail(
            "log_safe_results - invalid input (not a list)",
            f"Expected TypeError but got {type(e).__name__}",
        )

    # --- Test 10: log_safe_results with invalid input (list with non-dict) ---
    try:
        log_safe_results([{"valid": "dict"}, "not a dict"])
        validator.fail(
            "log_safe_results - invalid input (list with non-dict)",
            "Expected TypeError but no exception was raised",
        )
    except TypeError:
        validator.pass_(
            "log_safe_results - invalid input (list with non-dict)",
            "Correctly raised TypeError for list with non-dict element",
        )
    except Exception as e:
        validator.fail(
            "log_safe_results - invalid input (list with non-dict)",
            f"Expected TypeError but got {type(e).__name__}",
        )

    # --- Test 11: log_safe_results with empty list (valid) ---
    empty_result = log_safe_results([])
    validator.check("log_safe_results - empty list", expected=[], actual=empty_result)

    # Generate the final validation report and exit with appropriate code
    validator.report_and_exit()


if __name__ == "__main__":
    # Run validation instead of the original test code
    validate_log_utils()