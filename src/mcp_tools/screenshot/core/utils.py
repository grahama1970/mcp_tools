#!/usr/bin/env python3
"""
Utility Functions for Screenshot Module

This module provides common utility functions used by other core modules.
It includes functions for validation, error handling, and file operations.

This module is part of the Core Layer and should have no dependencies on
Presentation or Integration layers.

Sample input:
- Various utility function inputs

Expected output:
- Various utility function outputs
"""

import os
import time
import json
import platform
from typing import Dict, List, Any, Optional, Union, Tuple
from loguru import logger

from mcp_tools.screenshot.core.constants import REGION_PRESETS


def validate_quality(quality: int, min_quality: int, max_quality: int) -> int:
    """
    Validates and clamps quality value to acceptable range.
    
    Args:
        quality: Requested quality (1-100)
        min_quality: Minimum acceptable quality
        max_quality: Maximum acceptable quality
        
    Returns:
        int: Clamped quality value
    """
    original_quality = quality
    quality = max(min_quality, min(quality, max_quality))
    
    if quality != original_quality:
        logger.info(
            f"Adjusted quality from {original_quality} to {quality} "
            f"(min={min_quality}, max={max_quality})"
        )
    
    return quality


def validate_region(region: Optional[Union[List[int], str]]) -> Tuple[bool, Optional[str]]:
    """
    Validates region parameter format.
    
    Args:
        region: Region coordinates [x, y, width, height] or named preset
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    if region is None:
        return True, None
        
    if not isinstance(region, (list, str)):
        return False, f"Invalid region parameter: must be a list or string preset, got {type(region)}"
        
    if isinstance(region, list) and len(region) != 4:
        return False, f"Region must have 4 elements [x, y, width, height], got {len(region)}"
        
    if isinstance(region, str) and region not in REGION_PRESETS:
        valid_presets = ", ".join(REGION_PRESETS.keys())
        return False, f"String region must be a valid preset ({valid_presets}), got {region}"
        
    return True, None


def generate_filename(prefix: str = "screenshot", extension: str = "jpeg") -> str:
    """
    Generates a unique filename with timestamp.
    
    Args:
        prefix: Filename prefix
        extension: File extension without dot
        
    Returns:
        str: Generated filename
    """
    timestamp = int(time.time() * 1000)
    return f"{prefix}_{timestamp}.{extension}"


def ensure_directory(directory: str) -> bool:
    """
    Ensures directory exists, creating it if necessary.
    
    Args:
        directory: Directory path
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {directory}: {str(e)}")
        return False


def get_system_info() -> Dict[str, str]:
    """
    Get system information for debugging.
    
    Returns:
        Dict[str, str]: System information
    """
    info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
    }
    
    return info


def format_error_response(error_message: str, include_system_info: bool = False) -> Dict[str, Any]:
    """
    Creates a standardized error response.
    
    Args:
        error_message: Error message
        include_system_info: Whether to include system information
        
    Returns:
        Dict[str, Any]: Error response dictionary
    """
    response = {"error": error_message}
    
    if include_system_info:
        response["system_info"] = get_system_info()
        
    return response


def safe_file_operation(operation_name: str, func, *args, **kwargs) -> Tuple[bool, Optional[Any], Optional[str]]:
    """
    Safely execute a file operation with proper error handling.
    
    Args:
        operation_name: Name of the operation for error reporting
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple[bool, Optional[Any], Optional[str]]: (success, result, error_message)
    """
    try:
        result = func(*args, **kwargs)
        return True, result, None
    except FileNotFoundError as e:
        error = f"{operation_name} failed: File not found - {str(e)}"
        logger.error(error)
        return False, None, error
    except PermissionError as e:
        error = f"{operation_name} failed: Permission denied - {str(e)}"
        logger.error(error)
        return False, None, error
    except IOError as e:
        error = f"{operation_name} failed: I/O error - {str(e)}"
        logger.error(error)
        return False, None, error
    except Exception as e:
        error = f"{operation_name} failed: {str(e)}"
        logger.error(error, exc_info=True)
        return False, None, error


def parse_region_preset(preset: str, monitor_info: Dict[str, int]) -> Dict[str, int]:
    """
    Convert a region preset string to actual coordinates.
    
    Args:
        preset: Region preset name (right_half, left_half, etc.)
        monitor_info: Monitor information dictionary
        
    Returns:
        Dict[str, int]: Region coordinates dictionary
    """
    width = monitor_info["width"]
    height = monitor_info["height"]
    left = monitor_info.get("left", 0)
    top = monitor_info.get("top", 0)
    
    if preset == "right_half":
        return {
            "top": top,
            "left": left + width // 2,
            "width": width // 2,
            "height": height
        }
    elif preset == "left_half":
        return {
            "top": top,
            "left": left,
            "width": width // 2,
            "height": height
        }
    elif preset == "top_half":
        return {
            "top": top,
            "left": left,
            "width": width,
            "height": height // 2
        }
    elif preset == "bottom_half":
        return {
            "top": top + height // 2,
            "left": left,
            "width": width,
            "height": height // 2
        }
    else:
        # Default to full monitor
        return {
            "top": top,
            "left": left,
            "width": width,
            "height": height
        }


def list_coordinates_to_dict(coords: List[int]) -> Dict[str, int]:
    """
    Convert [x, y, width, height] list to a dictionary for MSS.
    
    Args:
        coords: [x, y, width, height] list
        
    Returns:
        Dict[str, int]: MSS-compatible region dictionary
    """
    if len(coords) != 4:
        raise ValueError(f"Coordinates must have 4 elements, got {len(coords)}")
        
    return {
        "left": coords[0],
        "top": coords[1],
        "width": coords[2],
        "height": coords[3]
    }


if __name__ == "__main__":
    """Validate utility functions"""
    import sys
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: validate_quality
    total_tests += 1
    test_quality = validate_quality(120, 30, 70)
    if test_quality != 70:
        all_validation_failures.append(f"validate_quality test: Expected 70, got {test_quality}")
        
    # Test 2: validate_region - valid cases
    total_tests += 1
    test_cases = [
        None,
        [0, 0, 100, 100],
        "right_half"
    ]
    
    for test_case in test_cases:
        is_valid, error = validate_region(test_case)
        if not is_valid:
            all_validation_failures.append(f"validate_region test: Valid case {test_case} failed: {error}")
            
    # Test 3: validate_region - invalid cases
    total_tests += 1
    invalid_test_cases = [
        "invalid_region",
        [0, 0],
        {"x": 0, "y": 0, "width": 100, "height": 100}
    ]
    
    for test_case in invalid_test_cases:
        is_valid, error = validate_region(test_case)
        if is_valid:
            all_validation_failures.append(f"validate_region test: Invalid case {test_case} passed when it should fail")
    
    # Test 4: generate_filename
    total_tests += 1
    filename = generate_filename("test", "png")
    if not filename.startswith("test_") or not filename.endswith(".png"):
        all_validation_failures.append(f"generate_filename test: Invalid format: {filename}")
    
    # Test 5: ensure_directory
    total_tests += 1
    test_dir = ".test_dir"
    success = ensure_directory(test_dir)
    if not success or not os.path.exists(test_dir):
        all_validation_failures.append(f"ensure_directory test: Failed to create directory")
    else:
        # Clean up
        os.rmdir(test_dir)
    
    # Test 6: get_system_info
    total_tests += 1
    sys_info = get_system_info()
    required_keys = ["platform", "python_version", "architecture"]
    for key in required_keys:
        if key not in sys_info:
            all_validation_failures.append(f"get_system_info test: Missing key {key}")
    
    # Test 7: format_error_response
    total_tests += 1
    error_resp = format_error_response("Test error", include_system_info=True)
    if "error" not in error_resp or error_resp["error"] != "Test error":
        all_validation_failures.append(f"format_error_response test: Invalid error message")
    if "system_info" not in error_resp:
        all_validation_failures.append(f"format_error_response test: Missing system_info")
    
    # Test 8: safe_file_operation
    total_tests += 1
    
    # Test with successful operation
    success, result, error = safe_file_operation(
        "string operation",
        lambda x: x.upper(),
        "test"
    )
    if not success or result != "TEST" or error is not None:
        all_validation_failures.append(f"safe_file_operation test: Failed on successful operation")
    
    # Test with failing operation
    success, result, error = safe_file_operation(
        "file operation",
        open,
        "nonexistent_file.txt",
        "r"
    )
    if success or result is not None or error is None:
        all_validation_failures.append(f"safe_file_operation test: Failed to handle error")
    
    # Test 9: parse_region_preset
    total_tests += 1
    monitor_info = {"width": 1920, "height": 1080, "left": 0, "top": 0}
    
    # Test right_half
    right_half = parse_region_preset("right_half", monitor_info)
    if right_half["left"] != 960 or right_half["width"] != 960:
        all_validation_failures.append(f"parse_region_preset test: Wrong right_half calculation")
    
    # Test 10: list_coordinates_to_dict
    total_tests += 1
    coords = [10, 20, 300, 400]
    region_dict = list_coordinates_to_dict(coords)
    if (region_dict["left"] != 10 or region_dict["top"] != 20 or 
            region_dict["width"] != 300 or region_dict["height"] != 400):
        all_validation_failures.append(f"list_coordinates_to_dict test: Wrong conversion")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Utility functions are validated and ready for use")
        sys.exit(0)  # Exit with success code
