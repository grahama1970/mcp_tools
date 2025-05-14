#!/usr/bin/env python3
"""
Constants for Screenshot Module

This module defines constants used throughout the screenshot functionality,
ensuring consistent configuration across the application.

This module is part of the Core Layer and should have no dependencies on
Presentation or Integration layers.

Sample input:
- None (module contains only constants)

Expected output:
- None (module contains only constants)
"""

from typing import Dict, Any

# Image settings for capture and processing
IMAGE_SETTINGS: Dict[str, Any] = {
    "MAX_WIDTH": 640,  # Maximum width for resized images
    "MAX_HEIGHT": 640,  # Maximum height for resized images
    "MIN_QUALITY": 30,  # Minimum JPEG compression quality
    "MAX_QUALITY": 70,  # Maximum JPEG compression quality
    "DEFAULT_QUALITY": 30,  # Default quality if none specified
    "MAX_FILE_SIZE": 350_000,  # Maximum file size in bytes (350kB)
}

# Regional capture presets
REGION_PRESETS = {
    "full": None,
    "right_half": "right_half",
    "left_half": "left_half",
    "top_half": "top_half",
    "bottom_half": "bottom_half",
}

# Default model for image description
DEFAULT_MODEL = "vertex_ai/gemini-2.5-pro-preview-05-06"

# Default prompt for image description
DEFAULT_PROMPT = "Describe this screenshot in detail."

# Logging settings
LOG_MAX_STR_LEN: int = 100  # Maximum string length for truncated logging


if __name__ == "__main__":
    """Validate module constants"""
    import sys
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Verify IMAGE_SETTINGS contains all required keys
    total_tests += 1
    required_keys = ["MAX_WIDTH", "MAX_HEIGHT", "MIN_QUALITY", "MAX_QUALITY", 
                     "DEFAULT_QUALITY", "MAX_FILE_SIZE"]
    missing_keys = [key for key in required_keys if key not in IMAGE_SETTINGS]
    if missing_keys:
        all_validation_failures.append(f"IMAGE_SETTINGS missing keys: {missing_keys}")
    
    # Test 2: Verify all numeric constants are positive
    total_tests += 1
    for key, value in IMAGE_SETTINGS.items():
        if not isinstance(value, (int, float)) or value <= 0:
            all_validation_failures.append(f"IMAGE_SETTINGS[{key}] should be positive number, got {value}")
    
    # Test 3: Verify quality range is valid
    total_tests += 1
    if not (1 <= IMAGE_SETTINGS["MIN_QUALITY"] <= IMAGE_SETTINGS["MAX_QUALITY"] <= 100):
        all_validation_failures.append(
            f"Invalid quality range: MIN_QUALITY={IMAGE_SETTINGS['MIN_QUALITY']}, "
            f"MAX_QUALITY={IMAGE_SETTINGS['MAX_QUALITY']}"
        )
    
    # Test 4: Verify logging settings
    total_tests += 1
    if not isinstance(LOG_MAX_STR_LEN, int) or LOG_MAX_STR_LEN <= 0:
        all_validation_failures.append(f"LOG_MAX_STR_LEN should be positive integer, got {LOG_MAX_STR_LEN}")
    
    # Test 5: Verify region presets
    total_tests += 1
    if "full" not in REGION_PRESETS or REGION_PRESETS["full"] is not None:
        all_validation_failures.append(f"REGION_PRESETS['full'] should be None")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Constants are valid and ready for use")
        sys.exit(0)  # Exit with success code