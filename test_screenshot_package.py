#!/usr/bin/env python3
"""
Comprehensive Test Script for Screenshot Package

This script tests all the key functionality in the mcp_tools.screenshot package
to ensure everything works as expected.
"""

import os
import sys
import time
import json
import traceback
from typing import Dict, Any, List, Tuple

from loguru import logger

# Configure logging
logger.remove()
logger.add(
    "logs/validation.log",
    rotation="10 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)
logger.add(sys.stderr, level="INFO")

# Import all components from the package
from mcp_tools.screenshot import (
    IMAGE_SETTINGS,
    capture_screenshot,
    get_screen_regions,
    resize_image_if_needed,
    compress_image_to_buffer,
    describe_image_content,
    prepare_image_for_multimodal,
    find_credentials_file,
    create_mcp_server
)


def test_constants() -> Tuple[int, List[str]]:
    """Test the constants module."""
    print("\n=== Testing Constants ===")
    total_tests = 0
    failures = []
    
    # Test 1: Verify IMAGE_SETTINGS has all required keys
    total_tests += 1
    required_keys = ["MAX_WIDTH", "MAX_HEIGHT", "MIN_QUALITY", "MAX_QUALITY", 
                     "DEFAULT_QUALITY", "MAX_FILE_SIZE"]
    missing_keys = [key for key in required_keys if key not in IMAGE_SETTINGS]
    if missing_keys:
        failures.append(f"IMAGE_SETTINGS missing keys: {missing_keys}")
    else:
        print("✓ IMAGE_SETTINGS contains all required keys")
    
    # Test 2: Verify quality values are in valid range
    total_tests += 1
    min_quality = IMAGE_SETTINGS["MIN_QUALITY"]
    max_quality = IMAGE_SETTINGS["MAX_QUALITY"]
    default_quality = IMAGE_SETTINGS["DEFAULT_QUALITY"]
    
    if not (1 <= min_quality <= 100 and 1 <= max_quality <= 100 and min_quality <= max_quality):
        failures.append(f"Invalid quality range: MIN_QUALITY={min_quality}, MAX_QUALITY={max_quality}")
    else:
        print(f"✓ Quality range is valid: {min_quality}-{max_quality}")
    
    # Test 3: Verify default quality is within range
    total_tests += 1
    if not (min_quality <= default_quality <= max_quality):
        failures.append(f"Default quality {default_quality} outside of range {min_quality}-{max_quality}")
    else:
        print(f"✓ Default quality {default_quality} is within range")
    
    return total_tests, failures


def test_image_processing() -> Tuple[int, List[str]]:
    """Test the image processing functionality."""
    print("\n=== Testing Image Processing ===")
    total_tests = 0
    failures = []
    
    try:
        from PIL import Image
        
        # Create a test image
        test_dir = ".test_images"
        os.makedirs(test_dir, exist_ok=True)
        
        # Test 1: Test image resizing
        total_tests += 1
        large_img = Image.new('RGB', (1920, 1080), color='red')
        resized_img = resize_image_if_needed(large_img, 640, 640)
        
        expected_width = 640
        expected_height = 360  # Maintaining aspect ratio
        
        if resized_img.size != (expected_width, expected_height):
            failures.append(f"Image resize failed: expected {expected_width}x{expected_height}, got {resized_img.size}")
        else:
            print(f"✓ Image resize works correctly: {large_img.size} -> {resized_img.size}")
        
        # Test 2: Test compression
        total_tests += 1
        compressed_bytes = compress_image_to_buffer(large_img, 60, 30, 1000000)
        
        if not isinstance(compressed_bytes, bytes) or len(compressed_bytes) == 0:
            failures.append("Image compression failed: no data returned")
        else:
            print(f"✓ Image compression works: produced {len(compressed_bytes)} bytes")
        
        # Clean up
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            
    except Exception as e:
        traceback.print_exc()
        failures.append(f"Image processing tests failed with exception: {str(e)}")
    
    return total_tests, failures


def test_capture() -> Tuple[int, List[str]]:
    """Test the screenshot capture functionality."""
    print("\n=== Testing Screenshot Capture ===")
    total_tests = 0
    failures = []
    
    # Test 1: Get screen regions
    total_tests += 1
    try:
        regions = get_screen_regions()
        if regions and "right_half" in regions:
            print(f"✓ Screen regions retrieved: {len(regions)} regions found")
        else:
            failures.append(f"Screen regions test failed: Regions missing or invalid")
    except Exception as e:
        failures.append(f"Screen regions test failed with exception: {str(e)}")
    
    # Test 2: Capture full screenshot
    total_tests += 1
    try:
        full_result = capture_screenshot(quality=30)
        if "error" in full_result:
            failures.append(f"Full screenshot test failed: {full_result['error']}")
        elif "file" not in full_result or not os.path.exists(full_result["file"]):
            failures.append("Full screenshot test failed: File not created")
        else:
            print(f"✓ Full screenshot captured: {full_result['file']}")
    except Exception as e:
        failures.append(f"Full screenshot test failed with exception: {str(e)}")
    
    # Test 3: Capture region screenshot
    total_tests += 1
    try:
        region_result = capture_screenshot(quality=30, region="right_half")
        if "error" in region_result:
            failures.append(f"Region screenshot test failed: {region_result['error']}")
        elif "file" not in region_result or not os.path.exists(region_result["file"]):
            failures.append("Region screenshot test failed: File not created")
        else:
            print(f"✓ Region screenshot captured: {region_result['file']}")
    except Exception as e:
        failures.append(f"Region screenshot test failed with exception: {str(e)}")
    
    return total_tests, failures


def test_description() -> Tuple[int, List[str]]:
    """Test the image description functionality."""
    print("\n=== Testing Image Description ===")
    total_tests = 0
    failures = []
    
    # Test 1: Find credentials file
    total_tests += 1
    try:
        credentials_file = find_credentials_file()
        if credentials_file is not None:
            print(f"✓ Credentials file found: {credentials_file}")
        else:
            print("⚠️ No credentials file found (warning, not failure)")
    except Exception as e:
        failures.append(f"Credentials file test failed with exception: {str(e)}")
    
    # Test 2: Image preparation for multimodal (if we have a screenshot)
    total_tests += 1
    try:
        # Find an existing screenshot
        screenshots_dir = "screenshots"
        if os.path.exists(screenshots_dir) and os.listdir(screenshots_dir):
            screenshot_files = [
                os.path.join(screenshots_dir, f) 
                for f in os.listdir(screenshots_dir) 
                if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")
            ]
            
            if screenshot_files:
                # Use the most recent screenshot
                test_image = sorted(screenshot_files, key=os.path.getmtime, reverse=True)[0]
                
                # Prepare the image
                img_b64 = prepare_image_for_multimodal(test_image)
                
                if not img_b64 or not isinstance(img_b64, str):
                    failures.append("Image preparation test failed: Invalid base64 data")
                else:
                    print(f"✓ Image preparation successful: {len(img_b64)} chars of base64 data")
            else:
                print("⚠️ No screenshots found for testing image preparation")
        else:
            print("⚠️ Screenshots directory not found or empty")
    except Exception as e:
        failures.append(f"Image preparation test failed with exception: {str(e)}")
    
    # Test 3: Actual API description call (only if credentials are available)
    total_tests += 1
    credentials_file = find_credentials_file()
    
    if credentials_file is None:
        print("⚠️ Skipping API test due to missing credentials file")
    elif not os.path.exists(credentials_file):
        print("⚠️ Skipping API test due to invalid credentials file path")
    else:
        try:
            # Find the most recent screenshot to test with
            screenshots_dir = "screenshots"
            screenshot_files = [
                os.path.join(screenshots_dir, f) 
                for f in os.listdir(screenshots_dir) 
                if f.endswith(".jpg") or f.endswith(".jpeg") or f.endswith(".png")
            ]
            
            if screenshot_files:
                test_image = sorted(screenshot_files, key=os.path.getmtime, reverse=True)[0]
                
                # Make the API call
                print(f"Making API call to describe image: {test_image}")
                result = describe_image_content(
                    image_path=test_image,
                    credentials_file=credentials_file
                )
                
                if "error" in result:
                    failures.append(f"API test failed: {result['error']}")
                elif "description" not in result:
                    failures.append("API test failed: No description returned")
                else:
                    print(f"✓ API call successful! Description: \"{result['description'][:100]}...\"")
            else:
                print("⚠️ No screenshots found for API testing")
        except Exception as e:
            traceback.print_exc()
            failures.append(f"API call failed with exception: {str(e)}")
    
    return total_tests, failures


def test_mcp_tools() -> Tuple[int, List[str]]:
    """Test creating the MCP server."""
    print("\n=== Testing MCP Server Creation ===")
    total_tests = 0
    failures = []
    
    # Test 1: Create MCP server
    total_tests += 1
    try:
        mcp = create_mcp_server("Test MCP Server")
        if mcp:
            print("✓ MCP server created successfully")
        else:
            failures.append("MCP server creation test failed: No server returned")
    except Exception as e:
        traceback.print_exc()
        failures.append(f"MCP server creation test failed with exception: {str(e)}")
    
    return total_tests, failures


def main():
    """Main test function."""
    print("===================================")
    print("TESTING SCREENSHOT PACKAGE")
    print("===================================")
    
    total_tests = 0
    all_failures = []
    
    # Run all tests
    constants_tests, constants_failures = test_constants()
    total_tests += constants_tests
    all_failures.extend(constants_failures)
    
    img_proc_tests, img_proc_failures = test_image_processing()
    total_tests += img_proc_tests
    all_failures.extend(img_proc_failures)
    
    capture_tests, capture_failures = test_capture()
    total_tests += capture_tests
    all_failures.extend(capture_failures)
    
    description_tests, description_failures = test_description()
    total_tests += description_tests
    all_failures.extend(description_failures)
    
    mcp_tests, mcp_failures = test_mcp_tools()
    total_tests += mcp_tests
    all_failures.extend(mcp_failures)
    
    # Print summary
    print("\n===================================")
    print("TEST SUMMARY")
    print("===================================")
    
    if all_failures:
        print(f"❌ VALIDATION FAILED - {len(all_failures)} of {total_tests} tests failed:")
        for failure in all_failures:
            print(f"  - {failure}")
        return 1
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Screenshot package is validated and ready for use")
        return 0


if __name__ == "__main__":
    """Run all tests and exit with appropriate code."""
    sys.exit(main())