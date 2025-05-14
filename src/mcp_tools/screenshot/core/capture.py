#!/usr/bin/env python3
"""
Screenshot Capture Module

This module provides functions for capturing screenshots of the screen or specific regions
using the MSS library instead of PyAutoGUI for improved performance and reliability.

This module is part of the Core Layer and should have no dependencies on
Presentation or Integration layers.

Sample input:
- quality=30, region=None (for full screen)
- quality=30, region=[100, 100, 300, 200] (for specific region)
- quality=30, region="right_half" (for right half of screen)

Expected output:
- Dictionary with:
  - content: List with a single image object (type, base64 data, MIME type).
  - file: Path to the saved screenshot file.
  - On error: error message as a string.
"""

import os
import time
import base64
import uuid
from typing import Dict, List, Union, Optional, Any, Tuple, cast

import mss
from PIL import Image
from loguru import logger

from mcp_tools.screenshot.core.constants import IMAGE_SETTINGS
from mcp_tools.screenshot.core.image_processing import resize_image_if_needed, compress_image_to_buffer
from mcp_tools.screenshot.core.mss import capture_monitor, capture_region, get_monitors


def capture_screenshot(
    quality: int = IMAGE_SETTINGS["DEFAULT_QUALITY"],
    region: Optional[Union[List[int], str]] = None,
    output_dir: str = "screenshots",
    include_raw: bool = False
) -> Dict[str, Any]:
    """
    Captures a screenshot of the entire desktop or a specified region.
    
    Args:
        quality: JPEG compression quality (1-100)
        region: Region coordinates [x, y, width, height] or "right_half"
        output_dir: Directory to save screenshot
        include_raw: Whether to also save the raw uncompressed PNG
        
    Returns:
        dict: Response containing:
            - content: List with image object (type, base64, MIME type)
            - file: Path to the saved screenshot file
            - raw_file: Path to raw PNG (if include_raw=True)
            - On error: error message as string
    """
    logger.info(f"Screenshot requested with quality={quality}, region={region}")

    try:
        # Validate region parameter
        if region is not None and not isinstance(region, (list, str)):
            return {"error": f"Invalid region parameter: must be a list or string preset, got {type(region)}"}

        if isinstance(region, list) and len(region) != 4:
            return {"error": f"Region must have 4 elements [x, y, width, height], got {len(region)}"}

        if isinstance(region, str) and region not in ["right_half", "left_half", "top_half", "bottom_half"]:
            return {"error": f"String region must be a valid preset, got {region}"}

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Generate unique filename with timestamp for easy sorting
        timestamp = int(time.time() * 1000)
        filename = f"screenshot_{timestamp}.jpeg"
        path = os.path.join(output_dir, filename)
        
        # Create raw file name if needed
        raw_path = None
        if include_raw:
            raw_filename = f"raw_screenshot_{timestamp}.png"
            raw_path = os.path.join(output_dir, raw_filename)

        # Capture screenshot
        with mss.mss() as sct:
            # Get primary monitor if no region is specified
            monitor = sct.monitors[1]  # Primary monitor
            logger.info(f"Selected monitor: {monitor}")

            if region == "right_half":
                # Calculate right half
                width = monitor["width"]
                height = monitor["height"]
                x = width // 2
                y = 0
                w = width // 2
                h = height
                logger.info(f"Taking right half screenshot at x={x}, y={y}, width={w}, height={h}")

                capture_region = {"top": y, "left": x, "width": w, "height": h}
                sct_img = sct.grab(capture_region)
            elif region == "left_half":
                # Calculate left half
                width = monitor["width"]
                height = monitor["height"]
                x = 0
                y = 0
                w = width // 2
                h = height
                logger.info(f"Taking left half screenshot at x={x}, y={y}, width={w}, height={h}")

                capture_region = {"top": y, "left": x, "width": w, "height": h}
                sct_img = sct.grab(capture_region)
            elif region == "top_half":
                # Calculate top half
                width = monitor["width"]
                height = monitor["height"]
                x = 0
                y = 0
                w = width
                h = height // 2
                logger.info(f"Taking top half screenshot at x={x}, y={y}, width={w}, height={h}")

                capture_region = {"top": y, "left": x, "width": w, "height": h}
                sct_img = sct.grab(capture_region)
            elif region == "bottom_half":
                # Calculate bottom half
                width = monitor["width"]
                height = monitor["height"]
                x = 0
                y = height // 2
                w = width
                h = height // 2
                logger.info(f"Taking bottom half screenshot at x={x}, y={y}, width={w}, height={h}")

                capture_region = {"top": y, "left": x, "width": w, "height": h}
                sct_img = sct.grab(capture_region)
            elif isinstance(region, list):
                x, y, w, h = region
                logger.info(f"Taking region screenshot at x={x}, y={y}, width={w}, height={h}")

                capture_region = {"top": y, "left": x, "width": w, "height": h}
                sct_img = sct.grab(capture_region)
            else:
                logger.info("Taking full screen screenshot")
                sct_img = sct.grab(monitor)

            # Convert to PIL Image
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            
            # Save raw PNG if requested
            if include_raw and raw_path:
                logger.info(f"Saving raw PNG to {raw_path}")
                img.save(raw_path, format="PNG")

        # Clamp quality to acceptable range
        original_quality = quality
        quality = max(IMAGE_SETTINGS["MIN_QUALITY"],
                      min(quality, IMAGE_SETTINGS["MAX_QUALITY"]))

        if quality != original_quality:
            logger.info(
                f"Adjusted quality from {original_quality} to {quality} "
                f"(min={IMAGE_SETTINGS['MIN_QUALITY']}, max={IMAGE_SETTINGS['MAX_QUALITY']})"
            )

        # Resize if needed
        img = resize_image_if_needed(
            img,
            IMAGE_SETTINGS["MAX_WIDTH"],
            IMAGE_SETTINGS["MAX_HEIGHT"]
        )

        # Compress to fit size limits
        img_bytes = compress_image_to_buffer(
            img,
            quality,
            IMAGE_SETTINGS["MIN_QUALITY"],
            IMAGE_SETTINGS["MAX_FILE_SIZE"]
        )

        # Save the compressed image
        with open(path, "wb") as f:
            f.write(img_bytes)

        # Encode to base64
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        logger.info(f"Image encoded as base64 string ({len(img_b64)} characters)")

        # Create response
        response = {
            "content": [
                {
                    "type": "image",
                    "data": img_b64,
                    "mimeType": "image/jpeg"
                }
            ],
            "file": path
        }
        
        # Add raw file path if saved
        if include_raw and raw_path:
            response["raw_file"] = raw_path

        logger.info(f"Screenshot captured successfully: {path}")
        return response

    except mss.exception.ScreenShotError as e:
        logger.error(f"Screenshot capture failed: {str(e)}", exc_info=True)
        return {"error": f"Screenshot capture failed: {str(e)}"}
    except IOError as e:
        logger.error(f"File I/O error: {str(e)}", exc_info=True)
        return {"error": f"File I/O error: {str(e)}"}
    except Exception as e:
        logger.error(f"Screenshot failed: {str(e)}", exc_info=True)
        return {"error": f"Screenshot failed: {str(e)}"}


def get_screen_regions() -> Dict[str, Dict[str, int]]:
    """
    Get information about available screen regions.
    
    Returns:
        Dict: Dictionary of available regions with their dimensions
    """
    regions = {}
    
    try:
        with mss.mss() as sct:
            # Get all monitors
            for i, monitor in enumerate(sct.monitors):
                if i == 0:  # Skip the "all monitors" entry
                    continue
                regions[f"monitor_{i}"] = {
                    "top": monitor["top"],
                    "left": monitor["left"],
                    "width": monitor["width"],
                    "height": monitor["height"]
                }
            
            # Add special regions
            primary = sct.monitors[1]  # Primary monitor
            width = primary["width"]
            height = primary["height"]
            
            # Add half screen regions
            regions["right_half"] = {
                "top": 0,
                "left": width // 2,
                "width": width // 2,
                "height": height
            }
            
            regions["left_half"] = {
                "top": 0,
                "left": 0,
                "width": width // 2,
                "height": height
            }
            
            regions["top_half"] = {
                "top": 0,
                "left": 0,
                "width": width,
                "height": height // 2
            }
            
            regions["bottom_half"] = {
                "top": height // 2,
                "left": 0,
                "width": width,
                "height": height // 2
            }
            
    except Exception as e:
        logger.error(f"Failed to get screen regions: {str(e)}")
        return {}
    
    return regions


if __name__ == "__main__":
    """Validate screenshot capture functions with real test data"""
    import sys
    import os
    import shutil
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Create test directory
    test_dir = ".test_screenshots"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Test 1: Capture full screen
        total_tests += 1
        result = capture_screenshot(quality=30, output_dir=test_dir)
        
        if "error" in result:
            all_validation_failures.append(f"Full screen capture test: {result['error']}")
        elif "file" not in result or not os.path.exists(result["file"]):
            all_validation_failures.append(f"Full screen capture test: File not created")
        elif "content" not in result or not result["content"][0]["data"]:
            all_validation_failures.append(f"Full screen capture test: No image content returned")
            
        # Test 2: Capture right half of screen
        total_tests += 1
        result = capture_screenshot(quality=30, region="right_half", output_dir=test_dir)
        
        if "error" in result:
            all_validation_failures.append(f"Right half capture test: {result['error']}")
        elif "file" not in result or not os.path.exists(result["file"]):
            all_validation_failures.append(f"Right half capture test: File not created")
        
        # Test 3: Invalid region parameter
        total_tests += 1
        result = capture_screenshot(quality=30, region="invalid_region", output_dir=test_dir)
        
        if "error" not in result:
            all_validation_failures.append(
                f"Invalid region test: Expected error for 'invalid_region', but got success"
            )
        
        # Test 4: Get screen regions
        total_tests += 1
        regions = get_screen_regions()
        
        if not regions or "right_half" not in regions:
            all_validation_failures.append(f"Get screen regions test: Failed to get regions")
        
        # Test 5: Raw image capture
        total_tests += 1
        result = capture_screenshot(quality=30, include_raw=True, output_dir=test_dir)
        
        if "error" in result:
            all_validation_failures.append(f"Raw image capture test: {result['error']}")
        elif "raw_file" not in result or not os.path.exists(result["raw_file"]):
            all_validation_failures.append(f"Raw image capture test: Raw file not created")
            
    finally:
        # Clean up test directory
        shutil.rmtree(test_dir, ignore_errors=True)
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Screenshot capture functions are validated and ready for use")
        sys.exit(0)  # Exit with success code