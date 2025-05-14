#!/usr/bin/env python3
"""
MSS (Screenshot) Low-Level Module

This module provides low-level wrapper functions for the MSS library,
focusing on direct capture operations without additional processing.

This module is part of the Core Layer and should have no dependencies on
Presentation or Integration layers.

Sample input:
- Capture parameters (monitor number, region)

Expected output:
- Raw screenshot data and monitor information
"""

import mss
import mss.tools
import platform
from typing import Dict, List, Tuple, Optional, Union, Any
from PIL import Image
from loguru import logger


def get_monitors() -> List[Dict[str, int]]:
    """
    Get information about all available monitors.
    
    Returns:
        List[Dict[str, int]]: List of monitor dictionaries with keys:
            - top, left, width, height, monitor number
    """
    try:
        with mss.mss() as sct:
            # Skip the first monitor (which is the "all monitors" combined view)
            return [dict(monitor, **{'monitor_num': i}) for i, monitor in enumerate(sct.monitors) if i > 0]
    except Exception as e:
        logger.error(f"Failed to get monitors: {str(e)}", exc_info=True)
        return []


def capture_monitor(monitor_num: int = 1) -> Optional[Tuple[Image.Image, Dict[str, int]]]:
    """
    Capture screenshot of a specific monitor.
    
    Args:
        monitor_num: Monitor number (1 is primary, 0 is all monitors)
        
    Returns:
        Optional[Tuple[Image.Image, Dict[str, int]]]: Tuple of:
            - PIL Image object
            - Monitor information dictionary
            Returns None on failure
    """
    try:
        with mss.mss() as sct:
            if monitor_num >= len(sct.monitors):
                logger.error(f"Monitor number {monitor_num} out of range (max {len(sct.monitors)-1})")
                return None
                
            monitor = sct.monitors[monitor_num]
            sct_img = sct.grab(monitor)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            return img, monitor
    except Exception as e:
        logger.error(f"Failed to capture monitor {monitor_num}: {str(e)}", exc_info=True)
        return None


def capture_region(region: Dict[str, int]) -> Optional[Image.Image]:
    """
    Capture screenshot of a specific region.
    
    Args:
        region: Dictionary with keys: top, left, width, height
        
    Returns:
        Optional[Image.Image]: PIL Image object, or None on failure
    """
    # Validate the region parameter
    required_keys = ["top", "left", "width", "height"]
    if not all(key in region for key in required_keys):
        missing = [key for key in required_keys if key not in region]
        logger.error(f"Region missing required keys: {missing}")
        return None
        
    try:
        with mss.mss() as sct:
            sct_img = sct.grab(region)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            return img
    except Exception as e:
        logger.error(f"Failed to capture region: {str(e)}", exc_info=True)
        return None


def get_system_info() -> Dict[str, str]:
    """
    Get system information relevant to screenshots.
    
    Returns:
        Dict[str, str]: System information
    """
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
    }
    
    # Add MSS version if available
    try:
        info["mss_version"] = mss.__version__
    except AttributeError:
        info["mss_version"] = "unknown"
        
    return info


if __name__ == "__main__":
    """Validate MSS wrapper functions"""
    import sys
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    try:
        # Test 1: Get monitors
        total_tests += 1
        monitors = get_monitors()
        
        if not monitors or not isinstance(monitors, list):
            all_validation_failures.append(f"Get monitors test: Expected list of monitors, got {type(monitors)}")
            
        # Test 2: Capture primary monitor
        total_tests += 1
        result = capture_monitor(1)
        
        if not result or not isinstance(result[0], Image.Image):
            all_validation_failures.append(f"Capture monitor test: Failed to capture primary monitor")
            
        # Test 3: Get system info
        total_tests += 1
        system_info = get_system_info()
        
        if not system_info or not isinstance(system_info, dict):
            all_validation_failures.append(f"Get system info test: Expected dict, got {type(system_info)}")
            
        if "platform" not in system_info:
            all_validation_failures.append(f"Get system info test: Missing 'platform' key")
            
        # Test 4: Capture specific region - take a small region from the primary monitor
        total_tests += 1
        try:
            if result:  # Use monitor info from test 2
                monitor = result[1]
                region = {
                    "left": monitor["left"] + 10,
                    "top": monitor["top"] + 10,
                    "width": 100,
                    "height": 100
                }
                
                region_result = capture_region(region)
                
                if not region_result or not isinstance(region_result, Image.Image):
                    all_validation_failures.append(f"Capture region test: Failed to capture region")
                    
                # On Retina displays, the actual capture size might be larger due to scaling
                # Check that image was produced rather than exact dimensions
                if region_result and (region_result.width < 50 or region_result.height < 50):
                    all_validation_failures.append(
                        f"Capture region test: Image size is too small, got {region_result.size}"
                    )
        except Exception as e:
            all_validation_failures.append(f"Capture region test failed: {str(e)}")
            
    except Exception as e:
        all_validation_failures.append(f"Unexpected error in validation: {str(e)}")
        
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("MSS wrapper functions are validated and ready for use")
        sys.exit(0)  # Exit with success code