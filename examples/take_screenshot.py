#!/usr/bin/env python3
"""
Example of using the screenshot module to take a screenshot

This script demonstrates how to use the screenshot module to capture a screenshot
of the entire screen, a specific region, or the right half of the screen.

Usage:
    python take_screenshot.py
"""

import os
import sys
import base64
from PIL import Image
from io import BytesIO

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the screenshot module
from src.mcp_tools.screenshot import capture_screenshot


def save_example_screenshot():
    """Take a screenshot of the right half of the screen and save it"""
    
    print("Taking screenshot of the right half of the screen...")
    result = capture_screenshot(quality=70, region="right_half")
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"Screenshot saved to: {result['file']}")
    
    # Optionally, display the image
    try:
        # Extract base64 data
        img_data = result["content"][0]["data"]
        
        # Decode base64 and open with PIL
        img_bytes = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_bytes))
        
        # Display image info
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")
        print(f"Image format: {img.format}")
    except Exception as e:
        print(f"Error displaying image: {str(e)}")


if __name__ == "__main__":
    save_example_screenshot()