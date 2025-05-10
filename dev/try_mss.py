#!/usr/bin/env python3
"""
Test script to take a screenshot using MSS library, focused on the right side of the screen
"""

import mss
import os
import sys
import time
import uuid

def take_screenshot():
    """Take a screenshot of the right side of the screen using MSS"""
    # Ensure screenshot directory exists
    os.makedirs("screenshots", exist_ok=True)
    
    # Generate unique filename with UUID and timestamp for easy sorting
    short_uuid = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
    timestamp = int(time.time() * 1000)
    filename = f"screenshots/{short_uuid}_mss_screenshot_{timestamp}.png"
    
    # Take screenshot of right side of screen
    with mss.mss() as sct:
        # Get screen size
        monitor = sct.monitors[1]  # Primary monitor
        screen_width = monitor["width"]
        
        # Define region (right half of screen)
        region = {
            "top": 0, 
            "left": screen_width // 2,  # Start at middle of screen
            "width": screen_width // 2,  # Take right half
            "height": monitor["height"]
        }
        
        # Capture the specified region
        img = sct.grab(region)
        
        # Save the screenshot
        mss.tools.to_png(img.rgb, img.size, output=filename)
        
        print(f"Screenshot saved to: {filename}")
        return filename

if __name__ == "__main__":
    take_screenshot()