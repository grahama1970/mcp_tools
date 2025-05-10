#!/usr/bin/env python
"""
Combined Screenshot and Description Tool

This script captures a screenshot and immediately sends it for AI description,
without returning the large base64 encoded string in the intermediate response.

It can capture the full screen or a specific region (like right half).
"""

import os
import time
import json
import base64
from datetime import datetime
from mss import mss
from PIL import Image, ImageOps
import io
import logging
from loguru import logger
import requests

# Configure logger
log_file = "logs/combined_screenshot.log"
os.makedirs(os.path.dirname(log_file), exist_ok=True)
logger.add(log_file, rotation="10 MB")

# Configuration
SCREENSHOT_DIR = "screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def capture_screenshot(region=None, quality=30):
    """Capture a screenshot with MSS.
    
    Args:
        region: List [x, y, width, height] or "right_half" for right half of the screen
        quality: JPEG quality (1-100)
        
    Returns:
        dict: Contains file path, image data, and result info
    """
    try:
        timestamp = int(time.time() * 1000)
        raw_filename = f"raw_screenshot_{timestamp}.png"
        jpeg_filename = f"screenshot_{timestamp}.jpeg"
        raw_filepath = os.path.join(SCREENSHOT_DIR, raw_filename)
        jpeg_filepath = os.path.join(SCREENSHOT_DIR, jpeg_filename)
        
        logger.info(f"Capturing screenshot to {raw_filepath}")
        
        with mss() as sct:
            if region == "right_half":
                # Get the first (primary) monitor
                monitor = sct.monitors[1]
                # Calculate the right half
                width = monitor["width"]
                height = monitor["height"]
                region = [width // 2, 0, width // 2, height]
                logger.info(f"Capturing right half of screen: {region}")
                
            # Capture based on region or full screen
            if isinstance(region, list) and len(region) == 4:
                x, y, width, height = region
                monitor = {"left": x, "top": y, "width": width, "height": height}
                img = sct.grab(monitor)
            else:
                img = sct.grab(sct.monitors[1])  # Primary monitor
                
            # Save raw PNG screenshot
            mss.tools.to_png(img.rgb, img.size, output=raw_filepath)
            
            # Convert to JPEG with specified quality
            png_image = Image.open(raw_filepath)
            
            # Save as JPEG
            png_image.save(jpeg_filepath, "JPEG", quality=quality)
            logger.info(f"Saved JPEG screenshot to {jpeg_filepath} with quality {quality}")
            
            # Get base64 but don't return it in the response - it's only for internal use
            buffered = io.BytesIO()
            png_image.save(buffered, format="JPEG", quality=quality)
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            return {
                "file": jpeg_filepath,
                "raw_file": raw_filepath,
                "base64_data": img_base64  # Only for internal use
            }
    
    except Exception as e:
        error_msg = f"Screenshot error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

def describe_image(screenshot_data, prompt="Describe this screenshot in detail.", model="vertex_ai/gemini-2.5-pro-preview-05-06"):
    """Get an AI description of the screenshot
    
    Args:
        screenshot_data: Dict containing the screenshot data
        prompt: Text prompt for the AI model
        model: AI model to use
        
    Returns:
        dict: Contains the AI description
    """
    try:
        logger.info(f"Getting description using prompt: {prompt}")
        
        if "error" in screenshot_data:
            return {"error": screenshot_data["error"]}
        
        # Here you would implement the call to the AI model (Gemini)
        # This is a placeholder as the actual implementation depends on your setup
        
        # For demonstration purposes
        description = "This is a placeholder description. In a real implementation, this would be the response from the AI model."
        
        # In a real implementation, you would use something like:
        """
        response = requests.post(
            "YOUR_AI_ENDPOINT",
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            json={
                "model": model,
                "prompt": prompt,
                "image": screenshot_data["base64_data"]
            }
        )
        description = response.json()["description"]
        """
        
        result = {
            "description": description,
            "filename": screenshot_data["file"],
            "confidence": 5
        }
        
        logger.info("Successfully obtained image description")
        return result
        
    except Exception as e:
        error_msg = f"Description error: {str(e)}"
        logger.error(error_msg)
        return {"error": error_msg}

def capture_and_describe(region=None, quality=30, prompt="Describe this screenshot in detail.", model="vertex_ai/gemini-2.5-pro-preview-05-06"):
    """Combined function to capture a screenshot and immediately get its description
    
    Args:
        region: List [x, y, width, height] or "right_half" for right half of the screen
        quality: JPEG quality (1-100)
        prompt: Text prompt for the AI model
        model: AI model to use
        
    Returns:
        dict: Contains the description and file info, but not the large base64 data
    """
    # Step 1: Capture the screenshot
    screenshot_data = capture_screenshot(region, quality)
    
    if "error" in screenshot_data:
        return {"error": screenshot_data["error"]}
    
    # Step 2: Get the description
    description_result = describe_image(screenshot_data, prompt, model)
    
    # Step 3: Return the combined result without the base64 data
    result = {
        "description": description_result.get("description", "No description available"),
        "file": screenshot_data["file"],
        "confidence": description_result.get("confidence", 0)
    }
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Capture a screenshot and get AI description")
    parser.add_argument("--region", default="right_half", help="Region to capture: 'right_half' or a list [x,y,width,height]")
    parser.add_argument("--quality", type=int, default=30, help="JPEG quality (1-100)")
    parser.add_argument("--prompt", default="Describe this screenshot in detail.", help="Prompt for the AI model")
    parser.add_argument("--model", default="vertex_ai/gemini-2.5-pro-preview-05-06", help="AI model to use")
    
    args = parser.parse_args()
    
    result = capture_and_describe(
        region=args.region,
        quality=args.quality,
        prompt=args.prompt,
        model=args.model
    )
    
    print(json.dumps(result, indent=2))