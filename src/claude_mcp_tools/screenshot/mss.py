#!/usr/bin/env python3
"""
Claude MCP Screenshot Tool using MSS library

A screenshot utility for Claude Desktop/Code using MSS instead of PyAutoGUI
"""

import os
import time
import base64
import uuid
from PIL import Image
import mss
from mcp.server.fastmcp import FastMCP
from ..utils.logging import truncate_large_value, setup_logger

# Configure logger
logger = setup_logger("mss_screenshot", "logs/mss_screenshot.log")

# Default settings
MAX_WIDTH = 640  # Reduced to ensure token limits
MAX_HEIGHT = 640  # Reduced to ensure token limits
MIN_QUALITY = 30
MAX_QUALITY = 70  # Lowered max quality to reduce file size
MAX_FILE_SIZE = 350_000  # Reduced to 350kB to stay within token limits

def capture_screenshot(quality: int = 30, region: list = None) -> dict:
    """
    Captures a screenshot of the entire desktop or a specified region, returning it as a base64-encoded JPEG.
    Uses MSS library instead of PyAutoGUI.

    Args:
        quality (int, optional): JPEG compression quality (1-100). Defaults to 30.
        region (list, optional): Region coordinates [x, y, width, height]. Defaults to None (full screen).

    Returns:
        dict: MCP-compliant response containing:
            - content: List with a single image object (type, base64 data, MIME type).
            - file: Path to the saved screenshot file.
            On error:
            - error: Error message as a string.
    """
    logger.info(f"Screenshot requested with quality={quality}, region={region}")

    try:
        # Ensure output directory exists
        outdir = "screenshots"
        os.makedirs(outdir, exist_ok=True)

        # Generate unique filename with UUID and timestamp for easy sorting
        short_uuid = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        timestamp = int(time.time() * 1000)
        filename = f"{short_uuid}_screenshot_{timestamp}.jpg"
        path = os.path.join(outdir, filename)
        
        # Capture screenshot using MSS
        with mss.mss() as sct:
            # Get primary monitor if no region is specified
            monitor = sct.monitors[1]  # Primary monitor
            
            if region and isinstance(region, list) and len(region) == 4:
                x, y, w, h = region
                logger.info(f"Taking region screenshot at x={x}, y={y}, width={w}, height={h}")
                capture_region = {
                    "top": y,
                    "left": x,
                    "width": w,
                    "height": h
                }
                sct_img = sct.grab(capture_region)
            else:
                logger.info("Taking full screen screenshot")
                sct_img = sct.grab(monitor)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        
        # Clamp quality
        original_quality = quality
        quality = max(MIN_QUALITY, min(quality, MAX_QUALITY))
        if quality != original_quality:
            logger.info(f"Adjusted quality from {original_quality} to {quality} (min={MIN_QUALITY}, max={MAX_QUALITY})")
        
        # Resize if needed - maintain aspect ratio
        width, height = img.size
        if width > MAX_WIDTH or height > MAX_HEIGHT:
            # Calculate scale factor to maintain aspect ratio
            scale_factor = min(MAX_WIDTH / width, MAX_HEIGHT / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
            img = img.resize((new_width, new_height), Image.LANCZOS)
            logger.debug(f"Resized image dimensions: {img.size}")
        
        # Save compressed JPEG to disk
        img.save(path, format="JPEG", quality=quality)
        logger.debug(f"Initial save with quality={quality}")
        
        # Read file and encode as base64, re-compress if needed to fit size limit
        with open(path, "rb") as f:
            img_bytes = f.read()
        
        compress_quality = quality
        compression_iterations = 0
        while len(img_bytes) > MAX_FILE_SIZE and compress_quality > MIN_QUALITY:
            compression_iterations += 1
            old_size = len(img_bytes)
            compress_quality = max(MIN_QUALITY, compress_quality - 10)
            logger.info(f"Image size ({old_size/1024:.1f} KB) exceeds limit ({MAX_FILE_SIZE/1024:.1f} KB), reducing quality to {compress_quality}")
            
            img.save(path, format="JPEG", quality=compress_quality)
            with open(path, "rb") as f:
                img_bytes = f.read()
            
            logger.debug(f"Compression iteration {compression_iterations}: size reduced from {old_size/1024:.1f} KB to {len(img_bytes)/1024:.1f} KB")
        
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        logger.info(f"Image encoded as base64 string ({len(img_b64)} characters)")
        
        # Create a response with the full base64 data
        response = {
            "content": [{"type": "image", "data": img_b64, "mimeType": "image/jpeg"}],
            "file": path,
        }
        
        # Create a log-safe version of the response with truncated base64 data
        log_safe_response = {
            "content": [
                {
                    "type": "image",
                    "data": truncate_large_value(img_b64, max_str_len=100),
                    "mimeType": "image/jpeg"
                }
            ],
            "file": path,
        }
        
        # Log the truncated response
        logger.info(f"Screenshot captured successfully: {path}")
        logger.debug(f"Screenshot response (truncated): {log_safe_response}")
        
        # Return the full response to the client
        return response
    except Exception as e:
        logger.error(f"Screenshot failed: {str(e)}", exc_info=True)
        # Return error in MCP-compliant format
        return {"error": f"Screenshot failed: {str(e)}"}


def create_mcp_tool():
    """Create and configure the MCP tool for standalone usage"""
    mcp = FastMCP("MSS Screenshot Tool")
    logger.info("Initialized FastMCP server for MSS screenshot tool")
    
    # Register the screenshot tool
    @mcp.tool()
    def screenshot(quality: int = 30, region: list = None) -> dict:
        return capture_screenshot(quality, region)
    
    return mcp


if __name__ == "__main__":
    mcp = create_mcp_tool()
    logger.info("Starting FastMCP server for MSS screenshot tool")
    mcp.run()