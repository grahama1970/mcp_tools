#!/usr/bin/env python3
r"""
Claude MCP Screenshot Tool (FastMCP Version)

A native screenshot utility for Claude Desktop/Code, leveraging the MCP Python SDK to capture and encode screenshots for agent-driven analysis.

--------------------------------------------------------
## Purpose and Agent-Driven Context

This tool enables Claude agents to capture desktop or region-specific screenshots when:
- Browser automation (e.g., Playwright) is blocked by bot detection.
- Content lies outside the browser (e.g., desktop apps, system dialogs, OS UI).
- The agent needs to analyze visual content and explain it to the user.

The agent invokes this tool based on user intent (e.g., "Take a screenshot of the right side of the screen and describe it") and receives a base64-encoded JPEG for further processing.

--------------------------------------------------------
## Agent Workflow Example

**User Prompt**:
Can you use your screenshot tool to capture the right side of the screen
and explain what you see?

**Agent Reasoning** (internal):
- Browser automation is inapplicable or blocked.
- Desktop screenshot required; invoke MCP screenshot tool.

**Agent Tool Call**:
```json
screenshot:screenshot (MCP)(quality: 30, region: [640,0,640,900])
```
- If the output exceeds token limits, the agent retries with a smaller region or lower quality.

**Agent Tool Result**:
- Receives base64-encoded JPEG.
- Analyzes the image and describes its contents to the user.

--------------------------------------------------------
## Compression and Resizing Rationale

- Large screenshots generate oversized base64 strings, risking Claude's 25,000-token result limit.
- This tool resizes and compresses images to ensure compatibility with token constraints.

--------------------------------------------------------
## Installation Instructions (with uv)

```bash
cd ~/claude-mcp-configs
uv venv
uv pip install pyautogui pillow loguru mcp
```

--------------------------------------------------------
## Third-Party Documentation

- PyAutoGUI: https://pyautogui.readthedocs.io/en/latest/
- Pillow: https://pillow.readthedocs.io/en/stable/
- Loguru: https://loguru.readthedocs.io/en/stable/
- MCP SDK: https://github.com/modelcontextprotocol/python-sdk

--------------------------------------------------------
## MCP Tool Input Example

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "quality": 30,
    "region": [100, 100, 500, 400]
  }
}
```

## MCP Tool Output Example

```json
{
  "content": [
    {
      "type": "image",
      "data": "<base64-encoded-image>",
      "mimeType": "image/jpeg"
    }
  ],
  "file": "screenshots/screenshot_123456789.jpg"
}
```
"""

import os
import time
import base64
import uuid
import pyautogui
from PIL import Image
from mcp.server.fastmcp import FastMCP
from log_utils import truncate_large_value
from validate_screenshot_tool import validate_screenshot_tool
from loguru import logger


# Configure logger - remove default stderr handler to avoid breaking MCP JSON-RPC
logger.remove()  # Remove default handler that outputs to stderr
logger.add(
    "logs/mcp_server.log",
    rotation="10 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Initialize FastMCP server for screenshot tool
mcp = FastMCP("Screenshot Tool")
logger.info("Initialized FastMCP server for screenshot tool")


@mcp.tool()
def screenshot(quality: int = 30, region: list = None) -> dict:
    """
    Captures a screenshot of the entire desktop or a specified region, returning it as a base64-encoded JPEG.

    Args:
        quality (int, optional): JPEG compression quality (1-100). Defaults to 30.
        region (list, optional): Region coordinates [x, y, width, height]. Defaults to None (full screen).

    Returns:
        dict: MCP-compliant response containing:
            - content: List with a single image object (type, base64 data, MIME type).
            - file: Path to the saved screenshot file.
            On error:
            - error: Error message as a string.

    Raises:
        Exception: If screenshot capture, file saving, or encoding fails.
    """
    # Compression/resize settings
    MAX_WIDTH = 640  # Reduced to ensure token limits
    MAX_HEIGHT = 640  # Reduced to ensure token limits
    MIN_QUALITY = 30
    MAX_QUALITY = 70  # Lowered max quality to reduce file size
    MAX_FILE_SIZE = 350_000  # Reduced to 350kB to stay within token limits

    logger.info(f"Screenshot requested with quality={quality}, region={region}")

    try:
        # Capture screenshot (region or full screen)
        if region and isinstance(region, list) and len(region) == 4:
            x, y, w, h = region
            logger.info(f"Taking region screenshot at x={x}, y={y}, width={w}, height={h}")
            img = pyautogui.screenshot(region=(x, y, w, h))
        else:
            logger.info("Taking full screen screenshot")
            img = pyautogui.screenshot()

        # Convert to RGB for JPEG compatibility
        img = img.convert("RGB")
        logger.debug(f"Original image dimensions: {img.size}")

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

        # Clamp quality
        original_quality = quality
        quality = max(MIN_QUALITY, min(quality, MAX_QUALITY))
        if quality != original_quality:
            logger.info(f"Adjusted quality from {original_quality} to {quality} (min={MIN_QUALITY}, max={MAX_QUALITY})")

        # Ensure output directory exists
        outdir = "screenshots"
        os.makedirs(outdir, exist_ok=True)

        # Generate unique filename with UUID and timestamp for easy sorting
        short_uuid = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        timestamp = int(time.time() * 1000)
        filename = f"{short_uuid}_screenshot_{timestamp}.jpg"
        path = os.path.join(outdir, filename)
        logger.info(f"Saving screenshot to {path}")

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


if __name__ == "__main__":
    # If VALIDATE_TOOL environment variable is set, run validation
    import os
    if os.environ.get("VALIDATE_TOOL"):
        logger.info("Running validation for screenshot tool")
        validate_screenshot_tool()
    else:
        # Start the FastMCP server
        logger.info("Starting FastMCP server")
        mcp.run()
