#!/usr/bin/env python3
"""
Standalone Screenshot Module with AI Description

A self-contained screenshot utility that captures screenshots and provides 
AI-powered image descriptions. This standalone script contains all necessary 
components from the mcp_tools package.

Third-party package documentation:
- MSS: https://python-mss.readthedocs.io/
- PIL: https://pillow.readthedocs.io/en/stable/
- LiteLLM: https://docs.litellm.ai/docs/
- MCP: https://github.com/anthropics/anthropic-mcp
- Typer: https://typer.tiangolo.com/

Sample input:
- python standalone_screenshot.py screenshot --quality 30 --region right_half
- python standalone_screenshot.py describe --prompt "What's visible in this image?"

Expected output:
- Screenshot file path and operation result
- For description: The AI-generated description of the image content
"""

import os
import sys
import time
import base64
import json
import io
import uuid
from typing import Dict, List, Union, Optional, Any, Tuple, cast

import mss
from PIL import Image
from loguru import logger
import typer
from litellm import completion
from mcp.server.fastmcp import FastMCP

# =====================================================================
# Constants
# =====================================================================

# Image settings for capture and processing
IMAGE_SETTINGS: Dict[str, Any] = {
    "MAX_WIDTH": 640,  # Maximum width for resized images
    "MAX_HEIGHT": 640,  # Maximum height for resized images
    "MIN_QUALITY": 30,  # Minimum JPEG compression quality
    "MAX_QUALITY": 70,  # Maximum JPEG compression quality
    "DEFAULT_QUALITY": 30,  # Default quality if none specified
    "MAX_FILE_SIZE": 350_000,  # Maximum file size in bytes (350kB)
}

# Logging settings
LOG_MAX_STR_LEN: int = 100  # Maximum string length for truncated logging

# Configure logger
logger.remove()  # Remove default handler that outputs to stderr
logger.add(
    "logs/standalone.log",
    rotation="10 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Define the response schema for image description
DESCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {
            "type": "string",
            "description": "Detailed description of the image content"
        },
        "filename": {
            "type": "string",
            "description": "The name of the image file"
        },
        "confidence": {
            "type": "integer",
            "description": "Confidence score (1-5) on accuracy of description given image quality and possible compression artifacts",
            "minimum": 1,
            "maximum": 5
        }
    },
    "required": ["description", "filename", "confidence"]
}

# Initialize typer app
app = typer.Typer(help="Claude MCP Screenshot Tool with MSS")

# Initialize FastMCP server for screenshot tool
mcp = FastMCP("MSS Screenshot Tool")
logger.info("Initialized FastMCP server for MSS screenshot tool")

# =====================================================================
# Image Processing Functions
# =====================================================================

def truncate_large_value(value, max_str_len=LOG_MAX_STR_LEN):
    """
    Truncates large string values for logging purposes.
    
    Args:
        value: The string value to truncate
        max_str_len: Maximum string length to allow
        
    Returns:
        Truncated string
    """
    if isinstance(value, str) and len(value) > max_str_len:
        return f"{value[:max_str_len]}... [truncated, {len(value)} chars total]"
    return value


def resize_image_if_needed(
    img: Image.Image, 
    max_width: int = IMAGE_SETTINGS["MAX_WIDTH"], 
    max_height: int = IMAGE_SETTINGS["MAX_HEIGHT"]
) -> Image.Image:
    """
    Resizes an image if it exceeds maximum dimensions while preserving aspect ratio.
    
    Args:
        img: PIL Image object to resize
        max_width: Maximum width allowed
        max_height: Maximum height allowed
        
    Returns:
        PIL.Image: Resized image or original if no resize needed
    """
    width, height = img.size
    if width <= max_width and height <= max_height:
        return img

    # Calculate scale factor to maintain aspect ratio
    scale_factor = min(max_width / width, max_height / height)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)

    logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
    return img.resize((new_width, new_height), Image.LANCZOS)


def ensure_rgb(img: Image.Image) -> Image.Image:
    """
    Converts image to RGB mode if needed for JPEG compatibility.
    
    Args:
        img: PIL Image object to convert
        
    Returns:
        PIL.Image: Image in RGB mode
    """
    if img.mode == 'RGBA':
        # Create a white background image
        background = Image.new('RGB', img.size, (255, 255, 255))
        # Paste the image using the alpha channel as mask
        background.paste(img, mask=img.split()[3])
        return background
    elif img.mode != 'RGB':
        # Convert any other mode to RGB
        return img.convert('RGB')
    return img


def compress_image_to_buffer(
    img: Image.Image, 
    initial_quality: int = IMAGE_SETTINGS["DEFAULT_QUALITY"],
    min_quality: int = IMAGE_SETTINGS["MIN_QUALITY"],
    max_file_size: int = IMAGE_SETTINGS["MAX_FILE_SIZE"]
) -> bytes:
    """
    Compresses an image to fit within size limits and returns as bytes buffer.
    Converts non-RGB images to RGB for JPEG compatibility.
    
    Args:
        img: PIL Image object to compress
        initial_quality: Initial JPEG quality setting (1-100)
        min_quality: Minimum quality to use if compression is needed
        max_file_size: Maximum file size in bytes
        
    Returns:
        bytes: Compressed image bytes
    """
    # Convert to RGB if needed
    img = ensure_rgb(img)

    # Set up buffer
    buffer = io.BytesIO()

    # Initial save with specified quality
    img.save(buffer, format="JPEG", quality=initial_quality)
    buffer.seek(0)
    img_bytes = buffer.getvalue()

    # Iterate compression if needed
    compress_quality = initial_quality
    compression_iterations = 0
    
    while len(img_bytes) > max_file_size and compress_quality > min_quality:
        compression_iterations += 1
        old_size = len(img_bytes)
        compress_quality = max(min_quality, compress_quality - 10)

        logger.info(
            f"Image size ({old_size / 1024:.1f} KB) exceeds limit "
            f"({max_file_size / 1024:.1f} KB), reducing quality to {compress_quality}"
        )

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=compress_quality)
        buffer.seek(0)
        img_bytes = buffer.getvalue()

        logger.debug(
            f"Compression iteration {compression_iterations}: "
            f"size reduced from {old_size / 1024:.1f} KB to {len(img_bytes) / 1024:.1f} KB"
        )

    return img_bytes


# =====================================================================
# Screenshot Capture Functions
# =====================================================================

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
            return {"error": f"Invalid region parameter: must be a list or 'right_half', got {type(region)}"}

        if isinstance(region, list) and len(region) != 4:
            return {"error": f"Region must have 4 elements [x, y, width, height], got {len(region)}"}

        if isinstance(region, str) and region != "right_half":
            return {"error": f"String region must be 'right_half', got {region}"}

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
            if include_raw:
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


# =====================================================================
# Image Description Functions
# =====================================================================

def prepare_image_for_multimodal(
    image_path: str, 
    max_width: int = IMAGE_SETTINGS["MAX_WIDTH"],
    max_height: int = IMAGE_SETTINGS["MAX_HEIGHT"], 
    initial_quality: int = IMAGE_SETTINGS["DEFAULT_QUALITY"]
) -> str:
    """
    Prepares an image for multimodal API calls:
    1. Opens the image file
    2. Resizes if needed
    3. Compresses to a reasonable size
    4. Returns base64 encoded string
    
    Args:
        image_path: Path to the image file
        max_width: Maximum image width
        max_height: Maximum image height
        initial_quality: Initial JPEG quality
        
    Returns:
        str: Base64-encoded image string
    """
    try:
        # Open the image
        img = Image.open(image_path)

        # Resize if needed
        img = resize_image_if_needed(img, max_width, max_height)

        # Compress the image
        img_bytes = compress_image_to_buffer(img, initial_quality)

        # Encode to base64
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        return img_b64
    except Exception as e:
        logger.error(f"Error preparing image: {str(e)}")
        raise


def find_credentials_file() -> Optional[str]:
    """
    Finds the credentials file by checking common locations.
    
    Returns:
        Optional[str]: Path to credentials file if found, None otherwise
    """
    # Check current working directory
    credentials_file = "vertex_ai_service_account.json"
    if os.path.exists(credentials_file):
        return credentials_file
    
    # Check project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    credentials_file = os.path.join(project_root, "vertex_ai_service_account.json")
    if os.path.exists(credentials_file):
        return credentials_file
    
    # Check home directory
    credentials_file = os.path.expanduser("~/.vertex_ai_service_account.json")
    if os.path.exists(credentials_file):
        return credentials_file
    
    return None


def describe_image_content(
    image_path: str, 
    model: str = "vertex_ai/gemini-2.5-pro-preview-05-06",
    prompt: str = "Describe this image in detail.",
    credentials_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Uses AI vision model to describe the content of an image
    
    Args:
        image_path: Path to the image file
        model: AI model to use
        prompt: Text prompt for image description
        credentials_file: Path to credentials file for API authentication
        
    Returns:
        dict: Description results with 'description', 'filename', 'confidence'
              or 'error' if description fails
    """
    try:
        # Prepare the image
        image_b64 = prepare_image_for_multimodal(image_path)

        # Extract the filename from the path
        filename = os.path.basename(image_path)

        # Prepare credentials if provided
        vertex_credentials = None
        if credentials_file and os.path.exists(credentials_file):
            try:
                with open(credentials_file, "r") as file:
                    vertex_credentials = json.load(file)
            except Exception as e:
                logger.error(f"Failed to load credentials: {str(e)}")

        # Construct messages with multimodal content
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{prompt} Respond with a JSON object that includes: 1) a 'description' field with your detailed description, 2) a 'filename' field with the value '{filename}', and 3) a 'confidence' field with a number from 1-5 (5 being highest) indicating your confidence in the accuracy of your description considering image quality and compression artifacts."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                ],
            }
        ]

        # Make the API call with JSON schema validation
        response = completion(
            model=model,
            messages=messages,
            vertex_ai=vertex_credentials,
            response_format={
                "type": "json_object",
                "response_schema": DESCRIPTION_SCHEMA,
                "enforce_validation": True
            }
        )
        
        # Parse the JSON response and return it
        result = response.choices[0].message.content
        # Clean up any potential JSON issues
        result = result.strip()
        if result.startswith('```json'):
            result = result[7:]
        if result.endswith('```'):
            result = result[:-3]
        
        return json.loads(result)
        
    except Exception as e:
        logger.error(f"Image description failed: {str(e)}")
        return {"error": f"Image description failed: {str(e)}"}


# =====================================================================
# MCP Tool Functions
# =====================================================================

@mcp.tool()
def screenshot(
    quality: int = IMAGE_SETTINGS["DEFAULT_QUALITY"],
    region: Optional[Union[List[int], str]] = None,
    include_description: bool = False,
    prompt: str = "Describe this screenshot in detail.",
    model: str = "vertex_ai/gemini-2.5-pro-preview-05-06"
) -> Dict[str, Any]:
    """
    Captures a screenshot of the entire desktop or a specified region, returning it as a base64-encoded JPEG.
    Optionally includes an AI-generated description.
    Uses MSS library instead of PyAutoGUI.

    Args:
        quality (int, optional): JPEG compression quality (1-100). Defaults to 30.
        region (list or str, optional): Region coordinates [x, y, width, height] in logical pixels,
                                      or "right_half" for right half of screen. Defaults to None (full screen).
        include_description (bool, optional): Whether to include an AI-generated description. Defaults to False.
        prompt (str, optional): The text prompt to send to the AI model for description. Defaults to "Describe this screenshot in detail."
        model (str, optional): AI model to use for image description. Defaults to Gemini Pro.

    Returns:
        dict: MCP-compliant response containing:
            - content: List with a single image object (type, base64 data, MIME type).
            - file: Path to the saved screenshot file.
            - result: If include_description is True, includes AI-generated description.
            On error:
            - error: Error message as a string.
    """
    logger.info(f"Screenshot requested with quality={quality}, region={region}, include_description={include_description}")

    # Capture screenshot
    result = capture_screenshot(quality=quality, region=region)
    
    # Check if screenshot failed
    if "error" in result:
        return result
        
    # If description is requested, add it
    if include_description:
        try:
            # Get the file path from the screenshot result
            image_path = result["file"]
            
            # Find credentials file
            credentials_file = find_credentials_file()
            
            # Get image description
            description_result = describe_image_content(
                image_path=image_path,
                model=model,
                prompt=prompt,
                credentials_file=credentials_file
            )
            
            if "error" in description_result:
                logger.error(f"Image description error: {description_result['error']}")
                result["description_error"] = description_result["error"]
            else:
                logger.info(f"Image description successful")
                result["result"] = description_result
                
                # Remove large base64 data to reduce response size when description succeeds
                if "content" in result:
                    del result["content"]
        
        except Exception as e:
            logger.error(f"Failed to get image description: {str(e)}")
            result["description_error"] = f"Failed to get image description: {str(e)}"
    
    return result


@mcp.tool()
def describe_screenshot(
    quality: int = IMAGE_SETTINGS["DEFAULT_QUALITY"], 
    region: Optional[Union[List[int], str]] = None,
    prompt: str = "Describe this screenshot in detail.",
    model: str = "vertex_ai/gemini-2.5-pro-preview-05-06"
) -> Dict[str, Any]:
    """
    Captures a screenshot and provides an AI description of the image content.
    
    Args:
        quality (int, optional): JPEG compression quality (1-100). Defaults to 30.
        region (list or str, optional): Region coordinates [x, y, width, height] in logical pixels,
                                      or "right_half" for right half of screen. Defaults to None (full screen).
        prompt (str, optional): The text prompt to send to the AI model. Defaults to "Describe this screenshot in detail."
        model (str, optional): AI model to use for image description. Defaults to Gemini Pro.
        
    Returns:
        dict: MCP-compliant response containing:
            - result: Dictionary with 'description', 'confidence', and metadata
            - image: The image data formatted as required by Claude
            - file: Path to the saved screenshot file
            On error:
            - error: Error message as a string.
    """
    logger.info(f"Screenshot with description requested - quality={quality}, region={region}, prompt={prompt}")
    
    # First take the screenshot
    screenshot_result = capture_screenshot(quality=quality, region=region)
    
    # Check if screenshot failed
    if "error" in screenshot_result:
        return screenshot_result
    
    # Get the file path from the screenshot result
    image_path = screenshot_result["file"]
    logger.info(f"Screenshot saved to {image_path}")
    
    # Find credentials file
    credentials_file = find_credentials_file()
    
    # Describe the image
    try:
        description_result = describe_image_content(
            image_path=image_path,
            model=model,
            prompt=prompt,
            credentials_file=credentials_file
        )
        
        if "error" in description_result:
            logger.error(f"Image description error: {description_result['error']}")
            # If description fails, still return the screenshot but with error
            return {
                "error": f"Image captured successfully but description failed: {description_result['error']}",
                "content": screenshot_result.get("content"),
                "file": image_path
            }
        
        logger.info(f"Image description successful")
        
        # Create a response combining screenshot and description, but WITHOUT image data
        response = {
            "result": description_result,
            "file": image_path
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Image description failed: {str(e)}", exc_info=True)
        # If description fails, still return the screenshot but with error
        return {
            "error": f"Image captured successfully but description failed: {str(e)}",
            "content": screenshot_result.get("content"),
            "file": image_path
        }


# =====================================================================
# CLI Commands
# =====================================================================

@app.command()
def screenshot_cmd(
    quality: int = typer.Option(
        IMAGE_SETTINGS["DEFAULT_QUALITY"], 
        help="JPEG compression quality (1-100)"
    ),
    region: Optional[str] = typer.Option(
        None, 
        help="Screen region to capture: 'full', 'right_half', 'left_half', 'top_half', 'bottom_half'"
    ),
    output: Optional[str] = typer.Option(
        None, 
        help="Output file path. If not provided, saves to screenshots directory."
    ),
    raw: bool = typer.Option(
        False, 
        help="Save raw PNG in addition to compressed JPEG"
    )
):
    """Take a screenshot of the screen or a specific region."""
    try:
        # Convert region string to appropriate format
        region_param = None
        if region == "full":
            region_param = None
        elif region in ["right_half", "left_half", "top_half", "bottom_half"]:
            region_param = region
        elif region:
            typer.echo(f"Unknown region: {region}. Using full screen.")
        
        # Set output directory from output parameter if provided
        output_dir = "screenshots"
        if output:
            output_dir = os.path.dirname(output) or "."
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        
        # Take screenshot
        result = capture_screenshot(
            quality=quality, 
            region=region_param, 
            output_dir=output_dir,
            include_raw=raw
        )
        
        if "error" in result:
            typer.echo(f"Error: {result['error']}")
            sys.exit(1)
        
        typer.echo(f"Screenshot saved to: {result['file']}")
        if raw and "raw_file" in result:
            typer.echo(f"Raw screenshot saved to: {result['raw_file']}")
            
    except Exception as e:
        typer.echo(f"Error: {str(e)}")
        sys.exit(1)


@app.command()
def describe(
    quality: int = typer.Option(
        IMAGE_SETTINGS["DEFAULT_QUALITY"], 
        help="JPEG compression quality (1-100)"
    ),
    region: Optional[str] = typer.Option(
        None, 
        help="Screen region to capture: 'full', 'right_half', 'left_half', 'top_half', 'bottom_half'"
    ),
    prompt: str = typer.Option(
        "Describe this screenshot in detail.", 
        help="Prompt for the AI description"
    ),
    model: str = typer.Option(
        "vertex_ai/gemini-2.5-pro-preview-05-06", 
        help="AI model to use for description"
    ),
    output_json: bool = typer.Option(
        False, 
        help="Output the description as JSON"
    )
):
    """Take a screenshot and get an AI description of the content."""
    try:
        # Convert region string to appropriate format
        region_param = None
        if region == "full":
            region_param = None
        elif region in ["right_half", "left_half", "top_half", "bottom_half"]:
            region_param = region
        elif region:
            typer.echo(f"Unknown region: {region}. Using full screen.")
        
        # Take screenshot
        screenshot_result = capture_screenshot(quality=quality, region=region_param)
        
        if "error" in screenshot_result:
            typer.echo(f"Error taking screenshot: {screenshot_result['error']}")
            sys.exit(1)
        
        typer.echo(f"Screenshot saved to: {screenshot_result['file']}")
        
        # Get credentials file
        credentials_file = find_credentials_file()
        if not credentials_file:
            typer.echo("Warning: Credentials file not found. Description may fail.")
        
        # Get description
        description_result = describe_image_content(
            image_path=screenshot_result["file"],
            model=model,
            prompt=prompt,
            credentials_file=credentials_file
        )
        
        if "error" in description_result:
            typer.echo(f"Error getting description: {description_result['error']}")
            sys.exit(1)
        
        # Output result
        if output_json:
            typer.echo(json.dumps(description_result, indent=2))
        else:
            typer.echo("\nImage Description:")
            typer.echo("-" * 40)
            typer.echo(description_result["description"])
            typer.echo("-" * 40)
            typer.echo(f"Confidence: {description_result['confidence']}/5")
            
    except Exception as e:
        typer.echo(f"Error: {str(e)}")
        sys.exit(1)


@app.command()
def run_server():
    """Run the MCP server for screenshot tools."""
    typer.echo("Starting MCP server for screenshot tools...")
    mcp.run()


@app.command()
def test(
    save_results: bool = typer.Option(True, help="Save test results to debug_results.json"),
    test_describe: bool = typer.Option(False, help="Test the describe functionality with API calls")
):
    """Run debug tests for screenshot functionality."""
    typer.echo("Running screenshot tests...")
    
    # Test cases covering key scenarios
    test_cases = [
        {"quality": 30, "region": None},  # Full screen
        {"quality": 30, "region": "right_half"},  # Right half
    ]

    results = []
    
    # Test screenshot function
    for i, test in enumerate(test_cases, 1):
        typer.echo(f"Test {i}: quality={test['quality']}, region={test['region']}")
        try:
            result = capture_screenshot(quality=test["quality"], region=test["region"])
            success = "error" not in result
            typer.echo(f"Result: {'Success' if success else 'Failed'}")
            results.append({"test": i, "type": "screenshot", "input": test, "result": "Success" if success else "Failed"})
        except Exception as e:
            typer.echo(f"Error: {str(e)}")
            results.append({"test": i, "type": "screenshot", "input": test, "error": str(e)})

    # Test describe_image_content if requested
    if test_describe:
        typer.echo("\nTesting AI description functionality...")
        
        # Take a test screenshot for description
        test_img_result = capture_screenshot(quality=30, region="right_half")
        
        if "error" in test_img_result:
            typer.echo(f"Failed to capture test image: {test_img_result['error']}")
        else:
            # Find credentials file
            credentials_file = find_credentials_file()
            if not credentials_file:
                typer.echo("Warning: Credentials file not found. Description test will likely fail.")
            
            # Test description
            typer.echo("Getting image description...")
            description_result = describe_image_content(
                image_path=test_img_result["file"],
                credentials_file=credentials_file
            )
            
            if "error" in description_result:
                typer.echo(f"Description failed: {description_result['error']}")
                results.append({"type": "describe", "result": "Failed", "error": description_result["error"]})
            else:
                typer.echo(f"Description successful. Confidence: {description_result.get('confidence', 'N/A')}/5")
                typer.echo(f"Description excerpt: {description_result.get('description', 'N/A')[:100]}...")
                results.append({"type": "describe", "result": "Success"})
    
    # Optionally save results to a file
    if save_results:
        with open("debug_results.json", "w") as f:
            json.dump(results, f, indent=2)
        typer.echo("Saved results to debug_results.json")
    
    typer.echo("\nAll tests completed.")


# =====================================================================
# Main Function
# =====================================================================

if __name__ == "__main__":
    """
    Validates and runs the standalone screenshot tool.
    
    Examples:
      python standalone_screenshot.py screenshot --quality 30 --region right_half
      python standalone_screenshot.py describe --prompt "What's in this image?"
      python standalone_screenshot.py run-server
      python standalone_screenshot.py test
    """
    # Run the application
    app()