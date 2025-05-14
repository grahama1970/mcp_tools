#!/usr/bin/env python3
"""
Claude MCP Screenshot Tool using MSS library

A screenshot utility for Claude Desktop/Code using MSS instead of PyAutoGUI
with built-in image description capabilities.

Agent Instruction for testing:
Carefully read CLAUDE.md, move into the project directory, and activate the .venv. 
Then use the describe_screenshot tool to capture the right half of the screen and 
provide an AI description of what's visible.
"""

import os
import sys
import time
import base64
import json
import io
import uuid
from PIL import Image
import mss
from mcp.server.fastmcp import FastMCP
from loguru import logger
import typer

# Constants for image capture and processing
IMAGE_SETTINGS = {
    "MAX_WIDTH": 640,  # Maximum width for resized images
    "MAX_HEIGHT": 640,  # Maximum height for resized images
    "MIN_QUALITY": 30,  # Minimum JPEG compression quality
    "MAX_QUALITY": 70,  # Maximum JPEG compression quality
    "DEFAULT_QUALITY": 30,  # Default quality if none specified
    "MAX_FILE_SIZE": 350_000,  # Maximum file size in bytes (350kB)
}

# Constants for logging
LOG_MAX_STR_LEN = 100  # Maximum string length for truncated logging

# Configure logger
logger.remove()  # Remove default handler that outputs to stderr
logger.add(
    "logs/mss_screenshot.log",
    rotation="10 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Initialize FastMCP server for screenshot tool
mcp = FastMCP("MSS Screenshot Tool")
logger.info("Initialized FastMCP server fr MSS screenshot tool")

# Initialize typer app
app = typer.Typer(help="Claude MCP Screenshot Tool with MSS")


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


def resize_image_if_needed(img, max_width, max_height):
    """
    Resizes an image if it exceeds maximum dimensions.
    
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


def compress_image_to_buffer(
    img, initial_quality=30, min_quality=30, max_file_size=350_000
):
    """
    Compresses an image to fit within size limits and returns as bytes buffer.
    Converts RGBA images to RGB for JPEG compatibility.
    
    Args:
        img: PIL Image object to compress
        initial_quality: Initial JPEG quality setting (1-100)
        min_quality: Minimum quality to use if compression is needed
        max_file_size: Maximum file size in bytes
        
    Returns:
        bytes: Compressed image bytes
    """
    # Convert RGBA images to RGB for JPEG compatibility
    if img.mode == 'RGBA':
        # Create a white background image
        background = Image.new('RGB', img.size, (255, 255, 255))
        # Paste the image using the alpha channel as mask
        background.paste(img, mask=img.split()[3])
        img = background
    elif img.mode != 'RGB':
        # Convert any other mode to RGB
        img = img.convert('RGB')

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


def prepare_image_for_multimodal(
    image_path, max_width=640, max_height=640, initial_quality=30
):
    """
    Prepares an image for multimodal API calls:
    1. Opens the image file
    2. Resizes if needed
    3. Compresses to a reasonable size
    4. Returns base64 encoded string
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


def describe_image_content(
    image_path, 
    model="vertex_ai/gemini-2.5-pro-preview-05-06",
    prompt="Describe this image in detail.",
    credentials_file=None
):
    """
    Uses AI vision model to describe the content of an image
    
    Args:
        image_path: Path to the image file
        model: AI model to use
        prompt: Text prompt for the image description
        credentials_file: Path to credentials file for API authentication
        
    Returns:
        dict: Description results with 'description', 'filename', and 'confidence'
    """
    if not DESCRIBE_IMAGE_AVAILABLE:
        return {"error": "Image description functionality is not available"}
    
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
                "response_schema": response_schema,
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


@mcp.tool()
def screenshot(quality: int = IMAGE_SETTINGS["DEFAULT_QUALITY"],
               region: list | str = None,
               include_description: bool = True,
               prompt: str = "Describe this screenshot in detail.",
               model: str = "vertex_ai/gemini-2.5-pro-preview-05-06") -> dict:
    """
    Captures a screenshot of the entire desktop or a specified region, returning it as a base64-encoded JPEG.
    Automatically includes an AI-generated description when include_description is True.
    Uses MSS library instead of PyAutoGUI.

    Args:
        quality (int, optional): JPEG compression quality (1-100). Defaults to 30.
        region (list or str, optional): Region coordinates [x, y, width, height] in logical pixels,
                                      or "right_half" for right half of screen. Defaults to None (full screen).
        include_description (bool, optional): Whether to include an AI-generated description. Defaults to True.
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

    try:
        # Validate region parameter
        if region is not None and not isinstance(region, (list, str)):
            return {"error": f"Invalid region parameter: must be a list or 'right_half', got {type(region)}"}

        if isinstance(region, list) and len(region) != 4:
            return {"error": f"Region must have 4 elements [x, y, width, height], got {len(region)}"}

        if isinstance(region, str) and region != "right_half":
            return {"error": f"String region must be 'right_half', got {region}"}

        # Ensure output directory exists
        outdir = "screenshots"
        os.makedirs(outdir, exist_ok=True)

        # Generate unique filename with timestamp for easy sorting
        timestamp = int(time.time() * 1000)
        filename = f"screenshot_{timestamp}.jpeg"
        path = os.path.join(outdir, filename)

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
            elif region is not None:
                x, y, w, h = region
                logger.info(f"Taking region screenshot at x={x}, y={y}, width={w}, height={h}")

                capture_region = {"top": y, "left": x, "width": w, "height": h}
                sct_img = sct.grab(capture_region)
            else:
                logger.info("Taking full screen screenshot")
                sct_img = sct.grab(monitor)

            # Convert to PIL Image
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")

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

        # If description is requested and available, skip including base64 in response
        if include_description and DESCRIBE_IMAGE_AVAILABLE:
            try:
                # Get the credentials file path - first check working directory
                credentials_file = "vertex_ai_service_account.json"
                if not os.path.exists(credentials_file):
                    # Then check project root
                    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    credentials_file = os.path.join(project_root, "vertex_ai_service_account.json")
                    if not os.path.exists(credentials_file):
                        logger.warning(f"Credentials file not found at {credentials_file}")
                        credentials_file = None

                # Describe the image
                description_result = describe_image_content(
                    image_path=path,
                    model=model,
                    prompt=prompt,
                    credentials_file=credentials_file
                )

                # Construct response with results but without the large base64 string
                response = {
                    "file": path
                }

                if "error" in description_result:
                    logger.error(f"Image description error: {description_result['error']}")
                    response["description_error"] = description_result['error']

                    # Only include the image data if description fails
                    response["content"] = [
                        {
                            "type": "image",
                            "data": img_b64,
                            "mimeType": "image/jpeg"
                        }
                    ]
                else:
                    logger.info(f"Image description result: {description_result}")
                    response["result"] = description_result
            except Exception as e:
                logger.error(f"Image description failed: {str(e)}", exc_info=True)
                response["description_error"] = f"Image description failed: {str(e)}"

                # Only include the image data if description fails
                response["content"] = [
                    {
                        "type": "image",
                        "data": img_b64,
                        "mimeType": "image/jpeg"
                    }
                ]
        else:
            # Standard response with image data if no description is requested
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

        # Create a log-safe version of the response with truncated base64 data
        log_safe_response = {k: v for k, v in response.items() if k != "content"}
        if "content" in response:
            log_safe_response["content"] = [
                {
                    "type": "image",
                    "data": truncate_large_value(img_b64),
                    "mimeType": "image/jpeg"
                }
            ]

        # Log the truncated response
        logger.info(f"Screenshot captured successfully: {path}")
        logger.debug(f"Screenshot response (truncated): {log_safe_response}")

        # Return the full response to the client
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


@mcp.tool()
def describe_screenshot(quality: int = IMAGE_SETTINGS["DEFAULT_QUALITY"], 
                       region: list | str = None,
                       prompt: str = "Describe this screenshot in detail.",
                       model: str = "vertex_ai/gemini-2.5-pro-preview-05-06") -> dict:
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
    
    if not DESCRIBE_IMAGE_AVAILABLE:
        return {"error": "Image description functionality is not available. Required modules not found."}
    
    try:
        # First take the screenshot
        screenshot_result = screenshot(quality=quality, region=region)
        
        # Check if screenshot failed
        if "error" in screenshot_result:
            return screenshot_result
        
        # Get the file path from the screenshot result
        image_path = screenshot_result["file"]
        logger.info(f"Screenshot saved to {image_path}")
        
        # Get the credentials file path - first check working directory
        credentials_file = "vertex_ai_service_account.json"
        if not os.path.exists(credentials_file):
            # Then check project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            credentials_file = os.path.join(project_root, "vertex_ai_service_account.json")
            if not os.path.exists(credentials_file):
                logger.warning(f"Credentials file not found at {credentials_file}")
                credentials_file = None
        
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
                    "content": screenshot_result["content"],
                    "file": image_path
                }
            
            logger.info(f"Image description result: {description_result}")
            
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
                "content": screenshot_result["content"],
                "file": image_path
            }
    
    except Exception as e:
        logger.error(f"Screenshot with description failed: {str(e)}", exc_info=True)
        return {"error": f"Screenshot with description failed: {str(e)}"}


def debug_tests(save_results=True, test_describe=False):
    """
    Runs debug tests for the MSS screenshot tool, simulating agent requests.

    Args:
        save_results (bool): If True, saves test results to 'debug_results.json'.
        test_describe (bool): If True, also tests the describe_screenshot functionality.

    Returns:
        list: List of test results (input parameters and outcome).
    """
    logger.info(f"Running debug tests for MSS screenshot tool (test_describe={test_describe})")

    # Test cases covering key scenarios and an edge case
    test_cases = [
        {"quality": 30, "region": None},  # Full screen
        {"quality": 30, "region": "right_half"},  # Right half
        {"quality": 30, "region": [100, 100, 300, 200]},  # Custom region
        {"quality": 30, "region": [0, 0, -100, 100]},  # Invalid region
    ]

    results = []
    
    # Part 1: Test screenshot function
    logger.info("Part 1: Testing basic screenshot functionality")
    for i, test in enumerate(test_cases, 1):
        logger.info(f"Test {i}: quality={test['quality']}, region={test['region']}")
        try:
            result = screenshot(quality=test["quality"], region=test["region"])
            logger.info(f"Test {i} result: {truncate_large_value(str(result))}")
            results.append({"test": i, "type": "screenshot", "input": test, "result": "Success"})
        except Exception as e:
            logger.error(f"Test {i} failed: {str(e)}")
            results.append({"test": i, "type": "screenshot", "input": test, "error": str(e)})

    # Part 2: Test describe_screenshot function (if requested and available)
    if test_describe and DESCRIBE_IMAGE_AVAILABLE:
        logger.info("Part 2: Testing screenshot with description functionality")
        
        # Subset of test cases for description testing (to avoid too many API calls)
        description_test_cases = [
            # Full screen with default prompt
            {"quality": 30, "region": None, "prompt": "Describe this screenshot in detail."},
            # Right half with custom prompt
            {"quality": 30, "region": "right_half", "prompt": "What's visible in this part of the screen?"},
        ]
        
        for i, test in enumerate(description_test_cases, 1):
            logger.info(f"Description Test {i}: {test}")
            try:
                result = describe_screenshot(**test)
                # Don't log the entire result as it could be very large
                log_safe_result = "Success" if "result" in result else "Failed"
                logger.info(f"Description Test {i} result: {log_safe_result}")
                results.append({"test": i, "type": "describe", "input": test, "success": "result" in result})
            except Exception as e:
                logger.error(f"Description Test {i} failed: {str(e)}")
                results.append({"test": i, "type": "describe", "input": test, "error": str(e)})
    
    # Optionally save results to a file
    if save_results:
        with open("debug_results.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved debug results to debug_results.json")

    return results


@app.command()
def run_server():
    """Start the FastMCP server for MSS screenshot tool."""
    logger.info("Starting FastMCP server for MSS screenshot tool")
    mcp.run()


@app.command()
def test(
    save_results: bool = typer.Option(True, help="Save test results to debug_results.json"),
    test_describe: bool = typer.Option(False, help="Test the describe functionality with API calls")
):
    """Run debug tests without starting the server."""
    logger.info("Running in test mode")
    debug_tests(save_results=save_results, test_describe=test_describe)


if __name__ == "__main__":
    """
    Agent Instruction for testing:
    Carefully read CLAUDE.md, move into the project directory, and activate the .venv. 
    Then use the describe_screenshot tool to capture the right half of the screen and 
    provide an AI description of what's visible.
    """
    app()