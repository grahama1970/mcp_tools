#!/usr/bin/env python3
"""
Claude MCP Screenshot Tool using MSS library

A screenshot utility for Claude Desktop/Code using MSS instead of PyAutoGUI
"""

import os
import time
import base64
import json
import io
import sys
from PIL import Image
import mss
from screeninfo import get_monitors
from mcp.server.fastmcp.server import FastMCP
from loguru import logger

# Add the project root to path so we can import our module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import our image description module
try:
    from src.claude_mcp_tools.llm_call.describe_image import describe_image
    from src.claude_mcp_tools.llm_call.initialize_litellm_cache import initialize_litellm_cache
    DESCRIBE_IMAGE_AVAILABLE = True
except ImportError:
    logger.warning("describe_image module not available, image description will be disabled")
    DESCRIBE_IMAGE_AVAILABLE = False

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


def get_project_path(*paths):
    """
    Returns an absolute path within the project directory.

    Args:
        *paths: Path components to join with the base directory

    Returns:
        str: Absolute path within the project
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, *paths)


# Configure logger
logger.remove()  # Remove default handler that outputs to stderr
log_path = get_project_path("logs", "mss_screenshot.log")
logger.add(
    log_path,
    rotation="10 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
)

# Initialize FastMCP server for screenshot tool
mcp = FastMCP("MSS Screenshot Tool")
logger.info("Initialized FastMCP server for MSS screenshot tool")


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


def validate_region(region):
    """
    Validates the region parameter for screenshot capture.

    Args:
        region: List of [x, y, width, height] values, "right_half", or None

    Returns:
        tuple: (is_valid, message)
    """
    if region is None or region == "right_half":
        return True, None

    if not isinstance(region, list):
        return False, f"Region must be a list, got {type(region)}"

    if len(region) != 4:
        return False, f"Region must have 4 elements [x, y, width, height], got {len(region)}"

    # Ensure all values are numeric
    if not all(isinstance(val, (int, float)) for val in region):
        return False, "All region values must be numeric"

    # Validate dimensions
    x, y, width, height = region
    if width <= 0 or height <= 0:
        return False, f"Width and height must be positive, got width={width}, height={height}"

    return True, None


def get_scale_factor(monitor_index=1):
    """
    Detect the display scaling factor using screeninfo.

    Args:
        monitor_index: Index of the monitor to check (default: 1 for primary).

    Returns:
        float: Scaling factor (e.g., 2.0 for Retina, 1.0 for non-scaled displays).
    """
    try:
        monitors = get_monitors()
        if monitor_index < 0 or monitor_index >= len(monitors):
            logger.warning(f"Invalid monitor index {monitor_index}. Available: {len(monitors)}")
            return 1.0
        
        monitor = monitors[monitor_index]
        # Calculate scaling factor (physical width / logical width)
        # screeninfo provides width (logical) and width_mm (physical measurement)
        if hasattr(monitor, 'dpi') and monitor.dpi:
            scale = monitor.dpi / 96.0  # Standard DPI is 96
        else:
            # Fallback: Estimate scaling from width/height
            scale = monitor.width / monitor.width_mm * 25.4 / 96.0  # Convert mm to inches, compare to 96 DPI
            if scale > 1.5:  # Typical Retina or high-DPI threshold
                scale = 2.0
            else:
                scale = 1.0
        
        logger.info(f"Detected scaling factor for monitor {monitor_index}: {scale}")
        return scale
    except Exception as e:
        logger.warning(f"Failed to detect scaling factor: {str(e)}. Using default: 1.0")
        return 1.0


def capture_screenshot(region=None):
    """
    Captures a screenshot using MSS library.

    Args:
        region: Optional region to capture [x, y, width, height] in logical pixels,
                or "right_half" for the right half of the screen.

    Returns:
        PIL.Image: The captured screenshot as a PIL Image
    """
    scale_factor = get_scale_factor(monitor_index=1)

    with mss.mss() as sct:
        # Get primary monitor
        monitor = sct.monitors[1]
        logger.info(f"Selected monitor: {monitor}")

        if region == "right_half":
            # Calculate right half in logical pixels
            logical_width = monitor["width"] // scale_factor
            logical_height = monitor["height"] // scale_factor
            region = [logical_width // 2, 0, logical_width // 2, logical_height]
            logger.info(f"Calculated right half region (logical): {region}")

        if region:
            x, y, w, h = region
            # Scale region for Retina/high-DPI displays
            scaled_region = [int(coord * scale_factor) for coord in region]
            x, y, w, h = scaled_region
            logger.info(f"Taking region screenshot at x={x}, y={y}, width={w}, height={h} (scaled x{scale_factor})")

            # Validate scaled region
            if (x < 0 or y < 0 or
                x + w > monitor["width"] or
                y + h > monitor["height"]):
                raise ValueError(f"Region out of bounds: {scaled_region}, monitor: {monitor}")

            capture_region = {"top": y, "left": x, "width": w, "height": h}
            sct_img = sct.grab(capture_region)
        else:
            logger.info("Taking full screen screenshot")
            sct_img = sct.grab(monitor)

        # Save raw screenshot for debugging
        raw_img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        raw_img.save(get_project_path("screenshots", f"raw_screenshot_{int(time.time() * 1000)}.png"))
        logger.info("Saved raw screenshot for debugging")

        # Convert to PIL Image
        return Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")


def resize_image_if_needed(img, max_width, max_height):
    """
    Resizes an image if it exceeds maximum dimensions.

    Args:
        img: PIL Image to resize
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
    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


def compress_image_to_size_limit(img, path, initial_quality, min_quality, max_file_size):
    """
    Compresses an image to fit within size limits.

    Args:
        img: PIL Image to compress
        path: Path to save the compressed image
        initial_quality: Initial JPEG quality setting (1-100)
        min_quality: Minimum quality to use
        max_file_size: Maximum file size in bytes

    Returns:
        tuple: (bytes, quality) - The image bytes and final quality used
    """
    # Initial save with specified quality
    img.save(path, format="JPEG", quality=initial_quality)

    # Read file to check size
    with open(path, "rb") as f:
        img_bytes = f.read()

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

        img.save(path, format="JPEG", quality=compress_quality)
        with open(path, "rb") as f:
            img_bytes = f.read()

        logger.debug(
            f"Compression iteration {compression_iterations}: "
            f"size reduced from {old_size / 1024:.1f} KB to {len(img_bytes) / 1024:.1f} KB"
        )

    return img_bytes, compress_quality


@mcp.tool()
def screenshot(quality: int = IMAGE_SETTINGS["DEFAULT_QUALITY"], region: list | str = None) -> dict:
    """
    Captures a screenshot of the entire desktop or a specified region, returning it as a base64-encoded JPEG.
    Uses MSS library instead of PyAutoGUI.

    Args:
        quality (int, optional): JPEG compression quality (1-100). Defaults to 30.
        region (list or str, optional): Region coordinates [x, y, width, height] in logical pixels,
                                       or "right_half" for right half of screen. Defaults to None (full screen).

    Returns:
        dict: MCP-compliant response containing:
            - content: List with a single image object (type, base64 data, MIME type).
            - file: Path to the saved screenshot file.
            On error:
            - error: Error message as a string.
    """
    logger.info(f"Screenshot requested with quality={quality}, region={region}")

    try:
        # Validate region parameter
        is_valid, error_message = validate_region(region)
        if not is_valid:
            return {"error": f"Invalid region parameter: {error_message}"}

        # Ensure output directory exists
        outdir = get_project_path("screenshots")
        os.makedirs(outdir, exist_ok=True)

        # Generate unique filename with timestamp for easy sorting
        timestamp = int(time.time() * 1000)
        filename = f"screenshot_{timestamp}.jpeg"
        path = os.path.join(outdir, filename)

        # Capture screenshot
        img = capture_screenshot(region)

        # Clamp quality to acceptable range
        original_quality = quality
        quality = max(IMAGE_SETTINGS["MIN_QUALITY"], min(quality, IMAGE_SETTINGS["MAX_QUALITY"]))
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
        img_bytes, final_quality = compress_image_to_size_limit(
            img,
            path,
            quality,
            IMAGE_SETTINGS["MIN_QUALITY"],
            IMAGE_SETTINGS["MAX_FILE_SIZE"]
        )

        # Encode to base64
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
        logger.info(f"Image encoded as base64 string ({len(img_b64)} characters)")

        # Construct MCP-compliant response (Claude expects proper "source" structure)
        response = {
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": img_b64
                    }
                }
            ],
            "file": path
        }

        # Create a log-safe version of the response with truncated base64 data
        log_safe_response = {
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/jpeg",
                        "data": truncate_large_value(img_b64)
                    }
                }
            ],
            "file": path
        }

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
        {"quality": 30, "region": "left_half"},  # Left half
        {"quality": 30, "region": "top_half"},  # Top half
        {"quality": 30, "region": "bottom_half"},  # Bottom half
        {
            "quality": 30,
            "region": [720, 0, 720, 900],
        },  # Specific region (right half equivalent)
        {"quality": 30, "region": [100, 100, 300, 200]},  # Small custom region
        {"quality": 30, "region": [0, 0, -100, 100]},  # Invalid region (negative width)
    ]

    results = []

    # Part 1: Test screenshot function
    logger.info("Part 1: Testing basic screenshot functionality")
    for i, test in enumerate(test_cases, 1):
        logger.info(f"Test {i}: quality={test['quality']}, region={test['region']}")
        try:
            result = screenshot(quality=test["quality"], region=test["region"])
            logger.info(f"Test {i} result: {truncate_large_value(str(result))}")
            results.append({"test": i, "type": "screenshot", "input": test, "result": result})
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
            # Small custom region
            {"quality": 30, "region": [100, 100, 300, 200], "prompt": "What UI elements do you see here?"}
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

        # Initialize litellm cache
        try:
            initialize_litellm_cache()
        except Exception as e:
            logger.error(f"Failed to initialize LiteLLM cache: {str(e)}")
            # Continue anyway as it might still work

        # Get the credentials file path
        credentials_file = get_project_path("vertex_ai_service_account.json")
        if not os.path.exists(credentials_file):
            logger.warning(f"Credentials file not found at {credentials_file}")
            credentials_file = None

        # Describe the image
        try:
            description_result = describe_image(
                image_input=image_path,
                model=model,
                prompt=prompt,
                credentials_file=credentials_file
            )

            logger.info(f"Image description result: {description_result}")

            # Create a response combining screenshot and description
            response = {
                "result": description_result,
                "image": screenshot_result["content"][0],  # Include the image data
                "file": image_path
            }

            return response

        except Exception as e:
            logger.error(f"Image description failed: {str(e)}", exc_info=True)
            # If description fails, still return the screenshot but with error
            return {
                "error": f"Image captured successfully but description failed: {str(e)}",
                "image": screenshot_result["content"][0],
                "file": image_path
            }

    except Exception as e:
        logger.error(f"Screenshot with description failed: {str(e)}", exc_info=True)
        return {"error": f"Screenshot with description failed: {str(e)}"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Claude MCP Screenshot Tool with MSS")
    parser.add_argument("--test", action="store_true", help="Run debug tests without starting the server")
    parser.add_argument("--test-describe", action="store_true", help="Test the describe functionality with API calls")
    parser.add_argument("--save-results", action="store_true", default=True, help="Save test results to debug_results.json")
    args = parser.parse_args()

    if args.test:
        logger.info("Running in test mode")
        debug_tests(save_results=args.save_results, test_describe=args.test_describe)
    else:
        logger.info("Starting FastMCP server for MSS screenshot tool")
        mcp.run()
