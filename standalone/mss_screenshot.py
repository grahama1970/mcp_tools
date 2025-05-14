#!/usr/bin/env python3
"""
Claude MCP Screenshot Tool using MSS library (Standalone Version)

A standalone screenshot utility for Claude Desktop/Code using the mcp_tools package.
This script provides both CLI and MCP server functionality.

Third-party package documentation:
- MSS: https://python-mss.readthedocs.io/
- MCP: https://github.com/anthropics/anthropic-mcp
- Typer: https://typer.tiangolo.com/

Sample input:
- python mss_screenshot.py run-server
- python mss_screenshot.py test --test-describe

Expected output:
- Running MCP server for screenshot tools
- Test results JSON file
"""

import os
import json
from loguru import logger

import typer
from mcp.server.fastmcp import FastMCP

# Import from the mcp_tools package using absolute imports
from mcp_tools.screenshot import (
    IMAGE_SETTINGS,
    capture_screenshot,
    describe_image_content,
    find_credentials_file
)

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
logger.info("Initialized FastMCP server for MSS screenshot tool")

# Initialize typer app
app = typer.Typer(help="Claude MCP Screenshot Tool with MSS")


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
        
        # Create a response combining screenshot and description
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
            # Don't log the entire result as it could be very large
            success = "error" not in result
            logger.info(f"Test {i} result: {'Success' if success else 'Failed'}")
            results.append({"test": i, "type": "screenshot", "input": test, "result": "Success" if success else "Failed"})
        except Exception as e:
            logger.error(f"Test {i} failed: {str(e)}")
            results.append({"test": i, "type": "screenshot", "input": test, "error": str(e)})

    # Part 2: Test describe_screenshot function (if requested)
    if test_describe:
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
                success = "result" in result and "error" not in result
                logger.info(f"Description Test {i} result: {'Success' if success else 'Failed'}")
                results.append({"test": i, "type": "describe", "input": test, "success": success})
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
    Validates and runs the standalone MSS screenshot tool.
    
    Examples:
      python mss_screenshot.py run-server  # Start the MCP server
      python mss_screenshot.py test        # Run debug tests
    """
    import sys
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Verify mcp_tools package is installed
    total_tests += 1
    try:
        import mcp_tools
        print(f"✓ mcp_tools package found (version: {getattr(mcp_tools, '__version__', 'unknown')})")
    except ImportError:
        all_validation_failures.append("mcp_tools package not found. Install with: uv pip install -e .")
    
    # Test 2: Verify required modules are available
    total_tests += 1
    try:
        from mcp_tools.screenshot import capture_screenshot, describe_image_content
        print("✓ Required mcp_tools.screenshot modules found")
    except ImportError as e:
        all_validation_failures.append(f"Required modules not found: {str(e)}")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Standalone screenshot tool is ready for use")
        app()  # Start the typer app