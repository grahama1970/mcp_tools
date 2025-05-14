#!/usr/bin/env python3
"""
MCP Wrappers for Screenshot Module

This module provides MCP-specific wrapper functions for the core screenshot
functionality, handling parameter validation and error formatting specific to MCP.

This module is part of the Integration Layer and can depend on both
Core Layer and Presentation Layer components.

Sample input:
- MCP function parameters

Expected output:
- MCP-compatible response dictionaries
"""

import os
import sys
import json
import traceback
from typing import Dict, List, Any, Optional, Union, Callable

from loguru import logger

from mcp_tools.screenshot.core.constants import (
    IMAGE_SETTINGS, 
    REGION_PRESETS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT
)
from mcp_tools.screenshot.core.capture import capture_screenshot, get_screen_regions
from mcp_tools.screenshot.core.description import describe_image_content, find_credentials_file
from mcp_tools.screenshot.core.utils import validate_quality, validate_region, format_error_response


def format_mcp_response(
    success: bool, 
    data: Optional[Dict[str, Any]] = None, 
    error: Optional[str] = None
) -> Dict[str, Any]:
    """
    Format a response in MCP-compatible format.
    
    Args:
        success: Whether the operation was successful
        data: Response data (for successful operations)
        error: Error message (for failed operations)
        
    Returns:
        Dict[str, Any]: MCP-compatible response
    """
    response = {"success": success}
    
    if success and data is not None:
        response.update(data)
    elif not success and error is not None:
        response["error"] = error
    
    return response


def screenshot_wrapper(
    quality: int = IMAGE_SETTINGS["DEFAULT_QUALITY"],
    region: Optional[Union[List[int], str]] = None,
    include_description: bool = False,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    MCP wrapper for screenshot functionality.
    
    Args:
        quality: JPEG compression quality (1-100)
        region: Region coordinates [x, y, width, height] or preset name
        include_description: Whether to include an AI description
        prompt: Text prompt for the AI description
        model: AI model to use for description
        
    Returns:
        Dict[str, Any]: MCP-compatible response
    """
    try:
        # Validate quality
        quality = validate_quality(
            quality, 
            IMAGE_SETTINGS["MIN_QUALITY"], 
            IMAGE_SETTINGS["MAX_QUALITY"]
        )
        
        # Validate region
        is_valid, error = validate_region(region)
        if not is_valid:
            return format_mcp_response(False, error=error)
        
        # Capture screenshot
        screenshot_result = capture_screenshot(quality=quality, region=region)
        
        # Check if screenshot failed
        if "error" in screenshot_result:
            return format_mcp_response(False, error=screenshot_result["error"])
            
        # If description is requested, add it
        if include_description:
            try:
                # Get the file path from the screenshot result
                image_path = screenshot_result["file"]
                
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
                    screenshot_result["description_error"] = description_result["error"]
                else:
                    logger.info(f"Image description successful")
                    screenshot_result["result"] = description_result
                    
                    # Remove large base64 data to reduce response size when description succeeds
                    if "content" in screenshot_result:
                        del screenshot_result["content"]
            
            except Exception as e:
                logger.error(f"Failed to get image description: {str(e)}")
                screenshot_result["description_error"] = f"Failed to get image description: {str(e)}"
        
        return format_mcp_response(True, data=screenshot_result)
        
    except Exception as e:
        error_message = f"Screenshot operation failed: {str(e)}"
        logger.error(error_message, exc_info=True)
        return format_mcp_response(False, error=error_message)


def describe_screenshot_wrapper(
    quality: int = IMAGE_SETTINGS["DEFAULT_QUALITY"], 
    region: Optional[Union[List[int], str]] = None,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    MCP wrapper for screenshot with description.
    
    Args:
        quality: JPEG compression quality (1-100)
        region: Region coordinates [x, y, width, height] or preset name
        prompt: Text prompt for the AI description
        model: AI model to use for description
        
    Returns:
        Dict[str, Any]: MCP-compatible response
    """
    try:
        # Validate quality
        quality = validate_quality(
            quality, 
            IMAGE_SETTINGS["MIN_QUALITY"], 
            IMAGE_SETTINGS["MAX_QUALITY"]
        )
        
        # Validate region
        is_valid, error = validate_region(region)
        if not is_valid:
            return format_mcp_response(False, error=error)
        
        # First take the screenshot
        screenshot_result = capture_screenshot(quality=quality, region=region)
        
        # Check if screenshot failed
        if "error" in screenshot_result:
            return format_mcp_response(False, error=screenshot_result["error"])
        
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
                return format_mcp_response(
                    False,
                    error=f"Image captured successfully but description failed: {description_result['error']}",
                    data={"file": image_path}
                )
            
            logger.info(f"Image description successful")
            
            # Create a response combining screenshot and description, but WITHOUT image data
            response = {
                "result": description_result,
                "file": image_path
            }
            
            return format_mcp_response(True, data=response)
            
        except Exception as e:
            logger.error(f"Image description failed: {str(e)}", exc_info=True)
            return format_mcp_response(
                False,
                error=f"Image captured successfully but description failed: {str(e)}",
                data={"file": image_path}
            )
            
    except Exception as e:
        error_message = f"Describe screenshot operation failed: {str(e)}"
        logger.error(error_message, exc_info=True)
        return format_mcp_response(False, error=error_message)


def regions_wrapper() -> Dict[str, Any]:
    """
    MCP wrapper for getting screen regions.
    
    Returns:
        Dict[str, Any]: MCP-compatible response with regions information
    """
    try:
        regions = get_screen_regions()
        return format_mcp_response(True, data={"regions": regions})
    except Exception as e:
        error_message = f"Get regions operation failed: {str(e)}"
        logger.error(error_message, exc_info=True)
        return format_mcp_response(False, error=error_message)


def describe_image_wrapper(
    image_path: str,
    prompt: str = DEFAULT_PROMPT,
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    MCP wrapper for describing an existing image.
    
    Args:
        image_path: Path to the image file
        prompt: Text prompt for the AI description
        model: AI model to use for description
        
    Returns:
        Dict[str, Any]: MCP-compatible response
    """
    try:
        # Validate image path
        if not os.path.exists(image_path):
            return format_mcp_response(False, error=f"Image file not found: {image_path}")
        
        if not os.path.isfile(image_path):
            return format_mcp_response(False, error=f"Not a file: {image_path}")
        
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
            return format_mcp_response(False, error=description_result["error"])
        
        return format_mcp_response(True, data={"result": description_result})
        
    except Exception as e:
        error_message = f"Describe image operation failed: {str(e)}"
        logger.error(error_message, exc_info=True)
        return format_mcp_response(False, error=error_message)


if __name__ == "__main__":
    """Validate MCP wrapper functions"""
    import sys
    
    # Configure logger if running as main script
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<level>{level}: {message}</level>",
        level="INFO",
        colorize=True
    )
    
    print("Testing MCP wrapper functions...")
    
    # Test format_mcp_response
    success_response = format_mcp_response(True, data={"result": "test"})
    error_response = format_mcp_response(False, error="Test error")
    
    print(f"Success response: {json.dumps(success_response)}")
    print(f"Error response: {json.dumps(error_response)}")
    
    print("\nNote: Complete testing requires actual screenshots and API calls.")
    print("For comprehensive testing, use the MCP server's testing functionality.")
