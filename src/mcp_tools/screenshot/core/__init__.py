"""
Core Layer for Screenshot Module

This package contains the core business logic for the screenshot functionality.
It provides pure functions for screenshot capture, image processing, and description.

The core layer is designed to be:
1. Independent of UI or integration concerns
2. Fully testable in isolation
3. Focused on business logic only
4. Self-validating through unit tests

Usage:
    from mcp_tools.screenshot.core import capture_screenshot, describe_image_content
    result = capture_screenshot(quality=70, region="right_half")
    description = describe_image_content(image_path=result["file"])
"""

# Core constants and settings
from mcp_tools.screenshot.core.constants import (
    IMAGE_SETTINGS, 
    REGION_PRESETS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT
)

# Screenshot capture
from mcp_tools.screenshot.core.capture import capture_screenshot, get_screen_regions

# Image processing
from mcp_tools.screenshot.core.image_processing import (
    resize_image_if_needed, 
    compress_image_to_buffer,
    process_image,
    ensure_rgb
)

# Image description
from mcp_tools.screenshot.core.description import (
    describe_image_content, 
    prepare_image_for_multimodal,
    find_credentials_file
)

# Low-level MSS wrapper
from mcp_tools.screenshot.core.mss import (
    get_monitors,
    capture_monitor,
    capture_region,
    get_system_info
)

# Utility functions
from mcp_tools.screenshot.core.utils import (
    validate_quality,
    validate_region,
    generate_filename,
    ensure_directory,
    format_error_response,
    safe_file_operation,
    parse_region_preset,
    list_coordinates_to_dict
)

__all__ = [
    # Constants
    'IMAGE_SETTINGS',
    'REGION_PRESETS',
    'DEFAULT_MODEL',
    'DEFAULT_PROMPT',
    
    # Screenshot capture
    'capture_screenshot',
    'get_screen_regions',
    
    # Image processing
    'resize_image_if_needed',
    'compress_image_to_buffer',
    'process_image',
    'ensure_rgb',
    
    # Image description
    'describe_image_content',
    'prepare_image_for_multimodal',
    'find_credentials_file',
    
    # MSS wrappers
    'get_monitors',
    'capture_monitor',
    'capture_region',
    'get_system_info',
    
    # Utilities
    'validate_quality',
    'validate_region',
    'generate_filename',
    'ensure_directory',
    'format_error_response',
    'safe_file_operation',
    'parse_region_preset',
    'list_coordinates_to_dict'
]