#!/usr/bin/env python3
"""
Image Processing for Screenshot Module

This module provides functions for processing captured screenshots including
resizing and compressing images to meet size constraints.

This module is part of the Core Layer and should have no dependencies on
Presentation or Integration layers.

Sample input:
- PIL Image object (1920x1080)
- Resize parameters: max_width=640, max_height=640
- Compression parameters: quality=30, min_quality=30, max_file_size=350000

Expected output:
- Resized image (640x360)
- Compressed image as bytes buffer (<350KB)
"""

import io
from typing import Tuple, Optional, Union, Dict, Any
from PIL import Image
from loguru import logger

from mcp_tools.screenshot.core.constants import IMAGE_SETTINGS


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


def process_image(
    img: Image.Image,
    quality: int = IMAGE_SETTINGS["DEFAULT_QUALITY"],
    resize: bool = True
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Complete pipeline for processing a screenshot image:
    1. Resize if needed and requested
    2. Compress to fit size requirements
    
    Args:
        img: PIL Image object to process
        quality: JPEG quality (1-100)
        resize: Whether to resize large images
        
    Returns:
        Tuple[bytes, Dict[str, Any]]: Processed image bytes and metadata
    """
    # Track original dimensions
    original_size = img.size
    
    # Resize if requested
    if resize:
        img = resize_image_if_needed(img)
    
    # Compress the image
    img_bytes = compress_image_to_buffer(img, quality)
    
    # Return the processed image and metadata
    metadata = {
        "original_size": original_size,
        "final_size": img.size,
        "bytes_size": len(img_bytes),
        "quality": quality
    }
    
    return img_bytes, metadata


if __name__ == "__main__":
    """Validate image processing functions with real test data"""
    import sys
    import os
    from PIL import Image
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Create test directory if it doesn't exist
    test_dir = ".test_images"
    os.makedirs(test_dir, exist_ok=True)
    
    try:
        # Test 1: Image resizing - create test image and resize
        total_tests += 1
        test_img = Image.new('RGB', (1920, 1080), color='red')
        resized_img = resize_image_if_needed(test_img)
        expected_size = (640, 360)  # Based on aspect ratio of 1920x1080
        
        if resized_img.size != expected_size:
            all_validation_failures.append(
                f"Image resize test: Expected size {expected_size}, got {resized_img.size}"
            )

        # Test 2: RGBA conversion
        total_tests += 1
        rgba_img = Image.new('RGBA', (100, 100), color=(255, 0, 0, 128))
        rgb_img = ensure_rgb(rgba_img)
        
        if rgb_img.mode != 'RGB':
            all_validation_failures.append(
                f"RGBA conversion test: Expected mode 'RGB', got '{rgb_img.mode}'"
            )

        # Test 3: Image compression under size limit
        total_tests += 1
        small_img = Image.new('RGB', (50, 50), color='blue')
        compressed_bytes = compress_image_to_buffer(small_img)
        
        if len(compressed_bytes) > IMAGE_SETTINGS["MAX_FILE_SIZE"]:
            all_validation_failures.append(
                f"Small image compression test: Expected size < {IMAGE_SETTINGS['MAX_FILE_SIZE']} bytes, "
                f"got {len(compressed_bytes)} bytes"
            )

        # Test 4: Complete pipeline
        total_tests += 1
        test_img = Image.new('RGB', (1920, 1080), color='green')
        processed_bytes, metadata = process_image(test_img)
        
        # Verify processing was performed
        if metadata["original_size"] != (1920, 1080) or metadata["final_size"] == (1920, 1080):
            all_validation_failures.append(
                f"Process image test: Expected resize from (1920, 1080), got {metadata}"
            )
        
        if len(processed_bytes) > IMAGE_SETTINGS["MAX_FILE_SIZE"]:
            all_validation_failures.append(
                f"Process image test: Expected size < {IMAGE_SETTINGS['MAX_FILE_SIZE']} bytes, "
                f"got {len(processed_bytes)} bytes"
            )
            
        # Test 5: Save processed image to disk to verify it's valid
        total_tests += 1
        test_output_path = os.path.join(test_dir, "test_processed.jpg")
        try:
            with open(test_output_path, "wb") as f:
                f.write(processed_bytes)
            
            # Verify we can read it back
            verification_img = Image.open(test_output_path)
            verification_img.load()  # Forces load to catch any issues
        except Exception as e:
            all_validation_failures.append(
                f"Image save test: Failed to save and read processed image: {str(e)}"
            )

    finally:
        # Cleanup test files (uncomment if you want to keep the files for manual inspection)
        import shutil
        shutil.rmtree(test_dir, ignore_errors=True)
        
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Image processing functions are validated and ready for use")
        sys.exit(0)  # Exit with success code