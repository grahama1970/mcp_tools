#!/usr/bin/env python3
"""
Image Description Module

This module provides functions for describing images using AI vision models.
It uses LiteLLM to support multiple model providers with a unified interface.

This module is part of the Core Layer and should have no dependencies on
Presentation or Integration layers.

Sample input:
- Image path: "/path/to/image.jpg"
- Model: "vertex_ai/gemini-2.5-pro-preview-05-06"
- Prompt: "Describe this image in detail."

Expected output:
- Dictionary with:
  - description: Detailed description of the image
  - filename: Name of the image file
  - confidence: Score (1-5) indicating description accuracy
  - On error: error message
"""

import os
import json
import base64
import sys
from typing import Dict, Any, Optional, Union

from loguru import logger
from PIL import Image
from litellm import completion, JSONSchemaValidationError

from mcp_tools.screenshot.core.constants import IMAGE_SETTINGS, DEFAULT_MODEL, DEFAULT_PROMPT
from mcp_tools.screenshot.core.image_processing import resize_image_if_needed, compress_image_to_buffer

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


def describe_image_content(
    image_path: str, 
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
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
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    credentials_file = os.path.join(project_root, "vertex_ai_service_account.json")
    if os.path.exists(credentials_file):
        return credentials_file
    
    # Check home directory
    credentials_file = os.path.expanduser("~/.vertex_ai_service_account.json")
    if os.path.exists(credentials_file):
        return credentials_file
    
    return None


if __name__ == "__main__":
    """Validate image description functionality with sample image"""
    import sys
    import os
    from PIL import Image
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Check for sample images
    sample_image_path = None
    sample_dirs = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sample_images"),
        "sample_images"
    ]
    
    for sample_dir in sample_dirs:
        if os.path.exists(sample_dir):
            for filename in os.listdir(sample_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    sample_image_path = os.path.join(sample_dir, filename)
                    break
    
    if not sample_image_path:
        print("⚠️ No sample images found for testing. Creating test image...")
        # Create a test image
        test_dir = ".test_images"
        os.makedirs(test_dir, exist_ok=True)
        test_img = Image.new('RGB', (100, 100), color='blue')
        sample_image_path = os.path.join(test_dir, "test_image.png")
        test_img.save(sample_image_path)
    
    # Test 1: Image preparation
    total_tests += 1
    try:
        img_b64 = prepare_image_for_multimodal(sample_image_path)
        if not img_b64 or not isinstance(img_b64, str):
            all_validation_failures.append(f"Image preparation test: Failed to prepare image")
    except Exception as e:
        all_validation_failures.append(f"Image preparation test: Exception: {str(e)}")
    
    # Test 2: Find credentials file
    total_tests += 1
    credentials_path = find_credentials_file()
    if not credentials_path:
        print("⚠️ Credentials search: No credentials file found")
    else:
        print(f"✓ Credentials search: Found at {credentials_path}")
    
    # Test 3: Image description with real API call
    total_tests += 1
    if sample_image_path:
        # Skip real API call if credentials not found
        if not credentials_path:
            print("⚠️ No credentials file found. API test cannot be performed.")
            # This is not a failure - just a warning
        else:
            print(f"Making real API call to describe image: {sample_image_path}")
            try:
                # Make an actual API call with real data
                result = describe_image_content(
                    image_path=sample_image_path,
                    credentials_file=credentials_path
                )
                
                # Validate the result structure
                if "error" in result:
                    all_validation_failures.append(f"API test failed: {result['error']}")
                elif "description" not in result:
                    all_validation_failures.append(f"API test failed: No description returned")
                elif "confidence" not in result or not isinstance(result["confidence"], int):
                    all_validation_failures.append(f"API test failed: Invalid confidence value")
                elif "filename" not in result:
                    all_validation_failures.append(f"API test failed: Filename missing")
                else:
                    print(f"✓ API call successful: \"{result['description'][:50]}...\" (confidence: {result['confidence']}/5)")
            except Exception as e:
                all_validation_failures.append(f"API call failed with exception: {str(e)}")
    else:
        all_validation_failures.append("No sample image found for testing")
    
    # Clean up test dir if created
    if 'test_dir' in locals() and os.path.exists(test_dir):
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
        print("Image description functions are validated and ready for use")
        sys.exit(0)  # Exit with success code