import os
import json
import base64
from PIL import Image
import io
from litellm import completion, JSONSchemaValidationError
from typing import Union, Dict, Any, Optional

from loguru import logger
from claude_mcp_tools.utils.json_utils import clean_json_string
from claude_mcp_tools.llm_call.initialize_litellm_cache import initialize_litellm_cache


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

    return img.resize((new_width, new_height), Image.Resampling.LANCZOS)


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
    while len(img_bytes) > max_file_size and compress_quality > min_quality:
        compress_quality = max(min_quality, compress_quality - 10)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=compress_quality)
        buffer.seek(0)
        img_bytes = buffer.getvalue()

    return img_bytes


def prepare_image_for_multimodal(
    image_input: Union[str, Image.Image, bytes],
    max_width: int = 640,
    max_height: int = 640,
    initial_quality: int = 30,
    filename: Optional[str] = None
) -> tuple:
    """
    Prepares an image for multimodal API calls, handling multiple input types:
    1. File path (str)
    2. PIL Image object
    3. Base64 encoded string
    4. Bytes

    Args:
        image_input: Image input (file path, PIL Image, base64 string, or bytes)
        max_width: Maximum width for resizing
        max_height: Maximum height for resizing
        initial_quality: Initial JPEG compression quality
        filename: Optional filename override (default: extracted from path if available)

    Returns:
        tuple: (base64_encoded_image_string, filename)
    """
    try:
        # Handle different input types
        img = None
        derived_filename = filename or "image.jpg"

        if isinstance(image_input, str):
            # Check if it's a base64 string
            if image_input.startswith(('data:image', 'iVBOR', '/9j/')):
                # Extract base64 data if it's a data URL
                if image_input.startswith('data:image'):
                    # Extract the base64 part after the comma
                    base64_data = image_input.split(',', 1)[1]
                    img_data = base64.b64decode(base64_data)
                else:
                    # Assume it's already base64 without data URL wrapper
                    img_data = base64.b64decode(image_input)
                img = Image.open(io.BytesIO(img_data))
            else:
                # Assume it's a file path
                img = Image.open(image_input)
                derived_filename = os.path.basename(image_input)
        elif isinstance(image_input, Image.Image):
            # Already a PIL Image
            img = image_input
        elif isinstance(image_input, bytes):
            # Bytes data
            img = Image.open(io.BytesIO(image_input))
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        # Resize if needed
        img = resize_image_if_needed(img, max_width, max_height)

        # Compress the image
        img_bytes = compress_image_to_buffer(img, initial_quality)

        # Encode to base64
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        return img_b64, derived_filename
    except Exception as e:
        logger.error(f"Error preparing image: {str(e)}")
        raise


# Define the response schema for image description
response_schema = {
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


def describe_image(
    image_input: Union[str, Image.Image, bytes],
    model: str = "vertex_ai/gemini-2.5-pro-preview-05-06",
    prompt: str = "Describe this image in detail.",
    credentials_file: Optional[str] = None,
    max_width: int = 640,
    max_height: int = 640,
    initial_quality: int = 30,
    filename: Optional[str] = None,
    return_raw_response: bool = False
) -> Union[Dict[str, Any], Any]:
    """
    Generic function to describe an image using AI vision models.
    Supports multiple input formats and models.

    Args:
        image_input: Image input (file path, PIL Image, base64 string, or bytes)
        model: AI model to use
        prompt: Text prompt for the image description
        credentials_file: Path to credentials file for API authentication
        max_width: Maximum width for resizing
        max_height: Maximum height for resizing
        initial_quality: Initial JPEG compression quality
        filename: Optional filename override (default: extracted from path if available)
        return_raw_response: If True, returns raw API response; if False, returns parsed JSON

    Returns:
        dict: Description results containing 'description', 'filename', and 'confidence'
             or raw API response if return_raw_response is True
    """
    # Prepare the image
    image_b64, derived_filename = prepare_image_for_multimodal(
        image_input,
        max_width=max_width,
        max_height=max_height,
        initial_quality=initial_quality,
        filename=filename
    )

    # Use provided filename or derived one
    final_filename = filename or derived_filename

    # Prepare credentials if provided
    vertex_credentials = None
    if credentials_file:
        with open(credentials_file, "r") as file:
            vertex_credentials = json.load(file)

    # Construct messages with multimodal content
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt} Respond with a JSON object that includes: 1) a 'description' field with your detailed description, 2) a 'filename' field with the value '{final_filename}', and 3) a 'confidence' field with a number from 1-5 (5 being highest) indicating your confidence in the accuracy of your description considering image quality and compression artifacts."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        }
    ]

    try:
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

        if return_raw_response:
            return response

        # Parse the JSON response
        response_json = clean_json_string(response.choices[0].message.content, return_dict=True)
        return response_json
    except JSONSchemaValidationError as e:
        logger.error(f"JSON schema validation error: {str(e)}")
        logger.debug(f"Raw response: {e.raw_response}")
        raise


# Example usage
if __name__ == "__main__":
    import sys
    from PIL import Image
    import io

    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0

    # Initialize LiteLLM cache
    initialize_litellm_cache()

    # Path to your service account credentials
    credentials_file = "vertex_ai_service_account.json"

    # Path to the image
    image_path = "sample_images/panda.png"

    # Example prompt
    prompt = "Describe this image in detail."

    # Test 1: File path input
    total_tests += 1
    print(f"Running test 1/{total_tests}: Image description from file path")

    try:
        response_json = describe_image(
            image_input=image_path,
            prompt=prompt,
            credentials_file=credentials_file
        )

        # Check that all required fields are present
        required_fields = ["description", "filename", "confidence"]
        for field in required_fields:
            if field not in response_json:
                all_validation_failures.append(f"Test 1: Required field '{field}' missing from response")

        # Check that confidence is between 1-5
        if "confidence" in response_json:
            confidence = response_json["confidence"]
            if not (isinstance(confidence, int) and 1 <= confidence <= 5):
                all_validation_failures.append(f"Test 1: Confidence value {confidence} is not an integer between 1-5")

        # Check that filename matches
        if "filename" in response_json:
            if response_json["filename"] != os.path.basename(image_path):
                all_validation_failures.append(f"Test 1: Filename '{response_json['filename']}' doesn't match expected '{os.path.basename(image_path)}'")

        print(f"Test 1 JSON Content: {response_json}")
    except Exception as e:
        all_validation_failures.append(f"Test 1: Exception occurred: {str(e)}")

    # Test 2: PIL Image object input
    total_tests += 1
    print(f"Running test 2/{total_tests}: Image description from PIL Image object")

    try:
        # Open image as PIL Image
        with Image.open(image_path) as img:
            # Make a copy to ensure we're not using a closed file
            img_copy = img.copy()

        response_json = describe_image(
            image_input=img_copy,
            prompt=prompt,
            credentials_file=credentials_file,
            filename="pil_image_test.jpg"  # Custom filename for testing
        )

        # Check that all required fields are present
        required_fields = ["description", "filename", "confidence"]
        for field in required_fields:
            if field not in response_json:
                all_validation_failures.append(f"Test 2: Required field '{field}' missing from response")

        # Check that confidence is between 1-5
        if "confidence" in response_json:
            confidence = response_json["confidence"]
            if not (isinstance(confidence, int) and 1 <= confidence <= 5):
                all_validation_failures.append(f"Test 2: Confidence value {confidence} is not an integer between 1-5")

        # Check that filename matches our custom name
        if "filename" in response_json:
            if response_json["filename"] != "pil_image_test.jpg":
                all_validation_failures.append(f"Test 2: Filename '{response_json['filename']}' doesn't match expected 'pil_image_test.jpg'")

        print(f"Test 2 JSON Content: {response_json}")
    except Exception as e:
        all_validation_failures.append(f"Test 2: Exception occurred: {str(e)}")

    # Test 3: Base64 input
    total_tests += 1
    print(f"Running test 3/{total_tests}: Image description from base64 string")

    try:
        # Create base64 string from file
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
            base64_data = base64.b64encode(img_bytes).decode("utf-8")

        response_json = describe_image(
            image_input=base64_data,
            prompt=prompt,
            credentials_file=credentials_file,
            filename="base64_test.jpg"  # Custom filename for testing
        )

        # Check that all required fields are present
        required_fields = ["description", "filename", "confidence"]
        for field in required_fields:
            if field not in response_json:
                all_validation_failures.append(f"Test 3: Required field '{field}' missing from response")

        # Check that confidence is between 1-5
        if "confidence" in response_json:
            confidence = response_json["confidence"]
            if not (isinstance(confidence, int) and 1 <= confidence <= 5):
                all_validation_failures.append(f"Test 3: Confidence value {confidence} is not an integer between 1-5")

        # Check that filename matches our custom name
        if "filename" in response_json:
            if response_json["filename"] != "base64_test.jpg":
                all_validation_failures.append(f"Test 3: Filename '{response_json['filename']}' doesn't match expected 'base64_test.jpg'")

        print(f"Test 3 JSON Content: {response_json}")
    except Exception as e:
        all_validation_failures.append(f"Test 3: Exception occurred: {str(e)}")

    # Test 4: Bytes input
    total_tests += 1
    print(f"Running test 4/{total_tests}: Image description from bytes")

    try:
        # Read image as bytes
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()

        response_json = describe_image(
            image_input=img_bytes,
            prompt=prompt,
            credentials_file=credentials_file,
            filename="bytes_test.jpg"  # Custom filename for testing
        )

        # Check that all required fields are present
        required_fields = ["description", "filename", "confidence"]
        for field in required_fields:
            if field not in response_json:
                all_validation_failures.append(f"Test 4: Required field '{field}' missing from response")

        # Check that confidence is between 1-5
        if "confidence" in response_json:
            confidence = response_json["confidence"]
            if not (isinstance(confidence, int) and 1 <= confidence <= 5):
                all_validation_failures.append(f"Test 4: Confidence value {confidence} is not an integer between 1-5")

        # Check that filename matches our custom name
        if "filename" in response_json:
            if response_json["filename"] != "bytes_test.jpg":
                all_validation_failures.append(f"Test 4: Filename '{response_json['filename']}' doesn't match expected 'bytes_test.jpg'")

        print(f"Test 4 JSON Content: {response_json}")
    except Exception as e:
        all_validation_failures.append(f"Test 4: Exception occurred: {str(e)}")

    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        sys.exit(0)  # Exit with success code
