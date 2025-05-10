from PIL import Image
import io
import base64
import os
import sys
import os

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.log_utils import truncate_large_value
from loguru import logger

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)

# Configure logger for validation - remove default stderr handler to avoid output conflicts
logger.remove()  # Remove default handler that outputs to stderr
logger.add(
    "logs/validation.log",
    rotation="10 MB",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

def validate_screenshot_tool():
    """
    Validate the screenshot tool functionality.
    Used for testing the base64 truncation functionality.
    """
    from utils.log_utils import ValidationTracker

    logger.info("Starting screenshot tool validation")
    validator = ValidationTracker("Screenshot Tool Module")

    # Create a test image with known dimensions
    logger.debug("Creating test image with dimensions 200x100")
    test_img = Image.new("RGB", (200, 100), color="red")
    test_bytes = None

    # Save to bytes
    img_buffer = io.BytesIO()
    test_img.save(img_buffer, format="JPEG", quality=75)
    test_bytes = img_buffer.getvalue()
    logger.debug(f"Test image saved as JPEG, size: {len(test_bytes)} bytes")

    # Encode as base64
    test_b64 = base64.b64encode(test_bytes).decode("utf-8")
    logger.debug(f"Test image encoded as base64, length: {len(test_b64)} characters")

    # Test 1: Check that truncate_large_value works on base64 data
    logger.info("Running test: base64 truncation")
    truncated_b64 = truncate_large_value(test_b64, max_str_len=50)
    truncation_result = len(truncated_b64) < len(test_b64) and "..." in truncated_b64

    validator.check(
        "base64 truncation",
        expected=True,
        actual=truncation_result,
        description="Base64 string should be truncated and contain '...'",
    )

    if truncation_result:
        logger.info("base64 truncation test passed")
        logger.debug(f"Original length: {len(test_b64)}, Truncated length: {len(truncated_b64)}")
    else:
        logger.error("base64 truncation test failed")
        logger.debug(f"Original length: {len(test_b64)}, Truncated length: {len(truncated_b64)}")

    # Test 2: Ensure our log-safe response structure is correct
    logger.info("Running test: log-safe response structure")
    log_safe_response = {
        "content": [
            {
                "type": "image",
                "data": truncate_large_value(test_b64, max_str_len=50),
                "mimeType": "image/jpeg",
            }
        ],
        "file": "test_path.jpg",
    }

    structure_result = (isinstance(log_safe_response, dict)
        and isinstance(log_safe_response["content"], list)
        and len(log_safe_response["content"]) == 1
        and log_safe_response["content"][0]["type"] == "image"
        and len(log_safe_response["content"][0]["data"]) < len(test_b64))

    validator.check(
        "log-safe response structure",
        expected=True,
        actual=structure_result,
        description="Log-safe response structure should be correct and contain truncated base64 data",
    )

    if structure_result:
        logger.info("log-safe response structure test passed")
    else:
        logger.error("log-safe response structure test failed")
        logger.debug(f"Response structure: {log_safe_response}")

    # Report validation results
    logger.info("Completing screenshot tool validation")
    validator.report_and_exit()
