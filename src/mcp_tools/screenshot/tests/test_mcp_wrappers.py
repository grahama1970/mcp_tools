#!/usr/bin/env python3
"""
Unit tests for integration/wrappers.py
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcp_tools.screenshot.mcp.wrappers import (
    format_mcp_response,
    screenshot_wrapper,
    describe_screenshot_wrapper
)


class TestIntegrationWrappers(unittest.TestCase):
    """Test cases for integration wrappers"""
    
    def test_format_mcp_response(self):
        """Test MCP response formatting"""
        # Test success response
        success_response = format_mcp_response(True, data={"result": "test"})
        self.assertTrue(success_response["success"])
        self.assertEqual(success_response["result"], "test")
        
        # Test error response
        error_response = format_mcp_response(False, error="Test error")
        self.assertFalse(error_response["success"])
        self.assertEqual(error_response["error"], "Test error")
    
    @patch('mcp_tools.screenshot.integration.wrappers.capture_screenshot')
    def test_screenshot_wrapper_success(self, mock_capture):
        """Test screenshot wrapper with success"""
        # Mock success result
        mock_capture.return_value = {
            "file": "/tmp/screenshot.jpg",
            "content": [{"type": "image", "data": "base64data", "mimeType": "image/jpeg"}]
        }
        
        # Call wrapper
        result = screenshot_wrapper(quality=70)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["file"], "/tmp/screenshot.jpg")
    
    @patch('mcp_tools.screenshot.integration.wrappers.capture_screenshot')
    def test_screenshot_wrapper_error(self, mock_capture):
        """Test screenshot wrapper with error"""
        # Mock error result
        mock_capture.return_value = {
            "error": "Screenshot failed"
        }
        
        # Call wrapper
        result = screenshot_wrapper(quality=70)
        
        # Check result
        self.assertFalse(result["success"])
        self.assertEqual(result["error"], "Screenshot failed")
    
    @patch('mcp_tools.screenshot.integration.wrappers.capture_screenshot')
    @patch('mcp_tools.screenshot.integration.wrappers.describe_image_content')
    def test_describe_screenshot_wrapper(self, mock_describe, mock_capture):
        """Test describe screenshot wrapper"""
        # Mock screenshot result
        mock_capture.return_value = {
            "file": "/tmp/screenshot.jpg",
            "content": [{"type": "image", "data": "base64data", "mimeType": "image/jpeg"}]
        }
        
        # Mock description result
        mock_describe.return_value = {
            "description": "This is a test image",
            "confidence": 4,
            "filename": "screenshot.jpg"
        }
        
        # Call wrapper
        result = describe_screenshot_wrapper(quality=70)
        
        # Check result
        self.assertTrue(result["success"])
        self.assertEqual(result["result"]["description"], "This is a test image")


if __name__ == "__main__":
    unittest.main()
