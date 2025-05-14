#!/usr/bin/env python3
"""
Unit tests for core/utils.py
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path to import module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

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


class TestCoreUtils(unittest.TestCase):
    """Test cases for core utility functions"""
    
    def test_validate_quality(self):
        """Test quality validation"""
        # Test within range
        self.assertEqual(validate_quality(50, 30, 70), 50)
        
        # Test below minimum
        self.assertEqual(validate_quality(20, 30, 70), 30)
        
        # Test above maximum
        self.assertEqual(validate_quality(80, 30, 70), 70)
    
    def test_validate_region(self):
        """Test region validation"""
        # Test valid cases
        self.assertEqual(validate_region(None), (True, None))
        self.assertEqual(validate_region([0, 0, 100, 100]), (True, None))
        self.assertEqual(validate_region("right_half"), (True, None))
        
        # Test invalid cases
        self.assertEqual(validate_region("invalid_region")[0], False)
        self.assertEqual(validate_region([0, 0])[0], False)
        self.assertEqual(validate_region({"x": 0, "y": 0})[0], False)
    
    def test_generate_filename(self):
        """Test filename generation"""
        # Test default parameters
        filename = generate_filename()
        self.assertTrue(filename.startswith("screenshot_"))
        self.assertTrue(filename.endswith(".jpeg"))
        
        # Test custom parameters
        filename = generate_filename("test", "png")
        self.assertTrue(filename.startswith("test_"))
        self.assertTrue(filename.endswith(".png"))
    
    def test_ensure_directory(self):
        """Test directory creation"""
        test_dir = ".test_dir"
        
        # Test directory creation
        self.assertTrue(ensure_directory(test_dir))
        self.assertTrue(os.path.exists(test_dir))
        
        # Clean up
        os.rmdir(test_dir)
    
    def test_format_error_response(self):
        """Test error response formatting"""
        # Test basic error
        response = format_error_response("Test error")
        self.assertEqual(response["error"], "Test error")
        
        # Test with system info
        response = format_error_response("Test error", include_system_info=True)
        self.assertEqual(response["error"], "Test error")
        self.assertTrue("system_info" in response)
    
    def test_safe_file_operation(self):
        """Test safe file operation wrapper"""
        # Test successful operation
        success, result, error = safe_file_operation(
            "string operation",
            lambda x: x.upper(),
            "test"
        )
        self.assertTrue(success)
        self.assertEqual(result, "TEST")
        self.assertIsNone(error)
        
        # Test failing operation
        success, result, error = safe_file_operation(
            "failing operation",
            lambda: 1/0
        )
        self.assertFalse(success)
        self.assertIsNone(result)
        self.assertTrue("failing operation failed" in error)
    
    def test_parse_region_preset(self):
        """Test region preset parsing"""
        monitor_info = {"width": 1920, "height": 1080, "left": 0, "top": 0}
        
        # Test right_half
        right_half = parse_region_preset("right_half", monitor_info)
        self.assertEqual(right_half["left"], 960)
        self.assertEqual(right_half["width"], 960)
        
        # Test left_half
        left_half = parse_region_preset("left_half", monitor_info)
        self.assertEqual(left_half["left"], 0)
        self.assertEqual(left_half["width"], 960)
        
        # Test top_half
        top_half = parse_region_preset("top_half", monitor_info)
        self.assertEqual(top_half["top"], 0)
        self.assertEqual(top_half["height"], 540)
        
        # Test bottom_half
        bottom_half = parse_region_preset("bottom_half", monitor_info)
        self.assertEqual(bottom_half["top"], 540)
        self.assertEqual(bottom_half["height"], 540)
    
    def test_list_coordinates_to_dict(self):
        """Test coordinates list to dict conversion"""
        coords = [10, 20, 300, 400]
        region_dict = list_coordinates_to_dict(coords)
        
        self.assertEqual(region_dict["left"], 10)
        self.assertEqual(region_dict["top"], 20)
        self.assertEqual(region_dict["width"], 300)
        self.assertEqual(region_dict["height"], 400)
        
        # Test invalid input
        with self.assertRaises(ValueError):
            list_coordinates_to_dict([10, 20])


if __name__ == "__main__":
    unittest.main()
