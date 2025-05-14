#!/usr/bin/env python3
"""
Unit tests for presentation/formatters.py
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from io import StringIO

# Add parent directory to path to import module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from mcp_tools.screenshot.cli.formatters import (
    console,
    print_error,
    print_warning,
    print_info,
    print_json
)


class TestPresentationFormatters(unittest.TestCase):
    """Test cases for presentation formatters"""
    
    def setUp(self):
        """Set up test environment"""
        # Redirect rich console output to StringIO
        self.console_output = StringIO()
        console.file = self.console_output
    
    def tearDown(self):
        """Tear down test environment"""
        # Restore console output
        console.file = sys.stdout
    
    def test_print_error(self):
        """Test error printing"""
        print_error("Test error")
        output = self.console_output.getvalue()
        
        self.assertIn("Error", output)
        self.assertIn("Test error", output)
    
    def test_print_warning(self):
        """Test warning printing"""
        print_warning("Test warning")
        output = self.console_output.getvalue()
        
        self.assertIn("Warning", output)
        self.assertIn("Test warning", output)
    
    def test_print_info(self):
        """Test info printing"""
        print_info("Test info")
        output = self.console_output.getvalue()
        
        self.assertIn("Info", output)
        self.assertIn("Test info", output)
    
    def test_print_json(self):
        """Test JSON printing"""
        test_data = {"name": "test", "value": 123}
        print_json(test_data)
        output = self.console_output.getvalue()
        
        self.assertIn("JSON Output", output)
        self.assertIn("test", output)
        self.assertIn("123", output)


if __name__ == "__main__":
    unittest.main()
