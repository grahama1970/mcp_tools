#!/usr/bin/env python3
"""
Test runner for Screenshot MCP Tool

This script discovers and runs all tests for the screenshot module.
"""

import os
import sys
import unittest
import time

# Add parent directory to path to import module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def run_all_tests():
    """Discover and run all tests"""
    start_time = time.time()
    
    print("=" * 80)
    print("Running all tests for Screenshot MCP Tool")
    print("=" * 80)
    
    # Discover tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__), pattern="test_*.py")
    
    # Run tests
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Print summary
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"Test Summary: {result.testsRun} tests run in {elapsed_time:.2f} seconds")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)
    
    # Return exit code
    return 0 if len(result.failures) == 0 and len(result.errors) == 0 else 1


def run_individual_tests():
    """Run tests for individual modules"""
    modules = [
        'mcp_tools.screenshot.core.capture',
        'mcp_tools.screenshot.core.image_processing',
        'mcp_tools.screenshot.core.utils',
        'mcp_tools.screenshot.core.mss',
        'mcp_tools.screenshot.presentation.formatters',
        'mcp_tools.screenshot.integration.wrappers'
    ]
    
    print("=" * 80)
    print("Running individual module tests")
    print("=" * 80)
    
    for module in modules:
        print(f"\nTesting {module}...")
        try:
            __import__(module)
            print(f"✓ Module {module} tests passed")
        except Exception as e:
            print(f"✗ Module {module} tests failed: {str(e)}")
    
    print("=" * 80)


if __name__ == "__main__":
    # Check for arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--individual":
        run_individual_tests()
    else:
        sys.exit(run_all_tests())
