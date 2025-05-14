#!/usr/bin/env python3
"""
Test script to verify the MCP screenshot package functionality.
"""

import os
import sys
import traceback

def test_imports():
    """Test importing all the necessary modules."""
    print("Testing imports...")
    try:
        from mcp_tools.screenshot import (
            IMAGE_SETTINGS,
            capture_screenshot,
            describe_image_content,
            find_credentials_file,
            create_mcp_server
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_creation():
    """Test creating the MCP server."""
    print("\nTesting MCP server creation...")
    try:
        from mcp_tools.screenshot import create_mcp_server
        mcp = create_mcp_server("Test MCP Server")
        print("✅ MCP server created successfully")
        return True
    except Exception as e:
        print(f"❌ MCP server creation error: {e}")
        traceback.print_exc()
        return False

def test_screenshot():
    """Test screenshot functionality."""
    print("\nTesting screenshot capture...")
    try:
        from mcp_tools.screenshot import capture_screenshot
        result = capture_screenshot(quality=30, region="right_half")
        
        if "error" in result:
            print(f"❌ Screenshot error: {result['error']}")
            return False
            
        if "file" not in result or not os.path.exists(result["file"]):
            print("❌ Screenshot file not created")
            return False
            
        print(f"✅ Screenshot captured successfully: {result['file']}")
        return True
    except Exception as e:
        print(f"❌ Screenshot error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing MCP screenshot package functionality...\n")
    
    results = []
    results.append(("Import Test", test_imports()))
    results.append(("MCP Server Creation Test", test_creation()))
    results.append(("Screenshot Test", test_screenshot()))
    
    # Print summary
    print("\n=== TEST SUMMARY ===")
    all_pass = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
        if not result:
            all_pass = False
    
    return 0 if all_pass else 1

if __name__ == "__main__":
    sys.exit(main())