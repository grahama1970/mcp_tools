#!/usr/bin/env python3
"""
Test script for CLI commands

This script tests the CLI commands by running them with various options
and checking the output. It's useful for end-to-end testing of the CLI.
"""

import os
import sys
import json
import subprocess
from typing import List, Dict, Any, Optional

# Add parent directory to path to import module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def run_command(command: List[str]) -> Dict[str, Any]:
    """
    Run a command and return the result.
    
    Args:
        command: Command to run as list of arguments
        
    Returns:
        Dict[str, Any]: Result with stdout, stderr, and return code
    """
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode
    }


def test_version_command():
    """Test version command"""
    print("\nTesting version command...")
    
    # Test human output
    result = run_command(["python", "-m", "mcp_tools.screenshot.cli.cli", "tools", "version"])
    if result["returncode"] == 0 and "Claude MCP Screenshot Tool" in result["stdout"]:
        print("✅ Human output test passed")
    else:
        print("❌ Human output test failed")
        print(f"Output: {result['stdout']}")
        print(f"Error: {result['stderr']}")
    
    # Test JSON output
    result = run_command(["python", "-m", "mcp_tools.screenshot.cli.cli", "tools", "version", "--json"])
    if result["returncode"] == 0:
        try:
            data = json.loads(result["stdout"])
            if data.get("success") and "data" in data:
                print("✅ JSON output test passed")
            else:
                print("❌ JSON output test failed: Invalid JSON structure")
                print(f"JSON: {data}")
        except json.JSONDecodeError:
            print("❌ JSON output test failed: Invalid JSON")
            print(f"Output: {result['stdout']}")
    else:
        print("❌ JSON output test failed: Command error")
        print(f"Error: {result['stderr']}")


def test_schema_command():
    """Test schema command"""
    print("\nTesting schema command...")
    
    # Test human output
    result = run_command(["python", "-m", "mcp_tools.screenshot.cli.cli", "tools", "schema"])
    if result["returncode"] == 0 and "Screenshot CLI Schema" in result["stdout"]:
        print("✅ Human output test passed")
    else:
        print("❌ Human output test failed")
        print(f"Output: {result['stdout']}")
        print(f"Error: {result['stderr']}")
    
    # Test JSON output
    result = run_command(["python", "-m", "mcp_tools.screenshot.cli.cli", "tools", "schema", "--format", "json"])
    if result["returncode"] == 0:
        try:
            data = json.loads(result["stdout"])
            if data.get("success") and "data" in data and "schema" in data["data"]:
                print("✅ JSON output test passed")
            else:
                print("❌ JSON output test failed: Invalid JSON structure")
                print(f"JSON: {data}")
        except json.JSONDecodeError:
            print("❌ JSON output test failed: Invalid JSON")
            print(f"Output: {result['stdout']}")
    else:
        print("❌ JSON output test failed: Command error")
        print(f"Error: {result['stderr']}")
    
    # Test MCP output
    result = run_command(["python", "-m", "mcp_tools.screenshot.cli.cli", "tools", "schema", "--format", "mcp"])
    if result["returncode"] == 0:
        try:
            data = json.loads(result["stdout"])
            if data.get("success") and "data" in data and "schema" in data["data"] and "functions" in data["data"]["schema"]:
                print("✅ MCP output test passed")
            else:
                print("❌ MCP output test failed: Invalid JSON structure")
                print(f"JSON: {data}")
        except json.JSONDecodeError:
            print("❌ MCP output test failed: Invalid JSON")
            print(f"Output: {result['stdout']}")
    else:
        print("❌ MCP output test failed: Command error")
        print(f"Error: {result['stderr']}")


def test_regions_command():
    """Test regions command"""
    print("\nTesting regions command...")
    
    # Test JSON output
    result = run_command(["python", "-m", "mcp_tools.screenshot.cli.cli", "tools", "regions", "--json"])
    if result["returncode"] == 0:
        try:
            data = json.loads(result["stdout"])
            if data.get("success") and "data" in data and "regions" in data["data"]:
                print("✅ JSON output test passed")
            else:
                print("❌ JSON output test failed: Invalid JSON structure")
                print(f"JSON: {data}")
        except json.JSONDecodeError:
            print("❌ JSON output test failed: Invalid JSON")
            print(f"Output: {result['stdout']}")
    else:
        print("❌ JSON output test failed: Command error")
        print(f"Error: {result['stderr']}")


if __name__ == "__main__":
    print("Testing CLI commands...")
    
    # Test commands
    test_version_command()
    test_schema_command()
    test_regions_command()
    
    print("\nAll tests completed!")
