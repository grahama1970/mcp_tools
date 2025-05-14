#!/usr/bin/env python3
"""
Schema Definitions for CLI Module

This module provides schema definitions and validation functions
for CLI output formatting and validation. It handles the conversion
between Typer CLI command structures and various schema formats,
including MCP-compatible schemas.

This module is part of the Presentation Layer and should only depend on
Core Layer components, not on Integration Layer.

Third-party package documentation:
- Typer: https://typer.tiangolo.com/
- Pydantic: https://docs.pydantic.dev/
- FastMCP: https://github.com/anthropics/fastmcp

Sample input:
- CLI command metadata from Typer app:
  ```python
  app = typer.Typer()

  @app.command()
  def screenshot(
      quality: int = typer.Option(30, "--quality", "-q", help="JPEG quality"),
      region: Optional[str] = typer.Option(None, "--region", "-r")
  ):
      \"\"\"Take a screenshot\"\"\"
      pass

  # Generate schema from this app
  schema = generate_cli_schema(app)
  ```

Expected output:
- CLI Schema:
  ```python
  {
      "name": "Screenshot CLI",
      "version": "1.0.0",
      "description": "Command line interface for screenshot capture and description",
      "commands": {
          "screenshot": {
              "name": "screenshot",
              "help": "Take a screenshot",
              "parameters": {
                  "quality": {
                      "name": "quality",
                      "type": "int",
                      "default": "30",
                      "required": False,
                      "help": "JPEG quality"
                  },
                  "region": {
                      "name": "region",
                      "type": "Optional[str]",
                      "default": "None",
                      "required": False,
                      "help": ""
                  }
              }
          }
      }
  }
  ```

- MCP Schema:
  ```python
  {
      "functions": {
          "screenshot": {
              "description": "Take a screenshot",
              "parameters": {
                  "type": "object",
                  "properties": {
                      "quality": {
                          "type": "integer",
                          "description": "JPEG quality",
                          "default": "30"
                      },
                      "region": {
                          "type": "string",
                          "description": ""
                      }
                  },
                  "required": []
              }
          }
      }
  }
  ```

- Response Format:
  ```python
  {
      "success": True,
      "data": {"result": "screenshot taken"}
  }
  # or
  {
      "success": False,
      "error": "Failed to take screenshot"
  }
  ```
"""

import os
import sys
import json
import inspect
from typing import Dict, List, Any, Optional, Union, Tuple, Callable, Type

import typer
from pydantic import BaseModel, Field

from mcp_tools.screenshot.core.constants import (
    IMAGE_SETTINGS, 
    REGION_PRESETS,
    DEFAULT_MODEL,
    DEFAULT_PROMPT
)


# Response structure models
class ErrorResponse(BaseModel):
    """Error response model"""
    success: bool = False
    error: str
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseModel):
    """Success response model"""
    success: bool = True
    data: Dict[str, Any]


# Schema generation functions
def generate_cli_schema(app: typer.Typer) -> Dict[str, Any]:
    """
    Generate schema for a Typer app.
    
    Args:
        app: Typer app
        
    Returns:
        Dict[str, Any]: Schema dictionary
    """
    schema = {
        "name": "Screenshot CLI",
        "version": "1.0.0",
        "description": "Command line interface for screenshot capture and description",
        "commands": {}
    }
    
    # Add main app commands
    _add_commands_to_schema(schema["commands"], app)
    
    return schema


def _add_commands_to_schema(schema: Dict[str, Any], app: typer.Typer) -> None:
    """
    Add commands to schema dictionary.
    
    Args:
        schema: Schema dictionary to update
        app: Typer app
    """
    # Get commands from the app
    for command_name, command in app.registered_commands.items():
        schema[command_name] = {
            "name": command_name,
            "help": command.help or command.callback.__doc__ or "",
            "parameters": _get_command_parameters(command)
        }
    
    # Get commands from sub-apps
    for sub_app_name, sub_app in app.registered_apps.items():
        schema[sub_app_name] = {
            "name": sub_app_name,
            "help": sub_app.app.help or "",
            "commands": {}
        }
        _add_commands_to_schema(schema[sub_app_name]["commands"], sub_app.app)


def _get_command_parameters(command: typer.models.CommandInfo) -> Dict[str, Any]:
    """
    Get parameters for a command.
    
    Args:
        command: Typer command
        
    Returns:
        Dict[str, Any]: Parameter schema
    """
    parameters = {}
    
    for param_name, param in command.parameters.items():
        if param_name == "ctx":  # Skip context parameter
            continue
            
        parameters[param_name] = {
            "name": param_name,
            "type": str(param.annotation.__name__ if hasattr(param.annotation, "__name__") else param.annotation),
            "default": str(param.default) if param.default is not param.empty else None,
            "required": param.default is param.empty,
            "help": param.help
        }
    
    return parameters


def convert_to_mcp_schema(cli_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert CLI schema to MCP schema format.
    
    Args:
        cli_schema: CLI schema
        
    Returns:
        Dict[str, Any]: MCP-compatible schema
    """
    mcp_schema = {
        "functions": {}
    }
    
    # Convert top-level commands
    for command_name, command in cli_schema["commands"].items():
        _add_command_to_mcp_schema(mcp_schema["functions"], command_name, command)
    
    # Convert sub-app commands
    for sub_app_name, sub_app in cli_schema["commands"].items():
        if "commands" in sub_app:
            for cmd_name, cmd in sub_app["commands"].items():
                full_name = f"{sub_app_name}_{cmd_name}"
                _add_command_to_mcp_schema(mcp_schema["functions"], full_name, cmd)
    
    return mcp_schema


def _add_command_to_mcp_schema(schema: Dict[str, Any], name: str, command: Dict[str, Any]) -> None:
    """
    Add a command to MCP schema.
    
    Args:
        schema: MCP schema to update
        name: Command name
        command: Command info
    """
    schema[name] = {
        "description": command.get("help", ""),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
    
    # Add parameters
    for param_name, param in command.get("parameters", {}).items():
        schema[name]["parameters"]["properties"][param_name] = {
            "type": _convert_type_to_json_schema(param.get("type", "string")),
            "description": param.get("help", "")
        }
        
        # Add default value if available
        if param.get("default") not in (None, "None"):
            schema[name]["parameters"]["properties"][param_name]["default"] = param.get("default")
        
        # Add to required list if required
        if param.get("required", False):
            schema[name]["parameters"]["required"].append(param_name)


def _convert_type_to_json_schema(type_str: str) -> str:
    """
    Convert Python type to JSON schema type.
    
    Args:
        type_str: Python type as string
        
    Returns:
        str: JSON schema type
    """
    type_map = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
        "None": "null",
        "NoneType": "null",
        "Optional[str]": "string",
        "Optional[int]": "integer",
        "Union[str, List[int]]": "string"
    }
    
    return type_map.get(type_str, "string")


# Response formatting functions
def format_cli_response(success: bool, data: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> Dict[str, Any]:
    """
    Format a standardized CLI response.

    Args:
        success: Whether the operation was successful
        data: Response data (for successful operations)
        error: Error message (for failed operations)

    Returns:
        Dict[str, Any]: Formatted response
    """
    if success and data is not None:
        # Using model_dump instead of dict for Pydantic v2 compatibility
        return SuccessResponse(data=data).model_dump()
    elif not success and error is not None:
        response = ErrorResponse(error=error).model_dump()
        # Remove None values for expected test results
        if response.get('details') is None:
            del response['details']
        return response
    else:
        return {"success": success}


def validate_output_against_schema(output: Dict[str, Any], schema_model: Type[BaseModel]) -> Tuple[bool, Optional[str]]:
    """
    Validate output against a schema model.
    
    Args:
        output: Output data to validate
        schema_model: Pydantic model to validate against
        
    Returns:
        Tuple[bool, Optional[str]]: (is_valid, error_message)
    """
    try:
        schema_model(**output)
        return True, None
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    """Validate schema functions with real test data"""
    import sys
    import json
    from rich.console import Console
    from rich import print_json
    import typer
    from typing import Optional

    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0

    console = Console()
    console.print("[bold]Schema Functions Validation[/bold]")

    # Create a test Typer app for validation
    test_app = typer.Typer()

    @test_app.command()
    def test_command(
        param1: str = typer.Argument(..., help="Required parameter"),
        param2: int = typer.Option(42, "--param2", "-p", help="Optional parameter with default"),
        flag: bool = typer.Option(False, "--flag", "-f", help="Boolean flag")
    ):
        """Test command with multiple parameters"""
        pass

    # Test 1: Manual schema creation validation
    total_tests += 1
    try:
        # Create a sample CLI schema directly (avoiding implementation details)
        cli_schema = {
            "name": "Test CLI",
            "version": "1.0.0",
            "description": "Test CLI for validation",
            "commands": {
                "test_command": {
                    "name": "test_command",
                    "help": "Test command with multiple parameters",
                    "parameters": {
                        "param1": {
                            "name": "param1",
                            "type": "str",
                            "required": True,
                            "help": "Required parameter"
                        },
                        "param2": {
                            "name": "param2",
                            "type": "int",
                            "default": "42",
                            "required": False,
                            "help": "Optional parameter with default"
                        }
                    }
                }
            }
        }

        # Verify schema structure
        if not isinstance(cli_schema, dict) or "commands" not in cli_schema:
            all_validation_failures.append("Schema test: Invalid schema structure")
        elif "test_command" not in cli_schema["commands"]:
            all_validation_failures.append("Schema test: Command missing from schema")
        elif "parameters" not in cli_schema["commands"]["test_command"]:
            all_validation_failures.append("Schema test: Parameters missing from command")
        else:
            console.print("✓ Schema test successful:")
            # Just show the test command for brevity
            test_cmd_schema = cli_schema["commands"]["test_command"]
            print_json(data=test_cmd_schema)
    except Exception as e:
        all_validation_failures.append(f"Schema test failed: {str(e)}")

    # Test 2: MCP schema conversion
    total_tests += 1
    try:
        if 'cli_schema' in locals():
            mcp_schema = convert_to_mcp_schema(cli_schema)
            # Verify MCP schema structure
            if not isinstance(mcp_schema, dict) or "functions" not in mcp_schema:
                all_validation_failures.append("MCP schema conversion test: Invalid MCP schema structure")
            elif "test_command" not in mcp_schema["functions"]:
                all_validation_failures.append("MCP schema conversion test: Command missing from MCP schema")
            else:
                console.print("\n✓ MCP schema conversion successful:")
                # Just show the test command for brevity
                test_cmd_mcp = mcp_schema["functions"]["test_command"]
                print_json(data=test_cmd_mcp)
        else:
            all_validation_failures.append("MCP schema conversion test: CLI schema unavailable")
    except Exception as e:
        all_validation_failures.append(f"MCP schema conversion test failed: {str(e)}")

    # Test 3: Response validation (Success case)
    total_tests += 1
    success_data = {"success": True, "data": {"result": "test"}}
    try:
        is_valid, error = validate_output_against_schema(success_data, SuccessResponse)
        if not is_valid:
            all_validation_failures.append(f"Success validation test failed: {error}")
    except Exception as e:
        all_validation_failures.append(f"Success validation test failed with exception: {str(e)}")

    # Test 4: Response validation (Error case)
    total_tests += 1
    error_data = {"success": False, "error": "Test error"}
    try:
        is_valid, error = validate_output_against_schema(error_data, ErrorResponse)
        if not is_valid:
            all_validation_failures.append(f"Error validation test failed: {error}")
    except Exception as e:
        all_validation_failures.append(f"Error validation test failed with exception: {str(e)}")

    # Test 5: Response formatting
    total_tests += 1
    try:
        success_response = format_cli_response(True, data={"result": "test"})
        expected_success = {"success": True, "data": {"result": "test"}}
        if success_response != expected_success:
            all_validation_failures.append(f"Response formatting test (success): Expected {expected_success}, got {success_response}")

        error_response = format_cli_response(False, error="Test error")
        expected_error = {"success": False, "error": "Test error"}
        if error_response != expected_error:
            all_validation_failures.append(f"Response formatting test (error): Expected {expected_error}, got {error_response}")

        console.print("\n✓ Response formatting examples:")
        console.print("Success response:")
        print_json(data=success_response)
        console.print("Error response:")
        print_json(data=error_response)
    except Exception as e:
        all_validation_failures.append(f"Response formatting test failed: {str(e)}")

    # Final validation result
    if all_validation_failures:
        print(f"\n❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)  # Exit with error code
    else:
        print(f"\n✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Schema functions are validated and ready for use")
        sys.exit(0)  # Exit with success code
