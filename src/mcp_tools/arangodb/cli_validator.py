#!/usr/bin/env python3
"""
CLI Validator for testing and debugging CLI commands

This module provides validation utilities for the CLI module, specifically:
1. Command-line interface functions in cli.py
2. Verification that all commands work as expected
3. Proper error handling and validation of inputs/outputs

It can be run directly to test the CLI functionality.
"""

import sys
import json
import subprocess
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import time
from datetime import datetime

# Add parent directory to path to allow imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import validation tracker
try:
    from src.arangodb.utils.validation_tracker import ValidationTracker
except ImportError:
    # If validation tracker doesn't exist in the expected location, create a minimal version here
    from datetime import datetime
    
    class ValidationTracker:
        """
        Simple validation tracking utility for testing and debugging.
        Tracks test results and provides reporting functions.
        """
        def __init__(self, module_name):
            self.module_name = module_name
            self.test_results = []
            self.total_tests = 0
            self.failed_tests = 0
            self.start_time = datetime.now()
        
        def check(self, condition, success_message, failure_message):
            """Check a condition and record the result"""
            self.total_tests += 1
            if condition:
                self.pass_(success_message)
                return True
            else:
                self.fail(failure_message)
                return False
        
        def pass_(self, message):
            """Record a passing test"""
            result = {"status": "PASS", "message": message, "timestamp": datetime.now()}
            self.test_results.append(result)
            print(f"✅ PASS: {message}")
            return result
        
        def fail(self, message):
            """Record a failing test"""
            self.failed_tests += 1
            result = {"status": "FAIL", "message": message, "timestamp": datetime.now()}
            self.test_results.append(result)
            print(f"❌ FAIL: {message}")
            return result
        
        def report_and_exit(self):
            """Generate a report of test results and exit with appropriate code"""
            end_time = datetime.now()
            duration = (end_time - self.start_time).total_seconds()
            
            passed_tests = self.total_tests - self.failed_tests
            
            print("\n" + "=" * 70)
            print(f"VALIDATION REPORT: {self.module_name}")
            print("=" * 70)
            print(f"Start time: {self.start_time}")
            print(f"End time: {end_time}")
            print(f"Duration: {duration:.2f} seconds")
            print(f"Tests passed: {passed_tests}/{self.total_tests} ({passed_tests/self.total_tests*100 if self.total_tests > 0 else 0:.1f}%)")
            print("-" * 70)
            
            if self.failed_tests > 0:
                print("\nFAILED TESTS:")
                for i, result in enumerate(self.test_results):
                    if result["status"] == "FAIL":
                        print(f"{i+1}. {result['message']}")
                print("\nSome tests failed - see details above")
                sys.exit(1)
            else:
                print("\nAll tests passed successfully!")
                sys.exit(0)

# Constants
CLI_MODULE_PATH = "cli.py"
CLI_VERIFICATION_PATH = "cli_verification.py"

def validate_cli_commands():
    """
    Validate the CLI commands functionality by testing command execution 
    and validating outputs against expected results.
    """
    validator = ValidationTracker("CLI Commands Module")
    
    # Test 1: CLI module exists and is importable
    try:
        import importlib.util
        cli_spec = importlib.util.find_spec("arangodb.cli")
        validator.check(
            cli_spec is not None,
            "CLI module is importable",
            f"CLI module not found or not importable"
        )
    except Exception as e:
        validator.fail(f"CLI module import check failed: {str(e)}")
    
    # Test 2: CLI has the expected commands
    expected_commands = [
        "search bm25", "search semantic", "search hybrid", "search keyword", "search tag", 
        "crud add-lesson", "crud get-lesson", "crud update-lesson", "crud delete-lesson",
        "graph add-relationship", "graph delete-relationship", "graph traverse",
        "memory store", "memory search", "memory related", "memory context"
    ]
    
    try:
        from src.arangodb.cli import app
        commands = []
        
        # Extract commands from Typer app
        for command in app.registered_commands:
            commands.append(command.name)
        
        # Extract commands from subcommands
        for group in app.registered_groups:
            group_app = group.typer_instance
            for command in group_app.registered_commands:
                commands.append(f"{group.name} {command.name}")
        
        # Validate that expected commands exist
        for expected_cmd in expected_commands:
            parts = expected_cmd.split()
            if len(parts) == 1:
                # Main command
                validator.check(
                    parts[0] in commands,
                    f"CLI has '{expected_cmd}' command",
                    f"Command '{expected_cmd}' not found in CLI"
                )
            else:
                # Subcommand
                found = False
                for cmd in commands:
                    if cmd == expected_cmd:
                        found = True
                        break
                validator.check(
                    found,
                    f"CLI has '{expected_cmd}' command",
                    f"Subcommand '{expected_cmd}' not found in CLI"
                )
    except Exception as e:
        validator.fail(f"CLI commands check failed: {str(e)}")
    
    # Test 3: Command-line execution works
    try:
        # Run a basic command (--help should be safe)
        cmd = f"python -m arangodb.cli --help"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        validator.check(
            result.returncode == 0,
            "CLI help command executes successfully",
            f"CLI help command failed with exit code {result.returncode}: {result.stderr}"
        )
        
        # Check that help output contains expected text
        help_text = result.stdout
        validator.check(
            "Usage:" in help_text,
            "CLI help output contains usage information",
            "CLI help output missing expected 'Usage:' section"
        )
        
        validator.check(
            "search" in help_text,
            "CLI help mentions 'search' commands",
            "CLI help output missing reference to 'search' commands"
        )
    except Exception as e:
        validator.fail(f"CLI execution check failed: {str(e)}")
    
    # Test 4: Input validation in main_callback
    try:
        from src.arangodb.cli import main_callback
        import typer
        from loguru import logger
        
        # Test valid log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        for level in valid_log_levels:
            try:
                # Create dummy context
                ctx = typer.Context(main_callback)
                # Call main_callback with valid log level
                main_callback(log_level=level)
                validator.pass_(f"main_callback accepts valid log level '{level}'")
            except Exception as e:
                validator.fail(f"main_callback failed with valid log level '{level}': {str(e)}")
        
        # Test invalid log level (should default to INFO and not raise exception)
        try:
            main_callback(log_level="INVALID_LEVEL")
            validator.pass_(f"main_callback handles invalid log level gracefully")
        except Exception as e:
            validator.fail(f"main_callback failed with invalid log level: {str(e)}")
            
    except Exception as e:
        validator.fail(f"main_callback validation failed: {str(e)}")
    
    # Test 5: get_db_connection error handling
    try:
        from src.arangodb.cli import get_db_connection
        
        # Temporarily set invalid environment variables to test error handling
        original_host = os.environ.get("ARANGO_HOST", "")
        os.environ["ARANGO_HOST"] = "http://invalid-host:9999"
        
        try:
            # This should fail but be caught
            get_db_connection()
            validator.fail("get_db_connection did not raise error with invalid host")
        except typer.Exit as e:
            # Should exit with code 1
            validator.check(
                e.exit_code == 1,
                "get_db_connection properly exits with code 1 on connection error",
                f"get_db_connection exited with unexpected code: {e.exit_code}"
            )
        finally:
            # Restore original environment variable
            if original_host:
                os.environ["ARANGO_HOST"] = original_host
            else:
                del os.environ["ARANGO_HOST"]
    except ImportError:
        validator.fail("Could not import get_db_connection from cli.py")
    except Exception as e:
        validator.fail(f"get_db_connection validation failed: {str(e)}")
    
    # Test 6: _display_results validation
    try:
        from src.arangodb.cli import _display_results
        
        # Test with empty results
        try:
            empty_data = {"results": [], "total": 0, "offset": 0}
            _display_results(empty_data, "Test", "score")
            validator.pass_("_display_results handles empty results")
        except Exception as e:
            validator.fail(f"_display_results failed with empty results: {str(e)}")
        
        # Test with malformed data
        try:
            malformed_data = "not a dict"
            _display_results(malformed_data, "Test", "score")
            validator.pass_("_display_results handles malformed data without crashing")
        except Exception as e:
            validator.fail(f"_display_results crashed with malformed data: {str(e)}")
        
        # Test with results but missing key fields
        try:
            missing_fields = {"results": [{"doc": {"problem": "test"}}]}
            _display_results(missing_fields, "Test", "score")
            validator.pass_("_display_results handles results with missing fields")
        except Exception as e:
            validator.fail(f"_display_results crashed with missing fields: {str(e)}")
            
    except ImportError:
        validator.fail("Could not import _display_results from cli.py")
    except Exception as e:
        validator.fail(f"_display_results validation failed: {str(e)}")
    
    # Test 7: Truncation in display functions
    try:
        from src.arangodb.cli import _display_results
        
        # Create a test result with very long fields
        long_problem = "A" * 1000  # 1000 characters
        test_data = {
            "results": [
                {
                    "score": 0.95,
                    "doc": {
                        "_key": "test_key",
                        "problem": long_problem,
                        "tags": ["test", "long", "content"]
                    }
                }
            ],
            "total": 1,
            "offset": 0
        }
        
        # This should not crash with long content
        try:
            _display_results(test_data, "Test", "score")
            validator.pass_("_display_results properly handles very long content fields")
        except Exception as e:
            validator.fail(f"_display_results crashed with long content: {str(e)}")
            
    except ImportError:
        validator.fail("Could not import display functions from cli.py")
    except Exception as e:
        validator.fail(f"Display functions validation failed: {str(e)}")
    
    # Output the test results
    validator.report_and_exit()

def validate_cli_verification():
    """
    Validate the CLI verification module which checks the functionality 
    of the CLI commands.
    """
    validator = ValidationTracker("CLI Verification Module")
    
    # Test 1: CLI verification module exists and is importable
    try:
        import importlib.util
        spec = importlib.util.find_spec("arangodb.cli_verification")
        validator.check(
            spec is not None,
            "CLI verification module is importable",
            f"CLI verification module not found or not importable"
        )
    except Exception as e:
        validator.fail(f"CLI verification module import check failed: {str(e)}")
    
    # Test 2: Verification module has expected test functions
    expected_functions = [
        "check_environment", 
        "initialize_test_data",
        "test_bm25_search",
        "test_semantic_search", 
        "test_hybrid_search",
        "test_tag_search",
        "test_memory_store",
        "test_memory_search",
        "test_memory_context",
        "run_all_tests"
    ]
    
    try:
        from arangodb.cli_verification import (
            check_environment, initialize_test_data, test_bm25_search,
            test_semantic_search, test_hybrid_search, test_tag_search,
            test_memory_store, test_memory_search, test_memory_context,
            run_all_tests
        )
        
        for func_name in expected_functions:
            func = locals().get(func_name)
            validator.check(
                callable(func),
                f"Function '{func_name}' exists and is callable",
                f"Function '{func_name}' is not callable or not defined"
            )
    except ImportError as e:
        validator.fail(f"Could not import expected functions from cli_verification: {str(e)}")
    except Exception as e:
        validator.fail(f"Function validation failed: {str(e)}")
    
    # Test 3: run_command function works correctly
    try:
        from arangodb.cli_verification import run_command
        
        # Test with a simple command that should succeed
        success, stdout, stderr = run_command("echo 'test'", check_output=True)
        validator.check(
            success is True,
            "run_command correctly reports success for valid command",
            f"run_command failed for valid command: stdout={stdout}, stderr={stderr}"
        )
        
        validator.check(
            "test" in stdout,
            "run_command captures stdout correctly",
            f"Expected 'test' in stdout but got: '{stdout}'"
        )
        
        # Test with a command that should fail
        success, stdout, stderr = run_command("command_that_does_not_exist", check_output=True)
        validator.check(
            success is False,
            "run_command correctly reports failure for invalid command",
            f"run_command reported success for invalid command"
        )
        
        validator.check(
            stderr != "",
            "run_command captures stderr for failed commands",
            f"Expected non-empty stderr for failed command but got empty string"
        )
        
    except ImportError:
        validator.fail("Could not import run_command from cli_verification")
    except Exception as e:
        validator.fail(f"run_command validation failed: {str(e)}")
    
    # Test 4: Test argument parsing
    try:
        from arangodb.cli_verification import parse_arguments
        
        # Check that arguments are parsed correctly
        # Save sys.argv
        original_argv = sys.argv
        try:
            # Test with no arguments
            sys.argv = ["cli_verification.py"]
            args = parse_arguments()
            validator.check(
                not args.force and not args.verbose and not args.json,
                "parse_arguments correctly handles default arguments",
                f"Unexpected default values: force={args.force}, verbose={args.verbose}, json={args.json}"
            )
            
            # Test with all arguments
            sys.argv = ["cli_verification.py", "--force", "--verbose", "--json"]
            args = parse_arguments()
            validator.check(
                args.force and args.verbose and args.json,
                "parse_arguments correctly handles all arguments",
                f"Failed to parse arguments: force={args.force}, verbose={args.verbose}, json={args.json}"
            )
        finally:
            # Restore sys.argv
            sys.argv = original_argv
    except ImportError:
        validator.fail("Could not import parse_arguments from cli_verification")
    except Exception as e:
        validator.fail(f"Argument parsing validation failed: {str(e)}")
    
    # Output the test results
    validator.report_and_exit()

if __name__ == "__main__":
    # Run the validation tests
    print("Validating CLI Commands Module...")
    validate_cli_commands()
    
    print("\nValidating CLI Verification Module...")
    validate_cli_verification()