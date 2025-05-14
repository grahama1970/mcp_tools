"""
Validation Tracker Framework

This module provides a reusable framework for tracking test results and validation
in a standardized way that complies with the Global Coding Standards.

Features:
- Track all validation failures without stopping at first failure
- Compare actual results against expected results
- Report counts of tests passed/failed
- Proper exit code handling (1 for any failures, 0 for success)
- Colored output for better visualization

Example usage:
```python
from arangodb.utils.validation_tracker import ValidationTracker

if __name__ == "__main__":
    # Create validator
    validator = ValidationTracker("Module Name")
    
    # Run tests
    result1 = some_function(test_input)
    validator.check("Test Case 1", expected=expected_value, actual=result1)
    
    # Test exception cases
    try:
        result2 = some_function(invalid_input)
        validator.fail("Test Case 2", "Expected exception but none was raised")
    except ValueError:
        validator.pass_("Test Case 2", "Correctly raised ValueError")
    
    # Report results and exit with appropriate code
    validator.report_and_exit()
```
"""
import sys
from typing import Any, Dict, List, Optional, Union, Tuple
import inspect
from datetime import datetime

class ValidationTracker:
    """
    A utility class to track validation results, report statistics, and manage exit codes.
    Designed to comply with Global Coding Standards for validation functions.
    """
    
    def __init__(self, module_name: str):
        """
        Initialize a validation tracker with a module name.
        
        Args:
            module_name: The name of the module being validated
        """
        self.module_name = module_name
        self.test_results: List[Dict[str, Any]] = []
        self.total_tests = 0
        self.failed_tests = 0
        self.start_time = datetime.now()
    
    def check(self, test_name: str, expected: Any, actual: Any, description: Optional[str] = None) -> bool:
        """
        Check if expected value matches actual value and record the result.
        
        Args:
            test_name: Name of the test case
            expected: Expected value
            actual: Actual value from the test
            description: Optional description of the test
            
        Returns:
            bool: True if test passed, False if it failed
        """
        self.total_tests += 1
        if expected == actual:
            self.test_results.append({
                "test_name": test_name,
                "result": "PASS",
                "expected": expected,
                "actual": actual,
                "description": description
            })
            return True
        else:
            self.failed_tests += 1
            self.test_results.append({
                "test_name": test_name,
                "result": "FAIL",
                "expected": expected,
                "actual": actual,
                "description": description
            })
            return False
    
    def pass_(self, test_name: str, description: Optional[str] = None) -> None:
        """
        Explicitly record a passed test.
        
        Args:
            test_name: Name of the test case
            description: Optional description of why the test passed
        """
        self.total_tests += 1
        self.test_results.append({
            "test_name": test_name,
            "result": "PASS",
            "description": description
        })
    
    def fail(self, test_name: str, description: Optional[str] = None) -> None:
        """
        Explicitly record a failed test.
        
        Args:
            test_name: Name of the test case
            description: Optional description of why the test failed
        """
        self.total_tests += 1
        self.failed_tests += 1
        self.test_results.append({
            "test_name": test_name,
            "result": "FAIL",
            "description": description
        })
    
    def report(self) -> str:
        """
        Generate a report of all test results with statistics.
        
        Returns:
            str: A formatted report of test results
        """
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        report_lines = [
            f"\n{'=' * 60}",
            f"VALIDATION REPORT FOR: {self.module_name}",
            f"{'=' * 60}",
            f"Timestamp: {end_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {duration:.2f} seconds",
            f"Total Tests: {self.total_tests}",
            f"Failed Tests: {self.failed_tests}",
            f"Passed Tests: {self.total_tests - self.failed_tests}",
            f"{'-' * 60}"
        ]
        
        # Only show detailed results if there are any tests
        if self.total_tests > 0:
            report_lines.append("DETAILED TEST RESULTS:")
            report_lines.append(f"{'-' * 60}")
            
            for idx, result in enumerate(self.test_results, 1):
                # Format header with result status and color
                status_marker = "✅" if result["result"] == "PASS" else "❌"
                header = f"{idx}. {status_marker} {result['test_name']} - {result['result']}"
                report_lines.append(header)
                
                # Add description if available
                if result.get("description"):
                    report_lines.append(f"   Description: {result['description']}")
                
                # Add expected/actual values if available
                if "expected" in result and "actual" in result:
                    report_lines.append(f"   Expected: {result['expected']}")
                    report_lines.append(f"   Actual: {result['actual']}")
                
                report_lines.append("")
        
        # Summary line
        if self.failed_tests == 0:
            report_lines.append(f"✅ VALIDATION PASSED - All {self.total_tests} tests produced expected results")
        else:
            report_lines.append(f"❌ VALIDATION FAILED - {self.failed_tests} of {self.total_tests} tests failed")
        
        report_lines.append(f"{'=' * 60}")
        return "\n".join(report_lines)
    
    def print_report(self) -> None:
        """Print the validation report to the console."""
        print(self.report())
    
    def report_and_exit(self) -> None:
        """
        Print the validation report and exit with appropriate code:
        - 0 for complete success (all tests pass)
        - 1 for any failures
        """
        self.print_report()
        if self.failed_tests > 0:
            sys.exit(1)
        else:
            sys.exit(0)
    
    @staticmethod
    def get_caller_info() -> Tuple[str, int, str]:
        """
        Get information about the caller of a function.
        Useful for automatically determining test locations.
        
        Returns:
            tuple: (filename, line_number, function_name)
        """
        caller = inspect.currentframe().f_back.f_back
        filename = caller.f_code.co_filename
        lineno = caller.f_lineno
        function = caller.f_code.co_name
        return (filename, lineno, function)


# Example usage in the main block
if __name__ == "__main__":
    import sys
    
    # Create a validation tracker for this module
    validator = ValidationTracker("ValidationTracker Module")
    
    # Example 1: Simple equality check
    test_value = "test string"
    validator.check("Simple equality test", expected="test string", actual=test_value)
    
    # Example 2: Deliberately failed test to demonstrate failure tracking
    validator.check("Intentional failure test", expected=10, actual=5, 
                   description="This test is designed to fail for demonstration")
    
    # Example 3: Explicit pass/fail methods
    validator.pass_("Manual pass test", "Explicitly marked as passing")
    validator.fail("Manual fail test", "Explicitly marked as failing")
    
    # Example 4: Exception handling pattern
    try:
        # This should raise a ValueError
        invalid_value = int("not a number")
        validator.fail("Exception test", "Expected ValueError but no exception was raised")
    except ValueError:
        validator.pass_("Exception test", "Correctly raised ValueError for invalid input")
    
    # Example 5: Dict comparison
    expected_dict = {"name": "John", "age": 30}
    actual_dict = {"name": "John", "age": 30}
    validator.check("Dict comparison test", expected=expected_dict, actual=actual_dict)
    
    # Generate final report and exit with appropriate code
    validator.report_and_exit()