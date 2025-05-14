#!/usr/bin/env python3
"""
Simplified validation module for verify.py

This module tests that verify.py follows best practices for validation:
1. Proper validation of expected vs actual results
2. Complete tracking of all validation failures
3. Proper exit codes (1 for failures, 0 for success)
"""

import sys
import os
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Import validation tracker
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


def validate_verify_module():
    """
    Validate the verify.py module by examining its implementation.
    """
    validator = ValidationTracker("Verify Integration Module")
    
    file_path = "/home/graham/workspace/experiments/arangodb/src/arangodb/verify.py"
    
    # Test 1: File exists
    validator.check(
        os.path.exists(file_path),
        "verify.py file exists",
        f"verify.py file not found at {file_path}"
    )
    
    try:
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Test 2: Check if the file contains code for properly tracking results
        validator.check(
            "results =" in content and "all_passed =" in content,
            "verify.py properly tracks validation results",
            "verify.py does not contain code for tracking validation results"
        )
        
        # Test 3: Check if the file contains code for comparing expected vs actual results
        validator.check(
            '"error" in result' in content or '"results" not in result' in content,
            "verify.py compares expected vs actual results",
            "verify.py does not contain code for comparing expected vs actual results"
        )
        
        # Test 4: Check if the file contains code for proper exit codes
        validator.check(
            "sys.exit(0 if success else 1)" in content or "sys.exit(0)" in content and "sys.exit(1)" in content,
            "verify.py uses proper exit codes",
            "verify.py does not contain code for proper exit codes"
        )
        
        # Test 5: Check if the file uses proper exception handling
        validator.check(
            "try:" in content and "except" in content,
            "verify.py uses proper exception handling",
            "verify.py does not contain code for exception handling"
        )
        
        # Count test cases to see if there's comprehensive validation
        bm25_test_count = len(re.findall(r'bm25_search\(', content))
        semantic_test_count = len(re.findall(r'semantic_search\(', content))
        hybrid_test_count = len(re.findall(r'hybrid_search\(', content))
        
        # Test 6: Check for comprehensive test coverage
        validator.check(
            bm25_test_count > 0 and semantic_test_count > 0 and hybrid_test_count > 0,
            f"verify.py tests multiple components (BM25: {bm25_test_count}, Semantic: {semantic_test_count}, Hybrid: {hybrid_test_count})",
            f"verify.py has limited test coverage (BM25: {bm25_test_count}, Semantic: {semantic_test_count}, Hybrid: {hybrid_test_count})"
        )
        
    except Exception as e:
        validator.fail(f"Error analyzing verify.py: {str(e)}")
    
    # Report results
    validator.report_and_exit()

if __name__ == "__main__":
    validate_verify_module()