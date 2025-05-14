# Task 1: Refactor Core Functions with Debug Capabilities

**Objective**: Create independently debuggable core functions for the screenshot tool.

**Pre-Task Setup**:
- Review existing core functions in capture.py, image_processing.py, and description.py

**Implementation Steps**:
- [ ] 1.1 Create skeletal structure for core.py module with detailed docstrings
- [ ] 1.2 Implement core screenshot capture function with proper error handling
- [ ] 1.3 Implement core image processing functions with debug output
- [ ] 1.4 Implement core description function with debug capability
- [ ] 1.5 Add debug demonstrations in `__main__` section
- [ ] 1.6 Create sample data validation function for testing
- [ ] 1.7 Test core functions with sample input/output
- [ ] 1.8 Git commit debugged core functions with message "Refactor: Core functions with independent debugging capabilities"

**Technical Specifications**:
- Each function must have:
  - Detailed docstring with Args/Returns sections
  - Type hints for all parameters and return values
  - Example usage in docstring showing sample input/output
  - Stand-alone debug/test capability in `__main__` section
- Error handling must be comprehensive and return structured errors
- No dependencies on CLI or MCP libraries in core functions

**Verification Method**:
- Run the module directly to execute self-tests: `python -m mcp_tools.screenshot.core`
- Verify output matches expected sample data
- Check error handling with invalid inputs

**Acceptance Criteria**:
- All functions run independently without CLI/MCP dependencies
- Debug demonstrations run successfully
- Documentation includes clear sample I/O
- Error cases are properly handled and return structured data

## Example Function Implementation

Each core function will include:

**Self-Contained Examples**:
```python
def capture_screenshot(quality: int = 30, region: Optional[Union[List[int], str]] = None) -> Dict[str, Any]:
    """
    Captures a screenshot of the desktop or a specific region.
    
    Args:
        quality: JPEG compression quality (1-100)
        region: Region coordinates [x, y, width, height] or "right_half"
        
    Returns:
        dict: Response with screenshot data
        
    Example:
        >>> result = capture_screenshot(quality=50, region="right_half")
        >>> print(result.keys())
        dict_keys(['file', 'dimensions', 'size'])
        >>> print(os.path.exists(result['file']))
        True
    """
```

**Self-Test in `__main__` Section**:
```python
if __name__ == "__main__":
    """Module self-test with sample inputs"""
    import sys
    
    def run_demo():
        print("=" * 40)
        print("DEMO: capture_screenshot")
        print("=" * 40)
        
        # Test 1: Basic full-screen capture
        print("\nTest 1: Full-screen capture")
        result = capture_screenshot(quality=30)
        print(f"Result keys: {list(result.keys())}")
        print(f"File exists: {os.path.exists(result['file'])}")
        print(f"Dimensions: {result['dimensions']}")
        print(f"File size: {result['size'] / 1024:.1f} KB")
        
        # Test 2: Right-half capture
        print("\nTest 2: Right-half capture")
        result = capture_screenshot(quality=30, region="right_half")
        print(f"Result keys: {list(result.keys())}")
        print(f"File exists: {os.path.exists(result['file'])}")
        print(f"Dimensions: {result['dimensions']}")
        print(f"File size: {result['size'] / 1024:.1f} KB")
        
        # Test 3: Error case - invalid region
        print("\nTest 3: Invalid region")
        result = capture_screenshot(quality=30, region="invalid")
        print(f"Error correctly returned: {'error' in result}")
        if 'error' in result:
            print(f"Error message: {result['error']}")
            
    # Run the demo if requested
    run_demo()
```
