# Task 6: Integration Testing with MCP Inspector

**Objective**: Perform end-to-end testing of the screenshot tool using MCP Inspector.

**Pre-Task Setup**:
- Ensure MCP Inspector is installed and working

**Implementation Steps**:
- [ ] 6.1 Start MCP server for testing
- [ ] 6.2 Connect with MCP Inspector
- [ ] 6.3 Test screenshot tool with various parameters
- [ ] 6.4 Test describe_screenshot tool with various parameters
- [ ] 6.5 Verify results match expected output
- [ ] 6.6 Check logs for expected events
- [ ] 6.7 Document any issues or unexpected behavior
- [ ] 6.8 Git commit final tested version with message "Complete: Fully tested MCP screenshot tool with end-to-end verification"

**Technical Specifications**:
- Use MCP Inspector to test tools interactively
- Test with various parameter combinations
- Verify output format matches expectations
- Check logs for detailed execution flow

**Verification Method**:
- Use MCP Inspector to interactively test tools
- Examine logs for detailed execution information
- Verify results match expected output

**Acceptance Criteria**:
- All tools function correctly through MCP Inspector
- Output format matches expectations
- Error cases are handled appropriately
- Logs provide detailed debugging information

## Testing Protocol

### 1. Start MCP Server

```bash
# Start server in debug mode
python -m mcp_tools.screenshot.run_screenshot_server --debug
```

### 2. Connect with MCP Inspector

Connect to `http://localhost:9001` with MCP Inspector and verify the server registers correctly.

### 3. Test Screenshot Tool

Test the screenshot tool with the following parameter combinations:

| Test Case | Quality | Region | Expected Result |
|-----------|---------|--------|-----------------|
| 1 | 30 | "full" | Full screenshot with default quality |
| 2 | 80 | "full" | Full screenshot with high quality |
| 3 | 30 | "right_half" | Right half of screen with default quality |
| 4 | 30 | "100,100,400,300" | Custom region with default quality |
| 5 | 30 | "invalid" | Error message for invalid region |

### 4. Test Describe Screenshot Tool

Test the describe_screenshot tool with the following files:

| Test Case | Input File | Expected Result |
|-----------|------------|-----------------|
| 1 | Generated screenshot | Valid description with confidence score |
| 2 | Non-existent file | Error message for file not found |
| 3 | Non-image file | Error message for invalid image |

### 5. Test Screenshot and Describe Combined Tool

Test the combined tool with default parameters and verify that it:
1. Successfully captures a screenshot
2. Successfully describes the screenshot
3. Returns both results in a structured format

### 6. Check for Error Handling

Verify that the following error cases are handled properly:
- Invalid parameters
- File system permission issues
- Missing dependencies
- Network errors (if applicable)

### 7. Documentation Review

During testing, verify that:
- All tool documentation is clear and accurate
- Examples match actual behavior
- Parameter descriptions are helpful and correct
- Error messages are informative and actionable

## Test Log Template

```
# MCP Screenshot Tool Integration Test Log

Date: YYYY-MM-DD
Tester: [Name]
Version: [Version]

## Environment
- OS: [Operating System]
- Python Version: [Python Version]
- MCP Inspector Version: [Version]

## Test Results

### Server Startup
- [ ] Server starts successfully
- [ ] Tools register correctly
- [ ] Debug logs are informative

### Screenshot Tool Tests
- [ ] Test Case 1: [Pass/Fail] - Notes: [Any observations]
- [ ] Test Case 2: [Pass/Fail] - Notes: [Any observations]
- [ ] Test Case 3: [Pass/Fail] - Notes: [Any observations]
- [ ] Test Case 4: [Pass/Fail] - Notes: [Any observations]
- [ ] Test Case 5: [Pass/Fail] - Notes: [Any observations]

### Describe Screenshot Tool Tests
- [ ] Test Case 1: [Pass/Fail] - Notes: [Any observations]
- [ ] Test Case 2: [Pass/Fail] - Notes: [Any observations]
- [ ] Test Case 3: [Pass/Fail] - Notes: [Any observations]

### Combined Tool Tests
- [ ] Screenshot and describe: [Pass/Fail] - Notes: [Any observations]

### Error Handling Tests
- [ ] Invalid parameters: [Pass/Fail] - Notes: [Any observations]
- [ ] File system issues: [Pass/Fail] - Notes: [Any observations]
- [ ] Missing dependencies: [Pass/Fail] - Notes: [Any observations]
- [ ] Network errors: [Pass/Fail] - Notes: [Any observations]

## Issues Found
1. [Issue description, severity, and steps to reproduce]
2. [Issue description, severity, and steps to reproduce]

## Conclusion
[Overall assessment of the tool's functionality and quality]
```
