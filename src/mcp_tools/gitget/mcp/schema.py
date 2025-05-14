"""JSON schema definitions for MCP integration of the gitget module.

This module provides JSON schema definitions for the MCP (Model Control Protocol)
integration of the gitget module, defining the input and output formats for
the MCP tool.

Links to third-party package documentation:
- JSON Schema: https://json-schema.org/understanding-json-schema/
- FastMCP Schema: https://fastmcp.readthedocs.io/en/latest/schema/

Sample input:
    {
        "url": "https://github.com/user/repo",
        "branch": "main",
        "output_dir": "./repos",
        "sparse_checkout": ["README.md", "src/"],
        "depth": 1,
        "include_git_dir": false
    }

Expected output:
    {
        "repository": {
            "url": "https://github.com/user/repo",
            "name": "repo",
            "owner": "user",
            "local_path": "/path/to/repo",
            "branch": "main",
            "file_count": 10,
            "total_size": 1024
        },
        "files": [
            {
                "name": "file.py",
                "path": "src/file.py",
                "size": 1024,
                "content_type": "text",
                "is_binary": false
            }
        ],
        "success": true
    }
"""

# Clone schema
CLONE_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "Repository URL (e.g., https://github.com/user/repo)"
        },
        "branch": {
            "type": "string",
            "description": "Branch to checkout",
            "default": "main"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory to store cloned repositories",
            "default": "./repos"
        },
        "sparse_checkout": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List of files/dirs to include in sparse checkout"
        },
        "depth": {
            "type": "integer",
            "description": "Depth of git history to clone (1 for shallow clone)",
            "default": 1,
            "minimum": 1
        },
        "include_git_dir": {
            "type": "boolean",
            "description": "Whether to include .git directory",
            "default": False
        },
        "force": {
            "type": "boolean",
            "description": "Force clone even if directory already exists",
            "default": False
        }
    },
    "required": ["url"],
    "additionalProperties": False
}

# Process schema
PROCESS_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "Repository URL (e.g., https://github.com/user/repo)"
        },
        "branch": {
            "type": "string",
            "description": "Branch to checkout",
            "default": "main"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory to store cloned repositories",
            "default": "./repos"
        },
        "sparse_checkout": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List of files/dirs to include in sparse checkout"
        },
        "depth": {
            "type": "integer",
            "description": "Depth of git history to clone (1 for shallow clone)",
            "default": 1,
            "minimum": 1
        },
        "include_content": {
            "type": "boolean",
            "description": "Whether to include file content in output",
            "default": False
        },
        "max_file_size": {
            "type": "integer",
            "description": "Maximum file size to process in bytes",
            "default": 1048576  # 1MB
        },
        "binary": {
            "type": "boolean",
            "description": "Whether to include binary files",
            "default": False
        },
        "output_file": {
            "type": "string",
            "description": "Path to save output (if not specified, returns in response)"
        }
    },
    "required": ["url"],
    "additionalProperties": False
}

# Extract schema
EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "Repository URL (e.g., https://github.com/user/repo)"
        },
        "branch": {
            "type": "string",
            "description": "Branch to checkout",
            "default": "main"
        },
        "output_dir": {
            "type": "string",
            "description": "Directory to store cloned repositories",
            "default": "./repos"
        },
        "sparse_checkout": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "List of files/dirs to include in sparse checkout"
        },
        "max_tokens": {
            "type": "integer",
            "description": "Maximum number of tokens per chunk",
            "default": 1024,
            "minimum": 1
        },
        "overlap": {
            "type": "integer",
            "description": "Number of tokens to overlap between chunks",
            "default": 200,
            "minimum": 0
        },
        "output_file": {
            "type": "string",
            "description": "Path to save output (if not specified, returns in response)"
        },
        "exclude_extensions": {
            "type": "array",
            "items": {
                "type": "string"
            },
            "description": "File extensions to exclude from processing",
            "default": [
                ".jpg", ".jpeg", ".png", ".gif", ".ico", ".svg", ".webp", ".bmp",
                ".pdf", ".zip", ".tar", ".gz", ".rar", ".7z", ".exe", ".bin"
            ]
        }
    },
    "required": ["url"],
    "additionalProperties": False
}

# Info schema
INFO_SCHEMA = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path to a local repository"
        }
    },
    "required": ["path"],
    "additionalProperties": False
}

# Combined schema for all commands
GITGET_SCHEMA = {
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["clone"],
                    "description": "Clone a Git repository"
                },
                "params": CLONE_SCHEMA
            },
            "required": ["command", "params"],
            "additionalProperties": False
        },
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["process"],
                    "description": "Process a Git repository and extract file information"
                },
                "params": PROCESS_SCHEMA
            },
            "required": ["command", "params"],
            "additionalProperties": False
        },
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["extract"],
                    "description": "Extract text chunks from a Git repository for processing with LLMs"
                },
                "params": EXTRACT_SCHEMA
            },
            "required": ["command", "params"],
            "additionalProperties": False
        },
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "enum": ["info"],
                    "description": "Show information about a local repository"
                },
                "params": INFO_SCHEMA
            },
            "required": ["command", "params"],
            "additionalProperties": False
        }
    ]
}

if __name__ == "__main__":
    # Validation function to test schemas with real data
    import sys
    import json
    from jsonschema import validate, ValidationError
    
    # List to track all validation failures
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Clone schema validation
    total_tests += 1
    try:
        # Valid clone params
        valid_clone = {
            "url": "https://github.com/user/repo",
            "branch": "develop",
            "output_dir": "./repos",
            "sparse_checkout": ["README.md", "src/"],
            "depth": 2,
            "include_git_dir": False,
            "force": True
        }
        
        # Validate against schema
        validate(instance=valid_clone, schema=CLONE_SCHEMA)
        
        # Test required fields
        try:
            invalid_clone = {
                "branch": "main",
                "output_dir": "./repos"
                # Missing url
            }
            validate(instance=invalid_clone, schema=CLONE_SCHEMA)
            all_validation_failures.append("Clone schema: Failed to catch missing required field")
        except ValidationError:
            # This is expected
            pass
            
        # Test additional properties
        try:
            invalid_clone = {
                "url": "https://github.com/user/repo",
                "invalid_param": "value"  # Not in schema
            }
            validate(instance=invalid_clone, schema=CLONE_SCHEMA)
            all_validation_failures.append("Clone schema: Failed to catch additional properties")
        except ValidationError:
            # This is expected
            pass
            
    except Exception as e:
        all_validation_failures.append(f"Clone schema: Unexpected exception: {str(e)}")
    
    # Test 2: Process schema validation
    total_tests += 1
    try:
        # Valid process params
        valid_process = {
            "url": "https://github.com/user/repo",
            "branch": "main",
            "output_dir": "./repos",
            "sparse_checkout": ["README.md", "src/"],
            "depth": 1,
            "include_content": True,
            "max_file_size": 2048,
            "binary": False,
            "output_file": "./output.json"
        }
        
        # Validate against schema
        validate(instance=valid_process, schema=PROCESS_SCHEMA)
        
        # Test required fields
        try:
            invalid_process = {
                "branch": "main",
                "output_dir": "./repos"
                # Missing url
            }
            validate(instance=invalid_process, schema=PROCESS_SCHEMA)
            all_validation_failures.append("Process schema: Failed to catch missing required field")
        except ValidationError:
            # This is expected
            pass
            
    except Exception as e:
        all_validation_failures.append(f"Process schema: Unexpected exception: {str(e)}")
    
    # Test 3: Extract schema validation
    total_tests += 1
    try:
        # Valid extract params
        valid_extract = {
            "url": "https://github.com/user/repo",
            "branch": "main",
            "output_dir": "./repos",
            "sparse_checkout": ["README.md", "src/"],
            "max_tokens": 2048,
            "overlap": 100,
            "output_file": "./chunks.json",
            "exclude_extensions": [".jpg", ".png"]
        }
        
        # Validate against schema
        validate(instance=valid_extract, schema=EXTRACT_SCHEMA)
        
        # Test required fields
        try:
            invalid_extract = {
                "branch": "main",
                "max_tokens": 2048
                # Missing url
            }
            validate(instance=invalid_extract, schema=EXTRACT_SCHEMA)
            all_validation_failures.append("Extract schema: Failed to catch missing required field")
        except ValidationError:
            # This is expected
            pass
            
    except Exception as e:
        all_validation_failures.append(f"Extract schema: Unexpected exception: {str(e)}")
    
    # Test 4: Info schema validation
    total_tests += 1
    try:
        # Valid info params
        valid_info = {
            "path": "./repos/user_repo"
        }
        
        # Validate against schema
        validate(instance=valid_info, schema=INFO_SCHEMA)
        
        # Test required fields
        try:
            invalid_info = {
                # Missing path
            }
            validate(instance=invalid_info, schema=INFO_SCHEMA)
            all_validation_failures.append("Info schema: Failed to catch missing required field")
        except ValidationError:
            # This is expected
            pass
            
    except Exception as e:
        all_validation_failures.append(f"Info schema: Unexpected exception: {str(e)}")
    
    # Test 5: Combined schema validation
    total_tests += 1
    try:
        # Test all command types
        commands = [
            {"command": "clone", "params": {"url": "https://github.com/user/repo"}},
            {"command": "process", "params": {"url": "https://github.com/user/repo"}},
            {"command": "extract", "params": {"url": "https://github.com/user/repo"}},
            {"command": "info", "params": {"path": "./repos/user_repo"}}
        ]
        
        for cmd in commands:
            validate(instance=cmd, schema=GITGET_SCHEMA)
        
        # Test invalid command
        try:
            invalid_command = {
                "command": "invalid",
                "params": {"url": "https://github.com/user/repo"}
            }
            validate(instance=invalid_command, schema=GITGET_SCHEMA)
            all_validation_failures.append("Combined schema: Failed to catch invalid command")
        except ValidationError:
            # This is expected
            pass
            
        # Test missing params
        try:
            invalid_params = {
                "command": "clone"
                # Missing params
            }
            validate(instance=invalid_params, schema=GITGET_SCHEMA)
            all_validation_failures.append("Combined schema: Failed to catch missing params")
        except ValidationError:
            # This is expected
            pass
            
    except Exception as e:
        all_validation_failures.append(f"Combined schema: Unexpected exception: {str(e)}")
    
    # Final validation result
    if all_validation_failures:
        print(f"❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)