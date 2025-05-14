#!/usr/bin/env python3
"""
CLI Verification Script for ArangoDB Integration

This script tests the functionality of the CLI commands defined in cli.py
to ensure they work as documented. It runs a series of commands and
verifies the output against expected results.

To run:
python -m complexity.arangodb.cli_verification [OPTIONS]

Options:
--force     Force initialization of test data even if it exists
--verbose   Display additional debugging information
--json      Output results in JSON format (default is text)
"""

import os
import sys
import json
import subprocess
import time
import argparse
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from colorama import init, Fore, Style

# Command prefixes to use
CLI_CMD = "python -m complexity.cli"
ARANGO_CLI_CMD = "python -m complexity.arangodb.cli"


def print_section(title: str, color=Fore.CYAN):
    """Print a section header with color"""
    print(f"\n{color}{'=' * 70}")
    print(f"{title}")
    print(f"{'=' * 70}{Style.RESET_ALL}\n")


def run_command(cmd: str, check_output: bool = True, capture_error: bool = True) -> Tuple[bool, str, str]:
    """
    Run a shell command and return success flag and output
    
    Args:
        cmd: Command to run
        check_output: Whether to check for successful output
        capture_error: Whether to capture stderr as well
        
    Returns:
        Tuple of (success, stdout, stderr)
    """
    try:
        if capture_error:
            result = subprocess.run(cmd, shell=True, check=False, text=True, capture_output=True)
            stdout = result.stdout
            stderr = result.stderr
        else:
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            stderr = ""
        
        if check_output and (result.returncode != 0 if capture_error else process.returncode != 0):
            return False, stdout, stderr
        
        return True, stdout, stderr
    
    except Exception as e:
        return False, "", str(e)


def check_environment():
    """Check if environment variables are set correctly"""
    print_section("CHECKING ENVIRONMENT", Fore.GREEN)
    
    # Required environment variables
    required_vars = [
        "ARANGO_HOST", 
        "ARANGO_USER", 
        "ARANGO_PASSWORD", 
        "ARANGO_DB_NAME"
    ]
    
    # Check each variable
    all_present = True
    for var in required_vars:
        value = os.environ.get(var, None)
        if value:
            # Mask password
            if var == "ARANGO_PASSWORD":
                display_val = "********"
            else:
                display_val = value
            print(f"{Fore.GREEN}✓ {var}: {display_val}{Style.RESET_ALL}")
        else:
            all_present = False
            print(f"{Fore.RED}✗ {var}: NOT SET{Style.RESET_ALL}")
    
    # Provide setup instructions if missing
    if not all_present:
        print(f"\n{Fore.YELLOW}⚠️ Missing environment variables - please set the following:{Style.RESET_ALL}")
        print("""
        export ARANGO_HOST="http://localhost:8529"
        export ARANGO_USER="root"
        export ARANGO_PASSWORD="openSesame"
        export ARANGO_DB_NAME="memory_bank"
        """)
        return False
    
    print(f"\n{Fore.GREEN}✓ Environment is properly configured{Style.RESET_ALL}")
    return True


def initialize_test_data(force: bool = False):
    """Initialize test data in the database"""
    print_section("INITIALIZING TEST DATA", Fore.GREEN)
    
    # Check if force flag is needed
    force_flag = "--force" if force else ""
    
    # Run the initialization command
    cmd = f"{CLI_CMD} init {force_flag}"
    print(f"Running: {cmd}")
    
    success, stdout, stderr = run_command(cmd)
    
    if success:
        if "initialized" in stdout.lower() or "successfully" in stdout.lower():
            print(f"{Fore.GREEN}✓ Test data initialized successfully{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.YELLOW}⚠️ Command succeeded but output unclear:{Style.RESET_ALL}")
            print(stdout)
            return True
    else:
        print(f"{Fore.RED}✗ Failed to initialize test data:{Style.RESET_ALL}")
        print(f"Error: {stderr if stderr else stdout}")
        return False


def test_bm25_search():
    """Test BM25 search functionality"""
    print_section("TESTING BM25 SEARCH", Fore.GREEN)
    
    # Run BM25 search with JSON output for easier parsing
    cmd = f"{CLI_CMD} search bm25 \"python error\" --top-n 3 --json-output"
    print(f"Running: {cmd}")
    
    success, stdout, stderr = run_command(cmd)
    
    if not success:
        print(f"{Fore.RED}✗ Command failed:{Style.RESET_ALL}")
        print(f"Error: {stderr if stderr else stdout}")
        return False
    
    # Try to parse JSON output
    try:
        results = json.loads(stdout)
        
        # Check results structure
        if not isinstance(results, dict):
            print(f"{Fore.RED}✗ Invalid result format (not a dictionary){Style.RESET_ALL}")
            return False
        
        if "results" not in results:
            print(f"{Fore.RED}✗ Missing 'results' field in response{Style.RESET_ALL}")
            return False
        
        # Check if we got any results
        result_count = len(results.get("results", []))
        if result_count == 0:
            print(f"{Fore.YELLOW}⚠️ Search returned zero results - possible issue{Style.RESET_ALL}")
            return False
        
        # Show search stats
        print(f"Found {result_count} results out of {results.get('total', 0)} total matches")
        print(f"Query: '{results.get('query', '')}'")
        print(f"Search time: {results.get('time', 0)*1000:.2f}ms")
        
        # Show first result
        if result_count > 0:
            first_result = results["results"][0]
            print(f"\nTop result:")
            print(f"  Document key: {first_result['doc'].get('_key', 'N/A')}")
            print(f"  Score: {first_result.get('score', 0):.5f}")
            
            # Try to find relevant content field
            content_fields = ["question", "problem", "content", "text", "title"]
            for field in content_fields:
                if field in first_result['doc'] and first_result['doc'][field]:
                    content = first_result['doc'][field]
                    # Truncate long content
                    if len(content) > 100:
                        content = content[:100] + "..."
                    print(f"  {field.capitalize()}: {content}")
                    break
        
        print(f"{Fore.GREEN}✓ BM25 search working correctly{Style.RESET_ALL}")
        return True
    
    except json.JSONDecodeError:
        print(f"{Fore.RED}✗ Failed to parse JSON output{Style.RESET_ALL}")
        print(f"Output: {stdout}")
        return False
    except Exception as e:
        print(f"{Fore.RED}✗ Error testing BM25 search: {str(e)}{Style.RESET_ALL}")
        return False


def test_semantic_search():
    """Test semantic search functionality"""
    print_section("TESTING SEMANTIC SEARCH", Fore.GREEN)
    
    # Run semantic search with JSON output
    cmd = f"{CLI_CMD} search semantic \"python error\" --top-n 3 --json-output"
    print(f"Running: {cmd}")
    
    success, stdout, stderr = run_command(cmd)
    
    if not success:
        print(f"{Fore.RED}✗ Command failed:{Style.RESET_ALL}")
        print(f"Error: {stderr if stderr else stdout}")
        return False
    
    # Try to parse JSON output
    try:
        results = json.loads(stdout)
        
        # Check results structure
        if not isinstance(results, dict):
            print(f"{Fore.RED}✗ Invalid result format (not a dictionary){Style.RESET_ALL}")
            return False
        
        if "results" not in results:
            print(f"{Fore.RED}✗ Missing 'results' field in response{Style.RESET_ALL}")
            return False
        
        # Check if we got any results
        result_count = len(results.get("results", []))
        if result_count == 0:
            print(f"{Fore.YELLOW}⚠️ Search returned zero results - possible issue{Style.RESET_ALL}")
            return False
        
        # Show search stats
        print(f"Found {result_count} results")
        print(f"Query: '{results.get('query', '')}'")
        print(f"Search time: {results.get('time', 0)*1000:.2f}ms")
        
        # Check if we're using ArangoDB vector search or fallback
        search_engine = results.get("search_engine", "unknown")
        if "arango" in search_engine.lower():
            print(f"{Fore.GREEN}✓ Using ArangoDB vector search: {search_engine}{Style.RESET_ALL}")
        else:
            print(f"{Fore.YELLOW}⚠️ Using fallback search method: {search_engine}{Style.RESET_ALL}")
        
        # Show first result
        if result_count > 0:
            first_result = results["results"][0]
            print(f"\nTop result:")
            print(f"  Document key: {first_result['doc'].get('_key', 'N/A')}")
            print(f"  Score: {first_result.get('similarity_score', 0):.5f}")
            
            # Try to find relevant content field
            content_fields = ["question", "problem", "content", "text", "title"]
            for field in content_fields:
                if field in first_result['doc'] and first_result['doc'][field]:
                    content = first_result['doc'][field]
                    # Truncate long content
                    if len(content) > 100:
                        content = content[:100] + "..."
                    print(f"  {field.capitalize()}: {content}")
                    break
        
        print(f"{Fore.GREEN}✓ Semantic search working (using {search_engine}){Style.RESET_ALL}")
        return True
    
    except json.JSONDecodeError:
        print(f"{Fore.RED}✗ Failed to parse JSON output{Style.RESET_ALL}")
        print(f"Output: {stdout}")
        return False
    except Exception as e:
        print(f"{Fore.RED}✗ Error testing semantic search: {str(e)}{Style.RESET_ALL}")
        return False


def test_hybrid_search():
    """Test hybrid search functionality"""
    print_section("TESTING HYBRID SEARCH", Fore.GREEN)
    
    # Run hybrid search with JSON output
    cmd = f"{CLI_CMD} search hybrid \"python error\" --top-n 3 --json-output"
    print(f"Running: {cmd}")
    
    success, stdout, stderr = run_command(cmd)
    
    if not success:
        print(f"{Fore.RED}✗ Command failed:{Style.RESET_ALL}")
        print(f"Error: {stderr if stderr else stdout}")
        return False
    
    # Try to parse JSON output
    try:
        results = json.loads(stdout)
        
        # Check results structure
        if not isinstance(results, dict):
            print(f"{Fore.RED}✗ Invalid result format (not a dictionary){Style.RESET_ALL}")
            return False
        
        if "results" not in results:
            print(f"{Fore.RED}✗ Missing 'results' field in response{Style.RESET_ALL}")
            return False
        
        # Check if we got any results
        result_count = len(results.get("results", []))
        if result_count == 0:
            print(f"{Fore.YELLOW}⚠️ Search returned zero results - possible issue{Style.RESET_ALL}")
            return False
        
        # Show search stats
        print(f"Found {result_count} results")
        print(f"Query: '{results.get('query', '')}'")
        print(f"Search time: {results.get('time', 0)*1000:.2f}ms")
        
        # Show first result
        if result_count > 0:
            first_result = results["results"][0]
            print(f"\nTop result:")
            print(f"  Document key: {first_result['doc'].get('_key', 'N/A')}")
            print(f"  RRF score: {first_result.get('score', 0):.5f}")
            
            # Check component scores
            if "bm25_score" in first_result:
                print(f"  BM25 score: {first_result.get('bm25_score', 0):.5f}")
            if "semantic_score" in first_result:
                print(f"  Semantic score: {first_result.get('semantic_score', 0):.5f}")
            
            # Try to find relevant content field
            content_fields = ["question", "problem", "content", "text", "title"]
            for field in content_fields:
                if field in first_result['doc'] and first_result['doc'][field]:
                    content = first_result['doc'][field]
                    # Truncate long content
                    if len(content) > 100:
                        content = content[:100] + "..."
                    print(f"  {field.capitalize()}: {content}")
                    break
        
        print(f"{Fore.GREEN}✓ Hybrid search working correctly{Style.RESET_ALL}")
        return True
    
    except json.JSONDecodeError:
        print(f"{Fore.RED}✗ Failed to parse JSON output{Style.RESET_ALL}")
        print(f"Output: {stdout}")
        return False
    except Exception as e:
        print(f"{Fore.RED}✗ Error testing hybrid search: {str(e)}{Style.RESET_ALL}")
        return False


def test_tag_search():
    """Test tag search functionality"""
    print_section("TESTING TAG SEARCH", Fore.GREEN)
    
    # Run tag search with JSON output
    cmd = f"{CLI_CMD} search tag python error --json-output"
    print(f"Running: {cmd}")
    
    success, stdout, stderr = run_command(cmd)
    
    if not success:
        print(f"{Fore.RED}✗ Command failed:{Style.RESET_ALL}")
        print(f"Error: {stderr if stderr else stdout}")
        return False
    
    # Try to parse JSON output
    try:
        results = json.loads(stdout)
        
        # Check results structure
        if not isinstance(results, list):
            print(f"{Fore.RED}✗ Invalid result format (not a list){Style.RESET_ALL}")
            return False
        
        # Check if we got any results
        result_count = len(results)
        if result_count == 0:
            print(f"{Fore.YELLOW}⚠️ Search returned zero results - possible issue{Style.RESET_ALL}")
            return False
        
        # Show search stats
        print(f"Found {result_count} results")
        
        # Show first result
        if result_count > 0:
            first_result = results[0]
            print(f"\nTop result:")
            print(f"  Document key: {first_result.get('_key', 'N/A')}")
            
            # Show tags
            if "tags" in first_result:
                print(f"  Tags: {', '.join(first_result['tags'])}")
            
            # Try to find relevant content field
            content_fields = ["question", "problem", "content", "text", "title"]
            for field in content_fields:
                if field in first_result and first_result[field]:
                    content = first_result[field]
                    # Truncate long content
                    if len(content) > 100:
                        content = content[:100] + "..."
                    print(f"  {field.capitalize()}: {content}")
                    break
        
        print(f"{Fore.GREEN}✓ Tag search working correctly{Style.RESET_ALL}")
        return True
    
    except json.JSONDecodeError:
        print(f"{Fore.RED}✗ Failed to parse JSON output{Style.RESET_ALL}")
        print(f"Output: {stdout}")
        return False
    except Exception as e:
        print(f"{Fore.RED}✗ Error testing tag search: {str(e)}{Style.RESET_ALL}")
        return False


def test_memory_store():
    """Test memory store functionality"""
    print_section("TESTING MEMORY STORE", Fore.GREEN)
    
    # Create a unique conversation ID
    conversation_id = f"test-{int(time.time())}"
    
    # Run memory store command
    user_msg = "How do I fix a Python IndexError?"
    agent_msg = "An IndexError in Python occurs when you try to access an index that's out of range. Check your array bounds and make sure you're not accessing beyond the length of your list or array."
    
    cmd = f'{CLI_CMD} memory store "{user_msg}" "{agent_msg}" --conversation-id {conversation_id} --json-output'
    print(f"Running memory store with conversation ID: {conversation_id}")
    
    success, stdout, stderr = run_command(cmd)
    
    if not success:
        print(f"{Fore.RED}✗ Command failed:{Style.RESET_ALL}")
        print(f"Error: {stderr if stderr else stdout}")
        return False, None
    
    # Try to parse JSON output
    try:
        result = json.loads(stdout)
        
        # Check result structure
        if not isinstance(result, dict):
            print(f"{Fore.RED}✗ Invalid result format (not a dictionary){Style.RESET_ALL}")
            return False, None
        
        if "user_message_key" not in result or "agent_message_key" not in result:
            print(f"{Fore.RED}✗ Missing message keys in response{Style.RESET_ALL}")
            return False, None
        
        # Extract message keys
        user_key = result["user_message_key"]
        agent_key = result["agent_message_key"]
        
        print(f"User message stored with key: {user_key}")
        print(f"Agent message stored with key: {agent_key}")
        print(f"Conversation ID: {conversation_id}")
        
        print(f"{Fore.GREEN}✓ Memory store working correctly{Style.RESET_ALL}")
        return True, conversation_id
    
    except json.JSONDecodeError:
        print(f"{Fore.RED}✗ Failed to parse JSON output{Style.RESET_ALL}")
        print(f"Output: {stdout}")
        return False, None
    except Exception as e:
        print(f"{Fore.RED}✗ Error testing memory store: {str(e)}{Style.RESET_ALL}")
        return False, None


def test_memory_search():
    """Test memory search functionality"""
    print_section("TESTING MEMORY SEARCH", Fore.GREEN)
    
    # Run memory search with JSON output
    cmd = f"{CLI_CMD} memory search \"python error\" --top-n 3 --json-output"
    print(f"Running: {cmd}")
    
    success, stdout, stderr = run_command(cmd)
    
    if not success:
        print(f"{Fore.RED}✗ Command failed:{Style.RESET_ALL}")
        print(f"Error: {stderr if stderr else stdout}")
        return False
    
    # Try to parse JSON output
    try:
        results = json.loads(stdout)
        
        # Check results structure
        if not isinstance(results, list):
            print(f"{Fore.RED}✗ Invalid result format (not a list){Style.RESET_ALL}")
            return False
        
        # Check if we got any results
        result_count = len(results)
        if result_count == 0:
            print(f"{Fore.YELLOW}⚠️ Search returned zero results - possible issue{Style.RESET_ALL}")
            return False
        
        # Show search stats
        print(f"Found {result_count} results")
        
        # Show first result
        if result_count > 0:
            first_result = results[0]
            print(f"\nTop result:")
            print(f"  Document key: {first_result.get('_key', 'N/A')}")
            print(f"  Score: {first_result.get('score', 0):.5f}")
            
            # Show content if available
            if "content" in first_result:
                content = first_result["content"]
                # Truncate long content
                if len(content) > 100:
                    content = content[:100] + "..."
                print(f"  Content: {content}")
            
            # Show tags if available
            if "tags" in first_result and first_result["tags"]:
                print(f"  Tags: {', '.join(first_result['tags'])}")
            
            # Show conversation ID if available
            if "conversation_id" in first_result:
                print(f"  Conversation ID: {first_result['conversation_id']}")
        
        print(f"{Fore.GREEN}✓ Memory search working correctly{Style.RESET_ALL}")
        return True
    
    except json.JSONDecodeError:
        print(f"{Fore.RED}✗ Failed to parse JSON output{Style.RESET_ALL}")
        print(f"Output: {stdout}")
        return False
    except Exception as e:
        print(f"{Fore.RED}✗ Error testing memory search: {str(e)}{Style.RESET_ALL}")
        return False


def test_memory_context(conversation_id: str):
    """Test memory context functionality"""
    print_section("TESTING MEMORY CONTEXT", Fore.GREEN)
    
    if not conversation_id:
        print(f"{Fore.YELLOW}⚠️ No conversation ID available - skipping test{Style.RESET_ALL}")
        return False
    
    # Run memory context with JSON output
    cmd = f"{CLI_CMD} memory context {conversation_id} --json-output"
    print(f"Running: {cmd}")
    
    success, stdout, stderr = run_command(cmd)
    
    if not success:
        print(f"{Fore.RED}✗ Command failed:{Style.RESET_ALL}")
        print(f"Error: {stderr if stderr else stdout}")
        return False
    
    # Try to parse JSON output
    try:
        results = json.loads(stdout)
        
        # Check results structure
        if not isinstance(results, list):
            print(f"{Fore.RED}✗ Invalid result format (not a list){Style.RESET_ALL}")
            return False
        
        # Check if we got any results
        result_count = len(results)
        if result_count == 0:
            print(f"{Fore.YELLOW}⚠️ Context returned zero messages - possible issue{Style.RESET_ALL}")
            return False
        
        # Show stats
        print(f"Found {result_count} messages in conversation {conversation_id}")
        
        # Show message details
        for i, msg in enumerate(results):
            role = msg.get("role", "unknown")
            content_preview = msg.get("content", "")[:50] + "..." if len(msg.get("content", "")) > 50 else msg.get("content", "")
            print(f"  {i+1}. {role.upper()}: {content_preview}")
        
        print(f"{Fore.GREEN}✓ Memory context working correctly{Style.RESET_ALL}")
        return True
    
    except json.JSONDecodeError:
        print(f"{Fore.RED}✗ Failed to parse JSON output{Style.RESET_ALL}")
        print(f"Output: {stdout}")
        return False
    except Exception as e:
        print(f"{Fore.RED}✗ Error testing memory context: {str(e)}{Style.RESET_ALL}")
        return False


def run_all_tests(args):
    """Run all verification tests"""
    # Check environment first
    if not check_environment():
        print(f"{Fore.RED}✗ Environment check failed - please fix environment variables{Style.RESET_ALL}")
        return False
    
    # Initialize test data
    if not initialize_test_data(args.force):
        print(f"{Fore.RED}✗ Data initialization failed - cannot continue{Style.RESET_ALL}")
        return False
    
    # Run search tests
    bm25_ok = test_bm25_search()
    semantic_ok = test_semantic_search()
    hybrid_ok = test_hybrid_search()
    tag_ok = test_tag_search()
    
    # Run memory tests
    memory_store_ok, conversation_id = test_memory_store()
    memory_search_ok = test_memory_search()
    memory_context_ok = test_memory_context(conversation_id) if conversation_id else False
    
    # Print summary
    print_section("VERIFICATION SUMMARY", Fore.GREEN)
    print(f"Environment check: {Fore.GREEN}✓{Style.RESET_ALL}")
    print(f"Data initialization: {Fore.GREEN}✓{Style.RESET_ALL}")
    print(f"BM25 search: {Fore.GREEN}✓{Style.RESET_ALL if bm25_ok else Fore.RED}✗{Style.RESET_ALL}")
    print(f"Semantic search: {Fore.GREEN}✓{Style.RESET_ALL if semantic_ok else Fore.RED}✗{Style.RESET_ALL}")
    print(f"Hybrid search: {Fore.GREEN}✓{Style.RESET_ALL if hybrid_ok else Fore.RED}✗{Style.RESET_ALL}")
    print(f"Tag search: {Fore.GREEN}✓{Style.RESET_ALL if tag_ok else Fore.RED}✗{Style.RESET_ALL}")
    print(f"Memory store: {Fore.GREEN}✓{Style.RESET_ALL if memory_store_ok else Fore.RED}✗{Style.RESET_ALL}")
    print(f"Memory search: {Fore.GREEN}✓{Style.RESET_ALL if memory_search_ok else Fore.RED}✗{Style.RESET_ALL}")
    print(f"Memory context: {Fore.GREEN}✓{Style.RESET_ALL if memory_context_ok else Fore.RED}✗{Style.RESET_ALL}")
    
    # Calculate overall result
    all_passed = all([bm25_ok, semantic_ok, hybrid_ok, tag_ok, memory_store_ok, memory_search_ok, memory_context_ok])
    
    if all_passed:
        print(f"\n{Fore.GREEN}✓ ALL TESTS PASSED - CLI is working correctly{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}✗ SOME TESTS FAILED - CLI needs fixes{Style.RESET_ALL}")
    
    return all_passed


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Verify ArangoDB CLI functionality")
    parser.add_argument("--force", action="store_true", help="Force initialization of test data")
    parser.add_argument("--verbose", action="store_true", help="Display additional debugging information")
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    return parser.parse_args()


if __name__ == "__main__":
    # Initialize colorama for cross-platform colored output
    init(autoreset=True)
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Run the tests
    try:
        if not args.json:
            print(f"{Fore.CYAN}ArangoDB CLI Verification{Style.RESET_ALL}")
            print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        success = run_all_tests(args)
        
        if args.json:
            # Output JSON result
            result = {
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "tests": {
                    "environment": True,
                    "init": True,
                    "bm25_search": success,
                    "semantic_search": success,
                    "hybrid_search": success,
                    "tag_search": success,
                    "memory_store": success,
                    "memory_search": success,
                    "memory_context": success
                }
            }
            print(json.dumps(result, indent=2))
        
        sys.exit(0 if success else 1)
    
    except Exception as e:
        if args.json:
            error_result = {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            print(json.dumps(error_result, indent=2))
        else:
            print(f"{Fore.RED}ERROR: {str(e)}{Style.RESET_ALL}")
        
        sys.exit(1)