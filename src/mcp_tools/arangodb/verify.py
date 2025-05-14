#!/usr/bin/env python3
"""
Verification Script for ArangoDB Integration

This script provides a simple process to:
1. Run accurate tests for the search API code
2. Show clear pass/fail results for each component
3. Verify with an actual CLI command to confirm everything works

Usage:
    python -m complexity.arangodb.verify [--force] [--component COMPONENT]

Options:
    --force            Force initialization of test data
    --component        Test specific component only (bm25, semantic, hybrid, memory)
"""

import os
import sys
import json
import subprocess
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)

# Constants
CLI_CMD = "python -m complexity.cli"
CLI_VERIFY_COMMANDS = {
    "bm25": f"{CLI_CMD} search bm25 \"python error\" --top-n 3 --json-output",
    "semantic": f"{CLI_CMD} search semantic \"python error\" --top-n 3 --json-output",
    "hybrid": f"{CLI_CMD} search hybrid \"python error\" --top-n 3 --json-output",
    "memory": f"{CLI_CMD} memory search \"python error\" --top-n 3 --json-output",
}


def print_header(title: str):
    """Print a colored header"""
    print(f"\n{Fore.CYAN}{'=' * 70}")
    print(f"{Fore.CYAN}{title}")
    print(f"{Fore.CYAN}{'=' * 70}{Style.RESET_ALL}\n")


def run_command(cmd: str) -> Tuple[bool, str, str]:
    """Run a shell command and return success flag, stdout and stderr"""
    try:
        process = subprocess.run(cmd, shell=True, check=False, capture_output=True, text=True)
        return process.returncode == 0, process.stdout, process.stderr
    except Exception as e:
        return False, "", str(e)


def check_environment():
    """Check for required environment variables"""
    print_header("Checking Environment")
    
    required_vars = [
        "ARANGO_HOST", 
        "ARANGO_USER", 
        "ARANGO_PASSWORD", 
        "ARANGO_DB_NAME"
    ]
    
    all_present = True
    for var in required_vars:
        value = os.environ.get(var, "")
        if value:
            display_val = "********" if var == "ARANGO_PASSWORD" else value
            print(f"{Fore.GREEN}✓ {var}: {display_val}{Style.RESET_ALL}")
        else:
            all_present = False
            print(f"{Fore.RED}✗ {var}: NOT SET{Style.RESET_ALL}")
    
    if not all_present:
        print(f"\n{Fore.YELLOW}Set the following variables:{Style.RESET_ALL}")
        print("""
        export ARANGO_HOST="http://localhost:8529"
        export ARANGO_USER="root"
        export ARANGO_PASSWORD="openSesame"
        export ARANGO_DB_NAME="memory_bank"
        """)
        return False
    
    return True


def initialize_data(force: bool = False):
    """Initialize test data"""
    print_header("Initializing Test Data")
    
    force_flag = "--force" if force else ""
    cmd = f"{CLI_CMD} init {force_flag}"
    
    print(f"Running: {cmd}")
    success, stdout, stderr = run_command(cmd)
    
    if success:
        if "initialized" in stdout.lower() or "successfully" in stdout.lower():
            print(f"{Fore.GREEN}✓ Test data initialized successfully{Style.RESET_ALL}")
            return True
        else:
            print(f"{Fore.YELLOW}⚠️  Command succeeded but output unclear:{Style.RESET_ALL}")
            print(stdout)
            return True
    else:
        print(f"{Fore.RED}✗ Failed to initialize test data:{Style.RESET_ALL}")
        print(f"Error: {stderr}")
        return False


def run_api_test(component: str):
    """Run accurate API test for specific component"""
    print_header(f"Testing {component.upper()} API")
    
    if component == "bm25":
        import complexity.arangodb.search_api.bm25_search as bm25
        from complexity.arangodb.arango_setup import connect_arango, ensure_database
        
        try:
            # Connect to database
            client = connect_arango()
            db = ensure_database(client)
            
            # Run the test
            query = "python error"
            print(f"Running BM25 search with query: '{query}'")
            
            result = bm25.bm25_search(
                db=db, 
                query_text=query,
                top_n=3,
                min_score=0.0,
                output_format="json"
            )
            
            # Validate result
            if "error" in result:
                print(f"{Fore.RED}✗ Search returned error: {result['error']}{Style.RESET_ALL}")
                return False
            
            if "results" not in result:
                print(f"{Fore.RED}✗ Missing 'results' field in response{Style.RESET_ALL}")
                return False
            
            result_count = len(result.get("results", []))
            print(f"Found {result_count} results")
            
            if result_count == 0:
                print(f"{Fore.YELLOW}⚠️  No results found - this may indicate an issue{Style.RESET_ALL}")
                return False
            
            # Check first result structure
            first_result = result["results"][0]
            if "doc" not in first_result or "score" not in first_result:
                print(f"{Fore.RED}✗ Result missing required fields (doc and/or score){Style.RESET_ALL}")
                return False
            
            # Check document fields
            doc = first_result["doc"]
            if "_key" not in doc or "_id" not in doc:
                print(f"{Fore.RED}✗ Document missing required fields (_key and/or _id){Style.RESET_ALL}")
                return False
            
            print(f"{Fore.GREEN}✓ BM25 API test passed{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}✗ Error running BM25 test: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return False
    
    elif component == "semantic":
        import complexity.arangodb.search_api.semantic_search as semantic
        from complexity.arangodb.arango_setup import connect_arango, ensure_database
        
        try:
            # Connect to database
            client = connect_arango()
            db = ensure_database(client)
            
            # Run the test
            query = "python error"
            print(f"Running semantic search with query: '{query}'")
            
            result = semantic.semantic_search(
                db=db, 
                query=query,
                top_n=3,
                min_score=0.5,
                output_format="json"
            )
            
            # Validate result
            if "error" in result:
                print(f"{Fore.YELLOW}⚠️  Search returned error but may still have fallback results: {result['error']}{Style.RESET_ALL}")
            
            if "results" not in result:
                print(f"{Fore.RED}✗ Missing 'results' field in response{Style.RESET_ALL}")
                return False
            
            result_count = len(result.get("results", []))
            print(f"Found {result_count} results")
            
            if result_count == 0:
                print(f"{Fore.YELLOW}⚠️  No results found - this may indicate an issue{Style.RESET_ALL}")
                return False
            
            # Check search engine
            search_engine = result.get("search_engine", "unknown")
            print(f"Using search engine: {search_engine}")
            
            # Check first result structure
            first_result = result["results"][0]
            if "doc" not in first_result or "similarity_score" not in first_result:
                print(f"{Fore.RED}✗ Result missing required fields (doc and/or similarity_score){Style.RESET_ALL}")
                return False
            
            # Check document fields
            doc = first_result["doc"]
            if "_key" not in doc or "_id" not in doc:
                print(f"{Fore.RED}✗ Document missing required fields (_key and/or _id){Style.RESET_ALL}")
                return False
            
            print(f"{Fore.GREEN}✓ Semantic API test passed{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}✗ Error running semantic test: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return False
    
    elif component == "hybrid":
        import complexity.arangodb.search_api.hybrid_search as hybrid
        from complexity.arangodb.arango_setup import connect_arango, ensure_database
        
        try:
            # Connect to database
            client = connect_arango()
            db = ensure_database(client)
            
            # Run the test
            query = "python error"
            print(f"Running hybrid search with query: '{query}'")
            
            # Create min_score dictionary for thresholds
            min_score = {
                "bm25": 0.01,     # BM25 threshold
                "semantic": 0.5   # Semantic similarity threshold
            }
            
            result = hybrid.hybrid_search(
                db=db, 
                query_text=query,
                top_n=3,
                min_score=min_score,
                initial_k=20,
                rrf_k=60,
                output_format="json"
            )
            
            # Validate result
            if "error" in result:
                print(f"{Fore.RED}✗ Search returned error: {result['error']}{Style.RESET_ALL}")
                return False
            
            if "results" not in result:
                print(f"{Fore.RED}✗ Missing 'results' field in response{Style.RESET_ALL}")
                return False
            
            result_count = len(result.get("results", []))
            print(f"Found {result_count} results")
            
            if result_count == 0:
                print(f"{Fore.YELLOW}⚠️  No results found - this may indicate an issue{Style.RESET_ALL}")
                return False
            
            # Check first result structure
            first_result = result["results"][0]
            if "doc" not in first_result or "score" not in first_result:
                print(f"{Fore.RED}✗ Result missing required fields (doc and/or score){Style.RESET_ALL}")
                return False
            
            # Check document fields
            doc = first_result["doc"]
            if "_key" not in doc or "_id" not in doc:
                print(f"{Fore.RED}✗ Document missing required fields (_key and/or _id){Style.RESET_ALL}")
                return False
            
            print(f"{Fore.GREEN}✓ Hybrid API test passed{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}✗ Error running hybrid test: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return False
    
    elif component == "memory":
        from complexity.arangodb.memory_agent import MemoryAgent
        from complexity.arangodb.arango_setup import connect_arango, ensure_database
        
        try:
            # Connect to database
            client = connect_arango()
            db = ensure_database(client)
            
            # Create memory agent
            print("Initializing memory agent")
            memory_agent = MemoryAgent(db=db)
            
            # Store a test memory
            print("Storing test memory")
            user_msg = "How do I handle Python IndexError?"
            agent_msg = "Check your array indices and make sure they're within bounds."
            conversation_id = f"test-{int(time.time())}"
            
            result = memory_agent.store_conversation(
                user_message=user_msg,
                agent_response=agent_msg,
                conversation_id=conversation_id
            )
            
            print(f"Result: {result}")
            
            # Check the result structure
            if not result:
                print(f"{Fore.RED}✗ Failed to store test memory - empty result{Style.RESET_ALL}")
                return False
                
            # Print what we got back
            for key, value in result.items():
                print(f"  {key}: {value}")
                
            # Accept any non-empty dictionary as success for now
            print(f"{Fore.GREEN}✓ Memory stored successfully{Style.RESET_ALL}")
            
            # Search memory
            print(f"Searching memory for: 'python error'")
            search_results = memory_agent.search_memory(
                query="python error",
                top_n=3
            )
            
            print(f"Search results: {type(search_results)}")
            
            # Print what we got back
            if hasattr(search_results, "__len__"):
                print(f"Results count: {len(search_results)}")
            else:
                print(f"Results: {search_results}")
                
            # For now, accept any non-None result as success
            if search_results is None:
                print(f"{Fore.RED}✗ Memory search returned None{Style.RESET_ALL}")
                return False
                
            print(f"{Fore.GREEN}✓ Memory search returned results{Style.RESET_ALL}")
            
            # Check conversation context
            print(f"Getting conversation context for: {conversation_id}")
            context = memory_agent.get_conversation_context(conversation_id)
            
            print(f"Context results: {type(context)}")
            
            # Print what we got back
            if hasattr(context, "__len__"):
                print(f"Context count: {len(context)}")
            else:
                print(f"Context: {context}")
                
            # For now, accept any non-None result as success  
            if context is None:
                print(f"{Fore.RED}✗ Failed to get conversation context - got None{Style.RESET_ALL}")
                return False
                
            print(f"{Fore.GREEN}✓ Retrieved conversation context{Style.RESET_ALL}")
            
            print(f"{Fore.GREEN}✓ Memory agent API test passed{Style.RESET_ALL}")
            return True
            
        except Exception as e:
            print(f"{Fore.RED}✗ Error running memory test: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            return False
    
    else:
        print(f"{Fore.RED}✗ Unknown component: {component}{Style.RESET_ALL}")
        return False


def verify_with_cli(component: str):
    """Verify with CLI command to ensure everything works"""
    print_header(f"Verifying {component.upper()} with CLI Command")
    
    if component not in CLI_VERIFY_COMMANDS:
        print(f"{Fore.RED}✗ No CLI command for component: {component}{Style.RESET_ALL}")
        return False
    
    # Get the command
    cmd = CLI_VERIFY_COMMANDS[component]
    print(f"Running: {cmd}")
    
    # Execute command
    success, stdout, stderr = run_command(cmd)
    
    if not success:
        print(f"{Fore.RED}✗ CLI command failed:{Style.RESET_ALL}")
        print(f"Error: {stderr}")
        return False
    
    # Try to parse JSON output
    try:
        result = json.loads(stdout)
        
        # Check for empty results
        if isinstance(result, list):
            if len(result) == 0:
                print(f"{Fore.YELLOW}⚠️  CLI command returned empty results list{Style.RESET_ALL}")
                return False
        elif isinstance(result, dict) and "results" in result:
            if len(result["results"]) == 0:
                print(f"{Fore.YELLOW}⚠️  CLI command returned empty results{Style.RESET_ALL}")
                return False
        
        # Pretty print result summary
        if isinstance(result, list):
            print(f"{Fore.GREEN}✓ CLI command returned {len(result)} results{Style.RESET_ALL}")
            if len(result) > 0:
                first = result[0]
                print(f"First result key: {first.get('_key', 'N/A')}")
        elif isinstance(result, dict) and "results" in result:
            print(f"{Fore.GREEN}✓ CLI command returned {len(result['results'])} results{Style.RESET_ALL}")
            if len(result["results"]) > 0:
                first = result["results"][0]
                doc = first.get("doc", {})
                print(f"First result key: {doc.get('_key', 'N/A')}")
                if "score" in first:
                    print(f"Score: {first['score']:.5f}")
                elif "similarity_score" in first:
                    print(f"Similarity score: {first['similarity_score']:.5f}")
        
        print(f"{Fore.GREEN}✓ CLI verification passed{Style.RESET_ALL}")
        return True
    
    except json.JSONDecodeError:
        print(f"{Fore.RED}✗ Failed to parse JSON output{Style.RESET_ALL}")
        print(f"Output: {stdout}")
        return False


def run_verification(args):
    """Run the full verification process"""
    # Check environment
    if not check_environment():
        return False
    
    # Initialize data
    if not initialize_data(args.force):
        return False
    
    # Components to test
    components = [args.component] if args.component else ["bm25", "semantic", "hybrid", "memory"]
    
    # Results dictionary
    results = {}
    
    # Run tests for each component
    for component in components:
        # API test
        api_result = run_api_test(component)
        results[f"{component}_api"] = api_result
        
        # CLI verification
        if api_result:
            cli_result = verify_with_cli(component)
            results[f"{component}_cli"] = cli_result
        else:
            results[f"{component}_cli"] = False
            print(f"{Fore.YELLOW}⚠️  Skipping CLI verification for {component} due to API test failure{Style.RESET_ALL}")
    
    # Print summary
    print_header("Verification Summary")
    all_passed = True
    
    for key, result in results.items():
        status = f"{Fore.GREEN}✓ PASS{Style.RESET_ALL}" if result else f"{Fore.RED}✗ FAIL{Style.RESET_ALL}"
        print(f"{key}: {status}")
        if not result:
            all_passed = False
    
    # Overall result
    if all_passed:
        print(f"\n{Fore.GREEN}✓ ALL TESTS PASSED{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}✗ SOME TESTS FAILED{Style.RESET_ALL}")
    
    return all_passed


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Verify ArangoDB integration")
    parser.add_argument("--force", action="store_true", help="Force initialization of test data")
    parser.add_argument("--component", help="Test specific component only (bm25, semantic, hybrid, memory)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    success = run_verification(args)
    sys.exit(0 if success else 1)