# Task 003: Memory Agent CLI Integration

## Objective

Integrate the Memory Agent functionality into the main CLI (`src/complexity/cli.py`) to enable storage, retrieval, and querying of agent conversation memories through the command line interface. This integration is crucial for making the Memory Agent accessible to other agents in the system.

## Status

- [ ] CLI Memory Command Group Implementation
- [ ] Store Conversation Command
- [ ] Search Memory Command
- [ ] Get Related Memories Command
- [ ] Get Conversation Context Command
- [ ] Tests and Documentation

## Technical Specifications

### Overview

The Memory Agent module provides functionality to store LLM agent conversations in ArangoDB with vector embeddings for semantic search and relationship management. The current implementation in `src/complexity/arangodb/memory_agent/memory_agent.py` needs to be exposed through the CLI to make it accessible to other agents.

### Requirements

1. Add a new command group called `memory` to the main CLI
2. Implement commands to:
   - Store conversations (`memory store`)
   - Search for relevant memories (`memory search`) 
   - Get related memories (`memory related`)
   - Get conversation context (`memory context`)
3. Ensure proper error handling and feedback
4. Maintain consistent CLI patterns with the existing commands
5. Allow both human-readable and JSON output formats

### Interface Design

New `memory` command group with the following subcommands:

1. `memory store`: Store a user-agent conversation exchange
   - Arguments: 
     - `conversation_id` (optional): ID to group related messages
     - `user_message`: The user's message content  
     - `agent_response`: The agent's response content
   - Options:
     - `--metadata/-m`: Additional metadata as JSON string
     - `--json-output/-j`: Output result as JSON

2. `memory search`: Search for relevant memories
   - Arguments:
     - `query`: The search query text
   - Options:
     - `--top-n/-n`: Number of results to return (default: 5)
     - `--collection/-c`: Collection to search in (default: agent_memories)
     - `--json-output/-j`: Output results as JSON

3. `memory related`: Get memories related to a specific memory
   - Arguments:
     - `memory_key`: Key of the memory to find related memories for
   - Options:
     - `--relationship-type/-r`: Type of relationship to filter by
     - `--max-depth/-d`: Maximum traversal depth (default: 1)
     - `--limit/-l`: Maximum number of results (default: 10)
     - `--json-output/-j`: Output results as JSON

4. `memory context`: Get the conversation context for a conversation ID
   - Arguments:
     - `conversation_id`: ID of the conversation
   - Options:
     - `--limit/-l`: Maximum number of messages to retrieve (default: 10)
     - `--json-output/-j`: Output results as JSON

## Implementation Tasks

### 1. CLI Memory Command Group Implementation

- [ ] **1.1 Create Memory Command Group**
  - Add a new Typer app for the memory command group
  - Add it to the main CLI app
  - Document the memory command group

### 2. Store Conversation Command

- [ ] **2.1 Implement memory store command**
  - Map CLI arguments to Memory Agent's store_conversation method
  - Handle JSON metadata parsing
  - Add proper error handling
  - Format output (both text and JSON)

- [ ] **2.2 Add examples and documentation**
  - Add detailed docstring with examples
  - Include when-to-use guidance

### 3. Search Memory Command

- [ ] **3.1 Implement memory search command**
  - Map CLI arguments to Memory Agent's search_memory method
  - Handle query embedding and search parameters
  - Format search results using the existing display_results utility
  - Add fallbacks for search failures

- [ ] **3.2 Add examples and documentation**
  - Add detailed docstring with examples
  - Include when-to-use guidance

### 4. Get Related Memories Command

- [ ] **4.1 Implement memory related command**
  - Map CLI arguments to Memory Agent's get_related_memories method
  - Handle relationship type filtering
  - Format related memory results
  - Add fallbacks for traversal failures

- [ ] **4.2 Add examples and documentation**
  - Add detailed docstring with examples
  - Include when-to-use guidance

### 5. Get Conversation Context Command

- [ ] **5.1 Implement memory context command**
  - Map CLI arguments to Memory Agent's get_conversation_context method
  - Format conversation context results
  - Add fallbacks for missing conversations

- [ ] **5.2 Add examples and documentation**
  - Add detailed docstring with examples
  - Include when-to-use guidance

### 6. Tests and Documentation

- [ ] **6.1 Add memory commands to initialization**
  - Update the CLI's initialization command to set up Memory Agent collections
  - Add sample memories to initialization command

- [ ] **6.2 Create test cases**
  - Create test cases for each memory command
  - Test error handling and edge cases

- [ ] **6.3 Update CLI usage documentation**
  - Add Memory Agent commands to CLI docstring
  - Update main README or usage documents

## Verification Methods

### Verification Approach

1. **Command Implementation Verification:**
   - Execute each memory command and verify behavior matches expectations
   - Check all parameters function as documented
   - Verify proper error handling

2. **Storage and Retrieval Verification:**
   - Store a conversation and verify it can be retrieved
   - Verify relationships are created between memories
   - Verify search works as expected

3. **Example Validation:**
   - Execute all examples from documentation
   - Verify output matches what is described

### Acceptance Criteria

- All memory commands correctly interact with the Memory Agent
- Commands provide consistent feedback in both text and JSON formats
- Error handling degrades gracefully
- Documentation clearly explains how to use each command
- Examples are accurate and representative

## Progress Tracking

**Start Date:** 2025-05-03
**Target Completion:** 2025-05-10
**Status:** Not Started

### Updates

- 2025-05-03: Task created