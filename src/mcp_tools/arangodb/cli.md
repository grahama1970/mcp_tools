# ArangoDB Lessons CLI Commands

This document provides a summary of the commands available in the `src/complexity/arangodb/cli.py` script.

## Search Commands

| Command                     | Description                                                                 | Key Arguments/Options                                                                 |
| :-------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| `search bm25`               | Find documents based on keyword relevance (BM25 algorithm).                 | `QUERY`, `--threshold`, `--top-n`, `--offset`, `--tags`, `--json-output`              |
| `search semantic`           | Find documents based on conceptual meaning (vector similarity).             | `QUERY`, `--threshold`, `--top-n`, `--tags`, `--json-output`                          |
| `search hybrid`             | Combine keyword (BM25) and semantic search results using RRF re-ranking.    | `QUERY`, `--top-n`, `--initial-k`, `--bm25-th`, `--sim-th`, `--tags`, `--json-output` |
| `search keyword`            | Find documents based on exact keyword matching within specified fields.     | `KEYWORDS...`, `--search-fields`, `--limit`, `--match-all`, `--json-output`           |
| `search tag`                | Find documents based on exact tag matching within the 'tags' array field.   | `TAGS...`, `--limit`, `--match-all`, `--json-output`                                  |

## CRUD Commands

| Command                     | Description                                                                 | Key Arguments/Options                                                                 |
| :-------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| `crud add-lesson`           | Add a new lesson document (vertex).                                         | `--data` or `--data-file` (required), `--json-output`                                 |
| `crud get-lesson`           | Retrieve a specific lesson document (vertex) by its _key.                   | `KEY` (required), `--json-output`                                                     |
| `crud update-lesson`        | Modify specific fields of an existing lesson document (vertex).             | `KEY` (required), `--data` or `--data-file` (required), `--json-output`               |
| `crud delete-lesson`        | Permanently remove a lesson document (vertex) and its associated edges.     | `KEY` (required), `--yes`, `--json-output`                                            |

## Graph Commands

| Command                     | Description                                                                 | Key Arguments/Options                                                                 |
| :-------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| `graph add-relationship`    | Create a directed link (relationship edge) between two lessons.             | `FROM_KEY`, `TO_KEY`, `--rationale` (req), `--type` (req), `--attributes`, `--json-output` |
| `graph delete-relationship` | Remove a specific relationship link (edge) between lessons.                 | `EDGE_KEY` (required), `--yes`, `--json-output`                                       |
| `graph traverse`            | Explore relationships between lessons via graph traversal.                  | `START_NODE_ID` (required), `--graph-name`, `--min-depth`, `--max-depth`, `--direction`, `--limit`, `--json-output` |

## Memory Agent Commands

| Command                     | Description                                                                 | Key Arguments/Options                                                                 |
| :-------------------------- | :-------------------------------------------------------------------------- | :------------------------------------------------------------------------------------ |
| `memory store`              | Store a conversation between user and agent in the memory database.         | `USER_MESSAGE` `AGENT_RESPONSE`, `--conversation-id`, `--metadata`, `--json-output`    |
| `memory search`             | Search for relevant memories using hybrid search.                           | `QUERY`, `--top-n`, `--tags`, `--json-output`                                         |
| `memory related`            | Find memories related to a specific memory through the knowledge graph.     | `MEMORY_KEY`, `--type`, `--max-depth`, `--limit`, `--json-output`                     |
| `memory context`            | Retrieve all messages in a conversation in chronological order.             | `CONVERSATION_ID`, `--limit`, `--json-output`                                         |

*Refer to the docstrings within `cli.py` or use `python -m src.complexity.arangodb.cli [COMMAND] --help` for detailed information on each command's arguments and options.*