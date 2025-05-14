"""
CLI search commands for ArangoDB.

This module provides CLI commands for executing searches against ArangoDB.
It includes BM25, semantic, tag, and hybrid search commands with various formatting options.

Links to third-party documentation:
- Typer: https://typer.tiangolo.com/
- Rich: https://rich.readthedocs.io/

Sample input:
    $ python -m mcp_tools.arangodb.cli search bm25 "python error handling" --limit 5 --format json
    $ python -m mcp_tools.arangodb.cli search tag python error-handling --match-all --limit 5

Expected output:
    A table or JSON output displaying search results with relevance scores.
"""

import typer
from typing import Dict, List, Any, Optional, Union
from loguru import logger
from arango.database import StandardDatabase

from mcp_tools.arangodb.cli.formatters import display_search_results
from mcp_tools.arangodb.core.search.bm25_search import bm25_search
from mcp_tools.arangodb.core.search.semantic_search import semantic_search
from mcp_tools.arangodb.core.search.tag_search import tag_search
from mcp_tools.arangodb.core.search.keyword_search import keyword_search
from mcp_tools.arangodb.core.search.glossary_search import (
    glossary_search,
    get_glossary_terms,
    add_glossary_terms,
    highlight_text_with_glossary
)

# Database connection helper to be properly implemented
from arango import ArangoClient

def get_db_connection() -> StandardDatabase:
    """Get a database connection."""
    # This should be properly implemented to match your connection setup
    # For now, just a placeholder
    client = ArangoClient(hosts="http://localhost:8529")
    db = client.db("_system", username="root", password="")
    return db

# Create search command app
app = typer.Typer(name="search", help="Search ArangoDB for documents", rich_markup_mode="rich")

@app.command("bm25")
def bm25_search_command(
    query: str = typer.Argument(..., help="Search query text"),
    collection: str = typer.Option("documents", "--collection", "-c", help="Collection to search"),
    view: str = typer.Option("documents_view", "--view", "-v", help="View to use for search"),
    min_score: float = typer.Option(0.0, "--min-score", "-m", help="Minimum score threshold"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    offset: int = typer.Option(0, "--offset", "-o", help="Offset for pagination"),
    tags: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Filter by tag (can specify multiple)"),
    filter: Optional[str] = typer.Option(None, "--filter", "-f", help="AQL filter expression"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
) -> None:
    """
    Execute a BM25 full-text search against ArangoDB.

    BM25 search uses a keyword-based algorithm to find relevant documents, similar to
    what's used in many search engines. It works best for keyword matching and supports
    filtering by tags or custom expressions.

    Examples:
        $ search bm25 "python error handling"
        $ search bm25 "database" --tag arangodb --tag tutorial --limit 5
        $ search bm25 "user interface" --filter "doc.validated == true" --format json
    """
    try:
        db = get_db_connection()

        # Run the search
        search_results = bm25_search(
            db=db,
            query_text=query,
            collection_name=collection,
            view_name=view,
            min_score=min_score,
            top_n=limit,
            offset=offset,
            tag_list=tags,
            filter_expr=filter
        )

        # Display results
        title = f"BM25 Search Results for '{query}'"
        display_search_results(
            results_data=search_results,
            title=title,
            score_field="score",
            output_format=format
        )
    except Exception as e:
        logger.exception(f"Error executing BM25 search: {e}")
        typer.echo(f"Error: {e}", err=True)


@app.command("semantic")
def semantic_search_command(
    query: str = typer.Argument(..., help="Search query text"),
    collection: str = typer.Option("documents", "--collection", "-c", help="Collection to search"),
    embedding_field: str = typer.Option("embedding", "--embedding-field", "-e", help="Field containing embedding vectors"),
    min_score: float = typer.Option(0.7, "--min-score", "-m", help="Minimum similarity threshold (0-1)"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    tags: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Filter by tag (can specify multiple)"),
    filter: Optional[str] = typer.Option(None, "--filter", "-f", help="AQL filter expression"),
    direct: bool = typer.Option(False, "--direct", help="Force direct vector search without filtering"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
) -> None:
    """
    Execute a semantic vector similarity search against ArangoDB.

    Semantic search uses embedding vectors to find documents with similar meaning
    rather than just matching keywords. It works by comparing the vector similarity
    between the query and documents.

    Examples:
        $ search semantic "how to implement error handling in Python"
        $ search semantic "climate change impacts" --tag science --min-score 0.8
        $ search semantic "database performance" --filter "doc.validated == true" --format json
    """
    try:
        db = get_db_connection()

        # Run the search
        collections = [collection] if collection else None

        search_results = semantic_search(
            db=db,
            query=query,
            collections=collections,
            filter_expr=filter,
            min_score=min_score,
            top_n=limit,
            tag_list=tags,
            embedding_field=embedding_field,
            force_direct=direct
        )

        # Add format for display
        if "format" not in search_results:
            search_results["format"] = format

        # Display results
        title = f"Semantic Search Results for '{query}'"
        display_search_results(
            results_data=search_results,
            title=title,
            score_field="similarity_score",
            output_format=format
        )
    except Exception as e:
        logger.exception(f"Error executing semantic search: {e}")
        typer.echo(f"Error: {e}", err=True)


@app.command("tag")
def tag_search_command(
    tags: List[str] = typer.Argument(..., help="One or more tags to search for (case-sensitive)."),
    collection: str = typer.Option("documents", "--collection", "-c", help="Collection to search"),
    match_all: bool = typer.Option(
        False,
        "--match-all",
        "-ma",
        help="Require all tags to match (AND logic) instead of any (OR logic).",
    ),
    limit: int = typer.Option(
        10, "--limit", "-l", help="Maximum number of results to return.", min=1
    ),
    offset: int = typer.Option(
        0, "--offset", "-o", help="Offset for pagination."
    ),
    filter: Optional[str] = typer.Option(None, "--filter", "-f", help="AQL filter expression"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
) -> None:
    """
    Find documents based on exact tag matching within the 'tags' array field.

    Tag search performs an exact, case-sensitive match against items in the document's `tags` array.
    This is useful for filtering documents by predefined categories or topics.

    Examples:
        $ search tag database python                # Find docs with tag "database" OR "python"
        $ search tag testing docker --match-all     # Find docs with BOTH "testing" AND "docker" tags
        $ search tag cli --format json              # Output results as JSON
        $ search tag api --filter "doc.validated == true"  # Only include validated documents
    """
    try:
        db = get_db_connection()

        # Collections must be in list format for tag_search
        collections = [collection] if collection else None

        # Run the search
        search_results = tag_search(
            db=db,
            tags=tags,
            collections=collections,
            filter_expr=filter,
            require_all_tags=match_all,
            limit=limit,
            offset=offset
        )

        # Add format for display if not already set
        if "format" not in search_results:
            search_results["format"] = format

        # Display results - pass tag_match_score as the score field
        title = f"Tag Search Results for {', '.join(tags)} ({'ALL' if match_all else 'ANY'})"
        display_search_results(
            results_data=search_results,
            title=title,
            score_field="tag_match_score",
            output_format=format
        )
    except Exception as e:
        logger.exception(f"Error executing tag search: {e}")
        typer.echo(f"Error: {e}", err=True)


@app.command("keyword")
def keyword_search_command(
    query: str = typer.Argument(..., help="Keyword to search for"),
    collection: str = typer.Option("documents", "--collection", "-c", help="Collection to search"),
    view: str = typer.Option("documents_view", "--view", "-v", help="View to use for search"),
    similarity: float = typer.Option(
        97.0, "--similarity", "-s", help="Similarity threshold (0-100) for fuzzy matching"
    ),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    tags: Optional[List[str]] = typer.Option(None, "--tag", "-t", help="Filter by tag (can specify multiple)"),
    fields: Optional[List[str]] = typer.Option(
        None, "--field", "-f", help="Specific fields to search (can specify multiple)"
    ),
    filter: Optional[str] = typer.Option(None, "--filter", help="AQL filter expression"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
) -> None:
    """
    Perform a keyword search with fuzzy matching capabilities.

    Keyword search uses RapidFuzz to find matches even with spelling variations
    and typos. It's great for finding documents that contain specific words or
    phrases, with a configurable similarity threshold.

    Examples:
        $ search keyword "python error"          # Find documents about "python error"
        $ search keyword javascript --field title --field content  # Search only in title and content fields
        $ search keyword pandas --similarity 90  # More lenient matching (lower threshold)
        $ search keyword "api usage" --tag coding --tag tutorial  # With tag filtering
    """
    try:
        db = get_db_connection()

        # Convert collection to list format
        collections = [collection] if collection else None

        # Execute the search
        search_results = keyword_search(
            db=db,
            search_term=query,
            collections=collections,
            view_name=view,
            similarity_threshold=similarity,
            top_n=limit,
            tags=tags,
            fields_to_search=fields,
            filter_expr=filter
        )

        # Add format for display
        if "format" not in search_results:
            search_results["format"] = format

        # Display results
        title = f"Keyword Search Results for '{query}'"
        display_search_results(
            results_data=search_results,
            title=title,
            score_field="keyword_score",
            output_format=format
        )
    except Exception as e:
        logger.exception(f"Error executing keyword search: {e}")
        typer.echo(f"Error: {e}", err=True)


# Create glossary subcommand group
glossary_app = typer.Typer(name="glossary", help="Glossary term management and search", rich_markup_mode="rich")
app.add_typer(glossary_app)


@glossary_app.command("search")
def glossary_search_command(
    text: str = typer.Argument(..., help="Text to search for glossary terms"),
    collection: str = typer.Option("glossary", "--collection", "-c", help="Glossary collection name"),
    show_positions: bool = typer.Option(False, "--positions", "-p", help="Include position information in results"),
    limit: int = typer.Option(100, "--limit", "-l", help="Maximum number of results"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
) -> None:
    """
    Find glossary terms within provided text.

    This command searches the text for any matches with terms in the glossary
    collection. This is useful for identifying domain-specific terminology
    within content.

    Examples:
        $ search glossary search "What is the relationship between primary colors and RGB?"
        $ search glossary search "Color theory combines hues and saturation values" --format json
        $ search glossary search "Complex technical terms" --collection custom_glossary
    """
    try:
        db = get_db_connection()

        # Execute the search
        search_results = glossary_search(
            db=db,
            text=text,
            collection_name=collection,
            include_positions=show_positions,
            top_n=limit
        )

        # Add format for display
        if "format" not in search_results:
            search_results["format"] = format

        # Display results
        title = f"Glossary Terms in '{text[:30] + '...' if len(text) > 30 else text}'"
        display_search_results(
            results_data=search_results,
            title=title,
            score_field="glossary_score",
            output_format=format
        )
    except Exception as e:
        logger.exception(f"Error executing glossary search: {e}")
        typer.echo(f"Error: {e}", err=True)


@glossary_app.command("list")
def glossary_list_command(
    collection: str = typer.Option("glossary", "--collection", "-c", help="Glossary collection name"),
    format: str = typer.Option("table", "--format", help="Output format: table or json"),
) -> None:
    """
    List all terms in the glossary.

    This command retrieves and displays all terms and definitions from the glossary collection.

    Examples:
        $ search glossary list
        $ search glossary list --format json
        $ search glossary list --collection custom_glossary
    """
    try:
        db = get_db_connection()

        # Get all terms
        results = get_glossary_terms(
            db=db,
            collection_name=collection
        )

        # Add format for display
        if "format" not in results:
            results["format"] = format

        # Display results
        title = f"All Glossary Terms in '{collection}'"
        display_search_results(
            results_data=results,
            title=title,
            score_field="glossary_score",
            output_format=format
        )
    except Exception as e:
        logger.exception(f"Error listing glossary terms: {e}")
        typer.echo(f"Error: {e}", err=True)


@glossary_app.command("add")
def glossary_add_command(
    term: str = typer.Argument(..., help="Term to add to the glossary"),
    definition: str = typer.Argument(..., help="Definition of the term"),
    collection: str = typer.Option("glossary", "--collection", "-c", help="Glossary collection name"),
) -> None:
    """
    Add a term to the glossary.

    This command adds a new term and its definition to the glossary collection.

    Examples:
        $ search glossary add "RGB" "Color model that uses red, green, and blue light"
        $ search glossary add "hue" "The pure color attribute" --collection color_glossary
    """
    try:
        db = get_db_connection()

        # Add the term
        result = add_glossary_terms(
            db=db,
            terms_dict={term: definition},
            collection_name=collection
        )

        if result.get("success", False):
            typer.echo(f"✅ Term '{term}' added successfully to the glossary.")
            typer.echo(f"The glossary now contains {result.get('total_terms', 0)} terms.")
        else:
            typer.echo(f"❌ Failed to add term: {result.get('error', 'Unknown error')}", err=True)
    except Exception as e:
        logger.exception(f"Error adding glossary term: {e}")
        typer.echo(f"Error: {e}", err=True)


@glossary_app.command("highlight")
def glossary_highlight_command(
    text: str = typer.Argument(..., help="Text to highlight with glossary terms"),
    collection: str = typer.Option("glossary", "--collection", "-c", help="Glossary collection name"),
    marker_start: str = typer.Option("**", "--marker-start", "-ms", help="Start marker for highlighting"),
    marker_end: str = typer.Option("**", "--marker-end", "-me", help="End marker for highlighting"),
) -> None:
    """
    Highlight glossary terms in the provided text.

    This command highlights any glossary terms found in the text using the specified markers.
    Default markers are double asterisks (Markdown bold).

    Examples:
        $ search glossary highlight "RGB is a primary color model used in displays"
        $ search glossary highlight "Color theory involves hue and saturation" --marker-start "<em>" --marker-end "</em>"
    """
    try:
        db = get_db_connection()

        # Highlight the text
        result = highlight_text_with_glossary(
            db=db,
            text=text,
            collection_name=collection,
            marker_start=marker_start,
            marker_end=marker_end
        )

        # Display the highlighted text
        typer.echo("\nOriginal text:")
        typer.echo(text)

        typer.echo("\nHighlighted text:")
        typer.echo(result.get("highlighted_text", text))

        # Show stats
        term_count = result.get("term_count", 0)
        terms = result.get("terms", [])
        if term_count > 0:
            typer.echo(f"\nFound {term_count} glossary terms:")
            for i, term_info in enumerate(terms, start=1):
                typer.echo(f"{i}. {term_info.get('term', 'Unknown')}: {term_info.get('definition', 'No definition')}")
        else:
            typer.echo("\nNo glossary terms found in the text.")
    except Exception as e:
        logger.exception(f"Error highlighting text: {e}")
        typer.echo(f"Error: {e}", err=True)


if __name__ == "__main__":
    import sys
    
    # Validation setup
    all_validation_failures = []
    total_tests = 0
    
    # Test 1: Ensure the CLI app is created successfully
    total_tests += 1
    try:
        # Check that the app is created
        if not isinstance(app, typer.Typer):
            raise ValueError("Typer app not created correctly")

        # Verify essential commands exist
        commands = app.registered_commands
        required_commands = ["bm25", "semantic", "tag", "keyword", "glossary"]

        for cmd_name in required_commands:
            if not any(cmd.name == cmd_name for cmd in commands):
                raise ValueError(f"{cmd_name.upper()} command not registered")

        # Log found commands
        cmd_names = [cmd.name for cmd in commands]
        print(f"CLI app has configured commands: {', '.join(cmd_names)}")
    except Exception as e:
        all_validation_failures.append(f"Test 1: CLI app check failed: {str(e)}")
    
    # Final validation result
    if all_validation_failures:
        print(f"\n❌ VALIDATION FAILED - {len(all_validation_failures)} of {total_tests} tests failed:")
        for failure in all_validation_failures:
            print(f"  - {failure}")
        sys.exit(1)
    else:
        print(f"\n✅ VALIDATION PASSED - All {total_tests} tests produced expected results")
        print("Function is validated and formal tests can now be written")
        sys.exit(0)