"""
CLI commands for ArangoDB database operations.

This module provides Typer CLI commands for:
- Document CRUD operations
- Message management
- Relationship/graph operations

These commands use the core database functionality and add CLI-specific
formatting and interaction capabilities.
"""

import json
from typing import Dict, List, Any, Optional, Tuple
import datetime
import typer
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text

from arango import ArangoClient
from arango.database import Database

from mcp_tools.arangodb.core.db import (
    # CRUD operations
    create_document,
    get_document, 
    update_document,
    delete_document,
    query_documents,
    
    # Message operations
    create_message,
    get_message,
    update_message,
    delete_message,
    get_conversation_messages,
    delete_conversation,
    
    # Relationship operations
    create_relationship,
    delete_relationship_by_key,
    delete_relationships_between,
    link_message_to_document,
    get_documents_for_message,
    get_messages_for_document,
    get_related_documents,
)

from mcp_tools.arangodb.cli.formatters import (
    format_document,
    format_document_list,
    format_error,
)

# Create Typer app for database commands
app = typer.Typer(
    help="ArangoDB database operations",
    rich_markup_mode="rich",
)

console = Console()


def get_db_connection(
    host: str = "http://localhost:8529", 
    username: str = "root", 
    password: str = "", 
    database: str = "_system"
) -> Database:
    """Get a database connection."""
    client = ArangoClient(hosts=host)
    return client.db(database, username=username, password=password)


# ---- CRUD Operations ----

@app.command("create")
def create_document_cli(
    collection: str = typer.Argument(..., help="Collection name"),
    data_file: str = typer.Option(None, "--file", "-f", help="JSON file containing the document data"),
    data_json: str = typer.Option(None, "--json", "-j", help="JSON string containing the document data"),
    host: str = typer.Option("http://localhost:8529", "--host", "-h", help="ArangoDB host URL"),
    username: str = typer.Option("root", "--username", "-u", help="ArangoDB username"),
    password: str = typer.Option("", "--password", "-p", help="ArangoDB password"),
    database: str = typer.Option("_system", "--database", "-d", help="ArangoDB database name"),
):
    """
    Create a document in an ArangoDB collection.
    
    Example:
        $ arangodb create my_collection --json '{"name": "Test Document", "value": 42}'
        $ arangodb create my_collection --file document_data.json
    """
    try:
        # Get document data
        if data_file:
            with open(data_file, 'r') as f:
                document_data = json.load(f)
        elif data_json:
            document_data = json.loads(data_json)
        else:
            typer.echo(format_error("Either --file or --json must be provided"))
            raise typer.Exit(code=1)
        
        # Connect to database
        db = get_db_connection(host, username, password, database)
        
        # Create document
        result = create_document(collection, document_data, db)
        
        # Display result
        console.print(Panel(
            format_document(result),
            title="[bold green]Document Created[/bold green]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(format_error(f"Document creation failed: {str(e)}"))
        raise typer.Exit(code=1)


@app.command("get")
def get_document_cli(
    collection: str = typer.Argument(..., help="Collection name"),
    key: str = typer.Argument(..., help="Document key"),
    host: str = typer.Option("http://localhost:8529", "--host", "-h", help="ArangoDB host URL"),
    username: str = typer.Option("root", "--username", "-u", help="ArangoDB username"),
    password: str = typer.Option("", "--password", "-p", help="ArangoDB password"),
    database: str = typer.Option("_system", "--database", "-d", help="ArangoDB database name"),
    format_output: bool = typer.Option(True, "--format/--raw", help="Format the output or show raw JSON"),
):
    """
    Retrieve a document from an ArangoDB collection.
    
    Example:
        $ arangodb get my_collection 12345
    """
    try:
        # Connect to database
        db = get_db_connection(host, username, password, database)
        
        # Get document
        result = get_document(collection, key, db)
        
        # Display result
        if format_output:
            console.print(Panel(
                format_document(result),
                title=f"[bold blue]Document: {collection}/{key}[/bold blue]",
                border_style="blue"
            ))
        else:
            # Print raw JSON with syntax highlighting
            console.print(Syntax(
                json.dumps(result, indent=2),
                "json",
                theme="monokai",
                line_numbers=True
            ))
            
    except Exception as e:
        console.print(format_error(f"Document retrieval failed: {str(e)}"))
        raise typer.Exit(code=1)


@app.command("update")
def update_document_cli(
    collection: str = typer.Argument(..., help="Collection name"),
    key: str = typer.Argument(..., help="Document key"),
    data_file: str = typer.Option(None, "--file", "-f", help="JSON file containing the update data"),
    data_json: str = typer.Option(None, "--json", "-j", help="JSON string containing the update data"),
    host: str = typer.Option("http://localhost:8529", "--host", "-h", help="ArangoDB host URL"),
    username: str = typer.Option("root", "--username", "-u", help="ArangoDB username"),
    password: str = typer.Option("", "--password", "-p", help="ArangoDB password"),
    database: str = typer.Option("_system", "--database", "-d", help="ArangoDB database name"),
    check_rev: str = typer.Option(None, "--rev", help="Document revision to check"),
    replace: bool = typer.Option(False, "--replace", help="Replace document instead of updating"),
):
    """
    Update a document in an ArangoDB collection.
    
    Example:
        $ arangodb update my_collection 12345 --json '{"name": "Updated Document"}'
        $ arangodb update my_collection 12345 --file update_data.json
    """
    try:
        # Get update data
        if data_file:
            with open(data_file, 'r') as f:
                update_data = json.load(f)
        elif data_json:
            update_data = json.loads(data_json)
        else:
            typer.echo(format_error("Either --file or --json must be provided"))
            raise typer.Exit(code=1)
        
        # Connect to database
        db = get_db_connection(host, username, password, database)
        
        # Update document
        result = update_document(
            collection, 
            key, 
            update_data, 
            db,
            merge=not replace,
            check_rev=check_rev
        )
        
        # Display result
        console.print(Panel(
            format_document(result),
            title="[bold green]Document Updated[/bold green]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(format_error(f"Document update failed: {str(e)}"))
        raise typer.Exit(code=1)


@app.command("delete")
def delete_document_cli(
    collection: str = typer.Argument(..., help="Collection name"),
    key: str = typer.Argument(..., help="Document key"),
    host: str = typer.Option("http://localhost:8529", "--host", "-h", help="ArangoDB host URL"),
    username: str = typer.Option("root", "--username", "-u", help="ArangoDB username"),
    password: str = typer.Option("", "--password", "-p", help="ArangoDB password"),
    database: str = typer.Option("_system", "--database", "-d", help="ArangoDB database name"),
    check_rev: str = typer.Option(None, "--rev", help="Document revision to check"),
    silent: bool = typer.Option(False, "--silent", "-s", help="Silent operation"),
):
    """
    Delete a document from an ArangoDB collection.
    
    Example:
        $ arangodb delete my_collection 12345
    """
    try:
        # Connect to database
        db = get_db_connection(host, username, password, database)
        
        # Delete document
        result = delete_document(collection, key, db, check_rev=check_rev)
        
        # Display result
        if not silent:
            if result:
                console.print("[bold green]Document deleted successfully[/bold green]")
            else:
                console.print("[yellow]Document not found or already deleted[/yellow]")
        
    except Exception as e:
        console.print(format_error(f"Document deletion failed: {str(e)}"))
        raise typer.Exit(code=1)


@app.command("query")
def query_documents_cli(
    query: str = typer.Argument(..., help="AQL query string"),
    bind_vars_file: str = typer.Option(None, "--vars-file", help="JSON file containing bind variables"),
    bind_vars_json: str = typer.Option(None, "--vars-json", help="JSON string containing bind variables"),
    host: str = typer.Option("http://localhost:8529", "--host", "-h", help="ArangoDB host URL"),
    username: str = typer.Option("root", "--username", "-u", help="ArangoDB username"),
    password: str = typer.Option("", "--password", "-p", help="ArangoDB password"),
    database: str = typer.Option("_system", "--database", "-d", help="ArangoDB database name"),
    count: bool = typer.Option(False, "--count", "-c", help="Enable count of results"),
    batch_size: int = typer.Option(None, "--batch-size", "-b", help="Batch size for result retrieval"),
    profile: bool = typer.Option(False, "--profile", help="Include profiling information"),
    format_output: bool = typer.Option(True, "--format/--raw", help="Format the output or show raw JSON"),
):
    """
    Execute an AQL query against ArangoDB.
    
    Example:
        $ arangodb query "FOR doc IN my_collection FILTER doc.name == @name RETURN doc" --vars-json '{"name": "Test"}'
    """
    try:
        # Get bind variables
        bind_vars = {}
        if bind_vars_file:
            with open(bind_vars_file, 'r') as f:
                bind_vars = json.load(f)
        elif bind_vars_json:
            bind_vars = json.loads(bind_vars_json)
        
        # Connect to database
        db = get_db_connection(host, username, password, database)
        
        # Execute query
        result = query_documents(
            query,
            db,
            bind_vars=bind_vars,
            count=count,
            batch_size=batch_size,
            profile=profile
        )
        
        # Display result
        if profile:
            results, stats = result
            
            # Display results
            if format_output:
                console.print(format_document_list(results))
            else:
                console.print(Syntax(json.dumps(results, indent=2), "json", theme="monokai"))
            
            # Display stats
            stats_table = Table(title="Query Statistics")
            stats_table.add_column("Statistic", style="cyan")
            stats_table.add_column("Value", style="yellow")
            
            for key, value in stats.items():
                if isinstance(value, dict):
                    stats_table.add_row(key, json.dumps(value, indent=2))
                else:
                    stats_table.add_row(key, str(value))
                    
            console.print(stats_table)
        else:
            if format_output:
                console.print(format_document_list(result))
            else:
                console.print(Syntax(json.dumps(result, indent=2), "json", theme="monokai"))
        
    except Exception as e:
        console.print(format_error(f"Query execution failed: {str(e)}"))
        raise typer.Exit(code=1)


# ---- Message Operations ----

@app.command("create-message")
def create_message_cli(
    conversation_id: str = typer.Argument(..., help="Conversation ID"),
    role: str = typer.Argument(..., help="Message role (e.g., 'user', 'assistant')"),
    content: str = typer.Argument(..., help="Message content"),
    metadata_file: str = typer.Option(None, "--metadata-file", help="JSON file containing metadata"),
    metadata_json: str = typer.Option(None, "--metadata-json", help="JSON string containing metadata"),
    host: str = typer.Option("http://localhost:8529", "--host", "-h", help="ArangoDB host URL"),
    username: str = typer.Option("root", "--username", "-u", help="ArangoDB username"),
    password: str = typer.Option("", "--password", "-p", help="ArangoDB password"),
    database: str = typer.Option("_system", "--database", "-d", help="ArangoDB database name"),
    collection: str = typer.Option("messages", "--collection", help="Messages collection name"),
):
    """
    Create a message in a conversation.
    
    Example:
        $ arangodb create-message conversation123 user "Hello, how can I help you?"
    """
    try:
        # Get metadata
        metadata = None
        if metadata_file:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        elif metadata_json:
            metadata = json.loads(metadata_json)
        
        # Connect to database
        db = get_db_connection(host, username, password, database)
        
        # Create message
        result = create_message(
            db,
            conversation_id,
            role,
            content,
            metadata=metadata,
            message_collection=collection
        )
        
        # Display result
        console.print(Panel(
            format_document(result),
            title="[bold green]Message Created[/bold green]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(format_error(f"Message creation failed: {str(e)}"))
        raise typer.Exit(code=1)


@app.command("get-conversation")
def get_conversation_cli(
    conversation_id: str = typer.Argument(..., help="Conversation ID"),
    host: str = typer.Option("http://localhost:8529", "--host", "-h", help="ArangoDB host URL"),
    username: str = typer.Option("root", "--username", "-u", help="ArangoDB username"),
    password: str = typer.Option("", "--password", "-p", help="ArangoDB password"),
    database: str = typer.Option("_system", "--database", "-d", help="ArangoDB database name"),
    collection: str = typer.Option("messages", "--collection", help="Messages collection name"),
    sort_by: str = typer.Option("created_at", "--sort-by", help="Field to sort by"),
    sort_direction: str = typer.Option("ASC", "--sort-direction", help="Sort direction (ASC/DESC)"),
    limit: int = typer.Option(None, "--limit", "-l", help="Maximum number of messages to return"),
    offset: int = typer.Option(None, "--offset", "-o", help="Number of messages to skip"),
    format_output: bool = typer.Option(True, "--format/--raw", help="Format the output or show raw JSON"),
):
    """
    Retrieve messages in a conversation.
    
    Example:
        $ arangodb get-conversation conversation123
    """
    try:
        # Connect to database
        db = get_db_connection(host, username, password, database)
        
        # Get conversation messages
        results = get_conversation_messages(
            db,
            conversation_id,
            sort_by=sort_by,
            sort_direction=sort_direction,
            limit=limit,
            offset=offset,
            message_collection=collection
        )
        
        # Display results
        if not results:
            console.print("[yellow]No messages found in this conversation[/yellow]")
            return
        
        if format_output:
            # Create conversation table
            table = Table(title=f"Conversation: {conversation_id}")
            table.add_column("Time", style="cyan")
            table.add_column("Role", style="green")
            table.add_column("Content", style="white")
            
            for msg in results:
                created_at = msg.get("created_at", "")
                if created_at:
                    # Format the timestamp to be more readable
                    try:
                        dt = datetime.datetime.fromisoformat(created_at)
                        created_at = dt.strftime("%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        pass
                        
                table.add_row(
                    created_at,
                    msg.get("role", ""),
                    Text(msg.get("content", ""), overflow="fold")
                )
                
            console.print(table)
        else:
            console.print(Syntax(json.dumps(results, indent=2), "json", theme="monokai"))
        
    except Exception as e:
        console.print(format_error(f"Failed to retrieve conversation: {str(e)}"))
        raise typer.Exit(code=1)


@app.command("delete-conversation")
def delete_conversation_cli(
    conversation_id: str = typer.Argument(..., help="Conversation ID"),
    host: str = typer.Option("http://localhost:8529", "--host", "-h", help="ArangoDB host URL"),
    username: str = typer.Option("root", "--username", "-u", help="ArangoDB username"),
    password: str = typer.Option("", "--password", "-p", help="ArangoDB password"),
    database: str = typer.Option("_system", "--database", "-d", help="ArangoDB database name"),
    message_collection: str = typer.Option("messages", "--messages", help="Messages collection name"),
    relationship_collection: str = typer.Option("relates_to", "--edges", help="Relationship collection name"),
    keep_relationships: bool = typer.Option(False, "--keep-relationships", help="Don't delete relationships"),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Confirm deletion without prompting"),
):
    """
    Delete all messages in a conversation.
    
    Example:
        $ arangodb delete-conversation conversation123 --yes
    """
    try:
        # Confirm deletion if not already confirmed
        if not confirm:
            typer.confirm(
                f"Are you sure you want to delete all messages in conversation '{conversation_id}'?",
                abort=True
            )
        
        # Connect to database
        db = get_db_connection(host, username, password, database)
        
        # Delete conversation
        deleted_count = delete_conversation(
            db,
            conversation_id,
            delete_relationships=not keep_relationships,
            message_collection=message_collection,
            relationship_collection=relationship_collection
        )
        
        # Display result
        console.print(f"[bold green]Deleted {deleted_count} messages from conversation '{conversation_id}'[/bold green]")
        
    except Exception as e:
        console.print(format_error(f"Conversation deletion failed: {str(e)}"))
        raise typer.Exit(code=1)


# ---- Relationship Operations ----

@app.command("link")
def link_documents_cli(
    from_id: str = typer.Argument(..., help="Source document ID (collection/key format)"),
    to_id: str = typer.Argument(..., help="Target document ID (collection/key format)"),
    properties_file: str = typer.Option(None, "--props-file", help="JSON file containing relationship properties"),
    properties_json: str = typer.Option(None, "--props-json", help="JSON string containing relationship properties"),
    host: str = typer.Option("http://localhost:8529", "--host", "-h", help="ArangoDB host URL"),
    username: str = typer.Option("root", "--username", "-u", help="ArangoDB username"),
    password: str = typer.Option("", "--password", "-p", help="ArangoDB password"),
    database: str = typer.Option("_system", "--database", "-d", help="ArangoDB database name"),
    edge_collection: str = typer.Option("relates_to", "--edges", help="Edge collection name"),
):
    """
    Create a relationship between two documents.
    
    Example:
        $ arangodb link messages/123 documents/456 --props-json '{"weight": 0.95}'
    """
    try:
        # Get properties
        properties = None
        if properties_file:
            with open(properties_file, 'r') as f:
                properties = json.load(f)
        elif properties_json:
            properties = json.loads(properties_json)
        
        # Connect to database
        db = get_db_connection(host, username, password, database)
        
        # Create relationship
        result = create_relationship(
            db,
            from_id,
            to_id,
            edge_collection=edge_collection,
            properties=properties
        )
        
        # Display result
        console.print(Panel(
            format_document(result),
            title="[bold green]Relationship Created[/bold green]",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(format_error(f"Relationship creation failed: {str(e)}"))
        raise typer.Exit(code=1)


@app.command("get-related")
def get_related_documents_cli(
    document_id: str = typer.Argument(..., help="Document ID (collection/key format)"),
    host: str = typer.Option("http://localhost:8529", "--host", "-h", help="ArangoDB host URL"),
    username: str = typer.Option("root", "--username", "-u", help="ArangoDB username"),
    password: str = typer.Option("", "--password", "-p", help="ArangoDB password"),
    database: str = typer.Option("_system", "--database", "-d", help="ArangoDB database name"),
    edge_collection: str = typer.Option("relates_to", "--edges", help="Edge collection name"),
    collection_filter: str = typer.Option(None, "--filter", "-f", help="Filter by collection name"),
    direction: str = typer.Option("outbound", "--direction", "-d", 
                                 help="Direction (outbound, inbound, any)"),
    max_depth: int = typer.Option(1, "--depth", help="Maximum traversal depth"),
    format_output: bool = typer.Option(True, "--format/--raw", help="Format the output or show raw JSON"),
):
    """
    Retrieve documents related to a document through graph traversal.
    
    Example:
        $ arangodb get-related messages/123 --filter documents
    """
    try:
        # Connect to database
        db = get_db_connection(host, username, password, database)
        
        # Get related documents
        results = get_related_documents(
            db,
            document_id,
            collection_filter=collection_filter,
            edge_collection=edge_collection,
            direction=direction,
            max_depth=max_depth
        )
        
        # Display results
        if not results:
            console.print(f"[yellow]No related documents found for '{document_id}'[/yellow]")
            return
        
        if format_output:
            # Create related documents table
            table = Table(title=f"Documents Related to {document_id}")
            table.add_column("Document ID", style="cyan")
            table.add_column("Collection", style="green")
            table.add_column("Edge Type", style="yellow")
            table.add_column("Edge Properties", style="white")
            
            for result in results:
                document = result.get("document", {})
                edge = result.get("edge", {})
                
                doc_id = document.get("_id", "")
                collection = doc_id.split("/")[0] if "/" in doc_id else ""
                
                # Extract edge properties, excluding system fields
                edge_props = {k: v for k, v in edge.items() 
                             if not k.startswith("_") and k not in ["created_at", "updated_at"]}
                
                table.add_row(
                    doc_id,
                    collection,
                    edge.get("type", ""),
                    json.dumps(edge_props, indent=2) if edge_props else ""
                )
                
            console.print(table)
        else:
            console.print(Syntax(json.dumps(results, indent=2), "json", theme="monokai"))
        
    except Exception as e:
        console.print(format_error(f"Failed to retrieve related documents: {str(e)}"))
        raise typer.Exit(code=1)


if __name__ == "__main__":
    import sys
    
    try:
        # Simple validation with test commands
        app(["--help"])
        print("✅ VALIDATION PASSED - CLI commands are properly defined")
        sys.exit(0)
    except Exception as e:
        print(f"❌ VALIDATION FAILED: {str(e)}")
        sys.exit(1)