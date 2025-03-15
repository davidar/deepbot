#!/usr/bin/env python3
"""CLI tool for indexing and searching Discord messages."""

import argparse
import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from llama_index.core.schema import NodeWithScore
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

from local_discord_index import LocalDiscordIndex
from message_store import MessageStore

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

logger = logging.getLogger("search_cli")
console = Console()


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Index and search Discord messages using semantic search"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Common arguments for Ollama configuration
    ollama_args = argparse.ArgumentParser(add_help=False)
    ollama_args.add_argument(
        "--model",
        default="nomic-embed-text",
        help="Ollama model to use for embeddings",
    )
    ollama_args.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="URL of the Ollama server",
    )

    # Index command
    index_parser = subparsers.add_parser(
        "index", help="Index messages from message store", parents=[ollama_args]
    )
    index_parser.add_argument(
        "--storage-path",
        default="./chroma_db",
        help="Path to store the vector database",
    )
    index_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for indexing",
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Search indexed messages", parents=[ollama_args]
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--storage-path",
        default="./chroma_db",
        help="Path to the vector database",
    )
    search_parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of results to return",
    )
    search_parser.add_argument("--channel", help="Filter by channel ID")
    search_parser.add_argument("--author", help="Filter by author name")

    return parser


def format_timestamp(timestamp_str: str) -> str:
    """Format ISO timestamp into a readable format."""
    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, AttributeError):
        return "Unknown time"


def display_search_results(nodes: list[NodeWithScore]) -> None:
    """Display search results in an IRC-like format."""
    if not nodes:
        console.print("[yellow]No messages found")
        return

    console.print("\n[bold blue]Search Results:[/]\n")

    for node in nodes:
        metadata = node.metadata or {}
        timestamp = format_timestamp(metadata.get("timestamp", ""))
        channel_id = metadata.get("channel_id", "unknown")
        author = metadata.get("author", "unknown")

        # Extract just the message content (skip the "Author:" prefix we added during indexing)
        text = node.text
        if "Message: " in text:
            text = text.split("Message: ", 1)[1].split("\n")[0]

        # Format attachments and embeds as metadata
        extras = []
        if metadata.get("has_attachments"):
            extras.append("📎 has attachments")
        if metadata.get("has_embeds"):
            extras.append("📌 has embeds")
        extra_info = f" ({', '.join(extras)})" if extras else ""

        # Build the message line
        header = Text()
        header.append(f"[{timestamp}] ", style="dim")
        header.append(f"#{channel_id} ", style="blue")
        header.append(f"<{author}> ", style="green")

        # Print with proper wrapping
        console.print(header, end="")
        console.print(text)
        if extras:
            console.print(f"  {extra_info}", style="dim")
        console.print()


def show_progress(processed: int, total: int) -> None:
    """Show indexing progress."""
    percentage = (processed / total) * 100 if total > 0 else 0
    console.print(f"[green]Progress: {percentage:.1f}% ({processed}/{total} messages)")


async def search_messages(
    query: str,
    storage_path: str,
    top_k: int,
    model: str,
    base_url: str,
    channel: Optional[str] = None,
    author: Optional[str] = None,
) -> None:
    """Search messages and display results."""
    message_store = MessageStore()
    index = LocalDiscordIndex(
        message_store, storage_path=storage_path, model_name=model, base_url=base_url
    )

    # Build filters
    filters: Dict[str, Any] = {}
    if channel:
        filters["channel_id"] = channel
    if author:
        filters["author"] = author

    # Perform search
    with console.status("[bold green]Searching messages..."):
        results = await index.search(query, top_k=top_k, **filters)

    display_search_results(results)


def index_messages(
    storage_path: str, batch_size: int, model: str, base_url: str
) -> None:
    """Index all messages from the message store."""
    message_store = MessageStore()
    index = LocalDiscordIndex(
        message_store, storage_path=storage_path, model_name=model, base_url=base_url
    )

    with console.status("[bold green]Indexing messages..."):
        index.index_messages(batch_size=batch_size, progress_callback=show_progress)


async def main() -> None:
    """Main entry point for the CLI."""
    parser = setup_argparse()
    args = parser.parse_args()

    if args.command == "index":
        index_messages(
            storage_path=args.storage_path,
            batch_size=args.batch_size,
            model=args.model,
            base_url=args.base_url,
        )
    elif args.command == "search":
        await search_messages(
            query=args.query,
            storage_path=args.storage_path,
            top_k=args.top_k,
            model=args.model,
            base_url=args.base_url,
            channel=args.channel,
            author=args.author,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    asyncio.run(main())
