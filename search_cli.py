#!/usr/bin/env python3
"""Command-line interface for searching messages."""

import argparse
import asyncio
import logging
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text

import config
from discord_types import StoredMessage
from message_indexer import MessageIndexer
from message_store import MessageStore
from utils.time_utils import parse_datetime

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed from INFO to DEBUG for more detailed logging
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)

# Silence noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

logger = logging.getLogger("deepbot.search_cli")
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
        default=config.EMBEDDING_MODEL_NAME,
        help="Ollama model to use for embeddings",
    )
    ollama_args.add_argument(
        "--base-url",
        default=config.API_URL,
        help="URL of the Ollama server",
    )

    # Index command
    index_parser = subparsers.add_parser(
        "index", help="Index messages from message store", parents=[ollama_args]
    )
    index_parser.add_argument(
        "--storage-path",
        default=config.SEARCH_INDEX_PATH,
        help="Path to store the vector database",
    )
    index_parser.add_argument(
        "--message-store-dir",
        default=config.MESSAGE_STORE_DIR,
        help="Path to the message store directory",
    )
    index_parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for indexing",
    )
    index_parser.add_argument(
        "--force",
        action="store_true",
        help="Force reindexing of all messages",
    )

    # Search command
    search_parser = subparsers.add_parser(
        "search", help="Search indexed messages", parents=[ollama_args]
    )
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument(
        "--storage-path",
        default=config.SEARCH_INDEX_PATH,
        help="Path to the vector database",
    )
    search_parser.add_argument(
        "--message-store-dir",
        default=config.MESSAGE_STORE_DIR,
        help="Path to the message store directory",
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
        dt = parse_datetime(timestamp_str)
        return dt.format("YYYY-MM-DD HH:mm:ss")
    except ValueError:
        return "Unknown time"


def display_search_results(messages: Dict[str, List[StoredMessage]]) -> None:
    """Display search results in an IRC-like format.

    Args:
        messages: Dictionary mapping channel IDs to lists of stored messages
    """
    if not messages:
        console.print("[yellow]No messages found")
        return

    console.print("\n[bold blue]Search Results:[/]\n")

    for channel_id, channel_messages in messages.items():
        for message in channel_messages:
            timestamp = format_timestamp(message.timestamp)
            author = message.author.name

            # Format attachments and embeds as metadata
            extras: list[str] = []
            if message.attachments:
                extras.append("📎 has attachments")
            if message.embeds:
                extras.append("📌 has embeds")
            extra_info = f" ({', '.join(extras)})" if extras else ""

            # Build the message line
            header = Text()
            header.append(f"[{timestamp}] ", style="dim")
            header.append(f"#{channel_id} ", style="blue")
            header.append(f"<{author}> ", style="green")

            # Print with proper wrapping
            console.print(header, end="")
            console.print(message.content)
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
    message_store_dir: str,
    top_k: int,
    model: str,
    base_url: str,
    channel: Optional[str] = None,
    author: Optional[str] = None,
) -> None:
    """Search messages and display results."""
    logger.debug(
        f"Initializing search with parameters: storage_path={storage_path}, message_store_dir={message_store_dir}, model={model}, base_url={base_url}"
    )

    # Create message indexer first
    indexer = MessageIndexer(
        storage_path=storage_path,
        model_name=model,
        base_url=base_url,
    )

    message_store = MessageStore(
        data_dir=message_store_dir,
        message_indexer=indexer,
    )

    # Build filters
    filters: Dict[str, Any] = {}
    if channel:
        filters["channel_id"] = channel
    if author:
        filters["author"] = author

    logger.debug(f"Applying search filters: {filters}")

    # Perform search
    with console.status("[bold green]Searching messages..."):
        logger.debug(f"Executing search query: '{query}' with top_k={top_k}")
        results = await message_store.search(query, top_k=top_k, **filters)
        logger.debug(f"Search returned {len(results)} results")

    display_search_results(results)


def index_messages(
    storage_path: str,
    message_store_dir: str,
    batch_size: int,
    model: str,
    base_url: str,
    force: bool = False,
) -> None:
    """Index all messages from the message store."""
    logger.debug(
        f"Initializing indexing with storage_path={storage_path}, message_store_dir={message_store_dir}, batch_size={batch_size}, model={model}"
    )

    with console.status("[bold green]Initializing message store..."):
        # Create message indexer first
        indexer = MessageIndexer(
            storage_path=storage_path,
            model_name=model,
            base_url=base_url,
        )

        message_store = MessageStore(
            data_dir=message_store_dir,
            message_indexer=indexer,
        )
        if not message_store.message_indexer:
            logger.warning("No message indexer configured")
            console.print("[yellow]No indexer configured - indexing skipped")
            return

    # Only force reindex if requested
    if force:
        with console.status("[bold green]Force reindexing all messages..."):
            try:
                logger.info("Starting force reindex of all messages")
                message_store.reindex_all_messages(progress_callback=show_progress)
                logger.info("Reindexing completed successfully")
                console.print("[green]Messages have been reindexed successfully!")
            except Exception as e:
                logger.error(f"Reindexing failed with error: {str(e)}", exc_info=True)
                console.print(f"[red]Error during indexing: {e}")
    else:
        logger.info(
            "Skipping force reindex - messages are indexed automatically when added"
        )
        console.print(
            "[green]Messages are indexed automatically when added to the store"
        )
        console.print("[yellow]Use --force to reindex all messages")


async def main() -> None:
    """Main entry point for the CLI."""
    parser = setup_argparse()
    args = parser.parse_args()

    if args.command == "index":
        index_messages(
            storage_path=args.storage_path,
            message_store_dir=args.message_store_dir,
            batch_size=args.batch_size,
            model=args.model,
            base_url=args.base_url,
            force=args.force,
        )
    elif args.command == "search":
        await search_messages(
            query=args.query,
            storage_path=args.storage_path,
            message_store_dir=args.message_store_dir,
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
