"""Command handlers for search functionality."""

import logging
from typing import Any, Dict, Optional

import discord
from discord.ext import commands

from message_store import MessageStore
from utils import format_search_results

Context = commands.Context[commands.Bot]

# Set up logging
logger = logging.getLogger("deepbot.command.search_commands")


class SearchCommands:
    """Handlers for search-related commands."""

    def __init__(
        self,
        message_store: MessageStore,
    ) -> None:
        """Initialize search commands.

        Args:
            message_store: The message store to search
        """
        self.message_store = message_store

    async def handle_search(
        self,
        ctx: Context,
        query: str,
        channel: Optional[discord.TextChannel],
        author: Optional[discord.Member],
        limit: int,
    ) -> None:
        """Handle the search command.

        Args:
            ctx: The command context
            query: The search query
            channel: Optional channel to filter results
            author: Optional author to filter results
            limit: Maximum number of results to return
        """
        # Build filters
        filters: Dict[str, Any] = {}
        if channel:
            filters["channel_id"] = str(channel.id)
        if author:
            filters["author"] = author.name

        try:
            # Perform search
            async with ctx.typing():  # pyright: ignore
                results = await self.message_store.search(query, top_k=limit, **filters)

            if not results:
                await ctx.send("-# No messages found matching your query.")
                return

            # Format results with context
            message_groups = format_search_results(results, self.message_store, ctx.bot)

            # Send each group of messages
            for group in message_groups:
                # Remove trailing blank line if it exists
                if group and not group[-1]:
                    group.pop()
                await ctx.send("\n".join(group))

        except Exception as e:
            logger.error(f"Error performing search: {e}")
            await ctx.send(f"-# An error occurred while searching: {e}")
