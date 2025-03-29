"""Command handlers for lorekeeper functionality."""

import logging
from typing import Optional

from discord.ext import commands

from llm_streaming import LLMResponseHandler
from lorekeeper import config as lore_config
from lorekeeper.lore_summary import LoreSummaryGenerator
from lorekeeper.vector_search import VectorSearch

Context = commands.Context[commands.Bot]

# Set up logging
logger = logging.getLogger("deepbot.command.lore_commands")


class LoreCommands:
    """Handlers for lorekeeper-related commands."""

    def __init__(
        self,
        llm_handler: LLMResponseHandler,
        qdrant_host: Optional[str] = None,
        qdrant_port: Optional[int] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialize lore commands.

        Args:
            llm_handler: The LLM response handler
            qdrant_host: The Qdrant server host (overrides config)
            qdrant_port: The Qdrant server port (overrides config)
            model: The Ollama model to use for summarization (overrides config)
        """
        # Get configuration values
        qdrant_config = lore_config.get_qdrant_config()
        lore_summary_config = lore_config.get_lore_summary_config()

        # Use provided values or defaults from config
        self.qdrant_host = qdrant_host or qdrant_config["host"]
        self.qdrant_port = qdrant_port or qdrant_config["port"]
        self.model = model or lore_summary_config["model"]

        # Initialize the vector search with config values
        self.vector_search = VectorSearch(
            qdrant_host=self.qdrant_host, qdrant_port=self.qdrant_port
        )

        # Create the summary generator using the bot's existing API client
        self.summary_generator = LoreSummaryGenerator(
            api_client=llm_handler.api_client, bot_user=llm_handler.bot_user
        )

        logger.info(f"Initialized lore commands with model {self.model}")
        logger.info(f"Using Qdrant at {self.qdrant_host}:{self.qdrant_port}")

    async def handle_lore(
        self,
        ctx: Context,
        *,
        query: str,
        limit: Optional[int] = None,
    ) -> None:
        """Handle the lore command.

        Args:
            ctx: The command context
            query: The search query
            limit: Maximum number of search results (optional, uses config default if not specified)
        """
        # Get the message object
        message = ctx.message

        # Use configured default if limit not provided
        limit = limit or lore_config.DEFAULT_SEARCH_LIMIT

        logger.info(f"Lore command received from user {ctx.author.name}: '{query}'")

        try:
            # Show that we're working on it
            async with ctx.typing():  # pyright: ignore
                # Add initial reaction to show we're working
                if ctx.bot.user:
                    await message.add_reaction("📇")

                # Retrieve guild_id for filtering if available
                guild_id = None
                if ctx.guild:
                    guild_id = str(ctx.guild.id)
                    logger.info(f"Guild ID for filtering: {guild_id}")
                else:
                    logger.info("No guild context available for filtering")

                # Perform search
                logger.info(f"Performing vector search for query '{query}'")
                results = self.vector_search.search(query, limit=limit)
                logger.info(f"Vector search returned {len(results)} raw results")

                # Filter by guild if in a guild
                # if guild_id and results:
                #     guild_filtered_results = [
                #         r for r in results if r.get("guild_id") == guild_id
                #     ]
                #     logger.info(
                #         f"Filtered results by guild: {len(guild_filtered_results)} results (from {len(results)} total)"
                #     )
                #     results = guild_filtered_results

                # Log some details about each result for debugging
                if results:
                    logger.info(f"Found {len(results)} results for query '{query}'")
                else:
                    logger.warning(f"No results found for query: '{query}'")

                if not results:
                    if ctx.bot.user:
                        await message.remove_reaction("📇", ctx.bot.user)
                        await message.add_reaction("❌")
                    await ctx.send("-# No lore found matching your query.")
                    return

                # Show that we found something
                if ctx.bot.user:
                    await message.remove_reaction("📇", ctx.bot.user)

                # Generate the summary using our integrated generator
                logger.info(f"Generating lore summary for {len(results)} results")
                await self.summary_generator.generate_summary(
                    results=results,
                    query=query,
                    channel=ctx.channel,
                    message=message,
                    model=self.model,
                )

        except Exception as e:
            logger.exception(f"Error accessing lore: {e}")
            if ctx.bot.user:
                try:
                    await message.remove_reaction("📇", ctx.bot.user)
                    await message.add_reaction("❌")
                except Exception:
                    pass  # Ignore reaction errors in error handler
            await ctx.send(f"-# An error occurred while accessing the lore: {e}")
