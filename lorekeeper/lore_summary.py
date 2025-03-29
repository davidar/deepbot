"""Lore summary generator using Ollama."""

import asyncio
import logging
from typing import Any, Dict, List

import ollama
from discord import ClientUser, Message, abc

from config import get_ollama_options

from . import config
from .conversation_formatter import format_lore_context

# Set up logging
logger = logging.getLogger("deepbot.lorekeeper")


class LoreSummaryGenerator:
    """Generates summaries of Discord conversations using Ollama."""

    def __init__(self, api_client: ollama.AsyncClient, bot_user: ClientUser) -> None:
        """Initialize the lore summary generator.

        Args:
            api_client: Ollama API client for text generation
            bot_user: The bot's Discord user
        """
        self.api_client = api_client
        self.bot_user = bot_user
        logger.info("Initialized LoreSummaryGenerator")

    async def generate_summary(
        self,
        results: List[Dict[str, Any]],
        query: str,
        channel: abc.Messageable,
        message: Message,
        model: str = None,
    ) -> None:
        """Generate a summary of search results and send to Discord.

        Args:
            results: The search results
            query: The original search query
            channel: The Discord channel to send the response to
            message: The original Discord message with the query
            model: The Ollama model to use (overrides config)
        """
        # Get configuration
        lore_config = config.get_lore_summary_config()
        model = model or lore_config["model"]

        logger.info(
            f"Starting summary generation with {len(results)} results using model {model}"
        )

        if not results:
            logger.warning("No results provided for summary generation")
            await channel.send("-# No lore records were found to ponder upon.")
            return

        # Format search results for the context
        context_text = format_lore_context(results)

        # Check the context size
        context_size = len(context_text)
        logger.info(f"Generated context of size {context_size} characters")

        if context_size < 10:
            logger.error(
                "Context size is suspiciously small, may indicate formatting issues"
            )

        # Prepare system prompt for the lore keeper character
        system_prompt = lore_config["system_template"].format(context=context_text)

        # Format messages for Ollama
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]

        reply = None
        try:
            # Show thinking reaction
            await message.add_reaction("📜")

            # Start a reply that will be updated with streaming content
            reply = await message.reply("-# The Keeper of Lore speaks...", silent=True)

            # Keep track of the accumulated content
            accumulated_content = ""

            # Stream the response, updating Discord message every few chunks
            update_frequency = 3  # Update message every 3 chunks
            chunk_counter = 0
            last_update_length = 0

            logger.info(f"Sending request to Ollama with model {model}")

            # Use streaming to show tokens as they're generated
            try:
                stream_response = await self.api_client.chat(
                    model=model,
                    messages=messages,
                    stream=True,
                    options=get_ollama_options(),
                )

                async for chunk in stream_response:
                    # Get new content
                    new_content = chunk["message"]["content"]
                    accumulated_content += new_content
                    chunk_counter += 1

                    # Update the Discord message periodically
                    if (
                        chunk_counter % update_frequency == 0
                        and len(accumulated_content) > last_update_length
                    ):
                        update_text = accumulated_content
                        # Truncate if too long for Discord
                        if len(update_text) > 1900:
                            update_text = update_text[:1900] + "..."

                        await reply.edit(content=update_text)
                        last_update_length = len(accumulated_content)

                        # Small delay to avoid rate limits
                        await asyncio.sleep(0.5)

            except Exception as stream_error:
                logger.exception(f"Error during streaming: {stream_error}")
                # Try fallback to non-streaming
                logger.info("Falling back to non-streaming response")
                try:
                    response = await self.api_client.chat(
                        model=model,
                        messages=messages,
                        stream=False,
                    )
                    accumulated_content = response.message.content
                    logger.info("Successfully got non-streaming response")
                except Exception as fallback_error:
                    logger.exception(f"Fallback also failed: {fallback_error}")
                    raise

            # Final update with complete text
            if accumulated_content and len(accumulated_content) > last_update_length:
                # Truncate if too long for Discord
                if len(accumulated_content) > 1900:
                    accumulated_content = accumulated_content[:1900] + "..."

                await reply.edit(content=accumulated_content)

            # Change reaction to indicate completion
            await message.remove_reaction("📜", self.bot_user)
            # await message.add_reaction("📚")

            logger.info(f"Successfully generated lore summary for query: '{query}'")

        except Exception as e:
            error_msg = f"Error generating lore summary: {str(e)}"
            logger.exception(error_msg)
            await message.remove_reaction("📜", self.bot_user)
            await message.add_reaction("❌")

            # Try to update the reply with the error
            try:
                if reply is not None:
                    await reply.edit(content=f"-# {error_msg}")
                else:
                    await message.reply(f"-# {error_msg}")
            except Exception as reply_error:
                logger.error(f"Error updating reply: {reply_error}")
                await message.reply(f"-# {error_msg}")
