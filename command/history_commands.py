"""Command handlers for conversation history."""

import io
import json
import logging

import discord
from discord.ext import commands

from context_builder import ContextBuilder
from message_history import MessageHistoryManager

Context = commands.Context[commands.Bot]

# Set up logging
logger = logging.getLogger("deepbot.command.history_commands")


class HistoryCommands:
    """Handlers for conversation history commands."""

    def __init__(
        self,
        message_history: MessageHistoryManager,
        context_builder: ContextBuilder,
    ) -> None:
        """Initialize history command handlers.

        Args:
            message_history: The message history manager instance
            context_builder: The context builder instance
        """
        self.message_history = message_history
        self.context_builder = context_builder

    async def handle_refresh(self, ctx: Context) -> None:
        """Handle the refresh command.

        Args:
            ctx: The command context
        """
        if not isinstance(ctx.channel, discord.TextChannel):
            await ctx.send(
                "-# This command can only be used in text channels, not in DMs"
            )
            return

        await ctx.send("-# Refreshing conversation history from channel messages...")

        try:
            # Clear existing history and re-initialize
            await self.message_history.initialize_channel(ctx.channel, refresh=True)

            history_count = self.message_history.get_history_length(ctx.channel.id)
            await ctx.send(
                f"-# Conversation history refreshed! Now tracking {history_count} messages"
            )
        except Exception as e:
            logger.error(f"Error refreshing history: {str(e)}")
            await ctx.send(f"-# Error refreshing history: {str(e)}")

    async def handle_raw(self, ctx: Context) -> None:
        """Handle the raw command.

        Args:
            ctx: The command context
        """
        channel_id = ctx.channel.id
        if not self.message_history.has_history(channel_id):
            await ctx.send("-# No conversation history found for this channel")
            return

        # Get the raw context that would be sent to the LLM
        messages = self.message_history.get_messages(channel_id)
        context = await self.context_builder.build_context(messages, ctx.channel)

        # Convert to JSON
        json_data = json.dumps(
            [msg.model_dump(exclude_none=True) for msg in context],
            indent=2,
        )

        # Create a discord.File object with the JSON data
        buffer = io.BytesIO(json_data.encode("utf-8"))
        file = discord.File(
            buffer,
            filename=f"conversation_history_{channel_id}.json",
        )

        await ctx.send("-# Here's the raw conversation history:", file=file)

    async def handle_wipe(self, ctx: Context) -> None:
        """Handle the wipe command.

        Args:
            ctx: The command context
        """
        if not isinstance(ctx.channel, discord.TextChannel):
            await ctx.send(
                "-# This command can only be used in text channels, not in DMs"
            )
            return

        # Set the reset timestamp to now
        self.context_builder.reset_history_from(ctx.channel.id, ctx.message.created_at)
        await ctx.send(
            "-# Conversation history has been wiped. Only messages from this point forward will be included in context."
        )

    async def handle_unwipe(self, ctx: Context) -> None:
        """Handle the unwipe command.

        Args:
            ctx: The command context
        """
        if not isinstance(ctx.channel, discord.TextChannel):
            await ctx.send(
                "-# This command can only be used in text channels, not in DMs"
            )
            return

        # Remove the reset timestamp
        self.context_builder.remove_reset(ctx.channel.id)
        await ctx.send(
            "-# Conversation history has been restored. All messages will now be included in context."
        )
