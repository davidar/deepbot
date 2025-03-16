"""Command handlers for response management."""

from discord.ext import commands

from llm_streaming import LLMResponseHandler

Context = commands.Context[commands.Bot]


class ResponseCommands:
    """Handlers for response-related commands."""

    def __init__(self, llm_handler: LLMResponseHandler) -> None:
        """Initialize response command handlers.

        Args:
            llm_handler: The LLM response handler instance
        """
        self.llm_handler = llm_handler

    async def handle_shutup(self, ctx: Context) -> None:
        """Handle the shutup command.

        Args:
            ctx: The command context
        """
        channel_id = ctx.channel.id
        self.llm_handler.stop_responses(channel_id)
        await ctx.send("-# 🤫 Stopped all responses in this channel")
