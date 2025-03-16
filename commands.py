"""Command handling for DeepBot."""

import logging
from typing import Optional

import discord
from discord.ext import commands

from command import (
    ExampleCommands,
    HistoryCommands,
    OptionCommands,
    PromptCommands,
    ReactionCommands,
    ResponseCommands,
    SearchCommands,
    UserCommands,
)
from context_builder import ContextBuilder
from llm_streaming import LLMResponseHandler
from message_history import MessageHistoryManager
from message_store import MessageStore
from reactions import ReactionManager
from user_management import UserManager

Context = commands.Context[commands.Bot]

# Set up logging
logger = logging.getLogger("deepbot.commands")


def _setup_option_commands(bot: commands.Bot, option_commands: OptionCommands) -> None:
    """Set up option-related commands.

    Args:
        bot: The Discord bot instance
        option_commands: The option command handler instance
    """

    @bot.command(name="options")
    async def options_command(
        ctx: Context,
        action: Optional[str] = None,
        option_name: Optional[str] = None,
        *,
        value: Optional[str] = None,
    ) -> None:
        """View or modify model options."""
        await option_commands.handle_options(ctx, action, option_name, value=value)


def _setup_history_commands(
    bot: commands.Bot, history_commands: HistoryCommands
) -> None:
    """Set up history-related commands.

    Args:
        bot: The Discord bot instance
        history_commands: The history command handler instance
    """

    @bot.command(name="refresh")
    async def refresh_history(ctx: Context) -> None:
        """Refresh the conversation history by fetching recent messages from the channel."""
        await history_commands.handle_refresh(ctx)

    @bot.command(name="raw")
    async def raw_history(ctx: Context) -> None:
        """Display the raw conversation history for debugging."""
        await history_commands.handle_raw(ctx)

    @bot.command(name="wipe")
    async def wipe_command(ctx: Context) -> None:
        """Wipe the conversation history to only include messages from this point forward."""
        await history_commands.handle_wipe(ctx)

    @bot.command(name="unwipe")
    async def unwipe_command(ctx: Context) -> None:
        """Restore access to all conversation history by removing the wipe point."""
        await history_commands.handle_unwipe(ctx)


def _setup_prompt_commands(bot: commands.Bot, prompt_commands: PromptCommands) -> None:
    """Set up prompt-related commands.

    Args:
        bot: The Discord bot instance
        prompt_commands: The prompt command handler instance
    """

    @bot.command(name="prompt")
    async def prompt_command(
        ctx: Context,
        action: Optional[str] = None,
        *,
        line: Optional[str] = None,
    ) -> None:
        """Manage the system prompt."""
        await prompt_commands.handle_prompt(ctx, action, line=line)


def _setup_example_commands(
    bot: commands.Bot, example_commands: ExampleCommands
) -> None:
    """Set up example conversation commands.

    Args:
        bot: The Discord bot instance
        example_commands: The example command handler instance
    """

    @bot.command(name="example")
    async def example_command(
        ctx: Context,
        action: Optional[str] = None,
        *,
        content: Optional[str] = None,
    ) -> None:
        """Manage the example conversation."""
        await example_commands.handle_example(ctx, action, content=content)


def _setup_response_commands(
    bot: commands.Bot, response_commands: ResponseCommands
) -> None:
    """Set up response-related commands.

    Args:
        bot: The Discord bot instance
        response_commands: The response command handler instance
    """

    @bot.command(name="shutup")
    async def shutup_command(ctx: Context) -> None:
        """Stop all responses in the current channel."""
        await response_commands.handle_shutup(ctx)


def _setup_reaction_commands(
    bot: commands.Bot, reaction_commands: ReactionCommands
) -> None:
    """Set up reaction-related commands.

    Args:
        bot: The Discord bot instance
        reaction_commands: The reaction command handler instance
    """

    @bot.command(name="reactions")
    async def reactions_command(ctx: Context, scope: str = "channel") -> None:
        """Display reaction statistics for the bot's messages."""
        await reaction_commands.handle_reactions(ctx, scope)


def _setup_user_commands(bot: commands.Bot, user_commands: UserCommands) -> None:
    """Set up user management commands.

    Args:
        bot: The Discord bot instance
        user_commands: The user command handler instance
    """

    @bot.command(name="ignore")
    async def ignore_command(ctx: Context, member: discord.Member) -> None:
        """Ignore messages from a user."""
        await user_commands.handle_ignore(ctx, member)

    @bot.command(name="unignore")
    async def unignore_command(ctx: Context, member: discord.Member) -> None:
        """Stop ignoring messages from a user."""
        await user_commands.handle_unignore(ctx, member)

    @bot.command(name="limit")
    async def limit_command(
        ctx: Context,
        member: discord.Member,
        consecutive_limit: Optional[int] = None,
    ) -> None:
        """Set a consecutive message limit for a user."""
        await user_commands.handle_limit(ctx, member, consecutive_limit)

    @bot.command(name="restrictions")
    async def restrictions_command(ctx: Context, member: discord.Member) -> None:
        """View current restrictions for a user."""
        await user_commands.handle_restrictions(ctx, member)


def _setup_error_handler(bot: commands.Bot) -> None:
    """Set up the command error handler.

    Args:
        bot: The Discord bot instance
    """

    @bot.event
    async def on_command_error(ctx: Context, error: Exception) -> None:
        """Handle command errors."""
        if isinstance(error, commands.CommandNotFound):
            # This is handled in on_message, so we can ignore it here
            pass
        elif isinstance(error, commands.MissingRequiredArgument):
            await ctx.send(f"-# Error: Missing required argument: {error.param}")
        elif isinstance(error, commands.BadArgument):
            await ctx.send(f"-# Error: Bad argument: {error}")
        else:
            logger.error(f"Command error: {error}")
            await ctx.send(f"-# Error executing command: {error}")


def _setup_search_commands(bot: commands.Bot, search_commands: SearchCommands) -> None:
    """Set up search-related commands.

    Args:
        bot: The Discord bot instance
        search_commands: The search command handler instance
    """

    @bot.command(name="search")
    async def search_command(
        ctx: Context,
        *,
        query: str,
        channel: Optional[discord.TextChannel] = None,
        author: Optional[discord.Member] = None,
        limit: int = 7,
    ) -> None:
        """Search for messages using semantic search.

        Args:
            query: The search query
            channel: Optional channel to filter results (mention the channel)
            author: Optional author to filter results (mention the user)
            limit: Maximum number of results to return
        """
        await search_commands.handle_search(ctx, query, channel, author, limit)


def setup_commands(
    bot: commands.Bot,
    message_history: MessageHistoryManager,
    context_builder: ContextBuilder,
    llm_handler: LLMResponseHandler,
    reaction_manager: ReactionManager,
    user_manager: UserManager,
    message_store: MessageStore,
) -> None:
    """Set up bot commands.

    Args:
        bot: The Discord bot instance
        message_history: The message history manager instance
        context_builder: The context builder instance
        llm_handler: The LLM response handler instance
        reaction_manager: The reaction manager instance
        user_manager: The user manager instance
        message_store: The message store instance
    """
    # Initialize command handlers
    option_commands = OptionCommands()
    history_commands = HistoryCommands(message_history, context_builder)
    user_commands = UserCommands(user_manager)
    prompt_commands = PromptCommands()
    example_commands = ExampleCommands()
    reaction_commands = ReactionCommands(reaction_manager)
    response_commands = ResponseCommands(llm_handler)
    search_commands = SearchCommands(message_store)

    # Set up commands by category
    _setup_option_commands(bot, option_commands)
    _setup_history_commands(bot, history_commands)
    _setup_prompt_commands(bot, prompt_commands)
    _setup_example_commands(bot, example_commands)
    _setup_response_commands(bot, response_commands)
    _setup_reaction_commands(bot, reaction_commands)
    _setup_user_commands(bot, user_commands)
    _setup_search_commands(bot, search_commands)
    _setup_error_handler(bot)
