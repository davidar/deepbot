"""Command handling for DeepBot."""

import io
import json
import logging
from typing import Literal, Optional, cast, get_type_hints

import discord
from discord.ext import commands

import config
import example_conversation
import system_prompt
from context_builder import ContextBuilder
from llm_streaming import LLMResponseHandler
from message_history import MessageHistoryManager
from reactions import ReactionManager

Bot = commands.Bot
Context = commands.Context

# Set up logging
logger = logging.getLogger("deepbot.commands")


def setup_commands(
    bot: Bot,
    message_history: MessageHistoryManager,
    context_builder: ContextBuilder,
    llm_handler: LLMResponseHandler,
    reaction_manager: ReactionManager,
) -> None:
    """Set up bot commands.

    Args:
        bot: The Discord bot instance
        message_history: The message history manager instance
        context_builder: The context builder instance
        llm_handler: The LLM response handler instance
        reaction_manager: The reaction manager instance
    """

    @bot.command(name="options")
    async def options_command(
        ctx: Context[Bot],
        action: Optional[str] = None,
        option_name: Optional[str] = None,
        *,  # Force remaining args to be keyword-only
        value: Optional[str] = None,
    ) -> None:
        """View or modify model options.

        Usage:
        !options - View all options
        !options get <option> - View specific option
        !options set <option> <value> - Set option value
        """
        if not action:
            # Display all options
            file = discord.File("model_options.json")
            await ctx.send("-# Current Model Options:", file=file)
            await ctx.send(
                "-# Use `options get <option>` or `options set <option> <value>` to modify options"
            )
            return

        if action.lower() == "get" and option_name:
            # Get specific option
            opt_value = config.load_model_options().get(option_name)
            if opt_value is not None:
                await ctx.send(f"-# Option `{option_name}` is set to `{opt_value}`")
            else:
                await ctx.send(f"-# Option `{option_name}` not found")
            return

        if action.lower() == "set" and option_name and value is not None:
            try:
                # Try to convert value to float first
                float_value = float(value)
                # If it's a whole number, convert to int
                if float_value.is_integer():
                    float_value = int(float_value)

                # Get valid options and their types directly from ModelOptions
                option_types = get_type_hints(config.ModelOptions)

                # Validate option name
                if option_name not in option_types:
                    raise KeyError(f"Invalid option name: {option_name}")

                # Validate type
                expected_type = option_types[option_name]
                if not isinstance(float_value, expected_type):
                    raise TypeError(
                        f"Option {option_name} expects type {expected_type.__name__}, got {type(float_value).__name__}"
                    )

                # Update the option
                options = config.load_model_options()
                options[option_name] = float_value  # type: ignore[literal-required]
                config.save_model_options(options)

                await ctx.send(f"-# Updated option `{option_name}` to `{float_value}`")
            except ValueError:
                await ctx.send(f"-# Invalid value, please provide a number")
            except KeyError as e:
                await ctx.send(f"-# {str(e)}")
            except TypeError as e:
                await ctx.send(f"-# {str(e)}")
            return

        await ctx.send(
            "-# Invalid command, use `options`, `options get <option>`, or `options set <option> <value>`"
        )

    @bot.command(name="refresh")
    async def refresh_history(ctx: Context[Bot]) -> None:
        """Refresh the conversation history by fetching recent messages from the channel."""
        if not isinstance(ctx.channel, discord.TextChannel):
            await ctx.send(
                "-# This command can only be used in text channels, not in DMs"
            )
            return

        await ctx.send("-# Refreshing conversation history from channel messages...")

        try:
            # Clear existing history and re-initialize
            await message_history.initialize_channel(ctx.channel, refresh=True)

            history_count = message_history.get_history_length(ctx.channel.id)
            await ctx.send(
                f"-# Conversation history refreshed! Now tracking {history_count} messages"
            )
        except Exception as e:
            logger.error(f"Error refreshing history: {str(e)}")
            await ctx.send(f"-# Error refreshing history: {str(e)}")

    @bot.command(name="raw")
    async def raw_history(ctx: Context[Bot]) -> None:
        """Display the raw conversation history for debugging."""
        channel_id = ctx.channel.id
        if not message_history.has_history(channel_id):
            await ctx.send("-# No conversation history found for this channel")
            return

        # Get the raw context that would be sent to the LLM
        messages = message_history.get_messages(channel_id)
        context = context_builder.build_context(messages, ctx.channel)

        # Convert to JSON
        json_data = json.dumps(
            [
                {
                    "role": msg.role,
                    "content": msg.content,
                }
                for msg in context
            ],
            indent=2,
        )

        # Create a discord.File object with the JSON data
        buffer = io.BytesIO(json_data.encode("utf-8"))
        file = discord.File(
            buffer,
            filename=f"conversation_history_{channel_id}.json",
        )

        await ctx.send("-# Here's the raw conversation history:", file=file)

    @bot.command(name="prompt")
    async def prompt_command(
        ctx: Context[Bot],
        action: Optional[str] = None,
        *,
        line: Optional[str] = None,
    ) -> None:
        """Manage the system prompt."""

        if not action:
            # Display current prompt as a file attachment
            file = discord.File("system_prompt.txt")
            await ctx.send("-# Current System Prompt:", file=file)
            await ctx.send(
                "-# Use `prompt add <line>` to add a line, `prompt remove <line>` to remove a line, or `prompt trim` to trim to max length"
            )
            return

        if action.lower() == "add" and line:
            # Add a new line and get any removed lines from trimming
            lines, removed_lines = system_prompt.add_line(line)

            logger.info(f"Added line to prompt: {line}")
            logger.info(f"Current line count: {len(lines)}")
            logger.info(f"Removed lines from add operation: {removed_lines}")

            message = [f"-# Added line to system prompt: `{line}`"]

            # If any lines were removed during trimming, show them
            if removed_lines:
                logger.info(f"Displaying {len(removed_lines)} removed lines to user")
                for line in removed_lines:
                    message.append(
                        f"-# Removed random line from system prompt: `{line}`"
                    )
            else:
                logger.info("No lines were removed during add operation")

            message.append(f"-# Updated prompt now has {len(lines)} lines")
            await ctx.send("\n".join(message))

        elif action.lower() == "remove" and line:
            # Remove a line
            original_lines = system_prompt.load_system_prompt()
            if line not in original_lines:
                await ctx.send(f"-# Line not found in system prompt: `{line}`")
                return

            lines = system_prompt.remove_line(line)
            message = [
                f"-# Removed line from system prompt: `{line}`",
                f"-# Updated prompt now has {len(lines)} lines",
            ]
            await ctx.send("\n".join(message))

        elif action.lower() == "trim":
            # Trim the prompt to max length
            max_lines = config.load_model_options()["max_prompt_lines"]
            lines = system_prompt.load_system_prompt()
            if len(lines) <= max_lines:
                await ctx.send(
                    f"-# Prompt is already within limit ({len(lines)} lines)"
                )
                return

            lines, removed_lines = system_prompt.trim_prompt(max_lines)
            message = [f"-# Trimmed prompt to {len(lines)} lines"]
            for line in removed_lines:
                message.append(f"-# Removed random line from system prompt: `{line}`")
            await ctx.send("\n".join(message))

        else:
            await ctx.send(
                "-# Invalid command, use `prompt`, `prompt add <line>`, `prompt remove <line>`, or `prompt trim`"
            )

    @bot.command(name="example")
    async def example_command(
        ctx: Context[Bot],
        action: Optional[str] = None,
        *,
        content: Optional[str] = None,
    ) -> None:
        """Manage the example conversation.

        Usage:
        !example - View current example conversation
        !example add <role> <content> - Add a new message
        !example remove <index> - Remove a message by index
        !example edit <index> [role] [content] - Edit a message
        """
        if not action:
            # Display current example conversation as a file attachment
            file = discord.File("example_conversation.json")
            await ctx.send("-# Current Example Conversation:", file=file)
            await ctx.send(
                "-# Use `example add <role> <content>`, `example remove <index>`, or `example edit <index> [role] [content]` to modify"
            )
            return

        if action.lower() == "add" and content:
            try:
                # Split content into role and message content
                role_str, *content_parts = content.split(maxsplit=1)
                if not content_parts:
                    await ctx.send("-# Please provide both role and content")
                    return
                message_content = content_parts[0]

                # Validate role
                if role_str not in ["user", "assistant", "system", "tool"]:
                    await ctx.send(
                        "-# Role must be one of: user, assistant, system, tool"
                    )
                    return

                messages = example_conversation.add_message(
                    cast(Literal["user", "assistant", "system", "tool"], role_str),
                    message_content,
                )
                await ctx.send(
                    f"-# Added new message to example conversation. Total messages: {len(messages)}"
                )
                return

            except Exception as e:
                logger.error(f"Error adding example message: {e}")
                await ctx.send(f"-# Error adding message: {str(e)}")
                return

        if action.lower() == "remove" and content:
            try:
                index = int(content)
                messages, removed = example_conversation.remove_message(index)
                if removed:
                    await ctx.send(
                        f"-# Removed message at index {index}. Total messages: {len(messages)}"
                    )
                else:
                    await ctx.send(f"-# No message found at index {index}")
                return

            except ValueError:
                await ctx.send("-# Please provide a valid index number")
                return
            except Exception as e:
                logger.error(f"Error removing example message: {e}")
                await ctx.send(f"-# Error removing message: {str(e)}")
                return

        if action.lower() == "edit" and content:
            try:
                # Split content into index and optional role/content
                parts = content.split(maxsplit=2)
                if len(parts) < 1:
                    await ctx.send("-# Please provide an index")
                    return

                index = int(parts[0])
                role: Optional[Literal["user", "assistant", "system", "tool"]] = None
                message_content = None

                if len(parts) > 1:
                    role_str = parts[1]
                    if role_str not in ["user", "assistant", "system", "tool"]:
                        await ctx.send(
                            "-# Role must be one of: user, assistant, system, tool"
                        )
                        return
                    role = cast(
                        Literal["user", "assistant", "system", "tool"], role_str
                    )

                if len(parts) > 2:
                    message_content = parts[2]

                messages, edited = example_conversation.edit_message(
                    index, role, message_content
                )
                if edited:
                    await ctx.send(f"-# Edited message at index {index}")
                else:
                    await ctx.send(f"-# No message found at index {index}")
                return

            except ValueError:
                await ctx.send("-# Please provide a valid index number")
                return
            except Exception as e:
                logger.error(f"Error editing example message: {e}")
                await ctx.send(f"-# Error editing message: {str(e)}")
                return

        await ctx.send(
            "-# Invalid command, use `example`, `example add <role> <content>`, `example remove <index>`, or `example edit <index> [role] [content]`"
        )

    @bot.command(name="shutup")
    async def shutup_command(ctx: Context[Bot]) -> None:
        """Stop all responses in the current channel."""
        channel_id = ctx.channel.id
        llm_handler.stop_responses(channel_id)
        await ctx.send("-# 🤫 Stopped all responses in this channel")

    @bot.command(name="reactions")
    async def reactions_command(ctx: Context[Bot], scope: str = "channel") -> None:
        """Display reaction statistics for the bot's messages.

        Args:
            scope: Either "channel" (default) or "global" to show stats across all channels
        """
        if scope.lower() not in ["channel", "global"]:
            await ctx.send('-# Invalid scope. Use "channel" or "global"')
            return

        if scope.lower() == "global":
            channel_scores = reaction_manager.get_global_stats()
            if not channel_scores:
                await ctx.send("-# No reaction data available yet")
                return

            message = ["-# Global reaction statistics:"]
            summary = reaction_manager.format_global_summary(channel_scores)
            for line in summary.split("\n"):
                if line.strip():
                    message.append(f"-# {line}")
            await ctx.send("\n".join(message))

        else:  # channel scope
            channel_id = ctx.channel.id
            message_reactions = reaction_manager.get_channel_stats(channel_id)

            if not message_reactions:
                await ctx.send("-# No reaction data available for this channel yet")
                return

            # Create a summary of reactions
            channel_name = (
                ctx.channel.name
                if isinstance(ctx.channel, discord.TextChannel)
                else "DM"
            )
            message = [f"-# Reaction statistics for #{channel_name}"]
            channel_summary = reaction_manager.format_reaction_summary(
                message_reactions
            )
            if channel_summary:
                for line in channel_summary.split("\n"):
                    if line.strip():
                        message.append(f"-# {line}")
            else:
                message.append("-# No reactions yet.")
            await ctx.send("\n".join(message))

    @bot.command(name="wipe")
    async def wipe_command(ctx: Context[Bot]) -> None:
        """Wipe the conversation history to only include messages from this point forward."""
        if not isinstance(ctx.channel, discord.TextChannel):
            await ctx.send(
                "-# This command can only be used in text channels, not in DMs"
            )
            return

        # Set the reset timestamp to now
        context_builder.reset_history_from(ctx.channel.id, ctx.message.created_at)
        await ctx.send(
            "-# Conversation history has been wiped. Only messages from this point forward will be included in context."
        )

    @bot.command(name="unwipe")
    async def unwipe_command(ctx: Context[Bot]) -> None:
        """Restore access to all conversation history by removing the wipe point."""
        if not isinstance(ctx.channel, discord.TextChannel):
            await ctx.send(
                "-# This command can only be used in text channels, not in DMs"
            )
            return

        # Remove the reset timestamp
        context_builder.remove_reset(ctx.channel.id)
        await ctx.send(
            "-# Conversation history has been restored. All messages will now be included in context."
        )

    # Add custom command error handler
    @bot.event
    async def on_command_error(ctx: Context[Bot], error: Exception) -> None:
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
