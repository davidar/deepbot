"""Command handling for DeepBot."""

import asyncio
import io
import json
import logging
from typing import Optional

import discord
from discord.ext import commands

import config
import system_prompt
from conversation import ConversationManager
from reactions import ReactionManager
from utils import get_server_name

Bot = commands.Bot
Context = commands.Context

# Set up logging
logger = logging.getLogger("deepbot.commands")


def setup_commands(
    bot: Bot,
    conversation_manager: ConversationManager,
    reaction_manager: ReactionManager,
) -> None:
    """Set up bot commands.

    Args:
        bot: The Discord bot instance
        conversation_manager: The conversation manager instance
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
            opt_value = config.get_option(option_name)
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

                # Update the option
                config.set_option(option_name, float_value)
                await ctx.send(f"-# Updated option `{option_name}` to `{float_value}`")
            except ValueError:
                await ctx.send(f"-# Invalid value, please provide a number")
            except KeyError:
                await ctx.send(f"-# Invalid option name: `{option_name}`")
            return

        await ctx.send(
            "-# Invalid command, use `options`, `options get <option>`, or `options set <option> <value>`"
        )

    @bot.command(name="reset")
    async def reset_history(ctx: Context[Bot]) -> None:
        """Reset the conversation history for the current channel."""
        channel_id = ctx.channel.id
        if conversation_manager.has_history(channel_id):
            conversation_manager.clear_history(channel_id)
            await ctx.send("-# Conversation history has been reset")
        else:
            await ctx.send("-# No conversation history to reset")

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
            conversation_manager.clear_history(ctx.channel.id)
            await conversation_manager.initialize_channel_history(ctx.channel)

            history_count = conversation_manager.get_history_length(ctx.channel.id)
            await ctx.send(
                f"-# Conversation history refreshed! Now tracking {history_count-1} messages (plus system message)"
            )
        except Exception as e:
            logger.error(f"Error refreshing history: {str(e)}")
            await ctx.send(f"-# Error refreshing history: {str(e)}")

    @bot.command(name="raw")
    async def raw_history(ctx: Context[Bot]) -> None:
        """Display the raw conversation history for debugging."""
        channel_id = ctx.channel.id
        if not conversation_manager.has_history(channel_id):
            await ctx.send("-# No conversation history found for this channel")
            return

        history = conversation_manager.get_history(channel_id)
        json_data = json.dumps(history, indent=2)

        # Create a discord.File object with the JSON data
        buffer = io.BytesIO(json_data.encode("utf-8"))
        file = discord.File(
            buffer,
            filename=f"conversation_history_{channel_id}.json",
        )

        await ctx.send("-# Here's the raw conversation history:", file=file)

    @bot.command(name="wipe")
    async def wipe_memory(ctx: Context[Bot]) -> None:
        """Temporarily wipe the bot's memory while keeping the system message."""
        channel_id = ctx.channel.id
        if conversation_manager.has_history(channel_id):
            conversation_manager.reset_to_initial(ctx.channel)
            await ctx.send(
                "-# 🧹 Memory wiped! I'm starting fresh, but I'll keep my personality intact!"
            )
        else:
            await ctx.send("-# No conversation history to wipe")

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

            # Update all channels with new system prompt
            logger.info("Updating channel prompts")
            for channel in bot.get_all_channels():
                if conversation_manager.has_history(channel.id):
                    new_prompt = system_prompt.get_system_prompt(
                        get_server_name(channel)
                    )
                    conversation_manager.update_system_prompt(channel.id, new_prompt)

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

            # Update all channels with new system prompt
            for channel in bot.get_all_channels():
                if conversation_manager.has_history(channel.id):
                    new_prompt = system_prompt.get_system_prompt(
                        get_server_name(channel)
                    )
                    conversation_manager.update_system_prompt(channel.id, new_prompt)

        elif action.lower() == "trim":
            # Trim the prompt to max length
            max_lines = int(config.get_option("max_prompt_lines", 60))
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

            # Update all channels with new system prompt
            for channel in bot.get_all_channels():
                if conversation_manager.has_history(channel.id):
                    new_prompt = system_prompt.get_system_prompt(
                        get_server_name(channel)
                    )
                    conversation_manager.update_system_prompt(channel.id, new_prompt)

        else:
            await ctx.send(
                "-# Invalid command, use `prompt`, `prompt add <line>`, `prompt remove <line>`, or `prompt trim`"
            )

    @bot.command(name="shutup")
    async def shutup_command(ctx: Context[Bot]) -> None:
        """Stop all responses in the current channel."""
        channel_id = ctx.channel.id

        # Cancel the response task if it exists
        task = conversation_manager.response_tasks.get(channel_id)
        if task is not None and not task.done():
            # Mark this task as shut up
            conversation_manager.shutup_tasks.add(task)
            task.cancel()
            conversation_manager.response_tasks[channel_id] = None

        # Clear the response queue
        while not conversation_manager.response_queues[channel_id].empty():
            try:
                conversation_manager.response_queues[channel_id].get_nowait()
            except asyncio.QueueEmpty:
                break

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
            summary = reaction_manager.format_reaction_summary(message_reactions)
            for line in summary.split("\n"):
                if line.strip():
                    message.append(f"-# {line}")
            await ctx.send("\n".join(message))

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
