"""Command handlers for system prompt management."""

import logging
from typing import Optional

import discord
from discord.ext import commands

import config
import system_prompt

Context = commands.Context[commands.Bot]

# Set up logging
logger = logging.getLogger("deepbot.command.prompt_commands")


class PromptCommands:
    """Handlers for system prompt commands."""

    def __init__(self) -> None:
        """Initialize prompt command handlers."""
        pass

    @staticmethod
    async def handle_prompt(
        ctx: Context,
        action: Optional[str] = None,
        *,
        line: Optional[str] = None,
    ) -> None:
        """Handle the prompt command.

        Args:
            ctx: The command context
            action: The action to perform (add/remove/trim)
            line: The line to add or remove
        """
        if not action:
            # Display current prompt as a file attachment
            file = discord.File("system_prompt.txt")
            await ctx.send("-# Current System Prompt:", file=file)
            await ctx.send(
                "-# Use `prompt add <line>` to add a line, "
                "`prompt remove <line>` to remove a line, or "
                "`prompt trim` to trim to max length"
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
