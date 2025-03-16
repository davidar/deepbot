"""Command handlers for example conversation management."""

import logging
from typing import List, Optional

from discord.ext import commands

import example_conversation

Context = commands.Context[commands.Bot]

# Set up logging
logger = logging.getLogger("deepbot.command.example_commands")


class ExampleCommands:
    """Handlers for example conversation commands."""

    def __init__(self) -> None:
        """Initialize example command handlers."""
        pass

    @staticmethod
    async def handle_example(
        ctx: Context,
        action: Optional[str] = None,
        *,
        content: Optional[str] = None,
    ) -> None:
        """Handle the example command.

        Args:
            ctx: The command context
            action: The action to perform (add/remove/edit)
            content: The content for the action
        """
        if not action:
            await ExampleCommands._display_examples(ctx)
            return

        if action.lower() == "add" and content:
            await ExampleCommands._add_example(ctx, content)
            return

        if action.lower() == "remove" and content:
            await ExampleCommands._remove_example(ctx, content)
            return

        if action.lower() == "edit" and content:
            await ExampleCommands._edit_example(ctx, content)
            return

        await ctx.send(
            "-# Invalid command, use:\n"
            "-# `example` - List all examples\n"
            "-# `example add <user_msg> | <bot_msg>` - Add a new example\n"
            "-# `example remove <number>` - Remove an example\n"
            "-# `example edit <number> <user_msg> | <bot_msg>` - Edit an example"
        )

    @staticmethod
    async def _display_examples(ctx: Context) -> None:
        """Display the current example conversation.

        Args:
            ctx: The command context
        """
        pairs = example_conversation.load_pairs()
        if not pairs:
            await ctx.send("-# No example conversation pairs yet")
            return

        messages = ["-# Current Example Conversation:"]
        for i, pair in enumerate(pairs, 1):
            messages.append(f"-# {i}. User: {pair.user} | Bot: {pair.assistant}")
        messages.append(
            "-# Use `example add <user_msg> | <bot_msg>`, "
            "`example remove <number>`, or "
            "`example edit <number> <user_msg> | <bot_msg>` to modify"
        )

        # Split into chunks if too long
        msg = "\n".join(messages)
        if len(msg) > 1900:  # Discord message length limit safety margin
            chunks: List[str] = []
            current_chunk: List[str] = []
            for line in messages:
                if len("\n".join(current_chunk + [line])) > 1900:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = [line]
                else:
                    current_chunk.append(line)
            if current_chunk:
                chunks.append("\n".join(current_chunk))
            for chunk in chunks:
                await ctx.send(chunk)
        else:
            await ctx.send(msg)

    @staticmethod
    async def _add_example(ctx: Context, content: str) -> None:
        """Add a new example conversation pair.

        Args:
            ctx: The command context
            content: The content containing user and bot messages
        """
        try:
            # Split content into user and assistant messages using | as delimiter
            parts = content.split("|", 1)
            if len(parts) != 2:
                await ctx.send(
                    "-# Please provide both user and assistant messages separated by |"
                )
                return

            user_msg = parts[0].strip()
            bot_msg = parts[1].strip()

            if not user_msg or not bot_msg:
                await ctx.send("-# Both user and bot messages must not be empty")
                return

            pairs = example_conversation.add_pair(user_msg, bot_msg)
            await ctx.send(f"-# Added new message pair #{len(pairs)}:")
            await ctx.send(f"-# User: {user_msg}\n-# Bot: {bot_msg}")

        except Exception as e:
            logger.error(f"Error adding example message pair: {e}")
            await ctx.send(f"-# Error adding message pair: {str(e)}")

    @staticmethod
    async def _remove_example(ctx: Context, content: str) -> None:
        """Remove an example conversation pair.

        Args:
            ctx: The command context
            content: The index of the pair to remove
        """
        try:
            # Convert to 0-based index
            index = int(content) - 1
            if index < 0:
                await ctx.send("-# Please provide a positive number")
                return

            pairs, removed = example_conversation.remove_pair(index)
            if removed:
                await ctx.send(
                    f"-# Removed message pair #{index + 1}:\n"
                    f"-# User: {removed.user}\n"
                    f"-# Bot: {removed.assistant}\n"
                    f"-# Total pairs remaining: {len(pairs)}"
                )
            else:
                await ctx.send(f"-# No message pair #{index + 1} found")

        except ValueError:
            await ctx.send("-# Please provide a valid number")
        except Exception as e:
            logger.error(f"Error removing example message pair: {e}")
            await ctx.send(f"-# Error removing message pair: {str(e)}")

    @staticmethod
    async def _edit_example(ctx: Context, content: str) -> None:
        """Edit an example conversation pair.

        Args:
            ctx: The command context
            content: The content containing index and new messages
        """
        try:
            # Split content into index and messages
            parts = content.split(maxsplit=1)
            if len(parts) < 2:
                await ctx.send("-# Please provide a number and messages")
                return

            # Convert to 0-based index
            index = int(parts[0]) - 1
            if index < 0:
                await ctx.send("-# Please provide a positive number")
                return

            msg_parts = parts[1].split("|", 1)
            edit_user_msg: Optional[str] = None
            edit_bot_msg: Optional[str] = None

            if len(msg_parts) == 2:
                # Both messages provided
                user_msg_str = msg_parts[0].strip()
                bot_msg_str = msg_parts[1].strip()
                if not user_msg_str and not bot_msg_str:
                    await ctx.send("-# At least one message must not be empty")
                    return
                # Convert empty strings to None
                edit_user_msg = user_msg_str if user_msg_str else None
                edit_bot_msg = bot_msg_str if bot_msg_str else None
            else:
                # Only one message provided - treat as user message
                user_msg_str = msg_parts[0].strip()
                if not user_msg_str:
                    await ctx.send("-# Message must not be empty")
                    return
                # Convert empty string to None
                edit_user_msg = user_msg_str if user_msg_str else None

            _, edited = example_conversation.edit_pair(
                index, edit_user_msg, edit_bot_msg
            )
            if edited:
                await ctx.send(
                    f"-# Edited message pair #{index + 1} to:\n"
                    f"-# User: {edited.user}\n"
                    f"-# Bot: {edited.assistant}"
                )
            else:
                await ctx.send(f"-# No message pair #{index + 1} found")

        except ValueError:
            await ctx.send("-# Please provide a valid number")
        except Exception as e:
            logger.error(f"Error editing example message pair: {e}")
            await ctx.send(f"-# Error editing message pair: {str(e)}")
