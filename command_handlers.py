"""Command handler implementations for DeepBot."""

import io
import json
import logging
from typing import Any, Dict, List, Optional

import discord
from discord.ext import commands

import config
import example_conversation
import system_prompt
from context_builder import ContextBuilder
from llm_streaming import LLMResponseHandler
from local_discord_index import LocalDiscordIndex
from message_history import MessageHistoryManager
from message_store import MessageStore
from reactions import ReactionManager
from user_management import UserManager
from utils import format_relative_time, resolve_channel_name, resolve_mentions

Context = commands.Context[commands.Bot]

# Set up logging
logger = logging.getLogger("deepbot.command_handlers")


class OptionCommands:
    """Handlers for model option commands."""

    def __init__(self) -> None:
        """Initialize option command handlers."""
        pass

    @staticmethod
    async def _display_options(ctx: Context) -> None:
        """Display all model options.

        Args:
            ctx: The command context
        """
        file = discord.File("model_options.json")
        await ctx.send("-# Current Model Options:", file=file)
        await ctx.send(
            "-# Use `options get <option>` or `options set <option> <value>` to modify options"
        )

    @staticmethod
    async def _get_option(ctx: Context, option_name: str) -> None:
        """Get the value of a specific option.

        Args:
            ctx: The command context
            option_name: The name of the option to get
        """
        opt_value = config.load_model_options().get(option_name)
        if opt_value is not None:
            await ctx.send(f"-# Option `{option_name}` is set to `{opt_value}`")
        else:
            await ctx.send(f"-# Option `{option_name}` not found")

    @staticmethod
    async def _set_option(ctx: Context, option_name: str, value: str) -> None:
        """Set the value of a specific option.

        Args:
            ctx: The command context
            option_name: The name of the option to set
            value: The value to set
        """
        try:
            # Try to convert value to float first
            float_value = float(value)
            # If it's a whole number, convert to int
            if float_value.is_integer():
                float_value = int(float_value)

            # Get valid options and their types
            option_types = config.get_model_option_types()

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
            await ctx.send("-# Invalid value, please provide a number")
        except KeyError as e:
            await ctx.send(f"-# {str(e)}")
        except TypeError as e:
            await ctx.send(f"-# {str(e)}")

    @staticmethod
    async def handle_options(
        ctx: Context,
        action: Optional[str] = None,
        option_name: Optional[str] = None,
        *,
        value: Optional[str] = None,
    ) -> None:
        """Handle the options command.

        Args:
            ctx: The command context
            action: The action to perform (get/set)
            option_name: The name of the option to get/set
            value: The value to set for the option
        """
        if not action:
            await OptionCommands._display_options(ctx)
            return

        if action.lower() == "get" and option_name:
            await OptionCommands._get_option(ctx, option_name)
            return

        if action.lower() == "set" and option_name and value is not None:
            await OptionCommands._set_option(ctx, option_name, value)
            return

        await ctx.send(
            "-# Invalid command, use `options`, `options get <option>`, or `options set <option> <value>`"
        )


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
        context = self.context_builder.build_context(messages, ctx.channel)

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


class UserCommands:
    """Handlers for user management commands."""

    def __init__(self, user_manager: UserManager) -> None:
        """Initialize user command handlers.

        Args:
            user_manager: The user manager instance
        """
        self.user_manager = user_manager

    async def handle_ignore(self, ctx: Context, member: discord.Member) -> None:
        """Handle the ignore command.

        Args:
            ctx: The command context
            member: The Discord member to ignore
        """
        self.user_manager.ignore_user(member.id)
        await ctx.send(f"-# Now ignoring messages from {member.display_name}")

    async def handle_unignore(self, ctx: Context, member: discord.Member) -> None:
        """Handle the unignore command.

        Args:
            ctx: The command context
            member: The Discord member to unignore
        """
        self.user_manager.unignore_user(member.id)
        await ctx.send(f"-# No longer ignoring messages from {member.display_name}")

    async def handle_limit(
        self,
        ctx: Context,
        member: discord.Member,
        consecutive_limit: Optional[int] = None,
    ) -> None:
        """Handle the limit command.

        Args:
            ctx: The command context
            member: The Discord member to limit
            consecutive_limit: Maximum consecutive messages allowed, or None to remove
        """
        try:
            self.user_manager.set_consecutive_limit(member.id, consecutive_limit)
            if consecutive_limit is None:
                await ctx.send(f"-# Removed message limit for {member.display_name}")
            else:
                await ctx.send(
                    f"-# Set consecutive message limit for {member.display_name} to {consecutive_limit} messages"
                )
        except ValueError as e:
            await ctx.send(f"-# Error: {str(e)}")

    async def handle_restrictions(self, ctx: Context, member: discord.Member) -> None:
        """Handle the restrictions command.

        Args:
            ctx: The command context
            member: The Discord member to check
        """
        restrictions = self.user_manager.get_user_restrictions(member.id)
        if not restrictions:
            await ctx.send(f"-# No restrictions set for {member.display_name}")
            return

        message = [f"-# Current restrictions for {member.display_name}:"]
        if restrictions.ignored:
            message.append("-# • User is ignored")
        if restrictions.consecutive_limit is not None:
            message.append(
                f"-# • Limited to {restrictions.consecutive_limit} consecutive messages"
            )
            if restrictions.consecutive_count > 0:
                message.append(
                    f"-# • Currently at {restrictions.consecutive_count} consecutive messages"
                )

        await ctx.send("\n".join(message))


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


class ExampleCommands:
    """Handlers for example conversation commands."""

    def __init__(self) -> None:
        """Initialize example command handlers."""
        pass

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


class ReactionCommands:
    """Handlers for reaction commands."""

    def __init__(self, reaction_manager: ReactionManager) -> None:
        """Initialize reaction command handlers.

        Args:
            reaction_manager: The reaction manager instance
        """
        self.reaction_manager = reaction_manager

    async def handle_reactions(self, ctx: Context, scope: str = "channel") -> None:
        """Handle the reactions command.

        Args:
            ctx: The command context
            scope: Either "channel" (default) or "global" to show stats across all channels
        """
        if scope.lower() not in ["channel", "global"]:
            await ctx.send('-# Invalid scope. Use "channel" or "global"')
            return

        if scope.lower() == "global":
            channel_scores = self.reaction_manager.get_global_stats()
            if not channel_scores:
                await ctx.send("-# No reaction data available yet")
                return

            message = ["-# Global reaction statistics:"]
            summary = self.reaction_manager.format_global_summary(channel_scores)
            for line in summary.split("\n"):
                if line.strip():
                    message.append(f"-# {line}")
            await ctx.send("\n".join(message))

        else:  # channel scope
            channel_id = ctx.channel.id
            message_reactions = self.reaction_manager.get_channel_stats(channel_id)

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
            channel_summary = self.reaction_manager.format_reaction_summary(
                message_reactions
            )
            if channel_summary:
                for line in channel_summary.split("\n"):
                    if line.strip():
                        message.append(f"-# {line}")
            else:
                message.append("-# No reactions yet.")
            await ctx.send("\n".join(message))


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


class SearchCommands:
    """Handlers for search-related commands."""

    def __init__(
        self,
        message_store: MessageStore,
        storage_path: str = "./chroma_db",
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ) -> None:
        """Initialize search commands.

        Args:
            message_store: The message store to search
            storage_path: Path to the vector database
            model_name: Ollama model to use for embeddings
            base_url: URL of the Ollama server
        """
        self.index = LocalDiscordIndex(
            message_store,
            storage_path=storage_path,
            model_name=model_name,
            base_url=base_url,
        )
        self.message_store = message_store

    def _format_reactions(self, reactions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Format reactions into a dictionary of emoji strings and counts.

        Args:
            reactions: List of reaction objects from Discord

        Returns:
            Dictionary mapping emoji strings to counts
        """
        formatted: Dict[str, int] = {}
        for r in reactions:
            emoji = r["emoji"]
            # Handle custom emoji objects
            if isinstance(emoji, dict):
                emoji_str = f"<:{emoji['name']}:{emoji['id']}>"
            else:
                emoji_str = str(emoji)
            formatted[emoji_str] = r["count"]
        return formatted

    def _format_extras(
        self,
        has_attachments: bool,
        has_embeds: bool,
        reactions: Optional[Dict[str, int]] = None,
    ) -> str:
        """Format extra information about a message.

        Args:
            has_attachments: Whether the message has attachments
            has_embeds: Whether the message has embeds
            reactions: Dictionary of reaction emojis and their counts

        Returns:
            Formatted string of extra information
        """
        extras: List[str] = []
        if has_attachments:
            extras.append("📎 has attachments")
        if has_embeds:
            extras.append("📌 has embeds")
        if reactions:
            reaction_str = " ".join(
                f"{emoji}{count}" for emoji, count in reactions.items()
            )
            if reaction_str:
                extras.append(reaction_str)
        return f" [{', '.join(extras)}]" if extras else ""

    def _resolve_stored_mentions(self, content: str, mentions: List[Any]) -> str:
        """Resolve mentions using stored user information.

        Args:
            content: The message content
            mentions: List of mentioned users with their info

        Returns:
            Content with mentions resolved to usernames
        """
        # Create a mapping of user IDs to names
        id_to_name = {
            mention.id: mention.nickname or mention.name for mention in mentions
        }

        # Replace <@ID> mentions with usernames
        for user_id, name in id_to_name.items():
            content = content.replace(f"<@{user_id}>", f"@{name}")
            content = content.replace(f"<@!{user_id}>", f"@{name}")  # Nickname mentions

        return content

    def _format_message(
        self,
        message_id: str,
        channel_id: str,
        timestamp: str,
        author: str,
        content: str,
        has_attachments: bool = False,
        has_embeds: bool = False,
        reactions: Optional[Dict[str, int]] = None,
        bot: Optional[commands.Bot] = None,
    ) -> str:
        """Format a message for display.

        Args:
            message_id: The message ID
            channel_id: The channel ID
            timestamp: The message timestamp
            author: The author's name
            content: The message content
            has_attachments: Whether the message has attachments
            has_embeds: Whether the message has embeds
            reactions: Dictionary of reaction emojis and their counts
            bot: The Discord bot instance for resolving mentions/channels

        Returns:
            Formatted message string
        """
        # Format timestamp and channel name
        relative_time = format_relative_time(timestamp)
        channel_name = resolve_channel_name(channel_id, bot)

        # Resolve mentions if bot is provided and collapse newlines
        if bot:
            content = resolve_mentions(content, bot)
            # Escape mentions by adding a zero-width space after @
            content = content.replace("@", "@\u200b")
        content = " ".join(content.split())

        # Format extra information
        extra_info = self._format_extras(has_attachments, has_embeds, reactions)

        # Build the message line
        return f"-# [{relative_time}] #{channel_name} <{author}> {content}{extra_info}"

    def _format_message_group(
        self,
        message: Any,
        channel_id: str,
        is_reply: bool = False,
        bot: Optional[commands.Bot] = None,
    ) -> str:
        """Format a single message with its extra information.

        Args:
            message: The Discord message object
            channel_id: The channel ID
            is_reply: Whether this message is a reply (for formatting)
            bot: The Discord bot instance for resolving mentions/channels

        Returns:
            Formatted message string
        """
        # Get reactions for the message
        reactions = (
            self._format_reactions(message.reactions) if message.reactions else None
        )

        # Format the content with mentions and newlines
        content = self._resolve_stored_mentions(message.content, message.mentions)
        content = content.replace("@", "@\u200b")  # Escape mentions
        content = " ".join(content.split())

        # Format timestamp
        relative_time = format_relative_time(message.timestamp)

        # Format the message line
        prefix = "  ↳ " if is_reply else ""
        extra_info = self._format_extras(
            bool(message.attachments), bool(message.embeds), reactions
        )
        return f"-# {prefix}[{relative_time}] #{resolve_channel_name(channel_id, bot)} <{message.author.name}> {content}{extra_info}"

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
                results = await self.index.search(query, top_k=limit, **filters)

            if not results:
                await ctx.send("-# No messages found matching your query.")
                return

            # Format results with context
            message_groups: List[List[str]] = []
            current_group: List[str] = []

            for node in results:
                metadata = node.metadata or {}
                channel_id = metadata.get("channel_id", "unknown")
                message_id = metadata.get("message_id", "unknown")

                # Get the message from the store
                message = self.message_store.get_message(channel_id, message_id)
                if not message:
                    continue

                # Get context messages
                channel_messages = self.message_store.get_channel_messages(channel_id)
                message_index = next(
                    (i for i, m in enumerate(channel_messages) if m.id == message_id),
                    -1,
                )

                # Start a new group for this result
                result_group: List[str] = []

                # Add context message first if it exists
                if message.reference:
                    # If it's a reply, add the referenced message first
                    ref_msg = self.message_store.get_message(
                        message.reference.channelId,
                        message.reference.messageId,
                    )
                    if ref_msg:
                        result_group.append(
                            self._format_message_group(
                                ref_msg, channel_id, is_reply=False, bot=ctx.bot
                            )
                        )
                elif message_index > 0:
                    # Otherwise, add the previous message first
                    prev_msg = channel_messages[message_index - 1]
                    result_group.append(
                        self._format_message_group(
                            prev_msg, channel_id, is_reply=False, bot=ctx.bot
                        )
                    )

                # Then add the main message
                result_group.append(
                    self._format_message_group(
                        message,
                        channel_id,
                        is_reply=bool(message.reference),
                        bot=ctx.bot,
                    )
                )

                # Add blank line between results
                result_group.append("")

                # Add this group to the current chunk if it fits, otherwise start a new chunk
                group_length = sum(
                    len(line) + 1 for line in result_group
                )  # +1 for newline
                current_length = sum(len(line) + 1 for line in current_group)

                if current_length + group_length > 1900:  # Leave room for formatting
                    if current_group:
                        message_groups.append(current_group)
                    current_group = result_group
                else:
                    current_group.extend(result_group)

            # Add the last group if it exists
            if current_group:
                message_groups.append(current_group)

            # Send each group of messages
            for group in message_groups:
                # Remove trailing blank line if it exists
                if group and not group[-1]:
                    group.pop()
                await ctx.send("\n".join(group))

        except Exception as e:
            logger.error(f"Error performing search: {e}")
            await ctx.send(f"-# An error occurred while searching: {e}")
