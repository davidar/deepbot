"""Command handlers for search functionality."""

import logging
from typing import Any, Dict, List, Optional, cast

import discord
from discord.ext import commands

from discord_types import StoredMessage
from message_store import MessageStore
from utils import format_relative_time, resolve_channel_name, resolve_mentions

Context = commands.Context[commands.Bot]

# Set up logging
logger = logging.getLogger("deepbot.command.search_commands")


class SearchCommands:
    """Handlers for search-related commands."""

    def __init__(
        self,
        message_store: MessageStore,
    ) -> None:
        """Initialize search commands.

        Args:
            message_store: The message store to search
        """
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
                if "id" in emoji and "name" in emoji:
                    name = cast(str, emoji["name"])
                    emoji_id = cast(str, emoji["id"])
                    emoji_str = f"<:{name}:{emoji_id}>"
                else:
                    emoji_name = emoji.get("name")
                    emoji_str = str(emoji_name if emoji_name is not None else "unknown")
            else:
                emoji_str = str(emoji)
            count = cast(int, r.get("count", 0))
            formatted[emoji_str] = count
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
        message: StoredMessage,
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

        # Try to get channel name from bot first, then fall back to stored info
        channel_name = "unknown"
        if bot:
            try:
                channel = bot.get_channel(int(channel_id))
                if isinstance(channel, discord.TextChannel):
                    channel_name = channel.name
            except ValueError:
                logger.warning(f"Invalid channel ID: {channel_id}")

        # Format the message line
        prefix = "  ↳ " if is_reply else ""
        extra_info = self._format_extras(
            bool(message.attachments), bool(message.embeds), reactions
        )
        return f"-# {prefix}[{relative_time}] #{channel_name} <{message.author.name}> {content}{extra_info}"

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
                results = await self.message_store.search(query, top_k=limit, **filters)

            if not results:
                await ctx.send("-# No messages found matching your query.")
                return

            # Format results with context
            message_groups: List[List[str]] = []
            current_group: List[str] = []

            # Process results from each channel
            for channel_id, messages in results.items():
                for message in messages:
                    # Get context messages
                    channel_messages = self.message_store.get_channel_messages(
                        channel_id
                    )
                    message_index = next(
                        (
                            i
                            for i, m in enumerate(channel_messages)
                            if m.id == message.id
                        ),
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

                    if (
                        current_length + group_length > 1900
                    ):  # Leave room for formatting
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
