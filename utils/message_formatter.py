"""Utilities for formatting Discord messages."""

import logging
from typing import Any, Dict, List, Optional

import discord
from discord.ext import commands

from discord_types import StoredMessage

from .time_utils import format_relative_time

logger = logging.getLogger("deepbot.utils.message_formatter")


def format_reactions(reactions: List[Dict[str, Any]]) -> Dict[str, int]:
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
                name = str(emoji["name"])
                emoji_id = str(emoji["id"])
                emoji_str = f"<:{name}:{emoji_id}>"
            else:
                emoji_name = emoji.get("name")
                emoji_str = str(emoji_name if emoji_name is not None else "unknown")
        else:
            emoji_str = str(emoji)
        count = int(r.get("count", 0))
        formatted[emoji_str] = count
    return formatted


def format_extras(
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
        reaction_str = " ".join(f"{emoji}{count}" for emoji, count in reactions.items())
        if reaction_str:
            extras.append(reaction_str)
    return f" [{', '.join(extras)}]" if extras else ""


def resolve_stored_mentions(content: str, mentions: List[Any]) -> str:
    """Resolve mentions using stored user information.

    Args:
        content: The message content
        mentions: List of mentioned users with their info

    Returns:
        Content with mentions resolved to usernames
    """
    # Create a mapping of user IDs to names
    id_to_name = {mention.id: mention.nickname or mention.name for mention in mentions}

    # Replace <@ID> mentions with usernames
    for user_id, name in id_to_name.items():
        content = content.replace(f"<@{user_id}>", f"@{name}")
        content = content.replace(f"<@!{user_id}>", f"@{name}")  # Nickname mentions

    return content


def format_message_group(
    message: StoredMessage,
    channel_id: str,
    is_reply: bool = False,
    bot: Optional[commands.Bot] = None,
    use_prefix: bool = True,
) -> str:
    """Format a single message with its extra information.

    Args:
        message: The Discord message object
        channel_id: The channel ID
        is_reply: Whether this message is a reply (for formatting)
        bot: The Discord bot instance for resolving mentions/channels
        use_prefix: Whether to include the "-# " prefix

    Returns:
        Formatted message string
    """
    # Get reactions for the message
    reactions = format_reactions(message.reactions) if message.reactions else None

    # Format the content with mentions and newlines
    content = resolve_stored_mentions(message.content, message.mentions)
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
    extra_info = format_extras(
        bool(message.attachments), bool(message.embeds), reactions
    )

    msg = f"{prefix}[{relative_time}] #{channel_name} <{message.author.name}> {content}{extra_info}"
    return f"-# {msg}" if use_prefix else msg


def format_search_results(
    results: Dict[str, List[StoredMessage]],
    message_store: Any,
    bot: Optional[commands.Bot] = None,
    use_prefix: bool = True,
) -> List[List[str]]:
    """Format search results into groups of messages.

    Args:
        results: Dictionary mapping channel IDs to lists of messages
        message_store: The message store instance for retrieving context
        bot: The Discord bot instance for resolving mentions/channels
        use_prefix: Whether to include the "-# " prefix in formatted messages

    Returns:
        List of message groups, where each group is a list of formatted message strings
    """
    message_groups: List[List[str]] = []
    current_group: List[str] = []

    # Process results from each channel
    for channel_id, messages in results.items():
        for message in messages:
            # Get context messages
            channel_messages = message_store.get_channel_messages(channel_id)
            message_index = next(
                (i for i, m in enumerate(channel_messages) if m.id == message.id),
                -1,
            )

            # Start a new group for this result
            result_group: List[str] = []

            # Add context message first if it exists
            if message.reference:
                # If it's a reply, add the referenced message first
                ref_msg = message_store.get_message(
                    message.reference.channelId,
                    message.reference.messageId,
                )
                if ref_msg:
                    result_group.append(
                        format_message_group(
                            ref_msg,
                            channel_id,
                            is_reply=False,
                            bot=bot,
                            use_prefix=use_prefix,
                        )
                    )
            elif message_index > 0:
                # Otherwise, add the previous message first
                prev_msg = channel_messages[message_index - 1]
                result_group.append(
                    format_message_group(
                        prev_msg,
                        channel_id,
                        is_reply=False,
                        bot=bot,
                        use_prefix=use_prefix,
                    )
                )

            # Then add the main message
            result_group.append(
                format_message_group(
                    message,
                    channel_id,
                    is_reply=bool(message.reference),
                    bot=bot,
                    use_prefix=use_prefix,
                )
            )

            # Add blank line between results
            result_group.append("")

            # Add this group to the current chunk if it fits, otherwise start a new chunk
            group_length = sum(len(line) + 1 for line in result_group)  # +1 for newline
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

    return message_groups
