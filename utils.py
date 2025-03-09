"""Utility functions used across the DeepBot modules."""

import re
from typing import Optional, Union

from discord.abc import GuildChannel, Messageable, PrivateChannel
from discord.channel import DMChannel, TextChannel
from discord.message import Message
from discord.threads import Thread


def get_channel_name(channel: Messageable) -> str:
    """Safely get channel name, handling both text channels and DMs.

    Args:
        channel: The Discord channel to get the name for

    Returns:
        str: The channel name, "DM" for direct messages, or "Unknown Channel"
    """
    if isinstance(channel, TextChannel):
        return channel.name
    elif isinstance(channel, DMChannel):
        return "DM"
    else:
        return "Unknown Channel"


def get_server_name(
    channel: Optional[Union[GuildChannel, Thread, PrivateChannel, Messageable]],
) -> str:
    """Get server name from channel, with fallback to DM chat.

    Args:
        channel: The Discord channel to get the server name for

    Returns:
        str: The server name or "DM chat" for direct messages
    """
    if channel and isinstance(channel, TextChannel):
        return channel.guild.name
    return "DM chat"


def clean_message_content(message: Message) -> str:
    """Clean up message content by replacing Discord mentions with usernames.

    Args:
        message: The Discord message to clean

    Returns:
        str: The cleaned message content with mentions replaced by readable names
    """
    content = message.content.strip()

    # Replace user mentions with usernames
    for user in message.mentions:
        mention_pattern = f"<@!?{user.id}>"
        username = user.display_name
        content = re.sub(mention_pattern, f"@{username}", content)

    # Replace channel mentions
    for channel in message.channel_mentions:
        channel_pattern = f"<#{channel.id}>"
        channel_name = channel.name
        content = re.sub(channel_pattern, f"#{channel_name}", content)

    # Replace role mentions
    if hasattr(message, "role_mentions"):
        for role in message.role_mentions:
            role_pattern = f"<@&{role.id}>"
            role_name = role.name
            content = re.sub(role_pattern, f"@{role_name}", content)

    return content
