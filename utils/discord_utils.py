"""Discord-specific utility functions."""

import re
from typing import Optional, Union

from discord.abc import GuildChannel, Messageable, PrivateChannel
from discord.channel import DMChannel, TextChannel
from discord.ext import commands
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


def resolve_channel_name(channel_id: str, bot: Optional[commands.Bot]) -> str:
    """Get channel name from ID.

    Args:
        channel_id: The channel ID
        bot: The Discord bot instance for resolving channel names

    Returns:
        Channel name or ID if not found
    """
    if not bot:
        return channel_id

    try:
        channel = bot.get_channel(int(channel_id))
        if isinstance(channel, (TextChannel, Thread)):
            return channel.name
    except (ValueError, AttributeError):
        pass
    return channel_id


def resolve_mentions(content: str, bot: commands.Bot) -> str:
    """Resolve user mentions to usernames.

    Args:
        content: Message content with mentions
        bot: The Discord bot instance for resolving mentions

    Returns:
        Content with mentions replaced by usernames
    """
    # Match Discord mention pattern
    mention_pattern = re.compile(r"<@!?(\d+)>")

    def replace_mention(match: re.Match[str]) -> str:
        user_id = int(match.group(1))
        user = bot.get_user(user_id)
        return f"@{user.name if user else 'unknown'}"

    return mention_pattern.sub(replace_mention, content)


def is_automated_message(content: str) -> bool:
    """Check if a message is an automated bot message.

    Args:
        content: The message content to check

    Returns:
        True if the message is automated (starts with -#), False otherwise
    """
    return content.strip().startswith("-#")
