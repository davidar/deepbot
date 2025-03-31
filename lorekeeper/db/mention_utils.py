"""Utilities for processing and resolving mentions in text."""

import logging
import re
from typing import Any, List, Optional, Tuple, TypeVar

from .typed_database import TypedDatabase

logger = logging.getLogger("deepbot.mention_utils")

# Define a type for author objects
T = TypeVar("T")


def strip_leading_zeros(id_str: str) -> str:
    """Strip leading zeros from an ID string.

    Args:
        id_str: The ID string to process

    Returns:
        The ID string with leading zeros removed
    """
    return id_str.lstrip("0") if id_str else id_str


def find_role_by_name(guild_ids: List[str], username: str) -> Tuple[bool, str, str]:
    """Find a role by name across multiple guilds.

    Args:
        guild_ids: List of guild IDs to search in
        username: Role name to search for

    Returns:
        Tuple of (found, role_mention, guild_id)
    """
    for g_id in guild_ids:
        try:
            # Get the roles collection and search for the role
            roles_collection = TypedDatabase.get_roles_collection(g_id)
            role = roles_collection.find_one({"name": username})

            if role:
                # Replace with proper role mention, stripping leading zeros
                role_id = strip_leading_zeros(role.id)
                role_mention = f"<@&{role_id}>"
                logger.info(
                    f"Processed role mention @{username} -> {role_mention} in guild {g_id}"
                )
                return True, role_mention, g_id
        except Exception as e:
            logger.debug(f"Error searching for role in guild {g_id}: {e}")
            continue

    return False, "", ""


def find_user_by_username(guild_id: str, username: str) -> Optional[Any]:
    """Find a user in a specific guild by nickname or name without suffix.

    Args:
        guild_id: The Discord guild ID to search in
        username: The username to search for

    Returns:
        User object if found, None otherwise
    """
    try:
        authors_collection = TypedDatabase.get_authors_collection(guild_id)

        # 1. Check exact match in nicknames list
        user = authors_collection.find_one({"nicknames": username})
        if user:
            logger.debug(f"Found user via nickname exact match: {username}")
            return user

        # 2. Check names list without numeric suffix
        # First try exact match in case the name doesn't have a suffix
        user = authors_collection.find_one({"names": username})
        if user:
            logger.debug(f"Found user via exact name match: {username}")
            return user

        # Try to match name part before "#" suffix
        name_prefix_pattern = f"^{re.escape(username)}#"
        user = authors_collection.find_one({"names": {"$regex": name_prefix_pattern}})
        if user:
            logger.debug(f"Found user via name prefix match: {username}")
            return user

        return None
    except Exception as e:
        logger.debug(f"Error searching for user in guild {guild_id}: {e}")
        return None


def find_user_across_guilds(
    guild_ids: List[str], username: str
) -> Tuple[bool, str, str]:
    """Find a user by username across multiple guilds.

    Args:
        guild_ids: List of guild IDs to search in
        username: Username to search for

    Returns:
        Tuple of (found, user_mention, guild_id)
    """
    for g_id in guild_ids:
        try:
            user = find_user_by_username(g_id, username)
            if user and hasattr(user, "id"):
                user_id = strip_leading_zeros(getattr(user, "id"))
                discord_mention = f"<@{user_id}>"
                logger.info(
                    f"Processed user mention @{username} -> {discord_mention} in guild {g_id}"
                )
                return True, discord_mention, g_id
        except Exception as e:
            logger.debug(f"Error searching for user in guild {g_id}: {e}")
            continue

    return False, "", ""


def resolve_mentions(content: str) -> str:
    """Process @username mentions in the content and convert to Discord mentions.

    Args:
        content: The text content to process

    Returns:
        The content with @username mentions replaced with proper Discord mentions
    """
    # Match @username patterns
    mention_pattern = r"@([\w.]+)"
    mentions = re.findall(mention_pattern, content)

    if not mentions:
        return content

    try:
        # Get all guilds collection to fetch all guild IDs
        guilds_collection = TypedDatabase.get_guilds_collection()
        guilds = guilds_collection.find({})
        guild_ids = [guild.id for guild in guilds]

        logger.debug(f"Searching for mentions across {len(guild_ids)} guilds")

        # Process each mention by searching across all guilds
        for username in mentions:
            # First check for roles
            role_found, role_mention, _ = find_role_by_name(guild_ids, username)
            if role_found:
                content = content.replace(f"@{username}", role_mention)
                continue

            # If no role found, look for users
            user_found, user_mention, _ = find_user_across_guilds(guild_ids, username)
            if user_found:
                content = content.replace(f"@{username}", user_mention)
                continue

            # If neither role nor user was found
            logger.debug(f"Mention @{username} not found in any guild")

        return content

    except Exception as e:
        logger.warning(f"Error processing mentions: {e}")
        return content  # Return the original content on error
