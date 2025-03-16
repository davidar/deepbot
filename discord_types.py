"""Type definitions for Discord message storage."""

import logging
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypeVar, Union

from discord import Embed, Emoji, Member, Message, PartialEmoji
from discord import Role as DiscordRole
from discord import Sticker, StickerItem, User

# Set up logging
logger = logging.getLogger("deepbot.discord_types")


def _format_timestamp(dt: Optional[datetime]) -> Optional[str]:
    """Format a datetime into a consistent ISO format with Z timezone.

    Args:
        dt: datetime object to format, or None

    Returns:
        ISO format string with Z timezone, or None if input is None
    """
    if dt is None:
        return None
    # Convert to UTC, format with 3 decimal places for microseconds, and use Z suffix
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _parse_timestamp(timestamp_str: str) -> datetime:
    """Parse an ISO format datetime string and ensure UTC timezone awareness.

    Args:
        timestamp_str: ISO format datetime string, with or without timezone info

    Returns:
        datetime object with UTC timezone
    """
    # Convert any timezone format to UTC
    if timestamp_str.endswith("Z"):
        # Remove Z and add UTC timezone
        dt = datetime.fromisoformat(timestamp_str[:-1])
        return dt.replace(tzinfo=timezone.utc)
    elif "+" in timestamp_str:
        # Parse with timezone and convert to UTC
        dt = datetime.fromisoformat(timestamp_str)
        return dt.astimezone(timezone.utc)
    else:
        # No timezone - assume UTC
        dt = datetime.fromisoformat(timestamp_str)
        return dt.replace(tzinfo=timezone.utc)


@dataclass
class Role:
    """Represents a Discord role."""

    id: str
    name: str
    color: Optional[str]
    position: int

    @classmethod
    def from_discord_role(cls, role: DiscordRole) -> "Role":
        """Create from a Discord role."""
        return cls(
            id=str(role.id),
            name=role.name,
            color=str(role.color) if role.color else None,
            position=role.position,
        )


@dataclass
class UserInfo:
    """Represents a Discord user with detailed information."""

    id: str
    name: str
    discriminator: str
    nickname: Optional[str]
    color: Optional[str]
    isBot: bool
    roles: List[Role]
    avatarUrl: str

    @classmethod
    def from_member(cls, member: Member) -> "UserInfo":
        """Create from a Discord member."""
        return cls(
            id=str(member.id),
            name=member.name,
            discriminator=member.discriminator,
            nickname=(
                member.display_name if member.display_name != member.name else None
            ),
            color=None,  # Would need to calculate from roles
            isBot=member.bot,
            roles=[
                Role.from_discord_role(r) for r in member.roles if not r.is_default()
            ],
            avatarUrl=str(member.avatar.url) if member.avatar else "",
        )

    @classmethod
    def from_user(cls, user: User) -> "UserInfo":
        """Create from a Discord user (with minimal information)."""
        return cls(
            id=str(user.id),
            name=user.name,
            discriminator=user.discriminator,
            nickname=None,
            color=None,
            isBot=user.bot,
            roles=[],  # Users don't have roles, only Members do
            avatarUrl=str(user.avatar.url) if user.avatar else "",
        )


@dataclass
class InlineEmoji:
    """Represents an inline emoji in a message."""

    id: str
    name: str
    code: str
    isAnimated: bool
    imageUrl: str


@dataclass
class Attachment:
    """Represents a message attachment."""

    id: str
    url: str
    fileName: str
    fileSizeBytes: int
    proxyUrl: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    contentType: Optional[str] = None


@dataclass
class MessageReference:
    """Represents a reference to another message (for replies)."""

    messageId: str
    channelId: str
    guildId: str


@dataclass
class ChannelInfo:
    """Represents Discord channel information."""

    id: str
    type: str
    categoryId: Optional[str]
    category: Optional[str]
    name: str
    topic: Optional[str]


@dataclass
class GuildInfo:
    """Represents Discord guild (server) information."""

    id: str
    name: str
    iconUrl: Optional[str]


@dataclass
class StoredMessage:
    """Represents a stored message with all its metadata."""

    id: str
    type: str  # "Default", "Reply", etc.
    timestamp: str  # ISO format UTC timestamp
    timestampEdited: Optional[str]  # ISO format UTC timestamp or None
    callEndedTimestamp: Optional[str]  # ISO format UTC timestamp or None
    isPinned: bool
    content: str
    author: UserInfo
    attachments: List[Attachment]
    embeds: List[Dict[str, Any]]
    stickers: List[Dict[str, Any]]
    reactions: List[Dict[str, Any]]
    mentions: List[UserInfo]
    reference: Optional[MessageReference]
    inlineEmojis: List[InlineEmoji]

    @staticmethod
    def _convert_emoji(emoji: Union[Emoji, PartialEmoji, str]) -> Dict[str, Any]:
        """Convert a Discord emoji to a dictionary format.

        Args:
            emoji: The Discord emoji to convert

        Returns:
            Dictionary representation of the emoji
        """
        if isinstance(emoji, (Emoji, PartialEmoji)):
            return {
                "id": str(emoji.id) if emoji.id else None,
                "name": emoji.name,
                "code": emoji.name,  # Add code field for consistency
                "isAnimated": bool(
                    emoji.animated if hasattr(emoji, "animated") else False
                ),
                "imageUrl": str(emoji.url) if hasattr(emoji, "url") else None,
            }
        else:  # Unicode emoji (str)
            return {
                "id": None,
                "name": emoji,
                "code": emoji,  # Add code field for consistency
                "isAnimated": False,
                "imageUrl": None,
            }

    @staticmethod
    def _convert_embed(embed: Embed) -> Dict[str, Any]:
        """Convert a Discord embed to a dictionary format.

        Args:
            embed: The Discord embed to convert

        Returns:
            Dictionary representation of the embed
        """
        return {
            "title": embed.title,
            "type": embed.type,
            "description": embed.description,
            "url": embed.url,
            "timestamp": _format_timestamp(embed.timestamp),
            "color": embed.colour.value if embed.colour else None,
            "footer": (
                {
                    "text": embed.footer.text if embed.footer else None,
                    "iconUrl": embed.footer.icon_url if embed.footer else None,
                }
                if embed.footer
                else None
            ),
            "image": (
                {
                    "url": embed.image.url,
                    "proxyUrl": embed.image.proxy_url,
                    "width": embed.image.width,
                    "height": embed.image.height,
                }
                if embed.image
                else None
            ),
            "thumbnail": (
                {
                    "url": embed.thumbnail.url,
                    "proxyUrl": embed.thumbnail.proxy_url,
                    "width": embed.thumbnail.width,
                    "height": embed.thumbnail.height,
                }
                if embed.thumbnail
                else None
            ),
            "video": (
                {
                    "url": embed.video.url,
                    "width": embed.video.width,
                    "height": embed.video.height,
                }
                if embed.video
                else None
            ),
            "provider": (
                {
                    "name": embed.provider.name,
                    "url": embed.provider.url,
                }
                if embed.provider
                else None
            ),
            "author": (
                {
                    "name": embed.author.name,
                    "url": embed.author.url,
                    "iconUrl": embed.author.icon_url,
                }
                if embed.author
                else None
            ),
            "fields": [
                {
                    "name": field.name,
                    "value": field.value,
                    "inline": field.inline,
                }
                for field in embed.fields
            ],
        }

    @staticmethod
    def _convert_sticker(sticker: Union[Sticker, StickerItem]) -> Dict[str, Any]:
        """Convert a Discord sticker to a dictionary format.

        Args:
            sticker: The Discord sticker to convert (can be Sticker or StickerItem)

        Returns:
            Dictionary representation of the sticker
        """
        return {
            "id": str(sticker.id),
            "name": sticker.name,
            "formatType": str(sticker.format),
            "description": getattr(sticker, "description", None),
            "url": str(sticker.url) if hasattr(sticker, "url") else None,
        }

    @staticmethod
    def _parse_inline_emoji(content: str) -> List[InlineEmoji]:
        """Parse inline emojis from message content.

        Args:
            content: The message content to parse

        Returns:
            List of parsed inline emojis
        """
        inline_emojis: List[InlineEmoji] = []
        for emoji in content.split():
            # Check if this is a custom emoji format <:name:id> or <a:name:id>
            if emoji.startswith("<") and emoji.endswith(">") and ":" in emoji:
                parts = emoji.strip("<>").split(":")
                if len(parts) == 3:  # Animated or regular custom emoji
                    is_animated = parts[0] == "a"
                    name = parts[1]
                    emoji_id = parts[2]
                    inline_emojis.append(
                        InlineEmoji(
                            id=emoji_id,
                            name=name,
                            code=emoji,
                            isAnimated=is_animated,
                            imageUrl=f"https://cdn.discordapp.com/emojis/{emoji_id}.{'gif' if is_animated else 'png'}",
                        )
                    )
        return inline_emojis

    @staticmethod
    async def _convert_reactions(message: Message) -> List[Dict[str, Any]]:
        """Convert Discord message reactions to a dictionary format.

        Args:
            message: The Discord message with reactions

        Returns:
            List of reaction dictionaries
        """
        reactions: List[Dict[str, Any]] = []
        for reaction in message.reactions:
            emoji_data = StoredMessage._convert_emoji(reaction.emoji)

            # Get users who reacted
            users: List[Dict[str, Any]] = []
            async for user in reaction.users():
                if isinstance(user, Member):
                    user_info = UserInfo.from_member(user)
                else:
                    user_info = UserInfo.from_user(user)
                user_dict = serialize_dataclass(user_info)
                if user_dict is not None:  # Add type check
                    users.append(user_dict)

            reaction_data: Dict[str, Any] = {
                "emoji": emoji_data,
                "count": reaction.count,
                "users": users,
            }
            reactions.append(reaction_data)
        return reactions

    @classmethod
    async def from_discord_message(cls, message: Message) -> "StoredMessage":
        """Create from a Discord message."""
        # Convert timestamps using utility function
        timestamp = _format_timestamp(message.created_at)
        if timestamp is None:  # This should never happen as created_at is required
            raise ValueError("Message created_at timestamp is required")
        edited_timestamp = _format_timestamp(message.edited_at)

        # Get message type
        msg_type = "Reply" if message.reference else "Default"

        # Convert author info - handle both Member and User objects
        author = message.author
        author_info = (
            UserInfo.from_member(author)
            if isinstance(author, Member)
            else UserInfo.from_user(author)
        )

        # Convert attachments
        attachments = [
            Attachment(
                id=str(a.id),
                url=a.url,
                fileName=a.filename,
                fileSizeBytes=a.size,
                proxyUrl=a.proxy_url,
                width=a.width,
                height=a.height,
                contentType=a.content_type,
            )
            for a in message.attachments
        ]

        # Convert mentions - handle both Member and User objects
        mentions: List[UserInfo] = []
        for user in message.mentions:
            if isinstance(user, Member):
                mentions.append(UserInfo.from_member(user))
            else:
                mentions.append(UserInfo.from_user(user))

        # Convert reactions
        reactions = await cls._convert_reactions(message)

        # Convert embeds
        embeds = [cls._convert_embed(embed) for embed in message.embeds]

        # Convert stickers
        stickers = [cls._convert_sticker(sticker) for sticker in message.stickers]

        # Convert inline emojis
        inline_emojis = cls._parse_inline_emoji(message.content)

        # Convert message reference if it exists
        reference = None
        if message.reference:
            reference = MessageReference(
                messageId=str(message.reference.message_id),
                channelId=str(message.reference.channel_id),
                guildId=(
                    str(message.reference.guild_id)
                    if message.reference.guild_id
                    else ""
                ),
            )

        return cls(
            id=str(message.id),
            type=msg_type,
            timestamp=timestamp,
            timestampEdited=edited_timestamp,
            callEndedTimestamp=None,
            isPinned=message.pinned,
            content=message.content,
            author=author_info,
            attachments=attachments,
            embeds=embeds,
            stickers=stickers,
            reactions=reactions,
            mentions=mentions,
            reference=reference,
            inlineEmojis=inline_emojis,
        )


T = TypeVar("T")


def serialize_dataclass(obj: Any) -> Optional[Dict[str, Any]]:
    """Convert a dataclass object to a dictionary.

    Args:
        obj: The object to convert

    Returns:
        Dictionary representation of the object, or None if obj is None
    """
    if obj is None:
        return None
    if not hasattr(obj, "__dataclass_fields__"):
        return obj if isinstance(obj, dict) else None
    result: Dict[str, Any] = {}
    try:
        dataclass_fields = fields(obj)
    except TypeError:
        return None

    for field in dataclass_fields:
        value = getattr(obj, field.name)
        if hasattr(value, "__dataclass_fields__"):
            result[field.name] = serialize_dataclass(value)
        elif isinstance(value, list):
            result[field.name] = [
                (
                    serialize_dataclass(item)
                    if hasattr(item, "__dataclass_fields__")
                    else item
                )
                for item in value
            ]
        else:
            result[field.name] = value
    return result
