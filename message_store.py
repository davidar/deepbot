"""Unified message storage system for DeepBot."""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, TypeVar, cast

from discord import Guild, Member, Message
from discord import Role as DiscordRole
from discord import TextChannel, User
from discord.abc import GuildChannel
from discord.emoji import Emoji
from discord.partial_emoji import PartialEmoji

# Set up logging
logger = logging.getLogger("deepbot.message_store")

T = TypeVar("T")


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


def serialize_dataclass(obj: Any) -> Optional[Dict[str, Any]]:
    """Serialize a dataclass to dict, preserving None values for certain fields."""
    if obj is None:
        return None
    if not hasattr(obj, "__dataclass_fields__"):
        return obj
    result: Dict[str, Any] = {}
    for field in obj.__dataclass_fields__:  # type: ignore
        value = getattr(obj, field)
        # Always include certain fields, even if None
        if value is not None or field in {
            "color",
            "timestampEdited",
            "callEndedTimestamp",
        }:
            if hasattr(value, "__dataclass_fields__"):
                result[field] = serialize_dataclass(value)
            elif isinstance(value, list):
                result[field] = [
                    (
                        serialize_dataclass(item)
                        if hasattr(item, "__dataclass_fields__")  # type: ignore
                        else item
                    )
                    for item in value  # type: ignore
                ]
            else:
                result[field] = value
    return result


@dataclass
class StoredMessage:
    """Represents a stored message with all its metadata."""

    id: str
    type: str  # "Default", "Reply", etc.
    timestamp: str
    timestampEdited: Optional[str]
    callEndedTimestamp: Optional[str]
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

    @classmethod
    async def from_discord_message(cls, message: Message) -> "StoredMessage":
        """Create from a Discord message."""
        # Convert timestamp to ISO format with timezone
        timestamp = message.created_at.astimezone(timezone.utc).isoformat()
        edited_timestamp = (
            message.edited_at.astimezone(timezone.utc).isoformat()
            if message.edited_at
            else None
        )

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
        reactions: List[Dict[str, Any]] = []
        for reaction in message.reactions:
            emoji = reaction.emoji
            # Handle both unicode emojis (str) and custom emojis (Emoji/PartialEmoji)
            if isinstance(emoji, (Emoji, PartialEmoji)):
                emoji_data: Dict[str, Any] = {
                    "id": str(emoji.id) if emoji.id else None,
                    "name": emoji.name,
                    "code": emoji.name,  # Add code field for consistency
                    "isAnimated": bool(
                        emoji.animated if hasattr(emoji, "animated") else False
                    ),
                    "imageUrl": str(emoji.url) if hasattr(emoji, "url") else None,
                }
            else:  # Unicode emoji (str)
                emoji_data = {
                    "id": None,
                    "name": emoji,
                    "code": emoji,  # Add code field for consistency
                    "isAnimated": False,
                    "imageUrl": None,
                }

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

        # Convert embeds
        embeds: List[Dict[str, Any]] = []
        for embed in message.embeds:
            embed_data: Dict[str, Any] = {
                "title": embed.title,
                "type": embed.type,
                "description": embed.description,
                "url": embed.url,
                "timestamp": embed.timestamp.isoformat() if embed.timestamp else None,
                "color": embed.color.value if embed.color else None,
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
            embeds.append(embed_data)

        # Convert stickers
        stickers: List[Dict[str, Any]] = []
        for sticker in message.stickers:
            sticker_data: Dict[str, Any] = {
                "id": str(sticker.id),
                "name": sticker.name,
                "formatType": str(sticker.format),
                # Some sticker types might not have description
                "description": getattr(sticker, "description", None),
                "url": str(sticker.url),
            }
            stickers.append(sticker_data)

        # Convert inline emojis (custom emojis in the message content)
        inline_emojis: List[InlineEmoji] = []
        for emoji in message.content.split():
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


class MessageStore:
    """Unified message storage system that handles messages in Discord Chat Exporter format."""

    def __init__(self, storage_dir: str = "message_store") -> None:
        """Initialize the message store.

        Args:
            storage_dir: Directory for storing channel message files
        """
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self._channel_messages: Dict[str, Dict[str, StoredMessage]] = {}
        self._channel_info: Dict[str, ChannelInfo] = {}
        self._guild_info: Optional[GuildInfo] = None
        self._channel_last_sync: Dict[str, datetime] = {}
        self._load_data()

    def _get_channel_file(self, channel_id: str) -> str:
        """Get the file path for a channel's messages."""
        return os.path.join(self.storage_dir, f"{channel_id}.json")

    def _load_data(self) -> None:
        """Load message data from storage directory."""
        try:
            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    channel_id = filename[:-5]  # Remove .json
                    file_path = os.path.join(self.storage_dir, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                        # Store guild info from first file we encounter
                        if not self._guild_info and "guild" in data:
                            self._guild_info = GuildInfo(**data["guild"])

                        # Store channel info
                        if "channel" in data:
                            self._channel_info[channel_id] = ChannelInfo(
                                **data["channel"]
                            )

                        # Store messages in a dictionary
                        messages: Dict[str, StoredMessage] = {}
                        for msg_data in data.get("messages", []):
                            # Convert nested structures
                            author_data = msg_data.pop("author")
                            author_roles = [
                                Role(**r) for r in author_data.pop("roles", [])
                            ]
                            author = UserInfo(**author_data, roles=author_roles)

                            mentions_data = msg_data.pop("mentions", [])
                            mentions: List[UserInfo] = []
                            for mention in mentions_data:
                                mention_roles = [
                                    Role(**r) for r in mention.pop("roles", [])
                                ]
                                mentions.append(
                                    UserInfo(**mention, roles=mention_roles)
                                )

                            attachments = [
                                Attachment(**a) for a in msg_data.pop("attachments", [])
                            ]
                            inline_emojis = [
                                InlineEmoji(**e)
                                for e in msg_data.pop("inlineEmojis", [])
                            ]

                            # Convert message reference if it exists
                            reference = None
                            if "reference" in msg_data:
                                reference_data = msg_data.pop("reference")
                                reference = MessageReference(**reference_data)

                            # Create the message and store by ID
                            stored_msg = StoredMessage(
                                **msg_data,
                                author=author,
                                mentions=mentions,
                                attachments=attachments,
                                inlineEmojis=inline_emojis,
                                reference=reference,
                            )
                            messages[stored_msg.id] = stored_msg

                        self._channel_messages[channel_id] = messages

            logger.info(f"Loaded messages from {len(self._channel_messages)} channels")
        except Exception as e:
            logger.error(f"Error loading message data: {str(e)}")
            raise  # Re-raise for testing

    def _save_channel_data(self, channel_id: str) -> None:
        """Save message data for a specific channel."""
        try:
            file_path = self._get_channel_file(channel_id)

            # Get messages as a sorted list for serialization
            messages = self._channel_messages.get(channel_id, {}).values()
            sorted_messages = sorted(
                messages,
                key=lambda m: datetime.fromisoformat(
                    m.timestamp.replace("Z", "+00:00")
                ),
            )

            data = {
                "guild": serialize_dataclass(self._guild_info),
                "channel": (
                    serialize_dataclass(self._channel_info[channel_id])
                    if channel_id in self._channel_info
                    else None
                ),
                "dateRange": {"after": None, "before": None},
                "exportedAt": datetime.now(timezone.utc).isoformat(),
                "messages": [serialize_dataclass(msg) for msg in sorted_messages],
                "messageCount": len(sorted_messages),
            }

            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Saved message data for channel {channel_id}")
        except Exception as e:
            logger.error(
                f"Error saving message data for channel {channel_id}: {str(e)}"
            )

    async def add_message(self, message: Message) -> None:
        """Add a new message to storage or update an existing one.

        Args:
            message: The Discord message to store
        """
        channel_id = str(message.channel.id)
        stored_msg = await StoredMessage.from_discord_message(message)

        if channel_id not in self._channel_messages:
            self._channel_messages[channel_id] = {}

            # Store channel info if we don't have it
            if channel_id not in self._channel_info:
                # Ensure we have a GuildChannel
                channel = cast(GuildChannel, message.channel)
                category = channel.category

                self._channel_info[channel_id] = ChannelInfo(
                    id=channel_id,
                    type="GuildTextChat",  # This might need to be more dynamic
                    categoryId=str(category.id) if category else None,
                    category=category.name if category else None,
                    name=str(channel.name),
                    topic=channel.topic if isinstance(channel, TextChannel) else None,
                )

            # Store guild info if we don't have it
            guild = cast(Guild, message.guild)
            if not self._guild_info and guild:
                self._guild_info = GuildInfo(
                    id=str(guild.id),
                    name=guild.name,
                    iconUrl=str(guild.icon.url) if guild.icon else None,
                )

        # Store or update the message
        self._channel_messages[channel_id][stored_msg.id] = stored_msg

    def get_message(self, channel_id: str, message_id: str) -> Optional[StoredMessage]:
        """Get a message by channel and message ID.

        Args:
            channel_id: The Discord channel ID
            message_id: The Discord message ID

        Returns:
            The stored message if found, None otherwise
        """
        return self._channel_messages.get(channel_id, {}).get(message_id)

    def get_channel_messages(
        self, channel_id: str, limit: Optional[int] = None
    ) -> Sequence[StoredMessage]:
        """Get messages from a channel.

        Args:
            channel_id: The Discord channel ID
            limit: Optional maximum number of messages to return (most recent)

        Returns:
            List of stored messages in chronological order
        """
        channel_dict = self._channel_messages.get(channel_id, {})
        messages = list(channel_dict.values())
        messages.sort(
            key=lambda m: datetime.fromisoformat(m.timestamp.replace("Z", "+00:00"))
        )
        if limit:
            return messages[-limit:]
        return messages

    async def initialize_channel(
        self, channel: TextChannel, refresh: bool = False
    ) -> None:
        """Initialize message history for a channel by fetching recent messages.

        Args:
            channel: The Discord channel to initialize history for
            refresh: Whether to refresh existing history
        """
        channel_id = str(channel.id)

        # Skip if history exists and refresh not requested
        if channel_id in self._channel_messages and not refresh:
            return

        try:
            # Clear existing messages if refreshing
            if refresh:
                self._channel_messages[channel_id] = {}

            logger.info(f"Fetching messages from channel {channel.name}")

            # Use the Discord API to fetch recent messages
            async for message in channel.history():
                await self.add_message(message)

            channel_dict = self._channel_messages.get(channel_id, {})
            logger.info(
                f"Initialized history for channel {channel.name} "
                f"with {len(channel_dict)} messages"
            )

        except Exception as e:
            logger.error(
                f"Error initializing history for channel {channel.name}: {str(e)}"
            )
            self._channel_messages[channel_id] = {}

    def save_all_channels(self) -> None:
        """Save all channel data to disk. For testing purposes."""
        for channel_id in self._channel_messages:
            self._save_channel_data(channel_id)

    def get_channel_ids(self) -> List[str]:
        """Get list of channel IDs. For testing purposes."""
        return list(self._channel_messages.keys())

    def _get_latest_message_time(self, channel_id: str) -> Optional[datetime]:
        """Get the timestamp of the most recent message in a channel.

        Args:
            channel_id: The Discord channel ID

        Returns:
            Datetime of the most recent message, or None if no messages exist
        """
        messages = self._channel_messages.get(channel_id, {})
        if not messages:
            return None

        # Find the latest message by comparing timestamps
        latest_time = None
        for msg in messages.values():
            msg_time = datetime.fromisoformat(msg.timestamp.replace("Z", "+00:00"))
            if latest_time is None or msg_time > latest_time:
                latest_time = msg_time

        return latest_time

    async def sync_channel(
        self, channel: TextChannel, overlap_minutes: int = 180
    ) -> None:
        """Synchronize messages for a channel since the last sync.

        This method will:
        1. Find messages newer than the most recent stored message
        2. Update metadata (like reactions) for recent messages
        3. Add new messages to storage

        Args:
            channel: The Discord channel to sync
            overlap_minutes: Number of minutes to overlap with existing messages for updating metadata
        """
        channel_id = str(channel.id)
        latest_time = self._get_latest_message_time(channel_id)

        try:
            logger.info(f"Syncing messages from channel {channel.name}")

            if latest_time:
                # Add overlap period to catch any edits/reactions on recent messages
                sync_after = latest_time - timedelta(minutes=overlap_minutes)

                logger.info(
                    f"Syncing messages after {sync_after.isoformat()} (with {overlap_minutes}m overlap)"
                )

                # Track progress
                message_count = 0
                new_messages = 0
                updated_messages = 0
                last_log_time = datetime.now()

                # Fetch messages after the sync point
                async for message in channel.history(after=sync_after, limit=None):
                    message_count += 1
                    stored_msg = self.get_message(channel_id, str(message.id))

                    if stored_msg:
                        # Message exists - update it if it's been edited or has reactions
                        if message.edited_at and (
                            not stored_msg.timestampEdited
                            or message.edited_at.isoformat()
                            != stored_msg.timestampEdited
                        ):
                            # Message was edited - update it
                            await self.add_message(message)
                            updated_messages += 1
                        elif message.reactions:
                            # Has reactions - update it
                            await self.add_message(message)
                            updated_messages += 1
                    else:
                        # New message - add it
                        await self.add_message(message)
                        new_messages += 1

                    # Log progress every 5 seconds
                    now = datetime.now()
                    if (now - last_log_time).total_seconds() >= 5:
                        logger.info(
                            f"Progress: processed {message_count} messages "
                            f"({new_messages} new, {updated_messages} updated)"
                        )
                        last_log_time = now

                logger.info(
                    f"Sync complete: processed {message_count} messages total "
                    f"({new_messages} new, {updated_messages} updated)"
                )
            else:
                # No existing messages - initialize the channel from newest to oldest
                logger.info("No existing messages found - initializing channel history")
                message_count = 0
                last_log_time = datetime.now()

                # Start from the most recent message
                async for message in channel.history(limit=None):
                    message_count += 1
                    await self.add_message(message)

                    # Log progress every 5 seconds
                    now = datetime.now()
                    if (now - last_log_time).total_seconds() >= 5:
                        logger.info(
                            f"Initial sync progress: {message_count} messages downloaded"
                        )
                        last_log_time = now

                logger.info(
                    f"Initial sync complete: downloaded {message_count} messages"
                )

            # Update last sync time
            self._channel_last_sync[channel_id] = datetime.now(timezone.utc)

            # Save changes
            self._save_channel_data(channel_id)

            channel_dict = self._channel_messages.get(channel_id, {})
            logger.info(
                f"Channel {channel.name} now has {len(channel_dict)} total messages stored"
            )

        except Exception as e:
            logger.error(f"Error syncing channel {channel.name}: {str(e)}")
            raise  # Re-raise for error handling
