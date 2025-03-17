"""SQLAlchemy-based storage management for Discord messages."""

import logging
from datetime import UTC
from typing import List, Optional

import pendulum
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from discord_types import Attachment as DiscordAttachment
from discord_types import ChannelInfo
from discord_types import InlineEmoji as DiscordInlineEmoji
from discord_types import MessageReference as DiscordMessageReference
from discord_types import Role, StoredMessage, UserInfo
from models import (
    Attachment,
    Base,
    Channel,
    Embed,
    InlineEmoji,
    Message,
    MessageMention,
    MessageReference,
    Reaction,
    Sticker,
    User,
    UserRole,
)
from time_tracking import ChannelMetadata
from utils.time_utils import parse_datetime

# Set up logging
logger = logging.getLogger("deepbot.sql_storage_manager")


class SQLStorageManager:
    """Manages SQLite storage of Discord messages and metadata."""

    def __init__(self, data_dir: str) -> None:
        """Initialize the storage manager.

        Args:
            data_dir: Directory to store the SQLite database
        """
        # Create database URL
        db_path = f"{data_dir}/messages.db"
        self.engine = create_engine(f"sqlite:///{db_path}")

        # Create all tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.Session = sessionmaker(bind=self.engine)

    def _convert_user_info(self, user_info: UserInfo) -> User:
        """Convert UserInfo to SQLAlchemy User model."""
        return User(
            id=user_info.id,
            name=user_info.name,
            discriminator=user_info.discriminator,
            nickname=user_info.nickname,
            color=user_info.color,
            isBot=user_info.isBot,
            avatarUrl=user_info.avatarUrl,
        )

    def _convert_message(self, message: StoredMessage, channel_id: str) -> Message:
        """Convert StoredMessage to SQLAlchemy Message model."""
        # Convert timestamps
        timestamp = parse_datetime(message.timestamp)
        timestamp_edited = (
            parse_datetime(message.timestampEdited) if message.timestampEdited else None
        )
        call_ended_timestamp = (
            parse_datetime(message.callEndedTimestamp)
            if message.callEndedTimestamp
            else None
        )

        # Create message model
        return Message(
            id=message.id,
            channel_id=channel_id,
            author_id=message.author.id,
            content=message.content or "",  # Ensure content is never None
            timestamp=timestamp,
            timestamp_edited=timestamp_edited,
            call_ended_timestamp=call_ended_timestamp,
            is_pinned=message.isPinned,
            type=message.type,
            attachments=[
                Attachment(
                    id=attachment.id,
                    url=attachment.url,
                    fileName=attachment.fileName,
                    fileSizeBytes=attachment.fileSizeBytes,
                    proxyUrl=attachment.proxyUrl,
                    width=attachment.width,
                    height=attachment.height,
                    contentType=attachment.contentType,
                )
                for attachment in message.attachments
            ],
            embeds=[
                Embed(
                    title=embed.get("title"),
                    type=embed.get("type", "rich"),
                    description=embed.get("description"),
                    url=embed.get("url"),
                    timestamp=(
                        parse_datetime(embed["timestamp"])
                        if embed.get("timestamp")
                        else None
                    ),
                    color=embed.get("color"),
                    footer_text=embed.get("footer", {}).get("text"),
                    footer_iconUrl=embed.get("footer", {}).get("iconUrl"),
                    image_url=embed.get("image", {}).get("url"),
                    image_proxyUrl=embed.get("image", {}).get("proxyUrl"),
                    image_width=embed.get("image", {}).get("width"),
                    image_height=embed.get("image", {}).get("height"),
                    thumbnail_url=embed.get("thumbnail", {}).get("url"),
                    thumbnail_proxyUrl=embed.get("thumbnail", {}).get("proxyUrl"),
                    thumbnail_width=embed.get("thumbnail", {}).get("width"),
                    thumbnail_height=embed.get("thumbnail", {}).get("height"),
                    video_url=embed.get("video", {}).get("url"),
                    video_width=embed.get("video", {}).get("width"),
                    video_height=embed.get("video", {}).get("height"),
                    provider_name=embed.get("provider", {}).get("name"),
                    provider_url=embed.get("provider", {}).get("url"),
                    author_name=embed.get("author", {}).get("name"),
                    author_url=embed.get("author", {}).get("url"),
                    author_iconUrl=embed.get("author", {}).get("iconUrl"),
                )
                for embed in message.embeds
            ],
            stickers=[
                Sticker(
                    id=sticker["id"],
                    name=sticker["name"],
                    formatType=sticker["formatType"],
                    description=sticker.get("description"),
                    url=sticker.get("url"),
                )
                for sticker in message.stickers
            ],
            reactions=[
                Reaction(
                    emoji_id=reaction["emoji"].get("id"),
                    emoji_name=reaction["emoji"]["name"],
                    emoji_code=reaction["emoji"]["code"],
                    isAnimated=reaction["emoji"].get("isAnimated", False),
                    emoji_imageUrl=reaction["emoji"].get("imageUrl"),
                    count=reaction["count"],
                )
                for reaction in message.reactions
            ],
            reference=(
                MessageReference(
                    message_id=message.id,
                    referenced_message_id=message.reference.messageId,
                    referenced_channel_id=message.reference.channelId,
                    referenced_guild_id=message.reference.guildId,
                )
                if message.reference
                else None
            ),
            inline_emojis=[
                InlineEmoji(
                    emoji_id=str(emoji.id),
                    name=emoji.name,
                    code=emoji.code,
                    isAnimated=emoji.isAnimated,
                    imageUrl=emoji.imageUrl,
                )
                for emoji in message.inlineEmojis
            ],
        )

    def _convert_channel(self, channel_info: ChannelInfo) -> Channel:
        """Convert ChannelInfo to SQLAlchemy Channel model."""
        return Channel(
            id=channel_info.id,
            name=channel_info.name,
            type=channel_info.type,
            topic=channel_info.topic,
            # Set default values for required fields that aren't in ChannelInfo
            guild_id=None,
            position=None,
            permissions_overwrites=[],
            parent_id=channel_info.categoryId,  # Use categoryId as parent_id
            nsfw=False,
            last_message_id=None,
            rate_limit_per_user=None,
            bitrate=None,
            user_limit=None,
        )

    def _convert_to_stored_message(self, message: Message) -> StoredMessage:
        """Convert SQLAlchemy Message model back to StoredMessage."""
        if not message or not message.author:
            raise ValueError("Message and author must not be None")

        # Convert timestamps to ISO format strings
        timestamp = message.timestamp.astimezone(UTC).isoformat()
        timestamp_edited = (
            message.timestamp_edited.astimezone(UTC).isoformat()
            if message.timestamp_edited
            else None
        )
        call_ended_timestamp = (
            message.call_ended_timestamp.astimezone(UTC).isoformat()
            if message.call_ended_timestamp
            else None
        )

        # Get author roles from UserRole table
        with Session(self.engine) as session:
            user_roles = (
                session.query(UserRole)
                .filter(UserRole.user_id == message.author.id)
                .all()
            )
            roles = [
                Role(
                    id=user_role.role.id,
                    name=user_role.role.name,
                    color=user_role.role.color,
                    position=user_role.role.position,
                )
                for user_role in user_roles
            ]

        # Convert author
        author = UserInfo(
            id=message.author.id,
            name=message.author.name,
            discriminator=message.author.discriminator,
            nickname=message.author.nickname,
            color=message.author.color,
            isBot=message.author.isBot,
            roles=roles,
            avatarUrl=message.author.avatarUrl,
        )

        # Convert mentions
        mentions = [UserInfo(**user.__dict__) for user in message.mentions]

        # Convert reference
        reference = (
            DiscordMessageReference(**message.reference.__dict__)
            if message.reference
            else None
        )

        # Convert attachments
        attachments = [
            DiscordAttachment(
                id=attachment.id,
                url=attachment.url,
                fileName=attachment.fileName,
                fileSizeBytes=attachment.fileSizeBytes,
                proxyUrl=attachment.proxyUrl,
                width=attachment.width,
                height=attachment.height,
                contentType=attachment.contentType,
            )
            for attachment in message.attachments
        ]

        # Convert embeds
        embeds = [
            {
                "title": embed.title,
                "type": embed.type,
                "description": embed.description,
                "url": embed.url,
                "timestamp": embed.timestamp.isoformat() if embed.timestamp else None,
                "color": embed.color,
                "footer": (
                    {
                        "text": embed.footer_text,
                        "iconUrl": embed.footer_iconUrl,
                    }
                    if embed.footer_text or embed.footer_iconUrl
                    else None
                ),
                "image": (
                    {
                        "url": embed.image_url,
                        "proxyUrl": embed.image_proxyUrl,
                        "width": embed.image_width,
                        "height": embed.image_height,
                    }
                    if embed.image_url
                    or embed.image_proxyUrl
                    or embed.image_width
                    or embed.image_height
                    else None
                ),
                "thumbnail": (
                    {
                        "url": embed.thumbnail_url,
                        "proxyUrl": embed.thumbnail_proxyUrl,
                        "width": embed.thumbnail_width,
                        "height": embed.thumbnail_height,
                    }
                    if embed.thumbnail_url
                    or embed.thumbnail_proxyUrl
                    or embed.thumbnail_width
                    or embed.thumbnail_height
                    else None
                ),
                "video": (
                    {
                        "url": embed.video_url,
                        "width": embed.video_width,
                        "height": embed.video_height,
                    }
                    if embed.video_url or embed.video_width or embed.video_height
                    else None
                ),
                "provider": (
                    {
                        "name": embed.provider_name,
                        "url": embed.provider_url,
                    }
                    if embed.provider_name or embed.provider_url
                    else None
                ),
                "author": (
                    {
                        "name": embed.author_name,
                        "url": embed.author_url,
                        "iconUrl": embed.author_iconUrl,
                    }
                    if embed.author_name or embed.author_url or embed.author_iconUrl
                    else None
                ),
            }
            for embed in message.embeds
        ]

        # Convert stickers
        stickers = [
            {
                "id": sticker.id,
                "name": sticker.name,
                "formatType": sticker.formatType,
                "description": sticker.description,
                "url": sticker.url,
            }
            for sticker in message.stickers
        ]

        # Convert reactions
        reactions = [
            {
                "emoji": {
                    "id": reaction.emoji_id,
                    "name": reaction.emoji_name,
                    "code": reaction.emoji_code,
                    "isAnimated": reaction.isAnimated,
                    "imageUrl": reaction.emoji_imageUrl,
                },
                "count": reaction.count,
            }
            for reaction in message.reactions
        ]

        # Convert inline emojis
        inline_emojis = [
            DiscordInlineEmoji(
                id=str(emoji.id),
                name=emoji.name,
                code=emoji.code,
                isAnimated=emoji.isAnimated,
                imageUrl=emoji.imageUrl,
            )
            for emoji in message.inline_emojis
        ]

        return StoredMessage(
            id=message.id,
            type=message.type,
            timestamp=timestamp,
            timestampEdited=timestamp_edited,
            callEndedTimestamp=call_ended_timestamp,
            isPinned=message.is_pinned,
            content=message.content or "",  # Ensure content is never None
            author=author,
            attachments=attachments,
            embeds=embeds,
            stickers=stickers,
            reactions=reactions,
            mentions=mentions,
            reference=reference,
            inlineEmojis=inline_emojis,
        )

    def get_channel_ids(self) -> List[str]:
        """Get all channel IDs."""
        with Session(self.engine) as session:
            channels = session.query(Channel.id).all()
            return [channel[0] for channel in channels]

    def get_message(self, channel_id: str, message_id: str) -> Optional[StoredMessage]:
        """Get a specific message by ID."""
        with Session(self.engine) as session:
            message = (
                session.query(Message)
                .filter(
                    Message.id == message_id,
                    Message.channel_id == channel_id,
                )
                .first()
            )
            return self._convert_to_stored_message(message) if message else None

    def get_channel_messages(
        self, channel_id: str, limit: Optional[int] = None
    ) -> List[StoredMessage]:
        """Get all messages for a channel."""
        with Session(self.engine) as session:
            query = (
                session.query(Message)
                .filter(Message.channel_id == channel_id)
                .order_by(Message.timestamp)
            )

            if limit:
                query = query.limit(limit)

            messages = query.all()
            return [self._convert_to_stored_message(msg) for msg in messages]

    def add_message(self, channel_id: str, message: StoredMessage) -> None:
        """Add a message to storage."""
        with Session(self.engine) as session:
            # Add or update author
            author = self._convert_user_info(message.author)
            session.merge(author)

            # Add or update channel
            channel = session.query(Channel).filter(Channel.id == channel_id).first()
            if not channel:
                channel = Channel(id=channel_id)
                session.add(channel)

            # Add or update message
            db_message = self._convert_message(message, channel_id)
            session.merge(db_message)

            # Update mentions
            for mention in message.mentions:
                db_mention = self._convert_user_info(mention)
                session.merge(db_mention)
                mention_assoc = MessageMention(
                    message_id=message.id,
                    user_id=mention.id,
                )
                session.merge(mention_assoc)

            session.commit()

    def get_channel_metadata(self, channel_id: str) -> Optional[ChannelMetadata]:
        """Get metadata for a channel."""
        with Session(self.engine) as session:
            channel = session.query(Channel).filter(Channel.id == channel_id).first()
            if not channel:
                return None

            # TODO: Implement proper metadata storage
            return ChannelMetadata(
                channel_id=channel_id,
                known_ranges=[],
                gaps=[],
                last_sync=pendulum.instance(channel.last_sync),
            )

    def ensure_channel_metadata(self, channel_id: str) -> None:
        """Ensure metadata exists for a channel."""
        with Session(self.engine) as session:
            channel = session.query(Channel).filter(Channel.id == channel_id).first()
            if not channel:
                channel = Channel(
                    id=channel_id,
                    last_sync=pendulum.now("UTC"),
                )
                session.add(channel)
                session.commit()
