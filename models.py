"""SQLAlchemy models for the message store."""

from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class Role(Base):
    """SQLAlchemy model for Discord roles."""

    __tablename__ = "roles"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    color: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    position: Mapped[int] = mapped_column(Integer)

    # Relationships
    user_roles: Mapped[List["UserRole"]] = relationship(back_populates="role")


class UserRole(Base):
    """Association table for user roles."""

    __tablename__ = "user_roles"

    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    role_id: Mapped[str] = mapped_column(
        String, ForeignKey("roles.id", ondelete="CASCADE"), primary_key=True
    )

    # Relationships
    user: Mapped["User"] = relationship(back_populates="user_roles")
    role: Mapped[Role] = relationship(back_populates="user_roles")


class User(Base):
    """SQLAlchemy model for Discord users."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    discriminator: Mapped[str] = mapped_column(String)
    nickname: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    color: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    isBot: Mapped[bool] = mapped_column(Boolean, default=False)
    avatarUrl: Mapped[str] = mapped_column(String)

    # Relationships
    messages: Mapped[List["Message"]] = relationship(back_populates="author")
    mentions: Mapped[List["Message"]] = relationship(
        secondary="message_mentions", back_populates="mentions"
    )
    user_roles: Mapped[List[UserRole]] = relationship(back_populates="user")


class InlineEmoji(Base):
    """SQLAlchemy model for inline emojis in messages."""

    __tablename__ = "inline_emojis"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[str] = mapped_column(
        String, ForeignKey("messages.id", ondelete="CASCADE")
    )
    emoji_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    name: Mapped[str] = mapped_column(String)
    code: Mapped[str] = mapped_column(String)
    isAnimated: Mapped[bool] = mapped_column(Boolean, default=False)
    imageUrl: Mapped[str] = mapped_column(String)

    # Relationships
    message: Mapped["Message"] = relationship(back_populates="inline_emojis")


class Attachment(Base):
    """SQLAlchemy model for message attachments."""

    __tablename__ = "attachments"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    message_id: Mapped[str] = mapped_column(
        String, ForeignKey("messages.id", ondelete="CASCADE")
    )
    url: Mapped[str] = mapped_column(String)
    fileName: Mapped[str] = mapped_column(String)
    fileSizeBytes: Mapped[int] = mapped_column(Integer)
    proxyUrl: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    contentType: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Relationships
    message: Mapped["Message"] = relationship(back_populates="attachments")


class MessageReference(Base):
    """SQLAlchemy model for message references (replies)."""

    __tablename__ = "message_references"

    message_id: Mapped[str] = mapped_column(
        String, ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True
    )
    referenced_message_id: Mapped[str] = mapped_column(
        String, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    referenced_channel_id: Mapped[str] = mapped_column(
        String, ForeignKey("channels.id", ondelete="SET NULL"), nullable=True
    )
    referenced_guild_id: Mapped[str] = mapped_column(String)

    # Relationships
    message: Mapped["Message"] = relationship(
        back_populates="reference",
        foreign_keys=[message_id],
    )
    referenced_message: Mapped[Optional["Message"]] = relationship(
        foreign_keys=[referenced_message_id],
    )
    referenced_channel: Mapped[Optional["Channel"]] = relationship(
        foreign_keys=[referenced_channel_id],
    )


class Reaction(Base):
    """SQLAlchemy model for message reactions."""

    __tablename__ = "reactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[str] = mapped_column(
        String, ForeignKey("messages.id", ondelete="CASCADE")
    )
    emoji_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    emoji_name: Mapped[str] = mapped_column(String)
    emoji_code: Mapped[str] = mapped_column(String)
    isAnimated: Mapped[bool] = mapped_column(Boolean, default=False)
    emoji_imageUrl: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    count: Mapped[int] = mapped_column(Integer)

    # Relationships
    message: Mapped["Message"] = relationship(back_populates="reactions")
    users: Mapped[List["ReactionUser"]] = relationship(back_populates="reaction")


class ReactionUser(Base):
    """Association table for reaction users."""

    __tablename__ = "reaction_users"

    reaction_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("reactions.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )

    # Relationships
    reaction: Mapped[Reaction] = relationship(back_populates="users")
    user: Mapped[User] = relationship()


class Sticker(Base):
    """SQLAlchemy model for message stickers."""

    __tablename__ = "stickers"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    message_id: Mapped[str] = mapped_column(
        String, ForeignKey("messages.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String)
    formatType: Mapped[str] = mapped_column(String)
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Relationships
    message: Mapped["Message"] = relationship(back_populates="stickers")


class Embed(Base):
    """SQLAlchemy model for message embeds."""

    __tablename__ = "embeds"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    message_id: Mapped[str] = mapped_column(
        String, ForeignKey("messages.id", ondelete="CASCADE")
    )
    title: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    type: Mapped[str] = mapped_column(String)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    color: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    footer_text: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    footer_iconUrl: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    image_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    image_proxyUrl: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    image_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    image_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    thumbnail_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    thumbnail_proxyUrl: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    thumbnail_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    thumbnail_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    video_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    video_width: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    video_height: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    provider_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    provider_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    author_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    author_url: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    author_iconUrl: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Relationships
    message: Mapped["Message"] = relationship(back_populates="embeds")
    fields: Mapped[List["EmbedField"]] = relationship(back_populates="embed")


class EmbedField(Base):
    """SQLAlchemy model for embed fields."""

    __tablename__ = "embed_fields"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    embed_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("embeds.id", ondelete="CASCADE")
    )
    name: Mapped[str] = mapped_column(String)
    value: Mapped[str] = mapped_column(Text)
    inline: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    embed: Mapped[Embed] = relationship(back_populates="fields")


class TimeRange(Base):
    """SQLAlchemy model for time ranges."""

    __tablename__ = "time_ranges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    channel_id: Mapped[str] = mapped_column(
        String, ForeignKey("channels.id", ondelete="CASCADE")
    )
    start: Mapped[datetime] = mapped_column(DateTime)
    end: Mapped[datetime] = mapped_column(DateTime)
    is_gap: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationship
    channel: Mapped["Channel"] = relationship(back_populates="time_ranges")


class Channel(Base):
    """SQLAlchemy model for Discord channels."""

    __tablename__ = "channels"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    type: Mapped[str] = mapped_column(String)
    guild_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    position: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    permissions_overwrites: Mapped[List[Dict[str, Any]]] = mapped_column(
        JSON, default=list
    )
    parent_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("channels.id", ondelete="SET NULL"), nullable=True
    )
    nsfw: Mapped[bool] = mapped_column(Boolean, default=False)
    last_message_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("messages.id", ondelete="SET NULL"), nullable=True
    )
    rate_limit_per_user: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    topic: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    bitrate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    user_limit: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    last_sync: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(UTC)
    )

    # Relationships
    messages: Mapped[List["Message"]] = relationship(
        back_populates="channel",
        foreign_keys="Message.channel_id",
    )
    time_ranges: Mapped[List[TimeRange]] = relationship(back_populates="channel")
    parent: Mapped[Optional["Channel"]] = relationship(
        remote_side=[id], backref="children"
    )
    last_message: Mapped[Optional["Message"]] = relationship(
        foreign_keys=[last_message_id],
    )


class Message(Base):
    """SQLAlchemy model for Discord messages."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    channel_id: Mapped[str] = mapped_column(
        String, ForeignKey("channels.id", ondelete="CASCADE")
    )
    author_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="SET NULL")
    )
    content: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime)
    timestamp_edited: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    call_ended_timestamp: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    is_pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    type: Mapped[str] = mapped_column(String)

    # Relationships
    channel: Mapped["Channel"] = relationship(
        back_populates="messages",
        foreign_keys=[channel_id],
    )
    author: Mapped[Optional["User"]] = relationship(
        back_populates="messages",
        foreign_keys=[author_id],
    )
    mentions: Mapped[List["User"]] = relationship(
        secondary="message_mentions",
        back_populates="mentions",
    )
    attachments: Mapped[List[Attachment]] = relationship(
        back_populates="message",
        foreign_keys="Attachment.message_id",
    )
    embeds: Mapped[List[Embed]] = relationship(
        back_populates="message",
        foreign_keys="Embed.message_id",
    )
    stickers: Mapped[List[Sticker]] = relationship(
        back_populates="message",
        foreign_keys="Sticker.message_id",
    )
    reactions: Mapped[List[Reaction]] = relationship(
        back_populates="message",
        foreign_keys="Reaction.message_id",
    )
    reference: Mapped[Optional[MessageReference]] = relationship(
        back_populates="message",
        foreign_keys="MessageReference.message_id",
    )
    inline_emojis: Mapped[List[InlineEmoji]] = relationship(
        back_populates="message",
        foreign_keys="InlineEmoji.message_id",
    )


class MessageMention(Base):
    """Association table for message mentions."""

    __tablename__ = "message_mentions"

    message_id: Mapped[str] = mapped_column(
        String, ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[str] = mapped_column(
        String, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
