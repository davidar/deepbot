"""Script to import JSON message files into SQLite database."""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pendulum
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from models import (
    Attachment,
    Base,
    Channel,
    Embed,
    EmbedField,
    InlineEmoji,
    Message,
    MessageMention,
    MessageReference,
    Reaction,
    ReactionUser,
    Role,
    Sticker,
    User,
    UserRole,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("deepbot.import_json")


def validate_channel_data(channel_data: Dict[str, Any]) -> bool:
    """Validate channel data from JSON file.

    Args:
        channel_data: Channel data dictionary

    Returns:
        True if data is valid, False otherwise
    """
    required_fields = ["id", "type", "name"]
    if not all(field in channel_data for field in required_fields):
        logger.error(f"Missing required fields in channel data: {required_fields}")
        return False
    return True


def validate_message_data(message_data: Dict[str, Any]) -> bool:
    """Validate message data from JSON file.

    Args:
        message_data: Message data dictionary

    Returns:
        True if data is valid, False otherwise
    """
    required_fields = ["id", "timestamp"]
    if not all(field in message_data for field in required_fields):
        logger.error(f"Missing required fields in message data: {required_fields}")
        return False
    return True


def load_json_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Load and parse a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed JSON data or None if file doesn't exist or is invalid
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data: Dict[str, Any] = json.load(f)

            # Validate required top-level fields
            if not all(field in data for field in ["channel", "messages"]):
                logger.error(f"Missing required fields in JSON file: {file_path}")
                return None

            # Validate channel data
            if not validate_channel_data(data["channel"]):
                logger.error(f"Invalid channel data in file: {file_path}")
                return None

            return data
    except FileNotFoundError:
        logger.warning(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {file_path}: {e}")
        return None


def convert_timestamp(timestamp_str: str) -> pendulum.DateTime:
    """Convert ISO format timestamp string to pendulum DateTime.

    Args:
        timestamp_str: ISO format timestamp string

    Returns:
        pendulum DateTime object
    """
    dt = pendulum.parse(timestamp_str)
    if not isinstance(dt, pendulum.DateTime):
        raise ValueError(f"Invalid timestamp format: {timestamp_str}")
    return dt


def process_roles(
    session: Session, user_id: str, roles_data: List[Dict[str, Any]]
) -> None:
    """Process and create roles for a user.

    Args:
        session: SQLAlchemy session
        user_id: ID of the user
        roles_data: List of role data dictionaries
    """
    for role_data in roles_data:
        # Create or update role
        role = session.query(Role).filter(Role.id == role_data["id"]).first()
        if not role:
            role = Role(
                id=role_data["id"],
                name=role_data["name"],
                color=role_data.get("color"),
                position=role_data["position"],
            )
            session.add(role)

        # Create user-role association
        user_role = UserRole(user_id=user_id, role_id=role.id)
        session.add(user_role)


def process_attachments(
    session: Session, message_id: str, attachments_data: List[Dict[str, Any]]
) -> None:
    """Process and create attachments for a message.

    Args:
        session: SQLAlchemy session
        message_id: ID of the message
        attachments_data: List of attachment data dictionaries
    """
    for attachment_data in attachments_data:
        attachment = Attachment(
            id=attachment_data["id"],
            message_id=message_id,
            url=attachment_data["url"],
            fileName=attachment_data["fileName"],
            fileSizeBytes=attachment_data["fileSizeBytes"],
            proxyUrl=attachment_data.get("proxyUrl"),
            width=attachment_data.get("width"),
            height=attachment_data.get("height"),
            contentType=attachment_data.get("contentType"),
        )
        session.add(attachment)


def process_embeds(
    session: Session, message_id: str, embeds_data: List[Dict[str, Any]]
) -> None:
    """Process and create embeds for a message.

    Args:
        session: SQLAlchemy session
        message_id: ID of the message
        embeds_data: List of embed data dictionaries
    """
    for embed_data in embeds_data:
        embed = Embed(
            message_id=message_id,
            title=embed_data.get("title"),
            type=embed_data.get("type", "rich"),
            description=embed_data.get("description"),
            url=embed_data.get("url"),
            timestamp=(
                convert_timestamp(embed_data["timestamp"])
                if embed_data.get("timestamp")
                else None
            ),
            color=embed_data.get("color"),
            footer_text=embed_data.get("footer", {}).get("text"),
            footer_iconUrl=embed_data.get("footer", {}).get("iconUrl"),
            image_url=embed_data.get("image", {}).get("url"),
            image_proxyUrl=embed_data.get("image", {}).get("proxyUrl"),
            image_width=embed_data.get("image", {}).get("width"),
            image_height=embed_data.get("image", {}).get("height"),
            thumbnail_url=embed_data.get("thumbnail", {}).get("url"),
            thumbnail_proxyUrl=embed_data.get("thumbnail", {}).get("proxyUrl"),
            thumbnail_width=embed_data.get("thumbnail", {}).get("width"),
            thumbnail_height=embed_data.get("thumbnail", {}).get("height"),
            video_url=embed_data.get("video", {}).get("url"),
            video_width=embed_data.get("video", {}).get("width"),
            video_height=embed_data.get("video", {}).get("height"),
            provider_name=embed_data.get("provider", {}).get("name"),
            provider_url=embed_data.get("provider", {}).get("url"),
            author_name=embed_data.get("author", {}).get("name"),
            author_url=embed_data.get("author", {}).get("url"),
            author_iconUrl=embed_data.get("author", {}).get("iconUrl"),
        )
        session.add(embed)
        session.flush()  # Get embed ID

        # Process embed fields
        for field_data in embed_data.get("fields", []):
            field = EmbedField(
                embed_id=embed.id,
                name=field_data["name"],
                value=field_data["value"],
                inline=field_data.get("inline", False),
            )
            session.add(field)


def process_reactions(
    session: Session, message_id: str, reactions_data: List[Dict[str, Any]]
) -> None:
    """Process and create reactions for a message.

    Args:
        session: SQLAlchemy session
        message_id: ID of the message
        reactions_data: List of reaction data dictionaries
    """
    for reaction_data in reactions_data:
        emoji_data = reaction_data["emoji"]
        reaction = Reaction(
            message_id=message_id,
            emoji_id=emoji_data.get("id"),
            emoji_name=emoji_data["name"],
            emoji_code=emoji_data["code"],
            isAnimated=emoji_data.get("isAnimated", False),
            emoji_imageUrl=emoji_data.get("imageUrl"),
            count=reaction_data["count"],
        )
        session.add(reaction)
        session.flush()  # Get reaction ID

        # Process reaction users
        for user_data in reaction_data.get("users", []):
            reaction_user = ReactionUser(
                reaction_id=reaction.id,
                user_id=user_data["id"],
            )
            session.add(reaction_user)


def process_stickers(
    session: Session, message_id: str, stickers_data: List[Dict[str, Any]]
) -> None:
    """Process and create stickers for a message.

    Args:
        session: SQLAlchemy session
        message_id: ID of the message
        stickers_data: List of sticker data dictionaries
    """
    for sticker_data in stickers_data:
        sticker = Sticker(
            id=sticker_data["id"],
            message_id=message_id,
            name=sticker_data["name"],
            formatType=sticker_data["formatType"],
            description=sticker_data.get("description"),
            url=sticker_data.get("url"),
        )
        session.add(sticker)


def process_inline_emojis(
    session: Session, message_id: str, emojis_data: List[Dict[str, Any]]
) -> None:
    """Process and create inline emojis for a message.

    Args:
        session: SQLAlchemy session
        message_id: ID of the message
        emojis_data: List of inline emoji data dictionaries
    """
    for emoji_data in emojis_data:
        emoji = InlineEmoji(
            message_id=message_id,
            emoji_id=emoji_data.get("id"),
            name=emoji_data["name"],
            code=emoji_data["code"],
            isAnimated=emoji_data.get("isAnimated", False),
            imageUrl=emoji_data["imageUrl"],
        )
        session.add(emoji)


def process_message_reference(
    session: Session, message_id: str, reference_data: Optional[Dict[str, Any]]
) -> None:
    """Process and create message reference.

    Args:
        session: SQLAlchemy session
        message_id: ID of the message
        reference_data: Message reference data dictionary
    """
    if reference_data:
        reference = MessageReference(
            message_id=message_id,
            referenced_message_id=reference_data["messageId"],
            referenced_channel_id=reference_data["channelId"],
            referenced_guild_id=reference_data["guildId"],
        )
        session.add(reference)


def process_mentions(
    session: Session, message_id: str, mentions_data: List[Dict[str, Any]]
) -> None:
    """Process and create message mentions.

    Args:
        session: SQLAlchemy session
        message_id: ID of the message
        mentions_data: List of mention data dictionaries
    """
    for mention_data in mentions_data:
        mention = MessageMention(
            message_id=message_id,
            user_id=mention_data["id"],
        )
        session.add(mention)


def import_channel_data(
    session: Session,
    file_data: Dict[str, Any],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> tuple[int, int]:
    """Import data for a single channel.

    Args:
        session: SQLAlchemy session
        file_data: Complete JSON data from file
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (messages_imported, users_imported)
    """
    messages_imported = 0
    users_imported = 0

    # Get channel data from file
    channel_data = file_data["channel"]
    channel_id = channel_data["id"]

    # Create or update channel
    channel = session.query(Channel).filter(Channel.id == channel_id).first()
    if not channel:
        channel = Channel(
            id=channel_id,
            name=channel_data["name"],
            type=channel_data["type"],
            guild_id=file_data.get("guild", {}).get("id"),
            position=channel_data.get("position"),
            permissions_overwrites=channel_data.get("permissionsOverwrites", []),
            parent_id=channel_data.get("categoryId"),
            nsfw=channel_data.get("nsfw", False),
            rate_limit_per_user=channel_data.get("rateLimitPerUser"),
            topic=channel_data.get("topic"),
            bitrate=channel_data.get("bitrate"),
            user_limit=channel_data.get("userLimit"),
            last_sync=convert_timestamp(
                file_data.get("exportedAt", pendulum.now("UTC").isoformat())
            ),
        )
        session.add(channel)
        session.commit()

    # Process messages
    messages = file_data.get("messages", [])
    total_messages = len(messages)
    logger.info(f"Processing {total_messages} messages for channel {channel_id}")

    for i, msg_data in enumerate(messages, 1):
        try:
            # Validate message data
            if not validate_message_data(msg_data):
                logger.error(f"Skipping invalid message data at index {i}")
                continue

            # Create or update author
            author_data = msg_data.get("author", {})
            author_id = author_data.get("id")
            if author_id:
                author = session.query(User).filter(User.id == author_id).first()
                if not author:
                    author = User(
                        id=author_id,
                        name=author_data.get("name", ""),
                        discriminator=author_data.get("discriminator", "0"),
                        nickname=author_data.get("nickname"),
                        color=author_data.get("color"),
                        isBot=author_data.get("isBot", False),
                        avatarUrl=author_data.get("avatarUrl", ""),
                    )
                    session.add(author)
                    users_imported += 1

                    # Process author roles
                    process_roles(session, author_id, author_data.get("roles", []))

            # Create message
            message = Message(
                id=msg_data["id"],
                channel_id=channel_id,
                author_id=author_id,
                content=msg_data.get("content", ""),
                timestamp=convert_timestamp(msg_data["timestamp"]),
                timestamp_edited=(
                    convert_timestamp(msg_data.get("timestampEdited"))
                    if msg_data.get("timestampEdited")
                    else None
                ),
                call_ended_timestamp=(
                    convert_timestamp(msg_data.get("callEndedTimestamp"))
                    if msg_data.get("callEndedTimestamp")
                    else None
                ),
                is_pinned=msg_data.get("isPinned", False),
                type=msg_data.get("type", "Default"),
            )
            session.add(message)
            session.flush()  # Get message ID

            # Process message components
            process_attachments(session, message.id, msg_data.get("attachments", []))
            process_embeds(session, message.id, msg_data.get("embeds", []))
            process_reactions(session, message.id, msg_data.get("reactions", []))
            process_stickers(session, message.id, msg_data.get("stickers", []))
            process_inline_emojis(session, message.id, msg_data.get("inlineEmojis", []))
            process_message_reference(session, message.id, msg_data.get("reference"))
            process_mentions(session, message.id, msg_data.get("mentions", []))

            messages_imported += 1

            # Commit every 100 messages to avoid memory issues
            if i % 100 == 0:
                session.commit()
                if progress_callback:
                    progress_callback(i, total_messages)

        except Exception as e:
            logger.error(f"Error processing message {msg_data.get('id')}: {e}")
            session.rollback()
            continue

    # Final commit for this channel
    session.commit()
    return messages_imported, users_imported


def import_data(
    data_dir: str,
    db_path: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """Import all JSON data into SQLite database.

    Args:
        data_dir: Directory containing JSON files
        db_path: Path to SQLite database file
        progress_callback: Optional callback for progress updates
    """
    # Create database and tables
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)

    # Get list of JSON files
    data_path = Path(data_dir)
    json_files = list(data_path.glob("*.json"))
    total_files = len(json_files)

    logger.info(f"Found {total_files} JSON files to process")

    total_messages = 0
    total_users = 0

    for i, file_path in enumerate(json_files, 1):
        logger.info(f"Processing file {i}/{total_files}: {file_path}")

        # Load and validate file data
        file_data = load_json_file(str(file_path))
        if not file_data:
            continue

        # Import channel data
        with Session(engine) as session:
            messages, users = import_channel_data(session, file_data, progress_callback)
            total_messages += messages
            total_users += users

        if progress_callback:
            progress_callback(i, total_files)

    logger.info(f"Import complete: {total_messages} messages, {total_users} users")


def main() -> None:
    """Main entry point for the import script."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Import JSON message files into SQLite database"
    )
    parser.add_argument("data_dir", help="Directory containing JSON files")
    parser.add_argument("db_path", help="Path to SQLite database file")
    args = parser.parse_args()

    def progress_callback(current: int, total: int) -> None:
        """Print progress updates."""
        print(f"Progress: {current}/{total} ({(current/total)*100:.1f}%)")

    import_data(args.data_dir, args.db_path, progress_callback)


if __name__ == "__main__":
    main()
