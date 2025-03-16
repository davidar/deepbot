"""File-based storage management for Discord messages."""

import json
import logging
import os
from typing import Any, Dict, List, Optional

import pendulum

from discord_types import (
    ChannelInfo,
    GuildInfo,
    MessageReference,
    Role,
    StoredMessage,
    UserInfo,
    serialize_dataclass,
)
from time_tracking import ChannelMetadata, TimeRange
from utils.time_utils import parse_datetime

# Set up logging
logger = logging.getLogger("deepbot.storage_manager")


class StorageManager:
    """Manages file-based storage of Discord messages and metadata."""

    def __init__(self, data_dir: str) -> None:
        """Initialize the storage manager.

        Args:
            data_dir: Directory to store message data and metadata
        """
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        # Initialize storage
        self.messages: Dict[str, Dict[str, StoredMessage]] = {}
        self.channel_metadata: Dict[str, ChannelMetadata] = {}
        self._guild_info: Optional[GuildInfo] = None
        self._channel_info: Dict[str, ChannelInfo] = {}

    def _get_channel_file(self, channel_id: str) -> str:
        """Get the file path for a channel's messages."""
        return os.path.join(self.data_dir, f"{channel_id}.json")

    def _get_metadata_file(self, channel_id: str) -> str:
        """Get the file path for a channel's metadata."""
        return os.path.join(self.data_dir, f"{channel_id}_metadata.json")

    def _load_metadata(self, channel_id: str) -> None:
        """Load metadata for a channel."""
        try:
            file_path = self._get_metadata_file(channel_id)
            logger.debug(f"Attempting to load metadata from {file_path}")

            if os.path.exists(file_path):
                logger.debug(f"Found metadata file for channel {channel_id}")
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.debug(f"Raw metadata content: {data}")

                    # Convert string timestamps back to datetime
                    known_ranges = [
                        TimeRange(
                            start=parse_datetime(r["start"]),
                            end=parse_datetime(r["end"]),
                        )
                        for r in data["known_ranges"]
                    ]
                    gaps = [
                        TimeRange(
                            start=parse_datetime(r["start"]),
                            end=parse_datetime(r["end"]),
                        )
                        for r in data["gaps"]
                    ]
                    last_sync = parse_datetime(data["last_sync"])

                    self.channel_metadata[channel_id] = ChannelMetadata(
                        channel_id=channel_id,
                        known_ranges=known_ranges,
                        gaps=gaps,
                        last_sync=last_sync,
                    )
                    logger.debug(
                        f"Successfully loaded metadata for channel {channel_id}"
                    )
            else:
                logger.debug(f"No metadata file found at {file_path}")
        except Exception as e:
            logger.error(
                f"Error loading metadata for channel {channel_id}: {str(e)}",
                exc_info=True,
            )
            self.channel_metadata[channel_id] = ChannelMetadata(
                channel_id=channel_id,
                known_ranges=[],
                gaps=[],
                last_sync=pendulum.now("UTC"),
            )

    def _save_metadata(self, channel_id: str) -> None:
        """Save metadata for a channel."""
        try:
            metadata = self.channel_metadata.get(channel_id)
            if not metadata:
                return

            file_path = self._get_metadata_file(channel_id)
            data = {
                "known_ranges": [
                    {
                        "start": r.start.to_iso8601_string(),
                        "end": r.end.to_iso8601_string(),
                    }
                    for r in metadata.known_ranges
                ],
                "gaps": [
                    {
                        "start": r.start.to_iso8601_string(),
                        "end": r.end.to_iso8601_string(),
                    }
                    for r in metadata.gaps
                ],
                "last_sync": metadata.last_sync.to_iso8601_string(),
            }
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata for channel {channel_id}: {str(e)}")

    def _load_guild_info(self, data: Dict[str, Any]) -> None:
        """Load guild information from data."""
        if not self._guild_info and "guild" in data and data["guild"]:
            self._guild_info = GuildInfo(**data["guild"])

    def _load_channel_info(self, channel_id: str, data: Dict[str, Any]) -> None:
        """Load channel information from data."""
        if "channel" in data and data["channel"]:
            self._channel_info[channel_id] = ChannelInfo(**data["channel"])

    def _convert_roles(self, roles_data: List[Dict[str, Any]]) -> List[Role]:
        """Convert role data to Role objects."""
        return [Role(**r) for r in roles_data]

    def _convert_user_info(self, user_data: Dict[str, Any]) -> UserInfo:
        """Convert user data to UserInfo object."""
        roles_data = user_data.pop("roles", [])
        roles = self._convert_roles(roles_data)

        # Ensure nickname field exists
        if "nickname" not in user_data:
            user_data["nickname"] = None

        return UserInfo(**user_data, roles=roles)

    def _convert_message_data(self, msg_data: Dict[str, Any]) -> StoredMessage:
        """Convert message data to StoredMessage object."""
        # Make a copy to avoid modifying the original
        msg_data = msg_data.copy()

        # Convert author
        author_data = msg_data.pop("author")
        author = self._convert_user_info(author_data)

        # Convert mentions
        mentions_data = msg_data.pop("mentions", [])
        mentions = [self._convert_user_info(mention) for mention in mentions_data]

        # Convert reference if it exists
        reference_data = msg_data.pop("reference", None)
        reference = None
        if reference_data:
            reference = MessageReference(
                messageId=reference_data["messageId"],
                channelId=reference_data["channelId"],
                guildId=reference_data["guildId"],
            )

        # Create and return the message
        return StoredMessage(
            **msg_data,
            author=author,
            mentions=mentions,
            reference=reference,
        )

    def _load_channel_messages(self, channel_id: str, data: Dict[str, Any]) -> None:
        """Load messages for a channel."""
        messages: Dict[str, StoredMessage] = {}
        for msg_data in data.get("messages", []):
            stored_msg = self._convert_message_data(msg_data)
            messages[stored_msg.id] = stored_msg
        self.messages[channel_id] = messages

    def load_all_data(self) -> None:
        """Load all message data from storage directory."""
        try:
            logger.debug(f"Starting to load data from {self.data_dir}")
            logger.debug(f"Directory contents: {os.listdir(self.data_dir)}")

            for filename in os.listdir(self.data_dir):
                if filename.endswith(".json") and not filename.endswith(
                    "_metadata.json"
                ):
                    channel_id = filename[:-5]  # Remove .json
                    file_path = os.path.join(self.data_dir, filename)
                    logger.debug(
                        f"Processing message file: {filename} for channel {channel_id}"
                    )

                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                        # Load guild and channel info
                        self._load_guild_info(data)
                        self._load_channel_info(channel_id, data)

                        # Load messages
                        self._load_channel_messages(channel_id, data)

                        # Load metadata
                        self._load_metadata(channel_id)

            logger.info(f"Loaded messages from {len(self.messages)} channels")
            logger.debug(
                f"Loaded metadata for channels: {list(self.channel_metadata.keys())}"
            )
        except Exception as e:
            logger.error(f"Error loading message data: {str(e)}", exc_info=True)
            raise

    def save_channel_data(self, channel_id: str) -> None:
        """Save message data for a specific channel."""
        try:
            file_path = self._get_channel_file(channel_id)

            # Get messages as a sorted list for serialization
            messages = self.messages.get(channel_id, {}).values()
            sorted_messages = sorted(
                messages,
                key=lambda m: parse_datetime(m.timestamp),
            )

            # Ensure we have guild info
            guild_data: Optional[Dict[str, Optional[str]]] = None
            if self._guild_info:
                guild_data = {
                    "id": self._guild_info.id,
                    "name": self._guild_info.name,
                    "iconUrl": self._guild_info.iconUrl,
                }

            # Ensure we have channel info
            channel_data: Optional[Dict[str, Optional[str]]] = None
            if channel_id in self._channel_info:
                channel_info = self._channel_info[channel_id]
                channel_data = {
                    "id": channel_info.id,
                    "type": channel_info.type,
                    "categoryId": channel_info.categoryId,
                    "category": channel_info.category,
                    "name": channel_info.name,
                    "topic": channel_info.topic,
                }

            # Prepare data for serialization
            data = {
                "exportedAt": pendulum.now("UTC").to_iso8601_string(),
                "guild": guild_data,
                "channel": channel_data,
                "messages": [serialize_dataclass(msg) for msg in sorted_messages],
            }

            # Save to file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            # Save metadata
            self._save_metadata(channel_id)
        except Exception as e:
            logger.error(f"Error saving data for channel {channel_id}: {str(e)}")

    def get_message(self, channel_id: str, message_id: str) -> Optional[StoredMessage]:
        """Get a specific message by ID."""
        return self.messages.get(channel_id, {}).get(message_id)

    def get_channel_messages(
        self, channel_id: str, limit: Optional[int] = None
    ) -> List[StoredMessage]:
        """Get all messages for a channel."""
        messages = list(self.messages.get(channel_id, {}).values())
        messages.sort(key=lambda m: parse_datetime(m.timestamp))
        if limit:
            return messages[:limit]
        return messages

    def add_message(self, channel_id: str, message: StoredMessage) -> None:
        """Add a message to storage."""
        if channel_id not in self.messages:
            self.messages[channel_id] = {}
        self.messages[channel_id][message.id] = message

    def get_channel_ids(self) -> List[str]:
        """Get all channel IDs."""
        return list(self.messages.keys())

    def get_channel_metadata(self, channel_id: str) -> Optional[ChannelMetadata]:
        """Get metadata for a channel."""
        return self.channel_metadata.get(channel_id)

    def ensure_channel_metadata(self, channel_id: str) -> None:
        """Ensure metadata exists for a channel."""
        if channel_id not in self.channel_metadata:
            self.channel_metadata[channel_id] = ChannelMetadata(
                channel_id=channel_id,
                known_ranges=[],
                gaps=[],
                last_sync=pendulum.now("UTC"),
            )
