"""Message history management for DeepBot."""

import logging
from typing import TYPE_CHECKING, List

from discord import Message

import config
from utils import get_channel_name

if TYPE_CHECKING:
    from discord.abc import MessageableChannel

# Set up logging
logger = logging.getLogger("deepbot.history")


class MessageHistoryManager:
    """Manages message history for Discord channels."""

    def __init__(self) -> None:
        """Initialize the message history manager."""
        self._message_history: dict[int, list[Message]] = {}

    def has_history(self, channel_id: int) -> bool:
        """Check if a channel has message history.

        Args:
            channel_id: The Discord channel ID

        Returns:
            True if the channel has history, False otherwise
        """
        return channel_id in self._message_history and bool(
            self._message_history[channel_id]
        )

    def get_messages(self, channel_id: int) -> List[Message]:
        """Get the message history for a channel.

        Args:
            channel_id: The Discord channel ID

        Returns:
            List of Discord messages
        """
        return self._message_history.get(channel_id, [])

    def get_history_length(self, channel_id: int) -> int:
        """Get the number of messages in a channel's history.

        Args:
            channel_id: The Discord channel ID

        Returns:
            Number of messages in history
        """
        return len(self._message_history.get(channel_id, []))

    def add_message(self, message: Message) -> None:
        """Add a message to the history.

        Args:
            message: The Discord message to add
        """
        channel_id = message.channel.id
        if channel_id not in self._message_history:
            self._message_history[channel_id] = []
        self._message_history[channel_id].append(message)

    async def initialize_channel(
        self, channel: "MessageableChannel", refresh: bool = False
    ) -> bool:
        """Initialize message history for a channel by fetching recent messages.

        Args:
            channel: The Discord channel to initialize history for
        """
        channel_id = channel.id

        # Skip if history already exists for this channel
        if self.has_history(channel_id) and not refresh:
            return False

        try:
            # Fetch recent messages from the channel
            message_limit = config.load_model_options()["history_fetch_limit"]
            self._message_history[channel_id] = []

            logger.info(
                f"Fetching up to {message_limit} messages from channel {get_channel_name(channel)}"
            )

            # Use the Discord API to fetch recent messages
            async for message in channel.history(limit=message_limit):
                self.add_message(message)

            # Sort messages by timestamp
            self._message_history[channel_id].sort(key=lambda m: m.created_at)

            logger.info(
                "Initialized history for channel {} with {} messages".format(
                    get_channel_name(channel), len(self._message_history[channel_id])
                )
            )
            return True

        except Exception as e:
            logger.error(
                "Error initializing history for channel {}: {}".format(
                    get_channel_name(channel), str(e)
                )
            )
            self._message_history[channel_id] = []
            return False
