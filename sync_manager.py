"""Channel synchronization and gap filling for Discord messages."""

import logging
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from discord import Message

if TYPE_CHECKING:
    from discord.abc import MessageableChannel

from discord_types import StoredMessage
from message_indexer import MessageIndexer
from storage_manager import StorageManager
from time_tracking import TimeRange
from utils import get_channel_name

# Set up logging
logger = logging.getLogger("deepbot.sync_manager")


class SyncManager:
    """Manages synchronization between Discord and local message storage."""

    def __init__(
        self,
        storage_manager: StorageManager,
        message_indexer: MessageIndexer | None,
    ) -> None:
        """Initialize the sync manager.

        Args:
            storage_manager: The storage manager instance
            message_indexer: Optional message indexer instance
        """
        self.storage_manager = storage_manager
        self.message_indexer = message_indexer

    async def _fill_gaps(
        self,
        channel: "MessageableChannel",
        channel_name: str,
        channel_id: str,
        recent_gaps: list[TimeRange],
    ) -> None:
        """Fill gaps in message history.

        Args:
            channel: The Discord channel
            channel_name: The channel name for logging
            channel_id: The channel ID
            recent_gaps: List of gaps to fill
        """
        logger.info(
            f"Found {len(recent_gaps)} gaps in recent history for channel {channel_name}"
        )

        # Fetch messages to fill the gaps
        message_count = 0
        for gap in recent_gaps:
            logger.info(
                f"Fetching messages from {gap.start.isoformat()} to {gap.end.isoformat()}"
            )
            async for message in channel.history(
                after=gap.start, before=gap.end, limit=None
            ):
                await self.add_message(message)
                message_count += 1

            # Update known range for the gap we just filled
            metadata = self.storage_manager.get_channel_metadata(channel_id)
            if metadata:
                metadata.add_known_range(TimeRange(start=gap.start, end=gap.end))

        if message_count > 0:
            logger.info(f"Filled gaps with {message_count} messages")

    async def _sync_recent_messages(
        self,
        channel: "MessageableChannel",
        channel_name: str,
        channel_id: str,
        latest_time: datetime,
    ) -> None:
        """Sync recent messages if we're behind.

        Args:
            channel: The Discord channel
            channel_name: The channel name for logging
            channel_id: The channel ID
            latest_time: Timestamp of the latest message
        """
        now = datetime.now(timezone.utc)
        time_since_last = now - latest_time
        if time_since_last > timedelta(
            minutes=5
        ):  # If we're more than 5 minutes behind
            logger.info(f"Syncing recent messages for channel {channel_name}")
            await self.sync_channel(channel, overlap_minutes=5)

    async def initialize_channel(self, channel: "MessageableChannel") -> None:
        """Initialize message history for a channel by fetching recent messages.

        This method will:
        1. Check for gaps in recent history
        2. Fetch messages to fill any gaps
        3. Update metadata about known ranges

        Args:
            channel: The Discord channel to initialize history for
        """
        channel_id = str(channel.id)
        channel_name = get_channel_name(channel)

        try:
            # Ensure we have metadata for this channel
            self.storage_manager.ensure_channel_metadata(channel_id)
            metadata = self.storage_manager.get_channel_metadata(channel_id)
            if not metadata:
                logger.error(f"Failed to create metadata for channel {channel_name}")
                return

            # Check for gaps in the last 24 hours
            recent_window = timedelta(hours=24)
            recent_gaps = metadata.get_recent_gaps(recent_window)

            if recent_gaps:
                await self._fill_gaps(channel, channel_name, channel_id, recent_gaps)
            else:
                # No gaps, but we should still check if we need to sync recent messages
                messages = self.storage_manager.get_channel_messages(channel_id)
                if messages:
                    latest_time = datetime.fromisoformat(messages[-1].timestamp)
                    await self._sync_recent_messages(
                        channel, channel_name, channel_id, latest_time
                    )
                else:
                    # No messages at all, do an initial sync
                    logger.info(
                        f"No messages found for channel {channel_name}, doing initial sync"
                    )
                    await self.sync_channel(channel)

            # Save changes
            self.storage_manager.save_channel_data(channel_id)

        except Exception as e:
            logger.error(
                f"Error initializing history for channel {channel_name}: {str(e)}"
            )
            raise  # Re-raise for error handling

    async def add_message(self, message: Message) -> None:
        """Add a new message to storage and index.

        Args:
            message: The Discord message to store
        """
        channel_id = str(message.channel.id)
        stored_msg = await StoredMessage.from_discord_message(message)

        # Store the message
        self.storage_manager.add_message(channel_id, stored_msg)

        # Index the message if indexing is enabled
        if self.message_indexer:
            self.message_indexer.index_message(stored_msg, channel_id)

    async def _sync_messages_after(
        self,
        channel: "MessageableChannel",
        sync_after: datetime,
        channel_id: str,
        channel_name: str,
    ) -> None:
        """Sync messages after a given timestamp.

        Args:
            channel: The Discord channel
            sync_after: Timestamp to sync messages after
            channel_id: The channel ID
            channel_name: The channel name for logging
        """
        # Track progress
        message_count = 0
        new_messages = 0
        updated_messages = 0
        last_log_time = datetime.now(timezone.utc)

        logger.info(f"Syncing messages after {sync_after.isoformat()}")

        # Fetch messages after the sync point
        async for message in channel.history(after=sync_after, limit=None):
            message_count += 1
            stored_msg = self.storage_manager.get_message(channel_id, str(message.id))

            if stored_msg:
                # Message exists - update it if it's been edited or has reactions
                if message.edited_at and (
                    not stored_msg.timestampEdited
                    or message.edited_at.isoformat() != stored_msg.timestampEdited
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
            now = datetime.now(timezone.utc)
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

    async def _initial_sync(
        self,
        channel: "MessageableChannel",
        channel_id: str,
        channel_name: str,
    ) -> None:
        """Perform initial sync for a channel with no existing messages.

        Args:
            channel: The Discord channel
            channel_id: The channel ID
            channel_name: The channel name for logging
        """
        logger.info("No existing messages found - initializing channel history")
        message_count = 0
        last_log_time = datetime.now(timezone.utc)

        # Start from the most recent message
        async for message in channel.history(limit=None):
            message_count += 1
            await self.add_message(message)

            # Log progress every 5 seconds
            now = datetime.now(timezone.utc)
            if (now - last_log_time).total_seconds() >= 5:
                logger.info(
                    f"Initial sync progress: {message_count} messages downloaded"
                )
                last_log_time = now

        # Update known range for initial sync
        messages = self.storage_manager.get_channel_messages(channel_id)
        if messages:
            first_msg = messages[0]
            last_msg = messages[-1]
            metadata = self.storage_manager.get_channel_metadata(channel_id)
            if metadata:
                metadata.add_known_range(
                    TimeRange(
                        start=datetime.fromisoformat(first_msg.timestamp),
                        end=datetime.fromisoformat(last_msg.timestamp),
                    )
                )

        logger.info(f"Initial sync complete: downloaded {message_count} messages")

    async def sync_channel(
        self, channel: "MessageableChannel", overlap_minutes: int = 180
    ) -> None:
        """Synchronize messages for a channel since the last sync.

        This method will:
        1. Find messages newer than the most recent stored message
        2. Update metadata (like reactions) for recent messages
        3. Add new messages to storage
        4. Track gaps in message history

        Args:
            channel: The Discord channel to sync
            overlap_minutes: Number of minutes to overlap with existing messages for updating metadata
        """
        channel_id = str(channel.id)
        channel_name = get_channel_name(channel)

        logger.info(f"Syncing messages from channel {channel_name}")

        try:
            # Initialize channel if needed
            self.storage_manager.ensure_channel_metadata(channel_id)
            metadata = self.storage_manager.get_channel_metadata(channel_id)
            if not metadata:
                logger.error(f"Failed to create metadata for channel {channel_name}")
                return

            messages = self.storage_manager.get_channel_messages(channel_id)
            now = datetime.now(timezone.utc)

            if messages:
                # Add overlap period to catch any edits/reactions on recent messages
                latest_time = datetime.fromisoformat(messages[-1].timestamp)
                sync_after = latest_time - timedelta(minutes=overlap_minutes)
                await self._sync_messages_after(
                    channel, sync_after, channel_id, channel_name
                )
                # Update known range for this sync
                metadata.add_known_range(TimeRange(start=sync_after, end=now))
            else:
                # No existing messages - initialize the channel from newest to oldest
                await self._initial_sync(channel, channel_id, channel_name)

            # Update last sync time
            metadata.last_sync = now

            # Save changes
            self.storage_manager.save_channel_data(channel_id)

            channel_messages = self.storage_manager.get_channel_messages(channel_id)
            logger.info(
                f"Channel {channel_name} now has {len(channel_messages)} total messages stored"
            )

        except Exception as e:
            logger.error(f"Error syncing channel {channel_name}: {str(e)}")
            raise  # Re-raise for error handling
