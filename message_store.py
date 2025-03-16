"""Main message store coordinating storage, indexing, and synchronization."""

import logging
import os
from typing import Any, Callable, Dict, List, Optional

import discord
from discord import Message

from discord_types import StoredMessage
from message_indexer import MessageIndexer
from storage_manager import StorageManager
from sync_manager import SyncManager

# Set up logging
logger = logging.getLogger("deepbot.message_store")


class MessageStore:
    """Main class coordinating message storage, indexing, and synchronization."""

    def __init__(
        self,
        data_dir: str,
        message_indexer: Optional[MessageIndexer] = None,
    ) -> None:
        """Initialize the message store.

        Args:
            data_dir: Directory to store message data and metadata
            message_indexer: Optional MessageIndexer instance for search functionality
        """
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

        # Initialize components
        self.storage_manager = StorageManager(data_dir)
        self.message_indexer = message_indexer
        self.sync_manager = SyncManager(self.storage_manager, self.message_indexer)

        # Load existing data
        self.storage_manager.load_all_data()

    def get_channel_ids(self) -> List[str]:
        """Get the list of all channel IDs in the store.

        Returns:
            List of channel IDs
        """
        return self.storage_manager.get_channel_ids()

    async def initialize_channel(self, channel: discord.TextChannel) -> None:
        """Initialize message history for a channel.

        Args:
            channel: The Discord channel to initialize
        """
        await self.sync_manager.initialize_channel(channel)

    async def sync_channel(self, channel: discord.TextChannel) -> None:
        """Synchronize messages for a channel.

        Args:
            channel: The Discord channel to sync
        """
        await self.sync_manager.sync_channel(channel)

    async def add_message(self, message: Message) -> None:
        """Add a new message to storage and index.

        Args:
            message: The Discord message to store
        """
        await self.sync_manager.add_message(message)
        self.storage_manager.save_channel_data(str(message.channel.id))

    def get_message(self, channel_id: str, message_id: str) -> Optional[StoredMessage]:
        """Get a message by channel and message ID.

        Args:
            channel_id: The Discord channel ID
            message_id: The Discord message ID

        Returns:
            The stored message if found, None otherwise
        """
        return self.storage_manager.get_message(channel_id, message_id)

    def get_channel_messages(
        self, channel_id: str, limit: Optional[int] = None
    ) -> List[StoredMessage]:
        """Get messages from a channel.

        Args:
            channel_id: The Discord channel ID
            limit: Optional maximum number of messages to return (most recent)

        Returns:
            List of stored messages in chronological order
        """
        return self.storage_manager.get_channel_messages(channel_id, limit)

    async def search(
        self, query: str, top_k: int = 5, **filters: Dict[str, Any]
    ) -> Dict[str, List[StoredMessage]]:
        """Search for messages matching the query.

        Args:
            query: The search query
            top_k: Maximum number of results to return per channel
            **filters: Optional filters to apply (e.g. channel_id, author_id)

        Returns:
            Dict mapping channel IDs to lists of matching messages

        Raises:
            RuntimeError: If search is called but indexing is not enabled
        """
        if not self.message_indexer:
            raise RuntimeError("Search is not available - indexing is not enabled")

        logger.debug(
            f"Searching with query: {query}, top_k: {top_k}, filters: {filters}"
        )
        nodes = await self.message_indexer.search(query, top_k, **filters)
        logger.debug(f"Vector store returned {len(nodes)} nodes")

        # Group results by channel
        results: Dict[str, List[StoredMessage]] = {}
        for i, node in enumerate(nodes):
            logger.debug(f"Processing node {i+1}/{len(nodes)}")
            metadata = node.metadata
            if not metadata:
                logger.debug(f"Node {i+1} has no metadata, skipping")
                continue

            logger.debug(f"Node {i+1} metadata: {metadata}")
            channel_id = metadata.get("channel_id")
            message_id = metadata.get("message_id")

            if not channel_id or not message_id:
                logger.debug(f"Node {i+1} missing channel_id or message_id, skipping")
                continue

            logger.debug(f"Looking up message {message_id} in channel {channel_id}")
            message = self.get_message(channel_id, message_id)

            if message:
                logger.debug(f"Found message {message_id} in store")
                if channel_id not in results:
                    results[channel_id] = []
                results[channel_id].append(message)
            else:
                logger.debug(f"Message {message_id} not found in store")

        logger.debug(f"Returning results from {len(results)} channels")
        return results

    def reindex_all_messages(
        self, progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> None:
        """Force reindexing of all messages in the store.

        Args:
            progress_callback: Optional callback function(processed_count, total_count) for progress updates

        Raises:
            RuntimeError: If indexing is not enabled
        """
        if not self.message_indexer:
            raise RuntimeError("Cannot reindex - indexing is not enabled")

        # Get total message count for progress reporting
        total_messages = 0
        processed_messages = 0
        for channel_id in self.get_channel_ids():
            messages = self.get_channel_messages(channel_id)
            total_messages += len(messages)

        # Index all messages
        for channel_id in self.get_channel_ids():
            messages = self.get_channel_messages(channel_id)
            for message in messages:
                self.message_indexer.index_message(message, channel_id)
                processed_messages += 1
                if progress_callback:
                    progress_callback(processed_messages, total_messages)

    def save_channel_data(self, channel_id: str) -> None:
        """Save message data for a specific channel.

        Args:
            channel_id: The Discord channel ID
        """
        self.storage_manager.save_channel_data(channel_id)
