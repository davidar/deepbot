"""Real-time message synchronization between Discord and MessageStore."""

import asyncio
import logging
from typing import Set

from discord import (
    Message,
    RawMessageDeleteEvent,
    RawMessageUpdateEvent,
    RawReactionActionEvent,
    TextChannel,
)
from discord.ext import commands, tasks

from message_store import MessageStore

# Set up logging
logger = logging.getLogger("deepbot.message_sync")


class MessageSyncManager:
    """Manages real-time synchronization between Discord events and MessageStore."""

    def __init__(
        self, bot: commands.Bot, message_store: MessageStore, sync_interval: int = 30
    ) -> None:
        """Initialize the sync manager.

        Args:
            bot: The Discord bot instance
            message_store: The message store to keep updated
            sync_interval: Minutes between background syncs (default: 30)
        """
        self.bot = bot
        self.message_store = message_store
        self.sync_interval = sync_interval
        self._sync_lock = asyncio.Lock()
        self._is_syncing = False
        self._active_channels: Set[int] = set()  # Track active channel IDs

        # Register event handlers
        self._register_handlers()

        # Start background sync
        self.background_sync.start()

    def _register_handlers(self) -> None:
        """Register all event handlers with the bot."""
        self.bot.add_listener(self.on_message, "on_message")
        self.bot.add_listener(self.on_raw_message_edit, "on_raw_message_edit")
        self.bot.add_listener(self.on_raw_message_delete, "on_raw_message_delete")
        self.bot.add_listener(self.on_raw_reaction_add, "on_raw_reaction_add")
        self.bot.add_listener(self.on_raw_reaction_remove, "on_raw_reaction_remove")

    def mark_channel_active(self, channel_id: int) -> None:
        """Mark a channel as active for syncing.

        Args:
            channel_id: The Discord channel ID
        """
        if channel_id not in self._active_channels:
            self._active_channels.add(channel_id)
            logger.debug(f"Marked channel {channel_id} as active for syncing")

    async def on_message(self, message: Message) -> None:
        """Handle new messages.

        Args:
            message: The new Discord message
        """
        try:
            # Only track channels where the bot sends messages
            if message.author == self.bot.user:
                self.mark_channel_active(message.channel.id)

            await self.message_store.add_message(message)
            logger.debug(f"Added new message {message.id} to store")
        except Exception as e:
            logger.error(f"Error adding message {message.id} to store: {str(e)}")

    async def on_raw_message_edit(self, payload: RawMessageUpdateEvent) -> None:
        """Handle message edits.

        Args:
            payload: The raw message update event data
        """
        try:
            channel_id = int(payload.data["channel_id"])
            if channel_id not in self._active_channels:
                return

            # Get the updated message object
            channel = self.bot.get_channel(channel_id)
            if not isinstance(channel, TextChannel):
                return

            message = await channel.fetch_message(payload.message_id)
            await self.message_store.add_message(message)
            logger.debug(f"Updated edited message {message.id} in store")
        except Exception as e:
            logger.error(
                f"Error updating edited message {payload.message_id} in store: {str(e)}"
            )

    async def on_raw_message_delete(self, payload: RawMessageDeleteEvent) -> None:
        """Handle message deletions.

        Args:
            payload: The raw message delete event data
        """
        if payload.channel_id not in self._active_channels:
            return

        # Note: Currently MessageStore doesn't support message deletion
        # This is a placeholder for when that functionality is added
        logger.debug(
            f"Message {payload.message_id} was deleted (not yet handled by store)"
        )

    async def on_raw_reaction_add(self, payload: RawReactionActionEvent) -> None:
        """Handle reaction additions.

        Args:
            payload: The raw reaction action event data
        """
        try:
            if payload.channel_id not in self._active_channels:
                return

            # Get the message that was reacted to
            channel = self.bot.get_channel(payload.channel_id)
            if not isinstance(channel, TextChannel):
                return

            message = await channel.fetch_message(payload.message_id)
            await self.message_store.add_message(message)
            logger.debug(f"Updated message {message.id} with new reaction in store")
        except Exception as e:
            logger.error(
                f"Error updating reaction for message {payload.message_id} in store: {str(e)}"
            )

    async def on_raw_reaction_remove(self, payload: RawReactionActionEvent) -> None:
        """Handle reaction removals.

        Args:
            payload: The raw reaction action event data
        """
        try:
            if payload.channel_id not in self._active_channels:
                return

            # Get the message that had a reaction removed
            channel = self.bot.get_channel(payload.channel_id)
            if not isinstance(channel, TextChannel):
                return

            message = await channel.fetch_message(payload.message_id)
            await self.message_store.add_message(message)
            logger.debug(f"Updated message {message.id} with removed reaction in store")
        except Exception as e:
            logger.error(
                f"Error updating removed reaction for message {payload.message_id} in store: {str(e)}"
            )

    @tasks.loop(minutes=30)
    async def background_sync(self) -> None:
        """Background task to periodically sync all active channels."""
        if self._is_syncing:
            logger.debug("Skipping background sync - another sync is in progress")
            return

        async with self._sync_lock:
            try:
                self._is_syncing = True
                logger.info(
                    f"Starting background sync of {len(self._active_channels)} active channels"
                )

                for channel_id in self._active_channels:
                    channel = self.bot.get_channel(channel_id)
                    if not isinstance(channel, TextChannel):
                        continue

                    try:
                        await self.message_store.sync_channel(channel)
                        logger.debug(
                            f"Background sync completed for channel {channel.name}"
                        )
                    except Exception as e:
                        logger.error(f"Error syncing channel {channel.name}: {str(e)}")
                        continue

                logger.info("Background sync completed for all active channels")
            finally:
                self._is_syncing = False

    @background_sync.before_loop
    async def before_background_sync(self) -> None:
        """Wait for bot to be ready before starting background sync."""
        await self.bot.wait_until_ready()

    def cog_unload(self) -> None:
        """Clean up when the cog is unloaded."""
        self.background_sync.cancel()
