"""Real-time message synchronization between Discord and MessageStore."""

import asyncio
import logging

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

    def _is_channel_active(self, channel_id: str) -> bool:
        """Check if a channel is being tracked in the message store.

        Args:
            channel_id: The Discord channel ID

        Returns:
            True if the channel exists in the message store
        """
        return channel_id in self.message_store.get_channel_ids()

    async def on_message(self, message: Message) -> None:
        """Handle new messages.

        Args:
            message: The new Discord message
        """
        try:
            channel_id = str(message.channel.id)
            if not self._is_channel_active(channel_id):
                return

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
            channel_id = str(payload.data["channel_id"])
            if not self._is_channel_active(channel_id):
                return

            # Get the updated message object
            channel = self.bot.get_channel(int(channel_id))
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
        if not self._is_channel_active(str(payload.channel_id)):
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
            channel_id = str(payload.channel_id)
            if not self._is_channel_active(channel_id):
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
            channel_id = str(payload.channel_id)
            if not self._is_channel_active(channel_id):
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
                active_channels = self.message_store.get_channel_ids()
                logger.info(
                    f"Starting background sync of {len(active_channels)} active channels"
                )

                for channel_id in active_channels:
                    channel = self.bot.get_channel(int(channel_id))
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
