"""Script to sync messages for a specific Discord channel."""

import argparse
import asyncio
import logging
from typing import Optional

import discord
from discord import TextChannel
from discord.ext import commands

import config
from message_store import MessageStore

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sync_channel")

# Verify Discord token is available
if not config.DISCORD_TOKEN:
    raise ValueError("Discord token not found in config")


class ChannelSyncBot(commands.Bot):
    """Simple bot to sync a specific channel."""

    def __init__(self, channel_id: int, store_dir: str) -> None:
        """Initialize the bot.

        Args:
            channel_id: The Discord channel ID to sync
            store_dir: Directory for the message store
        """
        # Set up intents like the main bot
        intents = discord.Intents.default()
        intents.message_content = True
        intents.messages = True
        intents.reactions = True

        super().__init__(command_prefix="!", intents=intents)

        self.target_channel_id = channel_id
        self.message_store = MessageStore(store_dir)
        self.channel: Optional[TextChannel] = None
        self.sync_complete = asyncio.Event()

    async def setup_hook(self) -> None:
        """Set up hook that runs after the bot is logged in."""
        self.loop.create_task(self.sync_task())

    async def sync_task(self) -> None:
        """Task to perform the sync after the bot is ready."""
        await self.wait_until_ready()

        try:
            # Get the channel
            channel = self.get_channel(self.target_channel_id)
            if not isinstance(channel, TextChannel):
                raise ValueError(
                    f"Channel {self.target_channel_id} not found or not a text channel"
                )
            self.channel = channel

            # Sync the channel
            logger.info(f"Starting sync for channel #{channel.name} ({channel.id})")
            await self.message_store.sync_channel(channel)
            logger.info("Sync completed successfully")

        except Exception as e:
            logger.error(f"Error during sync: {str(e)}")
            raise
        finally:
            # Signal completion and close the bot
            self.sync_complete.set()
            await self.close()


async def main(channel_id: int, store_dir: str) -> None:
    """Main entry point.

    Args:
        channel_id: The Discord channel ID to sync
        store_dir: Directory for the message store
    """
    bot = ChannelSyncBot(channel_id, store_dir)

    try:
        # Start the bot and wait for sync to complete
        async with bot:
            await bot.start(str(config.DISCORD_TOKEN))
            await bot.sync_complete.wait()
    except KeyboardInterrupt:
        logger.info("Sync cancelled by user")
        if not bot.is_closed():
            await bot.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync messages for a Discord channel")
    parser.add_argument("channel_id", type=int, help="Discord channel ID to sync")
    parser.add_argument(
        "--store-dir",
        type=str,
        default="message_store",
        help="Directory for message store (default: message_store)",
    )

    args = parser.parse_args()

    try:
        asyncio.run(main(args.channel_id, args.store_dir))
    except KeyboardInterrupt:
        logger.info("Sync cancelled by user")
    except Exception as e:
        logger.error(f"Sync failed: {str(e)}")
        exit(1)
