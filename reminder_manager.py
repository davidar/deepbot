"""Reminder management system for the bot."""

import datetime
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import discord
from discord import DMChannel, GroupChannel, TextChannel

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from llm_streaming import LLMResponseHandler

# Set up logging
logger = logging.getLogger("deepbot.reminders")


class ReminderManager:
    """Manager for handling reminders."""

    _instance = None
    REMINDERS_FILE = "reminders.json"

    def __new__(cls):
        """Ensure singleton pattern for ReminderManager."""
        if cls._instance is None:
            cls._instance = super(ReminderManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the reminder manager."""
        if getattr(self, "_initialized", False):
            return

        self.reminders: Dict[str, Dict[str, Any]] = {}
        self._initialized = True
        self._load_reminders()
        self.llm_handler: Optional["LLMResponseHandler"] = None

        logger.info("Reminder manager initialized")

    def set_llm_handler(self, llm_handler: "LLMResponseHandler") -> None:
        """Set the LLM handler for generating personalized reminders.

        Args:
            llm_handler: The LLM response handler
        """
        self.llm_handler = llm_handler
        logger.info("LLM handler set for reminder manager")

    def _load_reminders(self) -> None:
        """Load reminders from file."""
        try:
            if Path(self.REMINDERS_FILE).exists():
                with open(self.REMINDERS_FILE, "r") as f:
                    self.reminders = json.load(f)
                logger.info(f"Loaded {len(self.reminders)} reminders from file")
            else:
                logger.info("No reminders file found, starting with empty reminders")
        except Exception as e:
            logger.error(f"Error loading reminders: {str(e)}")
            self.reminders = {}

    def _save_reminders(self) -> None:
        """Save reminders to file."""
        try:
            with open(self.REMINDERS_FILE, "w") as f:
                json.dump(self.reminders, f, indent=2)
            logger.info(f"Saved {len(self.reminders)} reminders to file")
        except Exception as e:
            logger.error(f"Error saving reminders: {str(e)}")

    def add_reminder(
        self,
        reminder_id: str,
        channel_id: int,
        user_id: int,
        content: str,
        due_time: datetime.datetime,
        message_id: int,
    ) -> None:
        """Add a new reminder.

        Args:
            reminder_id: Unique ID for the reminder
            channel_id: Discord channel ID
            user_id: Discord user ID
            content: Reminder content
            due_time: When the reminder should trigger
            message_id: ID of the original message that set the reminder
        """
        self.reminders[reminder_id] = {
            "channel_id": channel_id,
            "user_id": user_id,
            "content": content,
            "due_time": due_time.isoformat(),
            "created_at": datetime.datetime.now().isoformat(),
            "message_id": message_id,
        }
        self._save_reminders()
        logger.info(f"Added reminder {reminder_id} due at {due_time}")

    def remove_reminder(self, reminder_id: str) -> bool:
        """Remove a reminder.

        Args:
            reminder_id: ID of the reminder to remove

        Returns:
            True if the reminder was removed, False otherwise
        """
        if reminder_id in self.reminders:
            del self.reminders[reminder_id]
            self._save_reminders()
            logger.info(f"Removed reminder {reminder_id}")
            return True
        return False

    def get_due_reminders(self) -> List[Dict[str, Any]]:
        """Get all reminders that are due.

        Returns:
            List of due reminders
        """
        now = datetime.datetime.now()
        due_reminders: List[Dict[str, Any]] = []

        for reminder_id, reminder in list(self.reminders.items()):
            due_time = datetime.datetime.fromisoformat(reminder["due_time"])
            if due_time <= now:
                due_reminder = reminder.copy()
                due_reminder["id"] = reminder_id
                due_reminders.append(due_reminder)

        return due_reminders

    async def _get_channel(
        self, bot: discord.Client, channel_id: int, reminder_id: str
    ) -> Optional[Union[TextChannel, DMChannel, GroupChannel]]:
        """Get the channel for sending a reminder.

        Args:
            bot: The Discord bot instance
            channel_id: The channel ID
            reminder_id: The reminder ID for logging

        Returns:
            The channel if found and valid, None otherwise
        """
        channel = bot.get_channel(channel_id)
        if not channel:
            logger.warning(f"Channel {channel_id} not found for reminder {reminder_id}")
            return None

        if not isinstance(channel, (TextChannel, DMChannel, GroupChannel)):
            logger.warning(f"Channel {channel_id} is not a text channel")
            return None

        return channel

    async def _fetch_original_message(
        self,
        channel: Union[TextChannel, DMChannel, GroupChannel],
        message_id: Optional[int],
        reminder_id: str,
    ) -> Optional[discord.Message]:
        """Fetch the original message that set the reminder.

        Args:
            channel: The Discord channel
            message_id: The message ID, if any
            reminder_id: The reminder ID for logging

        Returns:
            The original message if found, None otherwise
        """
        if not message_id:
            return None

        try:
            return await channel.fetch_message(message_id)
        except Exception as e:
            logger.warning(
                f"Could not fetch original message for reminder {reminder_id}: {str(e)}"
            )
            return None

    async def _get_user(
        self, bot: discord.Client, user_id: int, reminder_id: str
    ) -> Optional[discord.User]:
        """Get the user who set the reminder.

        Args:
            bot: The Discord bot instance
            user_id: The user ID
            reminder_id: The reminder ID for logging

        Returns:
            The user if found, None otherwise
        """
        user = bot.get_user(user_id)
        if not user:
            try:
                user = await bot.fetch_user(user_id)
            except Exception as e:
                logger.warning(
                    f"Could not fetch user for reminder {reminder_id}: {str(e)}"
                )
                return None
        return user

    async def _send_reminder(
        self,
        channel: Union[TextChannel, DMChannel, GroupChannel],
        original_message: Optional[discord.Message],
        user: Optional[discord.User],
        user_id: int,
        content: str,
        reminder_id: str,
    ) -> None:
        """Send the reminder message.

        Args:
            channel: The Discord channel
            original_message: The original message that set the reminder
            user: The user who set the reminder
            user_id: The user's ID
            content: The reminder content
            reminder_id: The reminder ID for logging
        """
        if self.llm_handler and original_message:
            # Add the original message directly to the LLM queue with reminder context
            self.llm_handler.add_reminder_to_queue(
                channel.id, original_message, content
            )
            logger.info(f"Added reminder {reminder_id} to LLM queue with context")
        else:
            # Fallback if we don't have the original message or LLM handler
            logger.warning(f"Using fallback for reminder {reminder_id}")

            if original_message:
                await original_message.reply(
                    f"{user.mention if user else f'<@{user_id}>'} Reminder: {content}"
                )
            else:
                await channel.send(
                    f"{user.mention if user else f'<@{user_id}>'} Reminder: {content}"
                )

            logger.info(f"Sent fallback reminder {reminder_id} to channel {channel.id}")

    async def process_due_reminder(
        self, bot: discord.Client, reminder: Dict[str, Any]
    ) -> None:
        """Process a due reminder.

        Args:
            bot: The Discord bot instance
            reminder: The reminder to process
        """
        reminder_id = reminder["id"]
        channel_id = reminder["channel_id"]
        user_id = reminder["user_id"]
        content = reminder["content"]

        # Safely convert message_id to int
        message_id_raw = reminder.get("message_id")
        try:
            message_id = int(message_id_raw) if message_id_raw is not None else None
        except (ValueError, TypeError):
            logger.warning(
                f"Invalid message_id for reminder {reminder_id}: {message_id_raw}"
            )
            message_id = None

        try:
            # Get the channel
            channel = await self._get_channel(bot, channel_id, reminder_id)
            if not channel:
                self.remove_reminder(reminder_id)
                return

            # Get the original message if possible
            original_message = await self._fetch_original_message(
                channel, message_id, reminder_id
            )

            # Get the user
            user = await self._get_user(bot, user_id, reminder_id)

            # Send the reminder
            await self._send_reminder(
                channel, original_message, user, user_id, content, reminder_id
            )

            # Remove the reminder after sending
            self.remove_reminder(reminder_id)

        except Exception as e:
            logger.error(f"Error processing reminder {reminder_id}: {str(e)}")


# Create a global instance of the reminder manager
reminder_manager = ReminderManager()
