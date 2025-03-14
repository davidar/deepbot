"""Reminder management system for the bot."""

import datetime
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

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
        message_id = reminder.get("message_id")

        try:
            # Get the channel
            channel = bot.get_channel(channel_id)
            if not channel:
                logger.warning(
                    f"Channel {channel_id} not found for reminder {reminder_id}"
                )
                self.remove_reminder(reminder_id)
                return

            # Check if the channel is a text channel that supports sending messages
            if not isinstance(channel, (TextChannel, DMChannel, GroupChannel)):
                logger.warning(f"Channel {channel_id} is not a text channel")
                self.remove_reminder(reminder_id)
                return

            # Get the original message if possible
            original_message = None
            if message_id:
                try:
                    original_message = await channel.fetch_message(message_id)
                except Exception as e:
                    logger.warning(
                        f"Could not fetch original message for reminder {reminder_id}: {str(e)}"
                    )

            # Get the user
            user = bot.get_user(user_id)
            if not user:
                try:
                    user = await bot.fetch_user(user_id)
                except Exception as e:
                    logger.warning(
                        f"Could not fetch user for reminder {reminder_id}: {str(e)}"
                    )

            # Process the reminder
            if self.llm_handler and original_message:
                # Add the original message directly to the LLM queue with reminder context
                self.llm_handler.add_reminder_to_queue(
                    channel_id, original_message, content
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

                logger.info(
                    f"Sent fallback reminder {reminder_id} to channel {channel_id}"
                )

            # Remove the reminder after sending
            self.remove_reminder(reminder_id)

        except Exception as e:
            logger.error(f"Error processing reminder {reminder_id}: {str(e)}")


# Create a global instance of the reminder manager
reminder_manager = ReminderManager()
