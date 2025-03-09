"""Reaction tracking and management for DeepBot."""

import json
import logging
import os
from typing import Any, Dict, List, Tuple

from discord import Message, Reaction, User

from utils import get_channel_name, get_server_name

# Set up logging
logger = logging.getLogger("deepbot.reactions")


class ReactionManager:
    """Manages reaction tracking and persistence for bot messages."""

    def __init__(self) -> None:
        """Initialize the reaction manager."""
        self.reaction_data: Dict[int, Dict[str, Any]] = {}
        self._load_reaction_data()

    def _load_reaction_data(self) -> None:
        """Load reaction data from file."""
        try:
            if os.path.exists("reaction_data.json"):
                with open("reaction_data.json", "r") as f:
                    data = json.load(f)
                    # Convert to our internal format
                    for message in data:
                        msg_id = message["id"]
                        self.reaction_data[msg_id] = {
                            "info": message["info"],
                            "reactions": message["reactions"],
                        }
                logger.info("Loaded reaction data from file")
        except Exception as e:
            logger.error(f"Error loading reaction data: {str(e)}")

    def _save_reaction_data(self) -> None:
        """Save reaction data to file."""
        try:
            # Convert to list format for saving
            messages_data: List[Dict[str, Any]] = []

            # Only include messages that have reactions
            for msg_id, data in self.reaction_data.items():
                if not data["reactions"]:  # Skip messages with no reactions
                    continue

                message_data = {
                    "id": msg_id,
                    "info": data["info"],
                    "reactions": data["reactions"],
                }
                messages_data.append(message_data)

            # Sort messages by timestamp (newest first)
            messages_data.sort(
                key=lambda x: float(x["info"]["timestamp"]), reverse=True
            )

            # Save with pretty formatting
            with open("reaction_data.json", "w") as f:
                json.dump(messages_data, f, indent=2)
            logger.info("Saved reaction data to file")
        except Exception as e:
            logger.error(f"Error saving reaction data: {str(e)}")

    def handle_reaction_add(self, reaction: Reaction, user: User) -> None:
        """Handle a reaction being added to a message.

        Args:
            reaction: The Discord reaction that was added
            user: The user who added the reaction
        """
        logger.info(
            f"Reaction add event received: {reaction.emoji} on message {reaction.message.id} by {user.name}"
        )

        message_id = reaction.message.id
        emoji = str(reaction.emoji)

        # Update reaction count
        if message_id not in self.reaction_data:
            self.reaction_data[message_id] = {
                "info": {
                    "content": reaction.message.content,
                    "channel_id": reaction.message.channel.id,
                    "channel_name": get_channel_name(reaction.message.channel),
                    "server_name": get_server_name(reaction.message.channel),
                    "timestamp": reaction.message.created_at.timestamp(),
                },
                "reactions": [],
            }

        for r in self.reaction_data[message_id]["reactions"]:
            if r["emoji"] == emoji:
                r["count"] += 1
                break
        else:
            # If emoji not found, add new reaction
            self.reaction_data[message_id]["reactions"].append(
                {"emoji": emoji, "count": 1}
            )

        # Save to file
        self._save_reaction_data()

    def get_channel_stats(self, channel_id: int) -> List[Tuple[int, int]]:
        """Get reaction statistics for a channel.

        Args:
            channel_id: The Discord channel ID

        Returns:
            List of tuples (message_id, total_reactions) sorted by reaction count
        """
        # Get all messages from this channel
        channel_messages = {
            msg_id: data
            for msg_id, data in self.reaction_data.items()
            if data["info"]["channel_id"] == channel_id
        }

        # Get most reacted messages
        message_reactions: List[Tuple[int, int]] = []
        for msg_id, data in channel_messages.items():
            total_reactions = sum(r["count"] for r in data["reactions"])
            if total_reactions > 0:
                message_reactions.append((msg_id, total_reactions))

        # Sort by reaction count
        message_reactions.sort(key=lambda x: x[1], reverse=True)
        return message_reactions

    def format_reaction_summary(
        self, message_reactions: List[Tuple[int, int]], limit: int = 5
    ) -> str:
        """Format reaction statistics into a summary string.

        Args:
            message_reactions: List of (message_id, count) tuples
            limit: Maximum number of messages to include

        Returns:
            Formatted summary string
        """
        if not message_reactions:
            return "No reactions yet in this channel."

        summary = ""
        for msg_id, count in message_reactions[:limit]:
            msg_data = self.reaction_data[msg_id]
            content = msg_data["info"]["content"]
            if len(content) > 100:
                content = content[:97] + "..."

            # Format reactions
            reactions: List[str] = []
            for r in msg_data["reactions"]:
                reactions.append(f"{r['emoji']} x {r['count']}")
            reactions_str = ", ".join(reactions)

            summary += f"{count} total reactions ({reactions_str}): {content}\n"
        return summary

    def initialize_bot_message(self, message: Message) -> None:
        """Initialize reaction tracking for a new bot message.

        Args:
            message: The Discord message to track reactions for
        """
        message_id = message.id
        if message_id not in self.reaction_data:
            self.reaction_data[message_id] = {
                "info": {
                    "content": message.content,
                    "channel_id": message.channel.id,
                    "channel_name": get_channel_name(message.channel),
                    "server_name": get_server_name(message.channel),
                    "timestamp": message.created_at.timestamp(),
                },
                "reactions": [],
            }

            # Initialize reaction counts from message reactions
            for reaction in message.reactions:
                emoji = str(reaction.emoji)
                self.reaction_data[message_id]["reactions"].append(
                    {"emoji": emoji, "count": reaction.count}
                )

            # Save to file
            self._save_reaction_data()
