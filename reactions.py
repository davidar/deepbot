"""Reaction tracking and management for DeepBot."""

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

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

    def _calculate_hot_score(self, reaction_count: int, timestamp: float) -> float:
        """Calculate a Reddit-like 'hot' score based on reactions and time.

        Args:
            reaction_count: Number of reactions
            timestamp: Unix timestamp of the message

        Returns:
            Float score where higher values mean "hotter" messages
        """
        # Similar to Reddit's algorithm but simplified
        # Score = log2(reactions + 1) + timestamp/45000
        # This means a 2x increase in reactions is worth about 12.5 hours of recency
        order = math.log2(max(reaction_count, 1))
        seconds = timestamp - 1134028003  # Reddit's epoch start
        return round(order + seconds / 45000, 7)

    def get_channel_stats(self, channel_id: int) -> List[Tuple[int, float]]:
        """Get reaction statistics for a channel.

        Args:
            channel_id: The Discord channel ID

        Returns:
            List of tuples (message_id, hot_score) sorted by score
        """
        # Get all messages from this channel
        channel_messages = {
            msg_id: data
            for msg_id, data in self.reaction_data.items()
            if data["info"]["channel_id"] == channel_id
        }

        # Get scored messages
        message_scores: List[Tuple[int, float]] = []
        for msg_id, data in channel_messages.items():
            total_reactions = sum(r["count"] for r in data["reactions"])
            if total_reactions > 0:
                score = self._calculate_hot_score(
                    total_reactions, data["info"]["timestamp"]
                )
                message_scores.append((msg_id, score))

        # Sort by score
        message_scores.sort(key=lambda x: x[1], reverse=True)
        return message_scores

    def get_global_stats(self) -> Dict[int, List[Tuple[int, float]]]:
        """Get reaction statistics across all channels, grouped by channel.

        Returns:
            Dict mapping channel_ids to lists of (message_id, hot_score) tuples
        """
        # Group messages by channel
        channel_messages: Dict[int, List[Tuple[int, float]]] = {}

        for msg_id, data in self.reaction_data.items():
            total_reactions = sum(r["count"] for r in data["reactions"])
            if total_reactions > 0:
                channel_id = data["info"]["channel_id"]
                score = self._calculate_hot_score(
                    total_reactions, data["info"]["timestamp"]
                )

                if channel_id not in channel_messages:
                    channel_messages[channel_id] = []
                channel_messages[channel_id].append((msg_id, score))

        # Sort each channel's messages by score
        for channel_id in channel_messages:
            channel_messages[channel_id].sort(key=lambda x: x[1], reverse=True)

        return channel_messages

    def format_reaction_summary(
        self,
        message_scores: List[Tuple[int, float]],
        limit: int = 5,
    ) -> Optional[str]:
        """Format reaction statistics into a summary string.

        Args:
            message_scores: List of (message_id, score) tuples
            limit: Maximum number of messages to include
            show_channel: Whether to show channel/server context (for global summaries)

        Returns:
            Formatted summary string
        """
        if not message_scores:
            return None

        summary = ""
        for msg_id, _score in message_scores[:limit]:
            msg_data = self.reaction_data[msg_id]
            content = msg_data["info"]["content"]
            if len(content) > 100:
                content = content[:97] + "..."

            total_reactions = sum(r["count"] for r in msg_data["reactions"])

            summary += f"{total_reactions} reactions: {content}\n"

        return summary

    def format_global_summary(
        self, channel_scores: Dict[int, List[Tuple[int, float]]], limit: int = 5
    ) -> str:
        """Format global reaction statistics into a summary string, grouped by channel.

        Args:
            channel_scores: Dict mapping channel_ids to lists of (message_id, score) tuples
            limit: Maximum number of messages to show per channel

        Returns:
            Formatted summary string
        """
        if not channel_scores:
            return "No reactions yet in any channel."

        summary: List[str] = []

        # Sort channels by their highest scoring message
        def channel_max_score(channel_id: int) -> float:
            scores = channel_scores[channel_id]
            return max(score for _, score in scores) if scores else 0

        sorted_channels = sorted(
            channel_scores.keys(), key=channel_max_score, reverse=True
        )

        for channel_id in sorted_channels:
            messages = channel_scores[channel_id]
            if not messages:
                continue

            # Get channel info from first message
            first_msg = self.reaction_data[messages[0][0]]
            channel_name = first_msg["info"]["channel_name"]
            server_name = first_msg["info"]["server_name"]

            summary.append(f"\n#{channel_name} ({server_name}):")
            body = self.format_reaction_summary(messages, limit)
            if body:
                summary.append(body)
            else:
                summary.append("No reactions yet.")

        return "\n".join(summary)

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
