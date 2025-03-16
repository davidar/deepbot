"""Reaction statistics tracking for DeepBot."""

import datetime
import logging
from typing import Dict, List, Tuple

from discord_types import StoredMessage
from message_store import MessageStore

# Set up logging
logger = logging.getLogger("deepbot.reactions")


class ReactionStats:
    """Manages reaction statistics for messages."""

    def __init__(self, message_store: MessageStore) -> None:
        """Initialize the reaction stats manager.

        Args:
            message_store: The message store instance to use
        """
        self.message_store = message_store

    def _calculate_hot_score(self, message: StoredMessage) -> float:
        """Calculate a "hot" score for a message based on reactions and time.

        Args:
            message: The message to calculate score for

        Returns:
            A float score where higher values indicate "hotter" messages
        """
        # Get total reactions
        total_reactions = float(
            sum(reaction["count"] for reaction in message.reactions)
        )

        # Get message age in hours
        msg_time = datetime.datetime.fromisoformat(
            message.timestamp.replace("Z", "+00:00")
        )
        current_time = datetime.datetime.now(datetime.timezone.utc)
        age_hours = float((current_time - msg_time).total_seconds()) / 3600.0

        # Simple decay formula: score = reactions / (age_hours + 2)^1.5
        # The +2 prevents division by zero and reduces the penalty for very new posts
        base = age_hours + 2.0
        return total_reactions / float(pow(base, 1.5))

    def get_channel_stats(self, channel_id: str) -> List[Tuple[StoredMessage, float]]:
        """Get reaction statistics for messages in a channel.

        Args:
            channel_id: The Discord channel ID

        Returns:
            List of (message, score) tuples sorted by score
        """
        # Get messages from store
        messages = self.message_store.get_channel_messages(channel_id)

        # Calculate scores for messages with reactions
        scored_messages: List[Tuple[StoredMessage, float]] = []
        for msg in messages:
            if msg.reactions:  # Only process messages that have reactions
                score = self._calculate_hot_score(msg)
                scored_messages.append((msg, score))

        # Sort by score descending
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        return scored_messages

    def get_global_stats(self) -> Dict[str, List[Tuple[StoredMessage, float]]]:
        """Get reaction statistics across all channels.

        Returns:
            Dict mapping channel IDs to lists of (message, score) tuples
        """
        stats: Dict[str, List[Tuple[StoredMessage, float]]] = {}
        for channel_id in self.message_store.get_channel_ids():
            channel_stats = self.get_channel_stats(channel_id)
            if channel_stats:  # Only include channels with reactions
                stats[channel_id] = channel_stats
        return stats

    def format_reaction_summary(self, message: StoredMessage) -> str:
        """Format a summary of reactions for a message.

        Args:
            message: The message to format reactions for

        Returns:
            A formatted string summarizing the reactions
        """
        if not message.reactions:
            return "No reactions"

        parts: List[str] = []
        for reaction in message.reactions:
            emoji = reaction["emoji"]
            count = reaction["count"]

            # Format the emoji representation
            if emoji["id"]:  # Custom emoji
                emoji_str = f"<{'a:' if emoji['isAnimated'] else ':'}{emoji['name']}:{emoji['id']}>"
            else:  # Unicode emoji
                emoji_str = emoji["name"]

            parts.append(f"{emoji_str}: {count}")

        return " | ".join(parts)

    def format_global_summary(self, top_n: int = 5) -> str:
        """Format a summary of top reactions across all channels.

        Args:
            top_n: Number of top messages to include per channel

        Returns:
            A formatted string with the global reaction summary
        """
        stats = self.get_global_stats()
        if not stats:
            return "No reactions found in any channel"

        lines: List[str] = []
        for channel_id, messages in stats.items():
            # Get channel info from first message's reference
            channel_name = channel_id
            if messages:
                first_msg = messages[0][0]  # First message from (msg, score) tuple
                if first_msg.reference:
                    channel_name = first_msg.reference.channelId

            lines.append(f"\n**#{channel_name}**")
            for msg, _score in messages[:top_n]:
                content = (
                    msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                )
                reaction_summary = self.format_reaction_summary(msg)
                lines.append(f"- {content}\n  {reaction_summary}")

        return "\n".join(lines)
