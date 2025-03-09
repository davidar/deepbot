"""Context building for LLM interactions."""

import datetime
import logging
from typing import TYPE_CHECKING, List, Literal, Optional, TypedDict

from discord import Message
from ollama import Message as LLMMessage

import config
from reactions import ReactionManager
from system_prompt import load_system_prompt
from utils import clean_message_content, get_server_name

if TYPE_CHECKING:
    from discord.abc import MessageableChannel


# Internal type for message grouping
class _GroupedMessage(TypedDict):
    """Internal type for message grouping that extends LLMMessage."""

    role: Literal["system", "assistant", "user"]
    content: str
    author_id: int


# Set up logging
logger = logging.getLogger("deepbot.context")


class ContextBuilder:
    """Builds LLM context from Discord messages."""

    def __init__(self, reaction_manager: ReactionManager) -> None:
        """Initialize the context builder.

        Args:
            reaction_manager: The reaction manager instance to use
        """
        self.reaction_manager = reaction_manager

    @staticmethod
    def is_automated_message(content: str) -> bool:
        """Check if a message is an automated bot message.

        Args:
            content: The message content to check

        Returns:
            True if the message is automated (starts with -#), False otherwise
        """
        return content.strip().startswith("-#")

    def _format_message(self, message: Message) -> Optional[_GroupedMessage]:
        """Format a Discord message for the LLM context.

        Args:
            message: The Discord message to format

        Returns:
            Formatted message dict or None if message should be skipped
        """
        content = message.content.strip()

        # Skip empty messages and automated bot messages
        if not content or (message.author.bot and self.is_automated_message(content)):
            return None

        # Clean up mentions and format content
        content = clean_message_content(message)
        if not message.author.bot:
            content = f"{message.author.display_name}: {content}"

        return _GroupedMessage(
            role="assistant" if message.author.bot else "user",
            content=content,
            author_id=message.author.id,
        )

    def get_system_prompt(self, channel: "MessageableChannel") -> LLMMessage:
        """Get the system prompt for a channel.

        Args:
            channel: The Discord channel

        Returns:
            System prompt message dict
        """
        # Get reaction stats for the channel
        channel_id = channel.id
        message_reactions = self.reaction_manager.get_channel_stats(channel_id)
        reaction_summary = self.reaction_manager.format_reaction_summary(
            message_reactions
        )

        # Format the complete prompt with server name, time and reactions
        server_name = get_server_name(channel)
        current_time = datetime.datetime.now().strftime("%A, %B %d, %Y")

        prompt = [
            f"# Discord Server: {server_name}",
            f"# Current Time: {current_time}",
            "",
        ]

        prompt.extend(load_system_prompt())

        if reaction_summary and reaction_summary != "No reactions yet.":
            prompt.append(f"\n# Channel Reactions:\n{reaction_summary}\n")

        return LLMMessage(
            role="system",
            content="\n".join(prompt),
        )

    def build_context(
        self, messages: List[Message], channel: "MessageableChannel"
    ) -> List[LLMMessage]:
        """Build LLM context from a list of Discord messages.

        Args:
            messages: List of Discord messages to build context from
            include_system_prompt: Whether to include the system prompt
            include_examples: Whether to include example conversation
            channel: The Discord channel (required if include_system_prompt is True)

        Returns:
            List of message dicts forming the LLM context
        """
        context: List[LLMMessage] = config.EXAMPLE_CONVERSATION.copy()

        # Process messages in chronological order
        messages = sorted(messages, key=lambda m: m.created_at)

        # Group adjacent messages from the same author
        current_group: Optional[_GroupedMessage] = None
        grouped_messages: List[LLMMessage] = []

        for message in messages:
            formatted = self._format_message(message)
            if formatted is None:
                continue

            if current_group is None:
                current_group = formatted
            elif (
                current_group["role"] == formatted["role"]
                and current_group["author_id"] == formatted["author_id"]
            ):
                # Add to current group
                current_group["content"] += f"\n\n{formatted['content']}"
            else:
                # Different author/role, add the current group and start a new one
                grouped_messages.append(
                    LLMMessage(
                        role=current_group["role"],
                        content=current_group["content"],
                    )
                )
                current_group = formatted

        # Add the last group if it exists
        if current_group is not None:
            grouped_messages.append(
                LLMMessage(
                    role=current_group["role"],
                    content=current_group["content"],
                )
            )

        # Add grouped messages to context
        context.extend(grouped_messages)

        # Get max_history from model options and apply limit to final context
        max_history = config.load_model_options()["max_history"]
        return [self.get_system_prompt(channel)] + context[-max_history:]
