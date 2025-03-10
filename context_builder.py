"""Context building for LLM interactions."""

import datetime
import logging
from typing import TYPE_CHECKING, List, Literal, Optional, TypedDict

from discord import Message
from discord.ext import commands
from ollama import Message as LLMMessage

import config
import example_conversation
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
        self._reset_timestamps: dict[int, datetime.datetime] = {}
        self._command_names: set[str] = set()

    def set_bot(self, bot: commands.Bot) -> None:
        """Set the bot instance to get command names from.

        Args:
            bot: The Discord bot instance
        """
        self._command_names = {cmd.name for cmd in bot.commands}
        logger.info(f"Updated command names: {self._command_names}")

    def reset_history_from(self, channel_id: int, timestamp: datetime.datetime) -> None:
        """Reset message history to only include messages after the given timestamp.

        Args:
            channel_id: The Discord channel ID
            timestamp: Messages before this timestamp will be excluded from context
        """
        self._reset_timestamps[channel_id] = timestamp

    def remove_reset(self, channel_id: int) -> None:
        """Remove the reset timestamp for a channel, allowing all messages to be included.

        Args:
            channel_id: The Discord channel ID
        """
        if channel_id in self._reset_timestamps:
            del self._reset_timestamps[channel_id]

    def _should_include_message(
        self, message: Message, reference_message: Optional[Message] = None
    ) -> bool:
        """Check if a message should be included based on reset timestamp and reference message.

        Args:
            message: The Discord message to check
            reference_message: The message that triggered the current interaction

        Returns:
            True if the message should be included, False otherwise
        """
        # First check reset timestamp
        channel_id = message.channel.id
        if channel_id in self._reset_timestamps:
            if message.created_at < self._reset_timestamps[channel_id]:
                return False

        # Then check reference message
        if reference_message:
            # If the reference message is a reply, get the message it's replying to
            if (
                reference_message.reference
                and reference_message.reference.resolved
                and isinstance(reference_message.reference.resolved, Message)
            ):
                # Include messages up to and including the original message, then the reference message
                return (
                    message.created_at
                    <= reference_message.reference.resolved.created_at
                    or message.id == reference_message.id
                )
            # Otherwise just include up to the reference message
            return message.created_at <= reference_message.created_at

        return True

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
        # Skip messages before reset timestamp
        if not self._should_include_message(message):
            return None

        # Check for commands in the original message content
        words = message.content.split()
        if (
            message.mentions
            and message.mentions[0].bot
            and len(words) > 1
            and words[1] in self._command_names
        ):
            return None

        # Clean up mentions and format content
        content = clean_message_content(message)

        # Skip empty messages and automated bot messages
        if not content or (message.author.bot and self.is_automated_message(content)):
            return None

        # Add username prefix if not a bot
        if not message.author.bot:
            content = f"{message.author.display_name}: {content}"

        # If this is a reply, add the referenced message in a quote block
        if message.reference and message.reference.resolved:
            try:
                referenced_message = message.reference.resolved
                if isinstance(referenced_message, Message):
                    # Clean up the referenced message content
                    ref_content = clean_message_content(referenced_message)
                    if not referenced_message.author.bot:
                        ref_content = (
                            f"{referenced_message.author.display_name}: {ref_content}"
                        )
                    # Add the quote block at the start of the message
                    content = f"> {ref_content}\n\n{content}"
            except AttributeError:
                # If the referenced message is deleted or inaccessible, just continue without the quote
                pass

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
        # channel_id = channel.id
        # message_reactions = self.reaction_manager.get_channel_stats(channel_id)
        # reaction_summary = self.reaction_manager.format_reaction_summary(
        #     message_reactions
        # )

        # Format the complete prompt with server name, time and reactions
        server_name = get_server_name(channel)
        current_time = datetime.datetime.now().strftime("%A, %B %d, %Y")

        prompt = [
            f"# Discord Server: {server_name}",
            f"# Current Time: {current_time}",
            "",
        ]

        prompt.extend(load_system_prompt())

        # if reaction_summary and reaction_summary != "No reactions yet.":
        #     prompt.append(f"\n# Channel Reactions:\n{reaction_summary}\n")

        return LLMMessage(
            role="system",
            content="\n".join(prompt),
        )

    def build_context(
        self,
        messages: List[Message],
        channel: "MessageableChannel",
        reference_message: Optional[Message] = None,
    ) -> List[LLMMessage]:
        """Build LLM context from a list of Discord messages.

        Args:
            messages: List of Discord messages to build context from
            channel: The Discord channel
            reference_message: The message that triggered the current interaction

        Returns:
            List of message dicts forming the LLM context
        """
        # Process messages in chronological order
        messages = sorted(messages, key=lambda m: m.created_at)

        # Group adjacent messages from the same author
        current_group: Optional[_GroupedMessage] = None
        grouped_messages: List[LLMMessage] = []

        for message in messages:
            formatted = self._format_message(message)
            if formatted is None or not self._should_include_message(
                message, reference_message
            ):
                continue

            if current_group is None:
                current_group = formatted
            elif (
                current_group["role"] == "assistant"
                and formatted["role"] == "assistant"
                and current_group["author_id"] == formatted["author_id"]
            ):
                # Add to current group only for assistant messages
                current_group["content"] += f"\n\n{formatted['content']}"
            else:
                # Different author/role or user message, add the current group and start a new one
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

        # Get max_history from model options and apply limit to final context
        max_history = config.load_model_options()["max_history"]
        return (
            [self.get_system_prompt(channel)]
            + example_conversation.load_example_conversation()
            + [
                LLMMessage(
                    role="system",
                    content="The messages above are provided only as examples, do not refer to them in conversation from now on.",
                )
            ]
            + grouped_messages[-max_history:]
        )
