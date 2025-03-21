"""Context building for LLM interactions."""

import logging
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    TypedDict,
    cast,
)

import pendulum
from discord import Message
from discord.ext import commands
from ollama import Message as LLMMessage
from pendulum import DateTime

import config
import example_conversation
from discord_types import StoredMessage
from message_store import MessageStore
from reactions import ReactionManager
from system_prompt import load_system_prompt
from tool_messages import is_tool_message, parse_repl_tool_message
from utils.discord_utils import (
    clean_message_content,
    get_server_name,
    is_automated_message,
)
from utils.message_formatter import format_message_group

if TYPE_CHECKING:
    from discord.abc import MessageableChannel


# Internal type for message grouping
class _GroupedMessage(TypedDict):
    """Internal type for message grouping that extends LLMMessage."""

    role: Literal["system", "assistant", "user", "tool"]
    content: str
    author_id: int
    tool_calls: Optional[Sequence[LLMMessage.ToolCall]]


class _GroupedMessageWithResponse(_GroupedMessage, total=False):
    """_GroupedMessage with optional response field."""

    _has_response: "_GroupedMessage"  # For tool messages with responses


# Set up logging
logger = logging.getLogger("deepbot.context")


class ContextBuilder:
    """Builds LLM context from Discord messages."""

    def __init__(
        self, reaction_manager: ReactionManager, message_store: MessageStore
    ) -> None:
        """Initialize the context builder.

        Args:
            reaction_manager: The reaction manager instance to use
            message_store: The message store instance to use for searching
        """
        self.reaction_manager = reaction_manager
        self.message_store = message_store
        self._reset_timestamps: dict[int, DateTime] = {}
        self._command_names: set[str] = set()

    def set_bot(self, bot: commands.Bot) -> None:
        """Set the bot instance to get command names from.

        Args:
            bot: The Discord bot instance
        """
        self._command_names = {cmd.name for cmd in bot.commands}
        logger.info(f"Updated command names: {self._command_names}")

    def reset_history_from(self, channel_id: int, timestamp: DateTime) -> None:
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
                # Include messages up to and including the original message
                return (
                    message.created_at
                    <= reference_message.reference.resolved.created_at
                )
            else:
                # Otherwise just include everything before the reference message
                return message.created_at < reference_message.created_at

        return True

    def _handle_tool_message(
        self, message: Message, content: str
    ) -> Optional[_GroupedMessageWithResponse]:
        """Handle a tool message and format it appropriately.

        Args:
            message: The Discord message
            content: The cleaned message content

        Returns:
            Formatted message dict or None if message should be skipped
        """
        try:
            # Parse the REPL-style tool message
            result = parse_repl_tool_message(content)
            if result:
                tool_name, tool_args, response_data = result

                # Create tool call message
                tool_call_message = _GroupedMessageWithResponse(
                    role="assistant",
                    content="",  # Empty content for tool calls
                    author_id=message.author.id,
                    tool_calls=[
                        LLMMessage.ToolCall(
                            function=LLMMessage.ToolCall.Function(
                                name=tool_name,
                                arguments=tool_args,
                            ),
                        ),
                    ],
                )

                # Create tool response message
                tool_response_message = _GroupedMessage(
                    role="tool",
                    content=response_data,
                    author_id=message.author.id,
                    tool_calls=None,
                )

                # Add the response message
                tool_call_message["_has_response"] = tool_response_message
                return tool_call_message

        except Exception as e:
            logger.warning(f"Error parsing REPL tool message: {str(e)}")

        # Fallback to treating as a regular message
        return _GroupedMessageWithResponse(
            role="assistant",
            content=content,
            author_id=message.author.id,
            tool_calls=None,
        )

    def _handle_reply(self, message: Message, content: str) -> str:
        """Handle a reply message by adding the referenced message as a quote.

        Args:
            message: The Discord message
            content: The current message content

        Returns:
            The updated message content with the quote
        """
        if not message.reference or not message.reference.resolved:
            return content

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
                return f"> {ref_content}\n\n{content}"
        except AttributeError:
            # If the referenced message is deleted or inaccessible, just continue without the quote
            pass

        return content

    def _format_message(self, message: Message) -> Optional[_GroupedMessage]:
        """Format a Discord message for the LLM context.

        Args:
            message: The Discord message to format

        Returns:
            Formatted message dict or None if message should be skipped
        """
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

        # Skip empty messages
        if not content:
            return None

        # Handle automated bot messages
        if message.author.bot:
            # Check if this is a Python REPL-style tool message
            if is_tool_message(content):
                return self._handle_tool_message(message, content)
            # Skip automated messages
            elif is_automated_message(content):
                return None

        # Add username prefix if not a bot
        if not message.author.bot:
            content = f"{message.author.display_name}: {content}"

        # Handle replies
        content = self._handle_reply(message, content)

        return _GroupedMessage(
            role="assistant" if message.author.bot else "user",
            content=content,
            author_id=message.author.id,
            tool_calls=None,
        )

    async def get_system_prompt(
        self,
        channel: "MessageableChannel",
        search_results: Optional[Dict[str, List[StoredMessage]]] = None,
        search_query: Optional[str] = None,
    ) -> LLMMessage:
        """Get the system prompt for a channel.

        Args:
            channel: The Discord channel
            search_results: Optional dictionary of search results to include in the prompt
            search_query: The search query that produced the results

        Returns:
            System prompt message dict
        """
        # Format the complete prompt with server name, time and reactions
        server_name = get_server_name(channel)
        current_time = pendulum.now("UTC").strftime("%A, %B %d, %Y")

        prompt = [
            f"# Discord Server: {server_name}",
            f"# Current Time: {current_time}",
            "",
        ]

        # Add search results if provided
        if search_results and search_query:
            prompt.append(
                f"# Relevant Context from the channel history related to '{search_query}':"
            )
            prompt.append("")
            for channel_id, messages in search_results.items():
                for message in messages:
                    # Format message without the -# prefix
                    formatted_msg = format_message_group(
                        message, channel_id, bot=None, use_prefix=False
                    )
                    prompt.append(formatted_msg)
            prompt.append("")
            prompt.append("# End of Relevant Context")
            prompt.append("")

        prompt.extend(load_system_prompt())

        return LLMMessage(
            role="system",
            content="\n".join(prompt),
        )

    def _group_messages(
        self,
        messages: List[Message],
        reference_message: Optional[Message] = None,
    ) -> List[_GroupedMessage]:
        """Group messages by author and handle special cases.

        Args:
            messages: List of Discord messages to group
            reference_message: The message that triggered the current interaction

        Returns:
            List of grouped messages
        """
        # Process messages in chronological order
        messages = sorted(messages, key=lambda m: m.created_at)

        # Group adjacent messages from the same author
        current_group: Optional[_GroupedMessage] = None
        grouped_messages: List[_GroupedMessage] = []

        for message in messages:
            formatted = self._format_message(message)
            if formatted is None or not self._should_include_message(
                message, reference_message
            ):
                continue

            # Handle special case for combined tool call and response messages
            if "_has_response" in formatted:
                # Add the tool call message
                grouped_messages.append(formatted)
                # Add the tool response message that's stored in _has_response
                grouped_messages.append(formatted["_has_response"])  # type: ignore
                # Reset current group
                current_group = None
                continue

            # Special handling for tool messages and tool calls
            if formatted["role"] == "tool" or (
                formatted["role"] == "assistant" and "tool_calls" in formatted
            ):
                # Tool messages and tool calls should be kept separate, not grouped
                grouped_messages.append(formatted)
                # Reset current group to ensure tool messages aren't grouped with other messages
                current_group = None
                continue

            if current_group is None:
                current_group = formatted
            elif (
                current_group["role"] == formatted["role"] == "assistant"  # type: ignore
                and current_group["author_id"] == formatted["author_id"]
            ):
                # Add to current group for regular assistant messages from the same author
                current_group["content"] += f"\n\n{formatted['content']}"
            else:
                # Different author/role or user message, add the current group and start a new one
                grouped_messages.append(current_group)
                current_group = formatted

        # Add the last group if it exists
        if current_group is not None:
            grouped_messages.append(current_group)

        return grouped_messages

    async def build_context(
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
        # Group messages
        grouped_messages = self._group_messages(messages, reference_message)

        # Get max_history from model options
        max_history = config.load_model_options()["max_history"]

        # Search for relevant messages if there's a reference message
        search_results = None
        search_query = None
        if reference_message:
            search_query = clean_message_content(reference_message)
            search_results = await self.message_store.search(search_query, top_k=5)

        # Build base context with search results included in system prompt
        context: List[LLMMessage] = []
        system_prompt = await self.get_system_prompt(
            channel, search_results, search_query
        )
        context.append(system_prompt)
        context.extend(example_conversation.load_example_conversation())
        context.append(
            LLMMessage(
                role="system",
                content=(
                    "The messages above are a distant memory. "
                    "You recall them, but they are not part of your current conversation."
                ),
            )
        )

        # Add conversation history
        for msg in grouped_messages[-max_history:]:
            context.append(cast(LLMMessage, msg))

        # Add reference message if present
        if reference_message:
            formatted_reference_message = self._format_message(reference_message)
            if formatted_reference_message is None:
                raise ValueError("Unable to format reference message")

            # Add system message
            context.append(
                LLMMessage(
                    role="system",
                    content="The messages above provide context for the conversation. Respond to the message below.",
                )
            )

            # Add reference message
            context.append(cast(LLMMessage, formatted_reference_message))

        return context
