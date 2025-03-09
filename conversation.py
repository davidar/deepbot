"""Conversation management and message processing for DeepBot."""

import asyncio
import logging
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import ollama
from discord import ClientUser, DMChannel, Message, TextChannel
from discord.abc import Messageable

import config
import system_prompt
from utils import clean_message_content, get_channel_name, get_server_name

if TYPE_CHECKING:
    from discord.abc import MessageableChannel

# Set up logging
logger = logging.getLogger("deepbot.conversation")


class LineStatus(Enum):
    """Status of a line being streamed."""

    ACCUMULATING = "accumulating"  # Line is still being built
    COMPLETE = "complete"  # Line is complete and ready to send

    def __str__(self) -> str:
        return self.value


class ConversationManager:
    """Manages conversation history and message processing."""

    def __init__(self, api_client: ollama.Client, bot_user: ClientUser) -> None:
        """Initialize the conversation manager.

        Args:
            api_client: The Ollama API client to use for responses
            bot_user: The bot's Discord user
        """
        self.api_client = api_client
        self.bot_user = bot_user
        self._conversation_history: Dict[int, List[Dict[str, str]]] = {}
        self.response_queues: Dict[int, asyncio.Queue[Message]] = {}
        self.response_tasks: Dict[int, Optional[asyncio.Task[None]]] = {}
        self.shutup_tasks: set[asyncio.Task[None]] = set()

    def has_history(self, channel_id: int) -> bool:
        """Check if a channel has conversation history.

        Args:
            channel_id: The Discord channel ID

        Returns:
            True if the channel has history, False otherwise
        """
        return channel_id in self._conversation_history and bool(
            self._conversation_history[channel_id]
        )

    def get_history(self, channel_id: int) -> List[Dict[str, str]]:
        """Get the conversation history for a channel.

        Args:
            channel_id: The Discord channel ID

        Returns:
            List of message dictionaries
        """
        return self._conversation_history.get(channel_id, [])

    def get_history_length(self, channel_id: int) -> int:
        """Get the number of messages in a channel's history.

        Args:
            channel_id: The Discord channel ID

        Returns:
            Number of messages in history
        """
        return len(self._conversation_history.get(channel_id, []))

    def clear_history(self, channel_id: int) -> None:
        """Clear the conversation history for a channel.

        Args:
            channel_id: The Discord channel ID
        """
        self._conversation_history[channel_id] = []

    def reset_to_initial(self, channel: "MessageableChannel") -> None:
        """Reset channel history to just the initial messages.

        Args:
            channel: The Discord channel
        """
        channel_id = channel.id
        self._conversation_history[channel_id] = self.get_initial_messages(channel)

    def add_message(self, channel_id: int, role: str, content: str) -> None:
        """Add a message to the conversation history.

        Args:
            channel_id: The Discord channel ID
            role: The role of the message sender ("user" or "assistant")
            content: The message content
        """
        # Don't add automated messages to history
        if role == "assistant" and self._is_automated_message(content):
            return

        if channel_id not in self._conversation_history:
            self._conversation_history[channel_id] = []
        self._conversation_history[channel_id].append(
            {"role": role, "content": content}
        )

    def is_duplicate_message(self, channel_id: int, role: str, content: str) -> bool:
        """Check if a message would be a duplicate of the last message.

        Args:
            channel_id: The Discord channel ID
            role: The role of the message sender
            content: The message content

        Returns:
            True if the message would be a duplicate, False otherwise
        """
        history = self._conversation_history.get(channel_id, [])
        return bool(
            history
            and history[-1]["role"] == role
            and history[-1]["content"] == content
        )

    def trim_history(self, channel_id: int, max_history: int) -> None:
        """Trim the conversation history to a maximum length.

        Args:
            channel_id: The Discord channel ID
            max_history: Maximum number of messages to keep (excluding system message)
        """
        if channel_id not in self._conversation_history:
            return

        history = self._conversation_history[channel_id]
        if len(history) <= max_history:
            return

        # Keep system message if it exists
        if history and history[0]["role"] == "system":
            self._conversation_history[channel_id] = [history[0]] + history[
                -max_history:
            ]
        else:
            self._conversation_history[channel_id] = history[-max_history:]

    def update_system_prompt(self, channel_id: int, new_prompt: str) -> None:
        """Update the system prompt for a channel.

        Args:
            channel_id: The Discord channel ID
            new_prompt: The new system prompt content
        """
        if (
            channel_id in self._conversation_history
            and self._conversation_history[channel_id]
            and self._conversation_history[channel_id][0]["role"] == "system"
        ):
            self._conversation_history[channel_id][0]["content"] = new_prompt

    def get_initial_messages(self, channel: Messageable) -> List[Dict[str, str]]:
        """Get the initial messages for a new conversation.

        Args:
            channel: The Discord channel

        Returns:
            List of message dictionaries with system prompt and example conversation
        """
        # Start with the system message
        system_message = {
            "role": "system",
            "content": system_prompt.get_system_prompt(get_server_name(channel)),
        }

        # Create a list with system message and example conversation
        initial_messages = [system_message]
        initial_messages.extend(config.EXAMPLE_CONVERSATION)

        return initial_messages

    def _is_automated_message(self, content: str) -> bool:
        """Check if a message is an automated bot message.

        Args:
            content: The message content to check

        Returns:
            True if the message is automated (starts with -#), False otherwise
        """
        return content.strip().startswith("-#")

    async def initialize_channel_history(
        self, channel: Union[TextChannel, DMChannel]
    ) -> None:
        """Initialize conversation history for a channel by fetching recent messages.

        Args:
            channel: The Discord channel to initialize history for
        """
        channel_id = channel.id

        # Skip if history already exists for this channel
        if (
            channel_id in self._conversation_history
            and self._conversation_history[channel_id]
        ):
            return

        # Initialize with system message and example interactions
        self._conversation_history[channel_id] = self.get_initial_messages(channel)

        try:
            # Fetch recent messages from the channel
            message_limit = int(config.get_option("history_fetch_limit", 50))
            all_messages: List[Dict[str, Any]] = []

            logger.info(
                f"Fetching up to {message_limit} messages from channel {get_channel_name(channel)}"
            )

            # Use the Discord API to fetch recent messages
            async for message in channel.history(limit=message_limit):
                # Skip empty messages and automated bot messages
                content = message.content.strip()
                if not content or (
                    message.author.bot and self._is_automated_message(content)
                ):
                    continue

                # Clean up the content by replacing Discord mentions with usernames
                content = clean_message_content(message)

                # Add to our list with metadata for sorting and grouping
                message_data = {
                    "role": "assistant" if message.author.bot else "user",
                    "content": (
                        f"{message.author.display_name}: {content}"
                        if not message.author.bot
                        else content
                    ),
                    "timestamp": float(message.created_at.timestamp()),
                    "author_id": str(message.author.id),
                }
                all_messages.append(message_data)

            # Sort messages by timestamp
            all_messages.sort(key=lambda m: m["timestamp"])

            # Group adjacent messages from the same author
            grouped_messages: List[Dict[str, Any]] = []
            current_group: Optional[Dict[str, Any]] = None

            for message_data in all_messages:
                if current_group is None:
                    # Start a new group
                    current_group = {
                        "role": str(message_data["role"]),
                        "content": str(message_data["content"]),
                        "author_id": str(message_data["author_id"]),
                    }
                elif current_group["author_id"] == str(message_data["author_id"]):
                    # Add to current group
                    current_group["content"] += "\n\n" + str(message_data["content"])
                else:
                    # Different author, add the current group and start a new one
                    grouped_messages.append(current_group)
                    current_group = {
                        "role": str(message_data["role"]),
                        "content": str(message_data["content"]),
                        "author_id": str(message_data["author_id"]),
                    }

            # Add the last group if it exists
            if current_group is not None:
                grouped_messages.append(current_group)

            # Add messages to history (without extra metadata)
            for message_data in grouped_messages:
                # Create a clean copy without the extra fields
                message_copy = {
                    "role": str(message_data["role"]),
                    "content": str(message_data["content"]),
                }
                self._conversation_history[channel_id].append(message_copy)

            # Trim if needed
            # +1 for system message
            max_history = int(config.get_option("max_history", 10))
            if len(self._conversation_history[channel_id]) > max_history + 1:
                self._conversation_history[channel_id] = [
                    self._conversation_history[channel_id][0]
                ] + self._conversation_history[channel_id][-max_history:]

            logger.info(
                f"Initialized history for channel {get_channel_name(channel)} with {len(grouped_messages)} message groups"
            )

        except Exception as e:
            logger.error(
                f"Error initializing history for channel {get_channel_name(channel)}: {str(e)}"
            )
            # Keep the initial messages at least
            self._conversation_history[channel_id] = self.get_initial_messages(channel)

    async def _stream_response_lines(
        self, channel_id: int
    ) -> AsyncGenerator[Tuple[LineStatus, str], None]:
        """Generator that yields line status and content as they stream in from the API.

        Args:
            channel_id: The Discord channel ID

        Yields:
            Tuples of (LineStatus, content) where:
            - LineStatus.ACCUMULATING indicates the start of a new line
            - LineStatus.COMPLETE indicates a complete line ready to send
        """
        try:
            logger.info(f"Starting streaming response for channel {channel_id}")
            logger.info(
                f"History length: {len(self._conversation_history[channel_id])} messages"
            )

            stream = self.api_client.chat(  # pyright: ignore
                model=str(config.MODEL_NAME),
                messages=self._conversation_history[channel_id],
                stream=True,
                keep_alive=-1,
                options=config.load_model_options(),
            )

            # Variables to track streaming state
            full_response = ""  # Complete response for history
            chunk_count = 0
            current_line = ""  # Current line being built
            has_non_whitespace = False

            # Process streaming response
            for chunk in stream:
                try:
                    chunk_count += 1

                    if "message" in chunk and "content" in chunk["message"]:
                        content = chunk["message"]["content"]
                        current_line += content
                        full_response += content

                        # Check if we've received non-whitespace content
                        if not has_non_whitespace and content.strip():
                            has_non_whitespace = True
                            # Signal start of new line with content
                            yield (LineStatus.ACCUMULATING, "")

                        # Log every 100 chunks
                        if chunk_count % 100 == 0:
                            logger.info(
                                f"Processed {chunk_count} chunks, current length: {len(full_response)}"
                            )

                        # Check for newlines in current line
                        while "\n" in current_line:
                            # Split at first newline
                            to_send, current_line = current_line.split("\n", 1)
                            has_non_whitespace = False  # Reset for next line

                            # Only yield if we have non-empty content
                            if to_send.strip():
                                yield (LineStatus.COMPLETE, to_send)

                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
                    continue

            logger.info(f"Stream completed. Total chunks: {chunk_count}")
            logger.info(f"Final response length: {len(full_response)} characters")

            # Handle any remaining text
            if current_line.strip():
                yield (LineStatus.COMPLETE, current_line)

            # Only store non-empty responses in history
            if full_response.strip():
                self._conversation_history[channel_id].append(
                    {"role": "assistant", "content": full_response}
                )

        except Exception as e:
            error_message = f"Error in streaming response: {str(e)}"
            logger.error(error_message)
            logger.error(f"Full error details: {repr(e)}")
            yield (LineStatus.COMPLETE, error_message)

    async def process_response_queue(self, channel_id: int) -> None:
        """Process the response queue for a channel.

        Args:
            channel_id: The Discord channel ID
        """
        while True:
            try:
                # Get the next message to respond to
                message = await self.response_queues[channel_id].get()

                # Process the response
                try:
                    await self._handle_streaming_response(message, channel_id)
                except Exception as e:
                    logger.error(f"Error processing response: {str(e)}")
                    await message.reply(f"-# Sorry, I encountered an error: {str(e)}")
                finally:
                    # Mark the task as done
                    self.response_queues[channel_id].task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in response queue processor: {str(e)}")
                # Don't break on error, just log and continue processing
                continue

    async def ensure_queue_processor(self, channel_id: int) -> None:
        """Ensure there's an active queue processor for the channel.

        Args:
            channel_id: The Discord channel ID
        """
        task = self.response_tasks.get(channel_id)
        if task is None or task.done():
            # Cancel existing task if it exists
            if task is not None and not task.done():
                task.cancel()
            # Create a new processor task
            self.response_tasks[channel_id] = asyncio.create_task(
                self.process_response_queue(channel_id)
            )

    async def _handle_streaming_response(
        self, message: Message, channel_id: int
    ) -> None:
        """Handle a streaming response by combining streaming and sending logic.

        Args:
            message: The Discord message to respond to
            channel_id: The Discord channel ID
        """
        # Get the current task
        current_task = asyncio.current_task()
        if current_task is None:
            logger.error("No current task found")
            return

        async with message.channel.typing():
            try:
                # Add typing reaction
                await message.add_reaction("⌨️")
                # Remove the thinking reaction
                try:
                    await message.remove_reaction("💭", self.bot_user)
                except:
                    # Reaction might not exist, that's okay
                    pass

                first_message = True
                line_count = 0
                async for status, line in self._stream_response_lines(channel_id):
                    # Check if this specific task has been told to shut up
                    if current_task in self.shutup_tasks:
                        logger.info(f"Task was told to shut up, stopping response")
                        break

                    if status == LineStatus.ACCUMULATING:
                        logger.info(f"Accumulating line")
                        # Start typing indicator
                        async with message.channel.typing():
                            pass
                    elif status == LineStatus.COMPLETE and line.strip():
                        logger.info(f"Sending line: {line}")
                        try:
                            if first_message:
                                # First message should be a reply
                                await message.reply(line)
                                first_message = False
                            else:
                                await message.channel.send(line)
                            line_count += 1

                            max_lines = int(config.get_option("max_response_lines", 10))
                            if line_count >= max_lines:
                                logger.info(
                                    "Reached maximum line limit, stopping response"
                                )
                                await message.channel.send(
                                    "-# Response truncated due to length limit"
                                )
                                break

                        except Exception as e:
                            logger.warning(f"Failed to send message chunk: {str(e)}")
            except Exception as e:
                logger.error(f"Error sending messages: {str(e)}")
                await message.reply(f"-# Error sending messages: {str(e)}")
            finally:
                # Remove the typing reaction
                try:
                    await message.remove_reaction("⌨️", self.bot_user)
                except:
                    # Reaction might not exist, that's okay
                    pass
                # Remove this task from the shutup set
                self.shutup_tasks.discard(current_task)
