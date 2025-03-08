import asyncio
import io
import json
import logging
import os
import re
from collections import defaultdict
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncGenerator,
    Dict,
    Generator,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
)

import discord
import ollama
from discord.abc import GuildChannel, Messageable, PrivateChannel
from discord.channel import DMChannel, TextChannel
from discord.ext import commands
from discord.message import Message
from discord.threads import Thread
from discord.user import ClientUser

Bot = commands.Bot
Context = commands.Context

if TYPE_CHECKING:
    from discord.abc import MessageableChannel

import config
import system_prompt
from system_prompt import get_system_prompt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger("deepbot")


# Event protocol for SSE client
class Event(Protocol):
    @property
    def data(self) -> str: ...


# SSE client protocol
class SSEClient(Protocol):
    def events(self) -> Generator[Event, None, None]: ...


class LineStatus(Enum):
    """Status of a line being streamed."""

    ACCUMULATING = "accumulating"  # Line is still being built
    COMPLETE = "complete"  # Line is complete and ready to send

    def __str__(self) -> str:
        return self.value


class DeepBot(commands.Bot):
    """Discord bot that uses streaming responses from Ollama."""

    def __init__(self) -> None:
        """Initialize the bot."""
        # Initialize with mention as command prefix
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message content
        intents.messages = True
        intents.reactions = True  # Required to track reactions

        # Use mention as the command prefix
        super().__init__(command_prefix=commands.when_mentioned, intents=intents)

        # Initialize Ollama client
        self.api_client = ollama.Client(host=config.API_URL)
        logger.info("Using Ollama API client")

        # Store conversation history for each channel
        self.conversation_history: Dict[int, List[Dict[str, str]]] = {}

        # Store response queues for each channel
        self.response_queues: Dict[int, asyncio.Queue[Message]] = defaultdict(
            asyncio.Queue
        )
        self.response_tasks: Dict[int, Optional[asyncio.Task[None]]] = defaultdict(
            lambda: None
        )

        # Track which tasks have been told to shut up
        self.shutup_tasks: Set[asyncio.Task[None]] = set()

        # Store reaction data for bot messages
        # Format: {message_id: {"info": {...}, "reactions": [{"emoji": str, "count": int}]}}
        self.reaction_data: Dict[int, Dict[str, Any]] = defaultdict(
            lambda: {"reactions": []}
        )

        # Discord message length limit with safety margin
        # Discord limit is 2000, leaving 50 chars as safety margin
        self.DISCORD_MESSAGE_LIMIT = 1950

        # Register commands
        self.add_commands()

        # Add custom command error handler
        self.add_error_handler()

        # Load reaction data from file if it exists
        self._load_reaction_data()

    def get_bot_user(self) -> ClientUser:
        if not self.user:
            raise RuntimeError("Bot user not initialized")
        return self.user

    def get_channel_name(self, channel: Messageable) -> str:
        """Safely get channel name, handling both text channels and DMs."""
        if isinstance(channel, TextChannel):
            return channel.name
        elif isinstance(channel, DMChannel):
            return "DM"
        else:
            return "Unknown Channel"

    def get_server_name(
        self,
        channel: Optional[Union[GuildChannel, Thread, PrivateChannel, Messageable]],
    ) -> str:
        """Get server name from channel, with fallback to DM chat."""
        if channel and isinstance(channel, TextChannel):
            return channel.guild.name
        return "DM chat"

    def is_mentioned_in(self, message: Message) -> bool:
        """Safely check if the bot is mentioned in a message."""
        try:
            return bool(self.get_bot_user().mentioned_in(message))
        except AttributeError:
            return (
                f"<@{self.get_bot_user().id}>" in message.content
                or f"<@!{self.get_bot_user().id}>" in message.content
            )

    def add_error_handler(self) -> None:
        """Add custom error handler for commands."""

        @self.event
        async def on_command_error(ctx: Context[Bot], error: Exception) -> None:
            """Handle command errors."""
            if isinstance(error, commands.CommandNotFound):
                # This is handled in on_message, so we can ignore it here
                pass
            elif isinstance(error, commands.MissingRequiredArgument):
                await ctx.send(f"Error: Missing required argument: {error.param}")
            elif isinstance(error, commands.BadArgument):
                await ctx.send(f"Error: Bad argument: {error}")
            else:
                logger.error(f"Command error: {error}")
                await ctx.send(f"Error executing command: {error}")

    def _get_reaction_stats(self, channel_id: int) -> List[Tuple[int, int]]:
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

    def _format_reaction_summary(
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

    def _get_initial_messages(
        self, channel: "MessageableChannel"
    ) -> List[Dict[str, str]]:
        """
        Get the initial messages for a new conversation.

        Args:
            channel: The Discord channel

        Returns:
            List of message dictionaries
        """
        # Start with the system message
        system_message = {
            "role": "system",
            "content": get_system_prompt(self.get_server_name(channel)),
        }

        # Add reaction summary if available
        channel_id = channel.id
        message_reactions = self._get_reaction_stats(channel_id)

        if message_reactions:
            reaction_summary = "\n\n**Most Reacted Messages in this Channel:**\n"
            reaction_summary += self._format_reaction_summary(
                message_reactions, limit=3
            )
            system_message["content"] += reaction_summary

        # Create a list with system message and example conversation
        initial_messages = [system_message]
        initial_messages.extend(config.EXAMPLE_CONVERSATION)

        return initial_messages

    def add_commands(self) -> None:
        """Add bot commands."""

        @self.command(name="reset")
        async def reset_history(ctx: Context[Bot]) -> None:
            """Reset the conversation history for the current channel."""
            channel_id = ctx.channel.id
            if channel_id in self.conversation_history:
                self.conversation_history[channel_id] = []
                await ctx.send("Conversation history has been reset.")
            else:
                await ctx.send("No conversation history to reset.")

        @self.command(name="refresh")
        async def refresh_history(ctx: Context[Bot]) -> None:
            """Refresh the conversation history by fetching recent messages from the channel."""
            if not isinstance(ctx.channel, discord.TextChannel):
                await ctx.send(
                    "This command can only be used in text channels, not in DMs."
                )
                return

            await ctx.send("Refreshing conversation history from channel messages...")

            try:
                # Clear existing history
                channel_id = ctx.channel.id
                self.conversation_history[channel_id] = []

                # Re-initialize with fresh data
                await self._initialize_channel_history(ctx.channel)

                history_count = len(self.conversation_history[ctx.channel.id])
                await ctx.send(
                    f"Conversation history refreshed! Now tracking {history_count-1} messages (plus system message)."
                )
            except Exception as e:
                logger.error(f"Error refreshing history: {str(e)}")
                await ctx.send(f"Error refreshing history: {str(e)}")

        @self.command(name="raw")
        async def raw_history(ctx: Context[Bot]) -> None:
            """Display the raw conversation history for debugging."""
            channel_id = ctx.channel.id
            if (
                channel_id not in self.conversation_history
                or not self.conversation_history[channel_id]
            ):
                await ctx.send("No conversation history found for this channel.")
                return

            history = self.conversation_history[channel_id]
            json_data = json.dumps(history, indent=2)

            # Create a discord.File object with the JSON data
            # Convert string to bytes for discord.File
            buffer = io.BytesIO(json_data.encode("utf-8"))
            file = discord.File(
                buffer,
                filename=f"conversation_history_{channel_id}.json",
            )

            await ctx.send("Here's the raw conversation history:", file=file)

        @self.command(name="info")
        async def info_command(ctx: Context[Bot]) -> None:
            """Display information about the bot configuration."""
            backend_type = "Ollama"

            info_text = (
                f"**DeepBot Configuration**\n\n"
                f"Backend: `{backend_type}`\n"
                f"API URL: `{config.API_URL}`\n"
                f"Model: `{config.MODEL_NAME}`\n"
                f"Max History: `{config.MAX_HISTORY}`\n"
                f"History Fetch Limit: `{config.HISTORY_FETCH_LIMIT}`\n"
                f"Temperature: `{config.TEMPERATURE}`\n"
                f"Max Tokens: `{config.MAX_TOKENS}`\n"
                f"Top P: `{config.TOP_P}`\n"
                f"Presence Penalty: `{config.PRESENCE_PENALTY}`\n"
                f"Frequency Penalty: `{config.FREQUENCY_PENALTY}`\n"
                f"Seed: `{config.SEED if config.SEED != -1 else 'None'}`\n"
            )
            await ctx.send(info_text)

        @self.command(name="history")
        async def history_command(ctx: Context[Bot]) -> None:
            """Display the current conversation history for the channel."""
            channel_id = ctx.channel.id
            if (
                channel_id not in self.conversation_history
                or not self.conversation_history[channel_id]
            ):
                await ctx.send("No conversation history found for this channel.")
                return

            history = self.conversation_history[channel_id]

            # Format the history
            history_text = f"**Conversation History ({len(history)} messages)**\n\n"

            for i, message in enumerate(history, 1):
                role = message["role"].capitalize()
                content = message["content"]

                # Format based on role
                if role == "System":
                    formatted_content = f"*{content}*"
                elif role == "User":
                    # Format user messages - content already includes username
                    formatted_content = content
                else:
                    formatted_content = content

                # Truncate long messages
                if len(formatted_content) > 200:
                    formatted_content = formatted_content[:197] + "..."

                # Add role prefix for assistant and system messages
                if role == "Assistant":
                    history_text += f"{i}. **{role}**: {formatted_content}\n\n"
                elif role == "System":
                    history_text += f"{i}. **{role}**: {formatted_content}\n\n"
                else:
                    history_text += f"{i}. {formatted_content}\n\n"

            await ctx.send(history_text)

        @self.command(name="wipe")
        async def wipe_memory(ctx: Context[Bot]) -> None:
            """Temporarily wipe the bot's memory while keeping the system message."""
            channel_id = ctx.channel.id
            if channel_id in self.conversation_history:
                # Keep only the initial messages (system message and examples)
                self.conversation_history[channel_id] = self._get_initial_messages(
                    ctx.channel
                )
                await ctx.send(
                    "🧹 Memory wiped! I'm starting fresh, but I'll keep my personality intact!"
                )
            else:
                await ctx.send("No conversation history to wipe.")

        @self.command(name="prompt")
        async def prompt_command(
            ctx: Context[Bot],
            action: Optional[str] = None,
            *,
            line: Optional[str] = None,
        ) -> None:
            """Manage the system prompt."""

            if not action:
                # Display current prompt as a file attachment
                file = discord.File("system_prompt.txt")
                await ctx.send("**Current System Prompt:**", file=file)
                await ctx.send(
                    "Use `prompt add <line>` to add a line or `prompt remove <line>` to remove a line."
                )
                return

            if action.lower() == "add" and line:
                # Add a new line
                lines = system_prompt.add_line(line)
                await ctx.send(
                    f"Added line to system prompt: `{line}`\n\nUpdated prompt now has {len(lines)} lines."
                )
                # Update all channels with new system prompt
                for channel_id in self.conversation_history:
                    if self.conversation_history[channel_id]:
                        channel = self.get_channel(channel_id)
                        if channel:  # Check if channel exists
                            self.conversation_history[channel_id][0]["content"] = (
                                get_system_prompt(self.get_server_name(channel))
                            )

            elif action.lower() == "remove" and line:
                # Remove a line
                original_lines = system_prompt.load_system_prompt()
                if line not in original_lines:
                    await ctx.send(f"Line not found in system prompt: `{line}`")
                    return

                lines = system_prompt.remove_line(line)
                await ctx.send(
                    f"Removed line from system prompt: `{line}`\n\nUpdated prompt now has {len(lines)} lines."
                )
                # Update all channels with new system prompt
                for channel_id in self.conversation_history:
                    if self.conversation_history[channel_id]:
                        channel = self.get_channel(channel_id)
                        if channel:  # Check if channel exists
                            self.conversation_history[channel_id][0]["content"] = (
                                get_system_prompt(self.get_server_name(channel))
                            )

            else:
                await ctx.send(
                    "Invalid command. Use `prompt` to view, `prompt add <line>` to add, or `prompt remove <line>` to remove."
                )

        @self.command(name="shutup")
        async def shutup_command(ctx: Context[Bot]) -> None:
            """Stop all responses in the current channel."""
            channel_id = ctx.channel.id

            # Cancel the response task if it exists
            task = self.response_tasks[channel_id]
            if task is not None and not task.done():
                # Mark this task as shut up
                self.shutup_tasks.add(task)
                task.cancel()
                self.response_tasks[channel_id] = None

            # Clear the response queue
            while not self.response_queues[channel_id].empty():
                try:
                    self.response_queues[channel_id].get_nowait()
                except asyncio.QueueEmpty:
                    break

            await ctx.send("🤫 Stopped all responses in this channel.")

        @self.command(name="reactions")
        async def reactions_command(ctx: Context[Bot]) -> None:
            """Display reaction statistics for the bot's messages in the current channel."""
            channel_id = ctx.channel.id
            message_reactions = self._get_reaction_stats(channel_id)

            if not message_reactions:
                await ctx.send("No reaction data available for this channel yet.")
                return

            # Create a summary of reactions
            channel_name = self.get_channel_name(ctx.channel)
            summary = f"**Reaction Statistics for #{channel_name}**\n\n"
            summary += self._format_reaction_summary(message_reactions)
            await ctx.send(summary)

    async def on_ready(self) -> None:
        """Event triggered when the bot is ready."""
        if self.user is None:
            logger.error("Bot user is None!")
            return

        logger.info(f"Logged in as {self.user.name} ({self.user.id})")
        logger.info(f"Using API URL: {config.API_URL}")

        # Log intents for debugging
        logger.info(f"Bot intents: {self.intents}")

        await self.change_presence(activity=discord.Game(name=f"with myself"))

        # Initialize history for all channels the bot can see
        logger.info(
            f"Starting to initialize history for all channels (fetch limit: {config.HISTORY_FETCH_LIMIT})"
        )

        for guild in self.guilds:
            logger.info(f"Connected to {guild.name}")

        logger.info(f"Bot is ready!")

    async def _initialize_channel_history(
        self, channel: Union[TextChannel, DMChannel]
    ) -> None:
        """Initialize conversation history for a channel by fetching recent messages."""
        channel_id = channel.id

        # Skip if history already exists for this channel
        if (
            channel_id in self.conversation_history
            and self.conversation_history[channel_id]
        ):
            return

        # Initialize with system message and example interactions
        self.conversation_history[channel_id] = self._get_initial_messages(channel)

        try:
            # Fetch recent messages from the channel
            message_limit = config.HISTORY_FETCH_LIMIT
            all_messages: List[Dict[str, Any]] = []

            logger.info(
                f"Fetching up to {message_limit} messages from channel {self.get_channel_name(channel)}"
            )

            # Use the Discord API to fetch recent messages
            async for message in channel.history(limit=message_limit):
                # Skip empty messages
                content = message.content.strip()
                if not content:
                    continue

                # Clean up the content by replacing Discord mentions with usernames
                content = self._clean_message_content(message)

                # Add to our list with metadata for sorting and grouping
                message_data = {
                    "role": (
                        "assistant" if message.author == self.get_bot_user() else "user"
                    ),
                    "content": (
                        f"{message.author.display_name}: {content}"
                        if message.author != self.get_bot_user()
                        else content
                    ),
                    "timestamp": float(message.created_at.timestamp()),
                    "author_id": str(message.author.id),
                    "is_directed": bool(
                        self.is_mentioned_in(message)
                        if message.author != self.get_bot_user()
                        else False
                    ),
                }
                all_messages.append(message_data)

                # If this is a bot message, store its info and initialize reaction counts
                if message.author == self.get_bot_user():
                    self.reaction_data[message.id] = {
                        "info": {
                            "content": content,
                            "channel_id": channel_id,
                            "channel_name": self.get_channel_name(channel),
                            "server_name": self.get_server_name(channel),
                            "timestamp": float(message.created_at.timestamp()),
                        },
                        "reactions": [],
                    }

                    # Initialize reaction counts from message reactions
                    for reaction in message.reactions:
                        emoji = str(reaction.emoji)
                        self.reaction_data[message.id]["reactions"].append(
                            {"emoji": emoji, "count": reaction.count}
                        )

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
                        "is_directed": bool(message_data["is_directed"]),
                    }
                elif current_group["author_id"] == str(
                    message_data["author_id"]
                ) and current_group["role"] == str(message_data["role"]):
                    # Add to current group
                    current_group["content"] += "\n\n" + str(message_data["content"])
                    # Update is_directed if this message is directed at the bot
                    if bool(message_data["is_directed"]):
                        current_group["is_directed"] = True
                else:
                    # Different author, add the current group and start a new one
                    grouped_messages.append(current_group)
                    current_group = {
                        "role": str(message_data["role"]),
                        "content": str(message_data["content"]),
                        "author_id": str(message_data["author_id"]),
                        "is_directed": bool(message_data["is_directed"]),
                    }

            # Add the last group if it exists
            if current_group is not None:
                grouped_messages.append(current_group)

            # Add messages to history (without extra metadata)
            for message_data in grouped_messages:
                # Create a clean copy without the extra fields we added
                message_copy = {
                    "role": str(message_data["role"]),
                    "content": str(message_data["content"]),
                }
                self.conversation_history[channel_id].append(message_copy)

            # Trim if needed
            # +1 for system message
            if len(self.conversation_history[channel_id]) > config.MAX_HISTORY + 1:
                self.conversation_history[channel_id] = [
                    self.conversation_history[channel_id][0]  # Keep system message
                ] + self.conversation_history[channel_id][-(config.MAX_HISTORY) :]

            # Save reaction data after initializing history
            self._save_reaction_data()

            logger.info(
                f"Initialized history for channel {self.get_channel_name(channel)} with {len(grouped_messages)} message groups"
            )

        except Exception as e:
            logger.error(
                f"Error initializing history for channel {self.get_channel_name(channel)}: {str(e)}"
            )
            # Keep the initial messages at least
            self.conversation_history[channel_id] = self._get_initial_messages(channel)

    async def _process_response_queue(self, channel_id: int) -> None:
        """Process the response queue for a channel."""
        while True:
            try:
                # Get the next message to respond to
                message = await self.response_queues[channel_id].get()

                # Process the response
                try:
                    await self._handle_streaming_response(message, channel_id)
                except Exception as e:
                    logger.error(f"Error processing response: {str(e)}")
                    await message.reply(f"Sorry, I encountered an error: {str(e)}")
                finally:
                    # Mark the task as done
                    self.response_queues[channel_id].task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in response queue processor: {str(e)}")
                # Don't break on error, just log and continue processing
                continue

    async def _ensure_queue_processor(self, channel_id: int) -> None:
        """Ensure there's an active queue processor for the channel."""
        task = self.response_tasks[channel_id]
        if task is None or task.done():
            # Cancel existing task if it exists
            if task is not None and not task.done():
                task.cancel()
            # Create a new processor task
            self.response_tasks[channel_id] = asyncio.create_task(
                self._process_response_queue(channel_id)
            )

    async def on_message(self, message: Message) -> None:
        """Event triggered when a message is received."""
        # Ignore messages from the bot itself
        if message.author == self.get_bot_user():
            return

        # Get or initialize conversation history for this channel
        channel_id = message.channel.id
        if channel_id not in self.conversation_history:
            # For new channels, initialize history by fetching recent messages
            if isinstance(message.channel, TextChannel):
                await self._initialize_channel_history(message.channel)
            else:
                # For DMs or other channel types, just add initial messages
                self.conversation_history[channel_id] = self._get_initial_messages(
                    message.channel
                )

        # Check if this message is directed at the bot
        is_dm = isinstance(message.channel, DMChannel)
        is_mentioned = self.is_mentioned_in(message)
        is_directed_at_bot = is_dm or is_mentioned

        # Only process commands if the bot is mentioned
        if is_mentioned:
            # Get the context to check if this is a valid command
            ctx = await self.get_context(message)

            # Log the command attempt for debugging
            content = message.content.strip()
            command_name = content.split()[1] if len(content.split()) > 1 else "unknown"

            if ctx.valid:
                logger.info(f"Processing valid command: {command_name}")
                # This is a valid command, process it
                await self.process_commands(message)
                return
            else:
                logger.info(f"Ignoring invalid command: {command_name}")

        # Add all messages to history, not just those directed at the bot
        # Format the username and message content
        content = message.content.strip()

        # Skip empty messages
        if not content:
            return

        # Clean up the content by replacing Discord mentions with usernames
        content = self._clean_message_content(message)

        # Check if this message is already the last message in history
        message_content = f"{message.author.display_name}: {content}"
        history = self.conversation_history[channel_id]
        is_duplicate = (
            history
            and history[-1]["role"] == "user"
            and history[-1]["content"] == message_content
        )

        # Add user message to history if it's not a duplicate
        if not is_duplicate:
            self.conversation_history[channel_id].append(
                {"role": "user", "content": message_content}
            )

        # Trim history if it exceeds the maximum length
        if len(self.conversation_history[channel_id]) > config.MAX_HISTORY:
            # Keep the system message if it exists
            if self.conversation_history[channel_id][0]["role"] == "system":
                self.conversation_history[channel_id] = [
                    self.conversation_history[channel_id][0]
                ] + self.conversation_history[channel_id][-(config.MAX_HISTORY - 1) :]
            else:
                self.conversation_history[channel_id] = self.conversation_history[
                    channel_id
                ][-config.MAX_HISTORY :]

        # Only respond to messages that mention the bot or are direct messages
        if not is_directed_at_bot:
            return

        # Remove the bot mention from the message content for processing
        try:
            clean_content = re.sub(
                f"<@!?{self.get_bot_user().id}>", "", content
            ).strip()
        except RuntimeError:
            logger.error("Bot user not initialized, cannot process message")
            return

        # If the message is empty after removing the mention, don't process it
        if not clean_content:
            return

        try:
            # Add message to response queue
            await self.response_queues[channel_id].put(message)
            # Ensure there's a queue processor running
            await self._ensure_queue_processor(channel_id)
            # Send acknowledgment
            await message.add_reaction("💭")
            logger.info(f"Added message to queue for channel {channel_id}")
        except Exception as e:
            logger.error(f"Error queueing response: {str(e)}")
            await message.reply(f"Sorry, I encountered an error: {str(e)}")

    def _clean_message_content(self, message: Message) -> str:
        """
        Clean up message content by replacing Discord mentions with usernames.

        Args:
            message: The Discord message

        Returns:
            Cleaned message content
        """
        content = message.content.strip()

        # Replace user mentions with usernames
        for user in message.mentions:
            mention_pattern = f"<@!?{user.id}>"
            username = user.display_name
            content = re.sub(mention_pattern, f"@{username}", content)

        # Replace channel mentions
        for channel in message.channel_mentions:
            channel_pattern = f"<#{channel.id}>"
            channel_name = channel.name
            content = re.sub(channel_pattern, f"#{channel_name}", content)

        # Replace role mentions
        if hasattr(message, "role_mentions"):
            for role in message.role_mentions:
                role_pattern = f"<@&{role.id}>"
                role_name = role.name
                content = re.sub(role_pattern, f"@{role_name}", content)

        return content

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
                f"History length: {len(self.conversation_history[channel_id])} messages"
            )

            stream = self.api_client.chat(  # pyright: ignore
                model=str(config.MODEL_NAME),
                messages=self.conversation_history[channel_id],
                stream=True,
                keep_alive=-1,
                options={
                    "temperature": float(config.TEMPERATURE),
                    "top_p": float(config.TOP_P),
                    "presence_penalty": float(config.PRESENCE_PENALTY),
                    "frequency_penalty": float(config.FREQUENCY_PENALTY),
                    "seed": int(config.SEED) if config.SEED != -1 else None,
                },
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

            # Store the complete response in conversation history
            self.conversation_history[channel_id].append(
                {"role": "assistant", "content": full_response}
            )

        except Exception as e:
            error_message = f"Error in streaming response: {str(e)}"
            logger.error(error_message)
            logger.error(f"Full error details: {repr(e)}")
            yield (LineStatus.COMPLETE, error_message)

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
                    await message.remove_reaction("💭", self.get_bot_user())
                except discord.errors.NotFound:
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

                            if line_count > 9:
                                logger.info(
                                    "Reached maximum line limit, stopping response"
                                )
                                await message.channel.send(
                                    "-# Response truncated due to length limit"
                                )
                                break

                        except discord.errors.HTTPException as e:
                            logger.warning(f"Failed to send message chunk: {str(e)}")
            except Exception as e:
                logger.error(f"Error sending messages: {str(e)}")
                await message.reply(f"Error sending messages: {str(e)}")
            finally:
                # Remove the typing reaction
                try:
                    await message.remove_reaction("⌨️", self.get_bot_user())
                except discord.errors.NotFound:
                    # Reaction might not exist, that's okay
                    pass
                # Remove this task from the shutup set
                self.shutup_tasks.discard(current_task)

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

    async def on_reaction_add(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """Handle reaction add events."""
        logger.info(
            f"Reaction add event received: {reaction.emoji} on message {reaction.message.id} by {user.name}"
        )

        # Ignore reactions from the bot itself
        if user == self.get_bot_user():
            logger.debug("Ignoring bot's own reaction")
            return

        # Only track reactions on messages from the bot
        if reaction.message.author != self.get_bot_user():
            logger.debug("Ignoring reaction on non-bot message")
            return

        message_id = reaction.message.id
        emoji = str(reaction.emoji)

        # Update reaction count
        if message_id not in self.reaction_data:
            self.reaction_data[message_id] = {
                "info": {
                    "content": reaction.message.content,
                    "channel_id": reaction.message.channel.id,
                    "channel_name": self.get_channel_name(reaction.message.channel),
                    "server_name": self.get_server_name(reaction.message.channel),
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

        logger.info(f"Reaction {emoji} added to message {message_id} by {user.name}")


def run_bot() -> None:
    """Run the bot."""
    bot = DeepBot()
    token = config.DISCORD_TOKEN
    if token is None:
        raise ValueError("DISCORD_TOKEN is not set in config")
    bot.run(token)


if __name__ == "__main__":
    run_bot()
