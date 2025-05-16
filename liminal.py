import asyncio
import logging
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import discord
import ollama
from discord.abc import Messageable as MessageableChannel
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord")

# Configuration
DISCORD_TOKEN: Optional[str] = os.getenv("DISCORD_TOKEN")
OLLAMA_MODEL_DEFAULT = "tbd-24b"
MONITORED_CHANNEL_NAME = "shoggoth"
WEBHOOK_URL: Optional[str] = os.getenv("WEBHOOK_URL")
BLACKLIST_FILE = "liminal_user_blacklist.txt"
CENSORED_TEXT = "[message removed]"
DELETED_USER = "Deleted User"

# How many messages to include in context
MESSAGE_HISTORY_LIMIT = 1

# Max messages bot sends before needing new user input
MAX_MESSAGES_PER_INTERACTION = 30


@dataclass
class ActiveCompletion:
    channel_id: int
    message_count: int
    task: Optional[asyncio.Task[None]] = None


# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


class IRCCompletionBot:
    def __init__(self):
        self.channel_histories: Dict[int, str] = {}
        self.webhook_url: Optional[str] = WEBHOOK_URL
        self.webhook_cache: Dict[int, discord.Webhook] = {}
        self.active_completion_state: Optional[ActiveCompletion] = None
        self.monitored_channel_name: str = MONITORED_CHANNEL_NAME
        self.monitored_channel_instance: Optional[discord.TextChannel] = None
        self.ollama_model: str = OLLAMA_MODEL_DEFAULT
        self.max_messages_per_interaction: int = MAX_MESSAGES_PER_INTERACTION
        self.api_client = ollama.AsyncClient()
        self.blacklisted_users = self._load_blacklist()

        # Add regex patterns for mentions and emojis
        self.mention_pattern = re.compile(r"<@!?(\d+)>")
        self.emoji_pattern = re.compile(r"<:([^:]+):(\d+)>")

    def _load_blacklist(self) -> set[str]:
        """Load blacklisted usernames from file"""
        try:
            if os.path.exists(BLACKLIST_FILE):
                with open(BLACKLIST_FILE, "r") as f:
                    return {line.strip() for line in f if line.strip()}
        except Exception as e:
            logger.error(f"Error loading blacklist: {e}")
        return set()

    def _resolve_mentions(self, message: discord.Message, content: str) -> str:
        """Replace <@123456> style mentions with @username format."""

        def replace_mention(match: re.Match[str]) -> str:
            user_id = match.group(1)
            # Try to find user in message mentions first
            for user in message.mentions:
                if str(user.id) == user_id:
                    return f"@{user.name}"
            # Fallback to getting user from bot's cache
            user = bot.get_user(int(user_id))
            return f"@{user.name if user else 'unknown'}"

        return self.mention_pattern.sub(replace_mention, content)

    def _resolve_emojis(self, content: str) -> str:
        """Replace <:emoji:123456> style emojis with :emoji: format."""

        def replace_emoji(match: re.Match[str]) -> str:
            emoji_name = match.group(1)
            return f":{emoji_name}:"

        return self.emoji_pattern.sub(replace_emoji, content)

    def format_message_as_irc(self, message: discord.Message) -> str:
        """Format a Discord message as IRC log line"""
        username: str = message.author.name
        content: str = message.content.strip()

        # Process mentions and emojis
        content = self._resolve_mentions(message, content)
        content = self._resolve_emojis(content)

        # Handle message references (replies)
        if message.reference and hasattr(message.reference, "resolved"):
            referenced_msg = message.reference.resolved
            if isinstance(referenced_msg, discord.Message) and referenced_msg.content:
                # Format the reply with the quoted message and the reply itself
                ref_username = (
                    referenced_msg.author.name if referenced_msg.author else "unknown"
                )
                ref_content = referenced_msg.content.strip()
                ref_content = self._resolve_mentions(referenced_msg, ref_content)
                ref_content = self._resolve_emojis(ref_content)

                # Return both the quote and the reply
                return f"<{username}> > {ref_content}\n<{username}> @{ref_username} {content}"

        # Handle multiline messages
        if "\n" in content:
            lines: List[str] = content.split("\n")
            return "\n".join(f"<{username}> {line}" for line in lines if line.strip())

        return f"<{username}> {content}"

    def parse_irc_line(self, line: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse an IRC-style line to extract username and message"""
        match = re.match(r"^<([^>]+)>\s*(.*)$", line.strip())
        if match:
            username, message = match.group(1), match.group(2)
            # Check if username is blacklisted
            if username in self.blacklisted_users:
                return DELETED_USER, CENSORED_TEXT
            return username, message
        return None, None

    async def get_webhook_for_channel(
        self, channel: MessageableChannel
    ) -> Optional[discord.Webhook]:
        """Get webhook for the channel - either from URL or try to find/create one"""
        if not isinstance(channel, discord.TextChannel):
            logger.warning(
                f"Webhooks are only supported in TextChannels, not {type(channel)}"
            )
            return None

        # If using a manual webhook URL
        if self.webhook_url:
            if channel.id not in self.webhook_cache:
                self.webhook_cache[channel.id] = discord.Webhook.from_url(
                    self.webhook_url, client=bot
                )
            return self.webhook_cache[channel.id]

        # Try to get existing webhook
        try:
            webhooks = await channel.webhooks()
            webhook: Optional[discord.Webhook] = discord.utils.get(
                webhooks, name="IRC Completion Bot"
            )

            if webhook:
                self.webhook_cache[channel.id] = webhook
                return webhook
            else:
                # Try to create one
                webhook = await channel.create_webhook(name="IRC Completion Bot")
                self.webhook_cache[channel.id] = webhook
                logger.info(f"Created webhook for channel {channel.name}")
                return webhook
        except discord.Forbidden:
            logger.error(f"No permission to access webhooks in {channel.name}")
            return None
        except Exception as e:
            logger.error(f"Error with webhooks: {e}")
            return None

    async def get_channel_history(
        self,
        channel: MessageableChannel,
        reference_message: Optional[discord.Message] = None,
        limit: int = 50,
    ) -> str:
        """Get recent channel history formatted as IRC log"""
        channel_identifier = getattr(
            channel, "name", str(getattr(channel, "id", "unknown"))
        )
        logger.info(
            f"Fetching history for channel {channel_identifier} with limit {limit}"
        )
        if reference_message:
            logger.info(
                f"Reference message content: {reference_message.content[:100]}..."
            )

        messages: List[discord.Message] = []
        if hasattr(channel, "history"):
            # If we have a reference message, get messages before it
            if reference_message:
                logger.info("Using reference message to fetch history")
                history_iterator = channel.history(
                    limit=limit, before=reference_message
                )
            else:
                logger.info("No reference message, fetching recent history")
                history_iterator = channel.history(limit=limit)
        else:
            logger.warning(
                f"Channel {getattr(channel, 'id', 'unknown channel')} of type {type(channel)} does not have history method or requires fetching."
            )
            return ""

        message_count = 0
        async for message_item in history_iterator:
            message_count += 1
            logger.debug(
                f"Found message {message_count}: {message_item.content[:100]}..."
            )

            # Only filter out empty messages
            if message_item.content:
                messages.append(message_item)
                logger.info(
                    f"Added message {len(messages)}: {message_item.content[:100]}..."
                )
            else:
                logger.debug(f"Skipped empty message")

        logger.info(
            f"Found {message_count} total messages, kept {len(messages)} messages"
        )

        # Reverse to get chronological order
        messages.reverse()

        # Add the reference message at the end if it has content
        if reference_message and reference_message.content:
            messages.append(reference_message)
            logger.info("Added reference message to history")

        # Format as IRC
        irc_lines: List[str] = []
        for msg in messages:
            irc_line = self.format_message_as_irc(msg)
            if irc_line:
                irc_lines.append(irc_line)

        formatted_history = "\n".join(irc_lines) + "\n"
        logger.info(f"Final formatted history has {len(irc_lines)} lines")
        logger.debug(f"Formatted history:\n{formatted_history}")

        return formatted_history

    def _get_channel_identifier(self, channel: MessageableChannel) -> str:
        """Get a human-readable identifier for a channel."""
        return getattr(channel, "name", str(getattr(channel, "id", "unknown")))

    async def _clear_bot_status(self) -> None:
        """Clear the bot's current status."""
        await bot.change_presence(activity=None)

    async def _set_bot_status(self, username: str) -> None:
        """Set the bot's status to show current username."""
        await bot.change_presence(activity=discord.Game(username))

    async def _send_webhook_message(
        self,
        webhook: discord.Webhook,
        message_content: str,
        username: str,
        completion_data: ActiveCompletion,
    ) -> None:
        """Send a message via webhook with proper blacklist checking."""
        if username in self.blacklisted_users:
            username = DELETED_USER
            message_content = CENSORED_TEXT

        await webhook.send(
            content=message_content,
            username=username,
            wait=True,
        )
        completion_data.message_count += 1

    def _check_message_limit(self, completion_data: ActiveCompletion) -> bool:
        """Check if message limit has been reached."""
        return completion_data.message_count >= self.max_messages_per_interaction

    async def stream_ollama_completion(
        self, prompt: str, channel: MessageableChannel
    ) -> None:
        """Stream completion from Ollama and send messages as they're parsed"""
        webhook = await self.get_webhook_for_channel(channel)
        current_username = None

        if self.active_completion_state is None:
            logger.error(
                f"Stream_ollama_completion called with no active_completion_state."
            )
            return

        completion_data = self.active_completion_state
        current_op_channel_id = completion_data.channel_id

        if not webhook:
            logger.error(
                f"Cannot send messages to channel {self._get_channel_identifier(channel)} as it has no send method."
            )
            return

        try:
            stream = await self.api_client.generate(
                model=self.ollama_model,
                prompt=prompt,
                stream=True,
            )

            current_line: str = ""
            messages_sent_this_stream = 0

            async for chunk in stream:
                try:
                    if self._check_message_limit(completion_data):
                        logger.info(
                            f"Message limit ({self.max_messages_per_interaction}) reached for channel {current_op_channel_id}."
                        )
                        if current_username:
                            await self._clear_bot_status()
                        break

                    if "response" in chunk:
                        current_line += chunk["response"]

                        # Check for complete username in current line
                        if "<" in current_line and ">" in current_line:
                            username_match = re.match(r"^<([^>]+)>", current_line)
                            if username_match:
                                new_username = username_match.group(1).strip()
                                if new_username != current_username:
                                    current_username = new_username
                                    await self._set_bot_status(current_username)

                        if "\n" in current_line:
                            lines = current_line.split("\n")
                            for i in range(len(lines) - 1):
                                if self._check_message_limit(completion_data):
                                    break

                                complete_line = lines[i].strip()
                                if complete_line:
                                    username, message_content = self.parse_irc_line(
                                        complete_line
                                    )
                                    if username and message_content:
                                        await self._send_webhook_message(
                                            webhook,
                                            message_content,
                                            username,
                                            completion_data,
                                        )
                                        messages_sent_this_stream += 1
                                        await asyncio.sleep(0.5)

                            current_line = lines[-1]

                        if chunk.get("done", False):
                            if current_line.strip() and not self._check_message_limit(
                                completion_data
                            ):
                                username, message_content = self.parse_irc_line(
                                    current_line
                                )
                                if username and message_content:
                                    await self._send_webhook_message(
                                        webhook,
                                        message_content,
                                        username,
                                        completion_data,
                                    )
                                    messages_sent_this_stream += 1
                            if current_username:
                                await self._clear_bot_status()
                            break
                except asyncio.CancelledError:
                    logger.info(
                        f"Stream processing cancelled for channel {current_op_channel_id}"
                    )
                    if current_username:
                        await self._clear_bot_status()
                    raise

            if (
                messages_sent_this_stream == 0
                and hasattr(channel, "send")
                and not self._check_message_limit(completion_data)
            ):
                await channel.send(
                    "No valid IRC-style responses were generated in this segment."
                )

        except asyncio.CancelledError:
            logger.info(f"Completion cancelled for channel {current_op_channel_id}")
            raise
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            if current_username:
                await self._clear_bot_status()
            if hasattr(channel, "send"):
                await channel.send(f"Error during Ollama generation: {e}")


# Create bot instance
irc_bot = IRCCompletionBot()


@bot.event
async def on_ready():
    logger.info(f"{bot.user.name if bot.user else 'Bot'} has connected to Discord!")

    found_channels: List[discord.TextChannel] = []
    for guild in bot.guilds:
        text_channels: List[discord.TextChannel] = guild.text_channels
        for channel in text_channels:
            if channel.name == irc_bot.monitored_channel_name:
                found_channels.append(channel)

    if not found_channels:
        logger.warning(
            f"No channel found with the name '{irc_bot.monitored_channel_name}'. Bot will not monitor any channel."
        )
        irc_bot.monitored_channel_instance = None
    elif len(found_channels) == 1:
        irc_bot.monitored_channel_instance = found_channels[0]
        logger.info(
            f"Monitoring single channel: {irc_bot.monitored_channel_instance.name} (ID: {irc_bot.monitored_channel_instance.id}) in guild '{irc_bot.monitored_channel_instance.guild.name}'"
        )
        webhook = await irc_bot.get_webhook_for_channel(
            irc_bot.monitored_channel_instance
        )
        if webhook:
            logger.info(
                f"Webhook access confirmed for {irc_bot.monitored_channel_instance.name}"
            )
        else:
            logger.warning(
                f"No webhook access for {irc_bot.monitored_channel_instance.name}. Bot may not be able to send IRC-style messages."
            )
    else:
        logger.error(
            f"CRITICAL: Found {len(found_channels)} channels matching the name '{irc_bot.monitored_channel_name}'. "
            f"The bot is designed to monitor only one channel. Please ensure the name is unique. "
            f"Bot will not monitor any channel. Found: {[f'{c.name} (ID: {c.id}) in {c.guild.name}' for c in found_channels]}"
        )
        irc_bot.monitored_channel_instance = None


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or irc_bot.monitored_channel_instance is None:
        return

    if message.channel.id == irc_bot.monitored_channel_instance.id:
        current_channel_id = message.channel.id
        channel_name = irc_bot.monitored_channel_instance.name

        logger.info(
            f"Processing message in channel {channel_name}: {message.content[:100]}..."
        )

        # Capture the state that this on_message invocation will create and manage.
        this_invocation_completion_state = ActiveCompletion(
            channel_id=current_channel_id,
            message_count=0,
        )

        if (
            irc_bot.active_completion_state is not None
            and irc_bot.active_completion_state.channel_id == current_channel_id
        ):
            logger.info(
                f"New message in monitored channel {channel_name} while bot is active. Cancelling previous task."
            )
            # Cancel the task
            if irc_bot.active_completion_state.task:
                old_task = irc_bot.active_completion_state.task
                old_task.cancel()
                try:
                    await old_task
                except asyncio.CancelledError:
                    pass

        logger.info(
            f"Setting up new completion for channel {channel_name} (ID: {current_channel_id})"
        )
        irc_bot.active_completion_state = this_invocation_completion_state

        current_channel: MessageableChannel = message.channel

        try:
            history = await irc_bot.get_channel_history(
                current_channel, reference_message=message, limit=MESSAGE_HISTORY_LIMIT
            )

            # Create and store the task
            completion_task = asyncio.create_task(
                irc_bot.stream_ollama_completion(history, current_channel)
            )
            this_invocation_completion_state.task = completion_task
            await completion_task

        except asyncio.CancelledError:
            logger.info(f"Completion task was cancelled for channel {channel_name}")
        except Exception as e:
            logger.error(
                f"Error processing message in {channel_name}: {e}", exc_info=True
            )
            if hasattr(current_channel, "send"):
                await current_channel.send(f"Error generating completion: {str(e)}")
        finally:
            # Only clear the global state if it's the exact same state object
            # that this specific on_message invocation created and managed.
            if irc_bot.active_completion_state is this_invocation_completion_state:
                logger.info(
                    f"Completion attempt for this message context ended for channel {channel_name} (ID: {current_channel_id})."
                )
                irc_bot.active_completion_state = None
            else:
                logger.info(
                    f"Completion attempt for this message context concluded for channel {channel_name} (ID: {current_channel_id}), "
                    f"but active_completion_state was already changed or cleared by a newer message. No action taken on state by this finally block."
                )

    await bot.process_commands(message)


if __name__ == "__main__":
    if DISCORD_TOKEN is None:
        logger.error("DISCORD_TOKEN not found in .env file. Please set it.")
    else:
        bot.run(DISCORD_TOKEN)
