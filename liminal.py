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
MONITORED_CHANNEL_NAME = "liminal-tbd"
WEBHOOK_URL: Optional[str] = os.getenv("WEBHOOK_URL")

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

        # Add regex patterns for mentions and emojis
        self.mention_pattern = re.compile(r"<@!?(\d+)>")
        self.emoji_pattern = re.compile(r"<:([^:]+):(\d+)>")

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

        # Handle multiline messages
        if "\n" in content:
            lines: List[str] = content.split("\n")
            return "\n".join(f"<{username}> {line}" for line in lines if line.strip())

        return f"<{username}> {content}"

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

    def parse_irc_line(self, line: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse an IRC-style line to extract username and message"""
        match = re.match(r"^<([^>]+)>\s*(.*)$", line.strip())
        if match:
            return match.group(1), match.group(2)
        return None, None

    async def stream_ollama_completion(
        self, prompt: str, channel: MessageableChannel
    ) -> None:
        """Stream completion from Ollama and send messages as they're parsed"""
        webhook = await self.get_webhook_for_channel(channel)
        current_username = None  # Initialize at the start of the method

        if self.active_completion_state is None:
            logger.error(
                f"Stream_ollama_completion called with no active_completion_state."
            )
            return

        completion_data = self.active_completion_state
        current_op_channel_id = completion_data.channel_id

        if not webhook:
            logger.error(
                f"Cannot send messages to channel {getattr(channel, 'id', 'unknown channel')} as it has no send method."
            )
            return

        try:
            # Use async Ollama client for streaming
            stream = await self.api_client.generate(
                model=self.ollama_model,
                prompt=prompt,
                stream=True,
            )

            current_line: str = ""
            messages_sent_this_stream = 0

            async for chunk in stream:
                try:
                    if (
                        completion_data.message_count
                        >= self.max_messages_per_interaction
                    ):
                        logger.info(
                            f"Message limit ({self.max_messages_per_interaction}) reached for channel {current_op_channel_id}."
                        )
                        # Clear the status when hitting message limit
                        if current_username:
                            await bot.change_presence(activity=None)
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
                                    # Update bot's status with current username
                                    await bot.change_presence(
                                        activity=discord.Game(
                                            name=f"as {current_username}"
                                        )
                                    )
                                    logger.debug(
                                        f"Updated status to: {current_username}"
                                    )

                        if "\n" in current_line:
                            lines = current_line.split("\n")
                            for i in range(len(lines) - 1):
                                if (
                                    completion_data.message_count
                                    >= self.max_messages_per_interaction
                                ):
                                    break

                                complete_line = lines[i].strip()
                                if complete_line:
                                    username, message_content = self.parse_irc_line(
                                        complete_line
                                    )
                                    if username and message_content:
                                        await webhook.send(
                                            content=message_content,
                                            username=username,
                                            wait=True,
                                        )
                                        completion_data.message_count += 1
                                        messages_sent_this_stream += 1
                                        await asyncio.sleep(0.5)

                            current_line = lines[-1]

                        if chunk.get("done", False):
                            if (
                                current_line.strip()
                                and completion_data.message_count
                                < self.max_messages_per_interaction
                            ):
                                username, message_content = self.parse_irc_line(
                                    current_line
                                )
                                if username and message_content:
                                    await webhook.send(
                                        content=message_content,
                                        username=username,
                                        wait=True,
                                    )
                                    completion_data.message_count += 1
                                    messages_sent_this_stream += 1
                            # Clear the status when done
                            if current_username:
                                await bot.change_presence(activity=None)
                            break
                except asyncio.CancelledError:
                    logger.info(
                        f"Stream processing cancelled for channel {current_op_channel_id}"
                    )
                    if current_username:
                        await bot.change_presence(activity=None)
                    raise

            if (
                messages_sent_this_stream == 0
                and hasattr(channel, "send")
                and completion_data.message_count < self.max_messages_per_interaction
            ):
                await channel.send(
                    "No valid IRC-style responses were generated in this segment."
                )

        except asyncio.CancelledError:
            logger.info(f"Completion cancelled for channel {current_op_channel_id}")
            raise
        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            # Make sure to clear status on error
            if current_username:
                await bot.change_presence(activity=None)
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


@bot.command()
async def test_irc(ctx: commands.Context[commands.Bot]):
    """Test the IRC formatting and parsing"""
    if irc_bot.monitored_channel_instance is None:
        await ctx.send(
            "Bot is not currently monitoring any channel. Configure with `set_channel` or ensure unique name on startup."
        )
        return

    if ctx.channel.id != irc_bot.monitored_channel_instance.id:
        await ctx.send(
            f"This command can only be run in the monitored channel: {irc_bot.monitored_channel_instance.mention}"
        )
        return

    if irc_bot.active_completion_state is not None:
        logger.info(
            f"Test IRC: Bot is already active in {irc_bot.monitored_channel_instance.name}. Aborting test."
        )
        await ctx.send(
            "Bot is already active. Please wait or interrupt it with a new message."
        )
        return

    test_history = """<Alice> Hey everyone!
<Bob> Hi Alice! How's it going?
<Alice> Pretty good! Working on a new project
<Charlie> What kind of project?"""

    current_channel: MessageableChannel = ctx.channel
    channel_id = ctx.channel.id  # Should be irc_bot.monitored_channel_instance.id

    logger.info(
        f"Starting test_irc for channel {irc_bot.monitored_channel_instance.name} (ID: {channel_id})"
    )
    irc_bot.active_completion_state = ActiveCompletion(
        channel_id=channel_id, message_count=0
    )

    await ctx.send("Sending test completion...")
    try:
        await irc_bot.stream_ollama_completion(test_history, current_channel)
    finally:
        if (
            irc_bot.active_completion_state
            and irc_bot.active_completion_state.channel_id == channel_id
        ):
            logger.info(
                f"Test IRC completion ended for channel {irc_bot.monitored_channel_instance.name} (ID: {channel_id})."
            )
            irc_bot.active_completion_state = None


@bot.command()
@commands.has_permissions(administrator=True)
async def set_channel(ctx: commands.Context[commands.Bot], *, channel_name: str):
    """Set the monitored channel by name (admin only). This will update the bot's single active channel."""
    if ctx.guild is None:
        await ctx.send("This command can only be used in a server.")
        return

    guild_channels: List[discord.TextChannel] = ctx.guild.text_channels
    channel_found: Optional[discord.TextChannel] = discord.utils.get(
        guild_channels, name=channel_name
    )

    if not channel_found:
        await ctx.send(f"Channel '{channel_name}' not found in this server.")
        irc_bot.monitored_channel_instance = None
        irc_bot.monitored_channel_name = channel_name  # Store the desired name
        await ctx.send(
            f"Bot will attempt to monitor '{channel_name}' on next restart or if it appears. Currently not monitoring."
        )
        return

    # Interrupt any ongoing completion if the monitored channel is changing
    if irc_bot.active_completion_state is not None:
        logger.info(
            f"Monitored channel changed by command. Interrupting ongoing completion if any."
        )
        irc_bot.active_completion_state = None  # Clear old state
        await asyncio.sleep(0.2)  # Give a moment for interruption to propagate

    irc_bot.monitored_channel_name = channel_name
    irc_bot.monitored_channel_instance = channel_found
    await ctx.send(f"Now monitoring channel: {channel_found.mention}")
    logger.info(
        f"Monitored channel set by command to: {channel_found.name} (ID: {channel_found.id})"
    )

    webhook = await irc_bot.get_webhook_for_channel(channel_found)
    if not webhook:
        await ctx.send(
            "⚠️ Warning: Cannot access webhooks in this channel. Please create a webhook manually or give the bot 'Manage Webhooks' permission."
        )
    else:
        await ctx.send("✅ Webhook access confirmed for the new channel.")


@bot.command()
@commands.has_permissions(administrator=True)
async def set_webhook(ctx: commands.Context[commands.Bot], webhook_url: str):
    """Set a manual webhook URL for all channels (admin only)"""
    irc_bot.webhook_url = webhook_url
    irc_bot.webhook_cache.clear()  # Clear cache to use new webhook
    await ctx.send("Manual webhook URL set. This will be used for all channels.")


@bot.command()
async def webhook_test(ctx: commands.Context[commands.Bot]):
    """Test webhook access in the current channel"""
    current_channel: MessageableChannel = ctx.channel
    webhook = await irc_bot.get_webhook_for_channel(current_channel)
    if webhook:
        await webhook.send(
            content="Webhook test successful!", username="Webhook Test", wait=True
        )
        await ctx.send("✅ Webhook is working!")
    else:
        await ctx.send(
            "❌ Cannot access webhooks in this channel. Please create one manually or give bot permissions, or use this command in a server text channel."
        )


@bot.command()
async def models(ctx: commands.Context[commands.Bot]):
    """List available Ollama models"""
    try:
        models_list = ollama.list()
        model_names: List[str] = [model["name"] for model in models_list["models"]]

        embed = discord.Embed(
            title="Available Ollama Models",
            description="\n".join(model_names) if model_names else "No models found",
            color=discord.Color.blue(),
        )
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send(f"Error fetching models: {e}")


@bot.command()
@commands.has_permissions(administrator=True)
async def set_model(ctx: commands.Context[commands.Bot], *, model_name: str):
    """Change the Ollama model (admin only)"""
    # Verify model exists
    try:
        models_list = ollama.list()
        available_models = [model["name"] for model in models_list["models"]]

        if model_name not in available_models:
            await ctx.send(
                f"Model '{model_name}' not found. Available models: {', '.join(available_models)}"
            )
            return

        irc_bot.ollama_model = model_name
        await ctx.send(f"Changed model to: {model_name}")
    except Exception as e:
        await ctx.send(f"Error changing model: {e}")


@bot.command()
async def status(ctx: commands.Context[commands.Bot]):
    """Check bot status"""
    embed = discord.Embed(
        title="IRC Completion Bot Status", color=discord.Color.green()
    )
    embed.add_field(name="Model", value=irc_bot.ollama_model, inline=True)

    if irc_bot.monitored_channel_instance:
        embed.add_field(
            name="Monitored Channel",
            value=f"{irc_bot.monitored_channel_instance.name} (ID: {irc_bot.monitored_channel_instance.id})",
            inline=True,
        )
        embed.add_field(
            name="Guild",
            value=irc_bot.monitored_channel_instance.guild.name,
            inline=True,
        )
    else:
        embed.add_field(
            name="Monitored Channel Name",
            value=f"'{irc_bot.monitored_channel_name}' (Not Found/Unique or Error)",
            inline=True,
        )
        embed.add_field(name="Guild", value="N/A", inline=True)

    embed.add_field(name="History Limit", value=MESSAGE_HISTORY_LIMIT, inline=True)

    is_active = irc_bot.active_completion_state is not None
    embed.add_field(
        name="Bot Active", value="✅ Yes" if is_active else "❌ No", inline=True
    )
    if is_active and irc_bot.active_completion_state:
        embed.add_field(
            name="Messages Sent (Current)",
            value=irc_bot.active_completion_state.message_count,
            inline=True,
        )
    else:
        embed.add_field(name="Messages Sent (Current)", value="N/A", inline=True)

    embed.add_field(
        name="Max Messages/Interaction",
        value=irc_bot.max_messages_per_interaction,
        inline=True,
    )

    if irc_bot.webhook_url:
        embed.add_field(name="Webhook Mode", value="Manual URL", inline=True)
    else:
        embed.add_field(name="Webhook Mode", value="Auto", inline=True)

    # Check Ollama connection
    try:
        ollama.list()
        embed.add_field(name="Ollama Status", value="✅ Connected", inline=True)
    except:
        embed.add_field(name="Ollama Status", value="❌ Disconnected", inline=True)

    await ctx.send(embed=embed)


if __name__ == "__main__":
    if DISCORD_TOKEN is None:
        logger.error("DISCORD_TOKEN not found in .env file. Please set it.")
    else:
        bot.run(DISCORD_TOKEN)
