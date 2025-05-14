import asyncio
import logging
import os
import re

import discord
import ollama
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("discord")

# Configuration
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
OLLAMA_MODEL = "tbd-24b"
MONITORED_CHANNEL_NAME = "liminal-tbd"
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
MESSAGE_HISTORY_LIMIT = 2  # How many messages to include in context

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)


class IRCCompletionBot:
    def __init__(self):
        self.channel_histories = {}
        self.webhook_url = WEBHOOK_URL
        self.webhook_cache = {}
        self.active_completions = set()
        self.monitored_channel_name = MONITORED_CHANNEL_NAME

    async def get_webhook_for_channel(self, channel):
        """Get webhook for the channel - either from URL or try to find/create one"""
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
            webhook = discord.utils.get(webhooks, name="IRC Completion Bot")

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

    def format_message_as_irc(self, message):
        """Format a Discord message as IRC log line"""
        username = message.author.name
        content = message.content.strip()

        # Handle multiline messages
        if "\n" in content:
            lines = content.split("\n")
            return "\n".join(f"<{username}> {line}" for line in lines if line.strip())

        return f"<{username}> {content}"

    async def get_channel_history(self, channel, limit=50):
        """Get recent channel history formatted as IRC log"""
        messages = []
        async for message in channel.history(limit=limit):
            if (
                message.content and not message.author.bot
            ):  # Skip bot messages and empty messages
                messages.append(message)

        # Reverse to get chronological order
        messages.reverse()

        # Format as IRC
        irc_lines = []
        for msg in messages:
            irc_line = self.format_message_as_irc(msg)
            if irc_line:
                irc_lines.append(irc_line)

        return "\n".join(irc_lines)

    def parse_irc_line(self, line):
        """Parse an IRC-style line to extract username and message"""
        match = re.match(r"^<([^>]+)>\s*(.*)$", line.strip())
        if match:
            return match.group(1), match.group(2)
        return None, None

    async def stream_ollama_completion(self, prompt, channel):
        """Stream completion from Ollama and send messages as they're parsed"""
        webhook = await self.get_webhook_for_channel(channel)

        if not webhook:
            await channel.send(
                "❌ Cannot send IRC-style messages: No webhook access. Please create a webhook manually or give the bot 'Manage Webhooks' permission."
            )
            return

        # Add a system prompt to encourage IRC-style responses
        system_prompt = """You are continuing an IRC chat log. Always respond in the format:
<username> message
You can have multiple users respond. Each line should be from one user. Stay in character based on the conversation context."""

        full_prompt = f"{system_prompt}\n\n{prompt}\n"

        try:
            # Use ollama library to stream the response
            stream = ollama.generate(
                model=OLLAMA_MODEL, prompt=full_prompt, stream=True
            )

            current_line = ""
            messages_sent = 0

            for chunk in stream:
                if "response" in chunk:
                    current_line += chunk["response"]

                    # Check for complete lines
                    if "\n" in current_line:
                        lines = current_line.split("\n")

                        # Process all complete lines
                        for i in range(len(lines) - 1):
                            complete_line = lines[i].strip()
                            if complete_line:
                                username, message = self.parse_irc_line(complete_line)
                                if username and message:
                                    # Send via webhook with custom username
                                    await webhook.send(
                                        content=message, username=username, wait=True
                                    )
                                    messages_sent += 1
                                    await asyncio.sleep(
                                        0.5
                                    )  # Small delay between messages

                        # Keep the incomplete line
                        current_line = lines[-1]

                    # Check if the stream is done
                    if chunk.get("done", False):
                        # Process any remaining text
                        if current_line.strip():
                            username, message = self.parse_irc_line(current_line)
                            if username and message:
                                await webhook.send(
                                    content=message, username=username, wait=True
                                )
                                messages_sent += 1
                        break

            if messages_sent == 0:
                await channel.send("No valid IRC-style responses were generated.")

        except Exception as e:
            logger.error(f"Error generating completion: {e}")
            raise


# Create bot instance
irc_bot = IRCCompletionBot()


@bot.event
async def on_ready():
    logger.info(f"{bot.user} has connected to Discord!")

    # Find the monitored channel
    for guild in bot.guilds:
        channel = discord.utils.get(
            guild.text_channels, name=irc_bot.monitored_channel_name
        )
        if channel:
            logger.info(f"Found monitored channel: {channel.name} in {guild.name}")
            # Check webhook access
            webhook = await irc_bot.get_webhook_for_channel(channel)
            if webhook:
                logger.info(f"Webhook access confirmed for {channel.name}")
            else:
                logger.warning(f"No webhook access for {channel.name}")


@bot.event
async def on_message(message):
    # Ignore bot's own messages
    if message.author.bot:
        return

    # Check if message is in monitored channel (by name)
    if message.channel.name == irc_bot.monitored_channel_name:
        # Avoid infinite loops - don't respond if we're already generating
        if message.channel.id in irc_bot.active_completions:
            return

        # Mark channel as active
        irc_bot.active_completions.add(message.channel.id)

        try:
            # Get channel history
            history = await irc_bot.get_channel_history(
                message.channel, limit=MESSAGE_HISTORY_LIMIT
            )

            # Add the current message to history
            current_message_irc = irc_bot.format_message_as_irc(message)
            if current_message_irc:
                history = (
                    f"{history}\n{current_message_irc}"
                    if history
                    else current_message_irc
                )

            # Stream completion from Ollama
            await irc_bot.stream_ollama_completion(history, message.channel)

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            await message.channel.send(f"Error generating completion: {str(e)}")
        finally:
            # Remove channel from active set
            irc_bot.active_completions.discard(message.channel.id)

    # Process commands
    await bot.process_commands(message)


@bot.command()
async def test_irc(ctx):
    """Test the IRC formatting and parsing"""
    test_history = """<Alice> Hey everyone!
<Bob> Hi Alice! How's it going?
<Alice> Pretty good! Working on a new project
<Charlie> What kind of project?"""

    await ctx.send("Sending test completion...")
    await irc_bot.stream_ollama_completion(test_history, ctx.channel)


@bot.command()
@commands.has_permissions(administrator=True)
async def set_channel(ctx, *, channel_name: str):
    """Set the monitored channel by name (admin only)"""
    # Find channel by name
    channel = discord.utils.get(ctx.guild.text_channels, name=channel_name)

    if not channel:
        await ctx.send(f"Channel '{channel_name}' not found in this server.")
        return

    irc_bot.monitored_channel_name = channel_name
    await ctx.send(f"Now monitoring channel: {channel.mention}")

    # Check webhook access
    webhook = await irc_bot.get_webhook_for_channel(channel)
    if not webhook:
        await ctx.send(
            "⚠️ Warning: Cannot access webhooks in this channel. Please create a webhook manually or give the bot 'Manage Webhooks' permission."
        )


@bot.command()
@commands.has_permissions(administrator=True)
async def set_webhook(ctx, webhook_url: str):
    """Set a manual webhook URL for all channels (admin only)"""
    irc_bot.webhook_url = webhook_url
    irc_bot.webhook_cache.clear()  # Clear cache to use new webhook
    await ctx.send("Manual webhook URL set. This will be used for all channels.")


@bot.command()
async def webhook_test(ctx):
    """Test webhook access in the current channel"""
    webhook = await irc_bot.get_webhook_for_channel(ctx.channel)
    if webhook:
        await webhook.send(
            content="Webhook test successful!", username="Webhook Test", wait=True
        )
        await ctx.send("✅ Webhook is working!")
    else:
        await ctx.send(
            "❌ Cannot access webhooks in this channel. Please create one manually or give bot permissions."
        )


@bot.command()
async def models(ctx):
    """List available Ollama models"""
    try:
        models_list = ollama.list()
        model_names = [model["name"] for model in models_list["models"]]

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
async def set_model(ctx, *, model_name: str):
    """Change the Ollama model (admin only)"""
    global OLLAMA_MODEL

    # Verify model exists
    try:
        models_list = ollama.list()
        available_models = [model["name"] for model in models_list["models"]]

        if model_name not in available_models:
            await ctx.send(
                f"Model '{model_name}' not found. Available models: {', '.join(available_models)}"
            )
            return

        OLLAMA_MODEL = model_name
        await ctx.send(f"Changed model to: {model_name}")
    except Exception as e:
        await ctx.send(f"Error changing model: {e}")


@bot.command()
async def status(ctx):
    """Check bot status"""
    # Find monitored channel
    monitored_channel = None
    for guild in bot.guilds:
        channel = discord.utils.get(
            guild.text_channels, name=irc_bot.monitored_channel_name
        )
        if channel:
            monitored_channel = channel
            break

    embed = discord.Embed(
        title="IRC Completion Bot Status", color=discord.Color.green()
    )
    embed.add_field(name="Model", value=OLLAMA_MODEL, inline=True)
    embed.add_field(
        name="Monitored Channel", value=irc_bot.monitored_channel_name, inline=True
    )
    embed.add_field(
        name="Channel Found", value="✅" if monitored_channel else "❌", inline=True
    )
    embed.add_field(name="History Limit", value=MESSAGE_HISTORY_LIMIT, inline=True)
    embed.add_field(
        name="Active Completions", value=len(irc_bot.active_completions), inline=True
    )

    # Check webhook status
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
    bot.run(DISCORD_TOKEN)
