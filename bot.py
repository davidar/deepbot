"""Core Discord bot implementation."""

import asyncio
import logging

import discord
import ollama
from discord.ext import commands

import config
from commands import setup_commands
from conversation import ConversationManager
from reactions import ReactionManager
from utils import clean_message_content

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()],
)
logger = logging.getLogger("deepbot")


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

    async def setup_hook(self) -> None:
        """Set up the bot's components after login."""
        if not self.user:
            raise RuntimeError("Bot user not initialized")

        # Initialize managers
        self.conversation_manager = ConversationManager(self.api_client, self.user)
        self.reaction_manager = ReactionManager()

        # Set up commands
        setup_commands(self, self.conversation_manager, self.reaction_manager)

        logger.info("Bot components initialized")

    async def on_ready(self) -> None:
        """Event triggered when the bot is ready."""
        if not self.user:
            logger.error("Bot user is None!")
            return

        logger.info(f"Logged in as {self.user.name} ({self.user.id})")
        logger.info(f"Using API URL: {config.API_URL}")

        # Log intents for debugging
        logger.info(f"Bot intents: {self.intents}")

        await self.change_presence(activity=discord.Game(name=f"with myself"))

        for guild in self.guilds:
            logger.info(f"Connected to {guild.name}")

        logger.info(f"Bot is ready!")

    async def on_message(self, message: discord.Message) -> None:
        """Event triggered when a message is received."""
        # Ignore messages from the bot itself
        if message.author == self.user:
            return

        # Get or initialize conversation history for this channel
        channel_id = message.channel.id
        if channel_id not in self.conversation_manager.conversation_history:
            # For new channels, initialize history by fetching recent messages
            if isinstance(message.channel, discord.TextChannel):
                await self.conversation_manager.initialize_channel_history(
                    message.channel
                )
            else:
                # For DMs or other channel types, just add initial messages
                self.conversation_manager.conversation_history[channel_id] = (
                    self.conversation_manager.get_initial_messages(message.channel)
                )

        # Check if this message is directed at the bot
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = bool(self.user and self.user.mentioned_in(message))
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
        content = clean_message_content(message)

        # Check if this message is already the last message in history
        message_content = f"{message.author.display_name}: {content}"
        history = self.conversation_manager.conversation_history[channel_id]
        is_duplicate = (
            history
            and history[-1]["role"] == "user"
            and history[-1]["content"] == message_content
        )

        # Add user message to history if it's not a duplicate
        if not is_duplicate:
            self.conversation_manager.conversation_history[channel_id].append(
                {"role": "user", "content": message_content}
            )

        # Trim history if it exceeds the maximum length
        max_history = int(config.get_option("max_history", 10))
        if (
            len(self.conversation_manager.conversation_history[channel_id])
            > max_history
        ):
            # Keep the system message if it exists
            if (
                self.conversation_manager.conversation_history[channel_id][0]["role"]
                == "system"
            ):
                self.conversation_manager.conversation_history[channel_id] = [
                    self.conversation_manager.conversation_history[channel_id][0]
                ] + self.conversation_manager.conversation_history[channel_id][
                    -(max_history - 1) :
                ]
            else:
                self.conversation_manager.conversation_history[channel_id] = (
                    self.conversation_manager.conversation_history[channel_id][
                        -max_history:
                    ]
                )

        # Only respond to messages that mention the bot or are direct messages
        if not is_directed_at_bot:
            return

        try:
            # Add message to response queue
            if channel_id not in self.conversation_manager.response_queues:
                self.conversation_manager.response_queues[channel_id] = asyncio.Queue()
            await self.conversation_manager.response_queues[channel_id].put(message)
            # Ensure there's a queue processor running
            await self.conversation_manager.ensure_queue_processor(channel_id)
            # Send acknowledgment
            await message.add_reaction("💭")
            logger.info(f"Added message to queue for channel {channel_id}")
        except Exception as e:
            logger.error(f"Error queueing response: {str(e)}")
            await message.reply(f"-# Sorry, I encountered an error: {str(e)}")

    async def on_reaction_add(
        self, reaction: discord.Reaction, user: discord.User
    ) -> None:
        """Handle reaction add events."""
        # Ignore reactions from the bot itself
        if user == self.user:
            return

        # Only track reactions on messages from the bot
        if reaction.message.author != self.user:
            return

        # Delegate to reaction manager
        self.reaction_manager.handle_reaction_add(reaction, user)


def run_bot() -> None:
    """Run the bot."""
    bot = DeepBot()
    token = config.DISCORD_TOKEN
    if token is None:
        raise ValueError("DISCORD_TOKEN is not set in config")
    bot.run(token)


if __name__ == "__main__":
    run_bot()
