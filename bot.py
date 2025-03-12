"""Core Discord bot implementation."""

import logging

import discord
import ollama
from discord.ext import commands

import config
from commands import setup_commands
from context_builder import ContextBuilder
from llm_streaming import LLMResponseHandler
from message_history import MessageHistoryManager
from reactions import ReactionManager
from tools import tool_registry
from user_management import UserManager
from utils import get_channel_name

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
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
        self.message_history = MessageHistoryManager()
        self.reaction_manager = ReactionManager()
        self.context_builder = ContextBuilder(self.reaction_manager)
        self.llm_handler = LLMResponseHandler(self.api_client, self.user)
        self.user_manager = UserManager()

        # Log available tools
        logger.info(
            f"Available tools: {[tool['function']['name'] for tool in tool_registry.get_tools()]}"
        )

        # Set up commands
        setup_commands(
            self,
            self.message_history,
            self.context_builder,
            self.llm_handler,
            self.reaction_manager,
            self.user_manager,
        )

        # Update context builder with command names after commands are set up
        self.context_builder.set_bot(self)

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
        channel_id = message.channel.id

        # Get or initialize message history for this channel
        if await self.message_history.initialize_channel(message.channel):
            logger.info(
                f"Initialized history for channel {get_channel_name(message.channel)}"
            )
        elif message.content.strip():
            self.message_history.add_message(message)

        # Ignore messages from the bot itself (but after adding to history)
        if message.author == self.user:
            return

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

        # Only respond to messages that mention the bot or are direct messages
        if not is_directed_at_bot:
            return

        # Check user restrictions for non-command messages directed at the bot
        can_message, reason = self.user_manager.can_message(message.author.id)
        if not can_message:
            # Send status message
            await message.reply(f"-# {reason}")
            return

        try:
            # Add message to response queue and start processing
            self.llm_handler.add_to_queue(channel_id, message)
            await self.llm_handler.ensure_queue_processor(
                channel_id,
                self.context_builder,
                self.message_history.get_messages(channel_id),
            )
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
