"""Core Discord bot implementation."""

import logging

import discord
import ollama
from discord.ext import commands, tasks

import config
from commands import setup_commands
from context_builder import ContextBuilder
from llm_streaming import LLMResponseHandler
from message_history import MessageHistoryManager
from message_indexer import MessageIndexer
from message_store import MessageStore
from reactions import ReactionManager
from reminder_manager import reminder_manager
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

        # Initialize message indexer for search functionality
        indexer = MessageIndexer(
            storage_path=config.SEARCH_INDEX_PATH,
            model_name=config.EMBEDDING_MODEL_NAME,
            base_url=config.API_URL,
        )

        # Initialize message store with search functionality
        self.message_store = MessageStore(
            data_dir=config.MESSAGE_STORE_DIR,
            message_indexer=indexer,
        )

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

        # Set the LLM handler for the reminder manager
        reminder_manager.set_llm_handler(self.llm_handler)

        # Start background tasks
        self.check_reminders.start()
        self.sync_messages.start()
        logger.info("Started background tasks")

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
            self.message_store,
        )

        # Update context builder with command names after commands are set up
        self.context_builder.set_bot(self)

        logger.info("Bot components initialized")

    @tasks.loop(minutes=1)
    async def check_reminders(self) -> None:
        """Check for due reminders and trigger them."""
        try:
            due_reminders = reminder_manager.get_due_reminders()
            for reminder in due_reminders:
                await reminder_manager.process_due_reminder(self, reminder)
        except Exception as e:
            logger.error(f"Error checking reminders: {str(e)}")

    @tasks.loop(minutes=30)
    async def sync_messages(self) -> None:
        """Periodically sync messages from all tracked channels."""
        try:
            channel_ids = self.message_store.get_channel_ids()
            logger.info(f"Starting periodic sync for {len(channel_ids)} channels")

            for channel_id in channel_ids:
                channel = self.get_channel(int(channel_id))
                if isinstance(channel, discord.TextChannel):
                    try:
                        await self.message_store.sync_channel(channel)
                        logger.info(f"Synced messages for channel #{channel.name}")
                    except Exception as e:
                        logger.error(f"Error syncing channel #{channel.name}: {str(e)}")
                        continue
        except Exception as e:
            logger.error(f"Error in periodic message sync: {str(e)}")

    @check_reminders.before_loop
    @sync_messages.before_loop
    async def before_background_tasks(self) -> None:
        """Wait until the bot is ready before starting background tasks."""
        await self.wait_until_ready()
        logger.info("Background tasks are ready to start")

    async def on_ready(self) -> None:
        """Event triggered when the bot is ready."""
        if not self.user:
            logger.error("Bot user is None!")
            return

        logger.info(f"Logged in as {self.user.name} ({self.user.id})")
        logger.info(f"Using API URL: {config.API_URL}")

        # Log intents for debugging
        logger.info(f"Bot intents: {self.intents}")

        await self.change_presence(activity=discord.Game(name="with myself"))

        # Initial sync of all tracked channels
        channel_ids = self.message_store.get_channel_ids()
        logger.info(f"Starting initial sync for {len(channel_ids)} channels")
        for channel_id in channel_ids:
            channel = self.get_channel(int(channel_id))
            if isinstance(channel, discord.TextChannel):
                try:
                    await self.message_store.sync_channel(channel)
                    logger.info(f"Initial sync completed for channel #{channel.name}")
                except Exception as e:
                    logger.error(
                        f"Error in initial sync for channel #{channel.name}: {str(e)}"
                    )
                    continue

        for guild in self.guilds:
            logger.info(f"Connected to {guild.name}")

        logger.info("Bot is ready!")

    async def close(self) -> None:
        """Close the bot and clean up resources."""
        # Cancel background tasks
        if self.check_reminders.is_running():
            self.check_reminders.cancel()
            logger.info("Cancelled reminder check task")
        if self.sync_messages.is_running():
            self.sync_messages.cancel()
            logger.info("Cancelled message sync task")

        # Call the parent close method
        await super().close()

    async def on_message(self, message: discord.Message) -> None:
        """Event triggered when a message is received."""
        channel_id = message.channel.id

        # Add message to store if it's in a tracked channel
        if str(channel_id) in self.message_store.get_channel_ids():
            await self.message_store.add_message(message)

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
