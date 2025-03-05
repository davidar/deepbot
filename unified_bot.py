import discord
from discord.ext import commands
import asyncio
import json
import logging
import re
import time
import os
from typing import Dict, List, Optional, Union, Any

from api_client import OpenAICompatibleClient
import config
from system_prompt import get_system_prompt

# Import echo backend if needed
if os.environ.get("USE_ECHO_BACKEND") == "1":
    from echo_backend import EchoClient

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("deepbot")

class DeepBot(commands.Bot):
    """Discord bot that uses streaming responses from an LLM API."""
    
    def __init__(self):
        """Initialize the bot."""
        # Initialize with mention as command prefix
        intents = discord.Intents.default()
        intents.message_content = True  # Required to read message content
        intents.messages = True
        
        # Use mention as the command prefix
        super().__init__(command_prefix=commands.when_mentioned, intents=intents)
        
        # Initialize the API client (either real or echo)
        if os.environ.get("USE_ECHO_BACKEND") == "1":
            self.api_client = EchoClient(config.API_URL, config.API_KEY)
            logger.info("Using ECHO backend for testing (no LLM connection)")
        else:
            self.api_client = OpenAICompatibleClient(config.API_URL, config.API_KEY)
            logger.info("Using OpenAI-compatible API client")
        
        # Store conversation history for each channel
        self.conversation_history: Dict[int, List[Dict[str, str]]] = {}
        
        # Discord message length limit with safety margin
        self.DISCORD_MESSAGE_LIMIT = 1950  # Discord limit is 2000, leaving 50 chars as safety margin
        
        # Register commands
        self.add_commands()
        
        # Add custom command error handler
        self.add_error_handler()
    
    def add_error_handler(self):
        """Add custom error handler for commands."""
        
        @self.event
        async def on_command_error(ctx, error):
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
    
    def _get_initial_messages(self, channel):
        """
        Get the initial messages for a new conversation, including:
        - System message customized for the channel
        - Example conversation from config
        
        Returns:
            List of message dictionaries
        """
        # Start with the system message
        system_message = {
            "role": "system",
            "content": get_system_prompt(
                server_name=channel.guild.name if isinstance(channel, discord.TextChannel) else "our DM chat",
                currentDateTime=config.get_current_datetime()
            )
        }
        
        # Create a list with system message and example conversation
        initial_messages = [system_message]
        initial_messages.extend(config.EXAMPLE_CONVERSATION)
        
        return initial_messages
    
    def add_commands(self):
        """Add bot commands."""
        
        @self.command(name="reset")
        async def reset_history(ctx):
            """Reset the conversation history for the current channel."""
            channel_id = ctx.channel.id
            if channel_id in self.conversation_history:
                self.conversation_history[channel_id] = []
                await ctx.send("Conversation history has been reset.")
            else:
                await ctx.send("No conversation history to reset.")
        
        @self.command(name="refresh")
        async def refresh_history(ctx):
            """Refresh the conversation history by fetching recent messages from the channel."""
            if not isinstance(ctx.channel, discord.TextChannel):
                await ctx.send("This command can only be used in text channels, not in DMs.")
                return
                
            await ctx.send("Refreshing conversation history from channel messages...")
            
            try:
                # Clear existing history
                channel_id = ctx.channel.id
                self.conversation_history[channel_id] = []
                
                # Re-initialize with fresh data
                await self._initialize_channel_history(ctx.channel)
                
                history_count = len(self.conversation_history[ctx.channel.id])
                await ctx.send(f"Conversation history refreshed! Now tracking {history_count-1} messages (plus system message).")
            except Exception as e:
                logger.error(f"Error refreshing history: {str(e)}")
                await ctx.send(f"Error refreshing history: {str(e)}")
        
        @self.command(name="debug")
        async def debug_history(ctx):
            """Debug command to show detailed information about the history initialization process."""
            if not isinstance(ctx.channel, discord.TextChannel):
                await ctx.send("This command can only be used in text channels, not in DMs.")
                return
                
            await ctx.send("Fetching channel messages for debugging...")
            
            try:
                channel = ctx.channel
                message_limit = config.HISTORY_FETCH_LIMIT
                
                # Count messages in the channel
                message_count = 0
                async for _ in channel.history(limit=message_limit):
                    message_count += 1
                
                # Count bot messages
                bot_message_count = 0
                async for message in channel.history(limit=message_limit):
                    if message.author == self.user:
                        bot_message_count += 1
                
                # Count directed messages
                directed_count = 0
                async for message in channel.history(limit=message_limit):
                    if message.author != self.user and self.user.mentioned_in(message):
                        directed_count += 1
                
                # Current history info
                channel_id = ctx.channel.id
                current_history_count = 0
                if channel_id in self.conversation_history:
                    current_history_count = len(self.conversation_history[channel_id])
                
                debug_info = (
                    f"**History Debug Information**\n\n"
                    f"Channel: `{channel.name}`\n"
                    f"Messages in channel (limit {message_limit}): `{message_count}`\n"
                    f"Bot messages: `{bot_message_count}`\n"
                    f"Messages directed at bot: `{directed_count}`\n"
                    f"Current history count: `{current_history_count}`\n"
                    f"History fetch limit: `{config.HISTORY_FETCH_LIMIT}`\n"
                    f"Max history: `{config.MAX_HISTORY}`\n"
                )
                
                await ctx.send(debug_info)
            except Exception as e:
                logger.error(f"Error in debug command: {str(e)}")
                await ctx.send(f"Error in debug command: {str(e)}")
        
        @self.command(name="raw")
        async def raw_history(ctx):
            """Display the raw conversation history for debugging."""
            channel_id = ctx.channel.id
            if channel_id not in self.conversation_history or not self.conversation_history[channel_id]:
                await ctx.send("No conversation history found for this channel.")
                return
            
            history = self.conversation_history[channel_id]
            raw_text = "**Raw Conversation History**\n\n```json\n"
            raw_text += json.dumps(history, indent=2)
            raw_text += "\n```"
            
            # Split into chunks if too long
            if len(raw_text) > 1900:  # Discord message limit is 2000
                chunks = [raw_text[i:i+1900] for i in range(0, len(raw_text), 1900)]
                for chunk in chunks:
                    await ctx.send(chunk)
            else:
                await ctx.send(raw_text)
        
        @self.command(name="commands")
        async def commands_help(ctx):
            """Display help information."""
            help_text = (
                f"**DeepBot Commands**\n\n"
                f"Mention me with `reset` - Reset the conversation history\n"
                f"Mention me with `refresh` - Refresh the conversation history from channel\n"
                f"Mention me with `commands` - Display this help message\n"
                f"Mention me with `info` - Display bot configuration\n"
                f"Mention me with `history` - Show current conversation history\n"
                f"Mention me with `raw` - Show raw conversation history for debugging\n"
                f"Mention me with `debug` - Show debug information about history\n"
                f"Mention me with `wipe` - Temporarily wipe the bot's memory while keeping the system message\n"
                f"Mention me with `prompt` - Show current system prompt\n"
                f"Mention me with `prompt add <line>` - Add a line to the system prompt\n"
                f"Mention me with `prompt remove <line>` - Remove a line from the system prompt\n\n"
                "To chat with me, mention me in a message or send me a direct message."
            )
            await ctx.send(help_text)
        
        @self.command(name="info")
        async def info_command(ctx):
            """Display information about the bot configuration."""
            backend_type = "Echo Backend (Test Mode)" if os.environ.get("USE_ECHO_BACKEND") == "1" else "OpenAI-compatible API"
            
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
        async def history_command(ctx):
            """Display the current conversation history for the channel."""
            channel_id = ctx.channel.id
            if channel_id not in self.conversation_history or not self.conversation_history[channel_id]:
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
                    # Format user messages - simpler now without "Message from" prefix
                    formatted_content = f"**User**: {content}"
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
        async def wipe_memory(ctx):
            """Temporarily wipe the bot's memory while keeping the system message."""
            channel_id = ctx.channel.id
            if channel_id in self.conversation_history:
                # Keep only the initial messages (system message and examples)
                self.conversation_history[channel_id] = self._get_initial_messages(ctx.channel)
                await ctx.send("🧹 Memory wiped! I'm starting fresh, but I'll keep my personality intact!")
            else:
                await ctx.send("No conversation history to wipe.")
        
        @self.command(name="prompt")
        async def prompt_command(ctx, action: str = None, *, line: str = None):
            """Manage the system prompt."""
            from system_prompt import get_system_prompt, add_line, remove_line, load_system_prompt
            
            if not action:
                # Display current prompt
                current_prompt = get_system_prompt()
                await ctx.send(f"**Current System Prompt:**\n```\n{current_prompt}\n```")
                return
            
            if action.lower() == "add" and line:
                # Add a new line
                lines = add_line(line)
                await ctx.send(f"Added line to system prompt: `{line}`\n\nUpdated prompt now has {len(lines)} lines.")
                # Update all channels with new system prompt
                for channel_id in self.conversation_history:
                    if self.conversation_history[channel_id]:
                        self.conversation_history[channel_id][0]["content"] = get_system_prompt()
            
            elif action.lower() == "remove" and line:
                # Remove a line
                original_lines = load_system_prompt()
                if line not in original_lines:
                    await ctx.send(f"Line not found in system prompt: `{line}`")
                    return
                
                lines = remove_line(line)
                await ctx.send(f"Removed line from system prompt: `{line}`\n\nUpdated prompt now has {len(lines)} lines.")
                # Update all channels with new system prompt
                for channel_id in self.conversation_history:
                    if self.conversation_history[channel_id]:
                        self.conversation_history[channel_id][0]["content"] = get_system_prompt()
            
            else:
                await ctx.send("Invalid command. Use `prompt` to view, `prompt add <line>` to add, or `prompt remove <line>` to remove.")
    
    async def on_ready(self):
        """Event triggered when the bot is ready."""
        logger.info(f"Logged in as {self.user.name} ({self.user.id})")
        logger.info(f"Using API URL: {config.API_URL}")
        logger.info(f"Using mode: {config.MODE}")
        if config.CHARACTER:
            logger.info(f"Using character: {config.CHARACTER}")
        
        # Set bot activity
        backend_type = "Echo" if os.environ.get("USE_ECHO_BACKEND") == "1" else config.MODE
        await self.change_presence(activity=discord.Game(name=f"Streaming with {backend_type} mode"))
        
        # Initialize history for all channels the bot can see
        logger.info(f"Starting to initialize history for all channels (fetch limit: {config.HISTORY_FETCH_LIMIT})")
        channel_count = 0
        message_count = 0
        
        for guild in self.guilds:
            logger.info(f"Initializing history for guild: {guild.name}")
            break
            for channel in guild.text_channels:
                if channel.permissions_for(guild.me).read_messages:
                    channel_count += 1
                    
                    # Get the current message count before initialization
                    channel_id = channel.id
                    current_count = 0
                    if channel_id in self.conversation_history:
                        current_count = len(self.conversation_history[channel_id])
                    
                    # Initialize the channel history
                    await self._initialize_channel_history(channel)
                    
                    # Get the new message count after initialization
                    new_count = 0
                    if channel_id in self.conversation_history:
                        new_count = len(self.conversation_history[channel_id])
                    
                    # Calculate how many messages were added
                    added_count = new_count - current_count
                    message_count += added_count
                    
                    logger.info(f"Initialized history for channel {channel.name}: {added_count} messages added")
        
        logger.info(f"Bot is ready! Initialized history for {channel_count} channels with {message_count} total messages")
    
    def _format_streaming_text(self, text):
        """
        Format streaming text for display in Discord.
        
        Content inside <think></think> tags is displayed with -# prefix,
        but the tags themselves are not shown.
        Regular content is displayed normally.
        """
        # Split the text into sections based on think tags
        sections = re.split(r'(<think>|</think>)', text)
        formatted_lines = []
        
        # Track if we're inside a think block
        in_think_block = False
        
        # Process each section
        for section in sections:
            if section == "<think>":
                # Start of think block - don't add this to output
                in_think_block = True
            elif section == "</think>":
                # End of think block - don't add this to output
                in_think_block = False
            elif section:  # Skip empty sections
                # Process the lines in this section
                for line in section.split('\n'):
                    if line.strip():  # Skip empty lines
                        if in_think_block:
                            # Inside think block, use small font
                            formatted_lines.append("-# " + line)
                        else:
                            # Regular content, use normal font
                            formatted_lines.append(line)
        
        # Join the formatted lines back into a single string
        return '\n'.join(formatted_lines)
    
    def _truncate_formatted_text(self, formatted_text, max_length=1950):
        """
        Intelligently truncate formatted text to fit within Discord's character limit.
        
        Prioritizes preserving regular content over thinking sections (small font lines).
        
        Args:
            formatted_text: The text with -# prefixes for thinking sections
            max_length: Maximum allowed length (default: 1950 to leave some buffer)
            
        Returns:
            Truncated text that fits within the character limit
        """
        # If text is already within limit, return it as is
        if len(formatted_text) <= max_length:
            return formatted_text
            
        # Split into lines
        lines = formatted_text.split('\n')
        
        # Separate thinking lines from regular lines
        thinking_lines = [line for line in lines if line.startswith("-# ")]
        regular_lines = [line for line in lines if not line.startswith("-# ")]
        
        # Calculate total length of regular content
        regular_content_length = sum(len(line) + 1 for line in regular_lines)  # +1 for newline
        
        # If regular content alone exceeds limit, we need to truncate it
        if regular_content_length > max_length:
            # Keep as much regular content as possible
            result = []
            current_length = 0
            
            for line in regular_lines:
                line_length = len(line) + 1  # +1 for newline
                if current_length + line_length <= max_length - 3:  # -3 for "..."
                    result.append(line)
                    current_length += line_length
                else:
                    # Truncate the last line if needed
                    remaining = max_length - current_length - 3
                    if remaining > 0:
                        result.append(line[:remaining] + "...")
                    else:
                        result[-1] = result[-1][:len(result[-1])-3] + "..."
                    break
                    
            return '\n'.join(result)
        
        # If we get here, we can keep all regular content and need to trim thinking sections
        
        # Calculate how much space we have for thinking sections
        available_space = max_length - regular_content_length
        
        # If we have very little space, just return regular content
        if available_space < 50:  # Arbitrary threshold
            return '\n'.join(regular_lines)
        
        # We need to select which thinking lines to keep
        # Strategy: Keep the first few and last few thinking lines
        
        # Calculate how many thinking lines we can keep
        thinking_lines_total_length = sum(len(line) + 1 for line in thinking_lines)
        
        if thinking_lines_total_length <= available_space:
            # We can keep all thinking lines
            # Reconstruct the original order
            result = []
            thinking_index = 0
            
            for line in lines:
                if line.startswith("-# "):
                    result.append(thinking_lines[thinking_index])
                    thinking_index += 1
                else:
                    result.append(line)
                    
            return '\n'.join(result)
        
        # We need to truncate thinking sections
        # Keep first and last thinking blocks intact if possible
        
        # Group thinking lines into blocks (consecutive thinking lines)
        thinking_blocks = []
        current_block = []
        
        for i, line in enumerate(lines):
            if line.startswith("-# "):
                current_block.append(line)
            elif current_block:
                thinking_blocks.append(current_block)
                current_block = []
                
        # Add the last block if it exists
        if current_block:
            thinking_blocks.append(current_block)
        
        # If we have multiple thinking blocks, try to keep first and last
        if len(thinking_blocks) > 1:
            first_block = thinking_blocks[0]
            last_block = thinking_blocks[-1]
            
            first_block_length = sum(len(line) + 1 for line in first_block)
            last_block_length = sum(len(line) + 1 for line in last_block)
            
            # If we can keep both first and last blocks
            if first_block_length + last_block_length <= available_space:
                # Keep first and last blocks, discard middle blocks
                kept_thinking_lines = first_block + last_block
                
                # Reconstruct with regular lines and kept thinking lines
                result = []
                kept_thinking_index = 0
                
                for line in lines:
                    if line.startswith("-# "):
                        if kept_thinking_index < len(kept_thinking_lines):
                            result.append(kept_thinking_lines[kept_thinking_index])
                            kept_thinking_index += 1
                        # Skip thinking lines we're not keeping
                    else:
                        result.append(line)
                
                return '\n'.join(result)
            
            # If we can't keep both blocks intact, prioritize the first block
            if first_block_length <= available_space:
                # Keep only the first thinking block
                result = []
                in_first_block = True
                
                for i, line in enumerate(lines):
                    if line.startswith("-# "):
                        if in_first_block and len(result) < len(first_block):
                            result.append(line)
                        # Skip other thinking lines
                    else:
                        result.append(line)
                        # Once we hit a regular line after the first thinking block,
                        # we're no longer in the first block
                        if in_first_block and any(line == block_line for block_line in first_block):
                            in_first_block = False
                
                return '\n'.join(result)
        
        # If we get here, we need a simpler approach: just keep as many thinking lines as will fit
        # Prioritize keeping the first few thinking lines
        
        result = []
        thinking_length = 0
        
        for line in lines:
            if line.startswith("-# "):
                line_length = len(line) + 1  # +1 for newline
                if thinking_length + line_length <= available_space:
                    result.append(line)
                    thinking_length += line_length
                # Skip thinking lines that won't fit
            else:
                result.append(line)
        
        return '\n'.join(result)
    
    def _convert_small_font_to_think_tags(self, content):
        """
        Convert small font lines (-# prefix) to <think></think> tags for the LLM.
        
        Lines with -# prefix are wrapped in <think></think> tags,
        while regular lines are preserved as is.
        """
        # Check if there are any lines with the small font prefix
        if not any(line.startswith("-# ") for line in content.split("\n")):
            return content
            
        # Split the content into lines
        lines = content.split("\n")
        processed_lines = []
        
        # Track consecutive small font lines
        small_font_lines = []
        
        # Process each line
        for line in lines:
            if line.startswith("-# "):
                # This is a small font line - add to the current group
                small_font_lines.append(line[3:])  # Remove the -# prefix
            else:
                # This is a regular line
                
                # If we have accumulated small font lines, wrap them in think tags
                if small_font_lines:
                    processed_lines.append("<think>")
                    processed_lines.extend(small_font_lines)
                    processed_lines.append("</think>")
                    small_font_lines = []
                
                # Add the regular line
                processed_lines.append(line)
        
        # If we have any remaining small font lines at the end, wrap them too
        if small_font_lines:
            processed_lines.append("<think>")
            processed_lines.extend(small_font_lines)
            processed_lines.append("</think>")
        
        # Join the processed lines back into a single string
        return "\n".join(processed_lines)
    
    async def _initialize_channel_history(self, channel):
        """Initialize conversation history for a channel by fetching recent messages."""
        channel_id = channel.id
        
        # Skip if history already exists for this channel
        if channel_id in self.conversation_history and self.conversation_history[channel_id]:
            return
        
        # Initialize with system message and example interactions
        self.conversation_history[channel_id] = self._get_initial_messages(channel)
        
        try:
            # Fetch recent messages from the channel
            message_limit = config.HISTORY_FETCH_LIMIT
            messages = []
            
            logger.info(f"Fetching up to {message_limit} messages from channel {channel.name}")
            
            # Use the Discord API to fetch recent messages
            async for message in channel.history(limit=message_limit):
                # Skip messages from the bot itself to avoid duplicates
                # We'll handle bot responses separately
                if message.author == self.user:
                    continue
                
                # Skip empty messages
                content = message.content.strip()
                if not content:
                    continue
                
                # Clean up the content by replacing Discord mentions with usernames
                content = self._clean_message_content(message)
                
                # Check if message was directed at the bot
                is_directed_at_bot = self.user.mentioned_in(message)
                
                # Add to our temporary list - simplified format without "Message from" prefix
                messages.append({
                    "role": "user",
                    "content": content,
                    "timestamp": message.created_at.timestamp(),
                    "id": message.id,
                    "is_directed": is_directed_at_bot
                })
            
            # Now look for bot responses to all messages
            bot_responses = []
            async for response in channel.history(limit=message_limit):
                if response.author == self.user and response.reference and response.reference.message_id:
                    # Process the response content to convert small font lines back to <think> tags
                    content = self._convert_small_font_to_think_tags(response.content)
                    
                    bot_responses.append({
                        "role": "assistant",
                        "content": content,
                        "timestamp": response.created_at.timestamp(),
                        "reply_to": response.reference.message_id
                    })
            
            # Match bot responses to user messages
            matched_responses = []
            for response in bot_responses:
                # Find the user message this response is replying to
                for message in messages:
                    if message["id"] == response["reply_to"]:
                        # Add the response right after the user message
                        matched_responses.append(response)
                        break
            
            # Sort all messages by timestamp
            all_messages = messages + matched_responses
            all_messages.sort(key=lambda m: m["timestamp"])
            
            # Add messages to history (without extra metadata)
            for message in all_messages:
                # Create a clean copy without the extra fields we added
                message_copy = {
                    "role": message["role"],
                    "content": message["content"]
                }
                self.conversation_history[channel_id].append(message_copy)
            
            # Trim if needed
            if len(self.conversation_history[channel_id]) > config.MAX_HISTORY + 1:  # +1 for system message
                self.conversation_history[channel_id] = [self.conversation_history[channel_id][0]] + self.conversation_history[channel_id][-(config.MAX_HISTORY):]
            
            logger.info(f"Initialized history for channel {channel.name} with {len(messages)} messages")
            
        except Exception as e:
            logger.error(f"Error initializing history for channel {channel.name}: {str(e)}")
            # Keep the initial messages at least
            self.conversation_history[channel_id] = self._get_initial_messages(channel)
    
    async def on_message(self, message):
        """Event triggered when a message is received."""
        # Ignore messages from the bot itself
        if message.author == self.user:
            return
        
        # Get or initialize conversation history for this channel
        channel_id = message.channel.id
        if channel_id not in self.conversation_history:
            # # For new channels, initialize history by fetching recent messages
            # if isinstance(message.channel, discord.TextChannel):
            #     await self._initialize_channel_history(message.channel)
            # else:
            #     # For DMs or other channel types, just add initial messages
                self.conversation_history[channel_id] = self._get_initial_messages(message.channel)
        
        # Check if this message is directed at the bot
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.user.mentioned_in(message)
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
        username = message.author.display_name
        content = message.content.strip()
        
        # Skip empty messages
        if not content:
            return
            
        # Clean up the content by replacing Discord mentions with usernames
        content = self._clean_message_content(message)
        
        # Add user message to history - simplified format without "Message from" prefix
        self.conversation_history[channel_id].append({
            "role": "user",
            "content": content
        })
        
        # Trim history if it exceeds the maximum length
        if len(self.conversation_history[channel_id]) > config.MAX_HISTORY:
            # Keep the system message if it exists
            if self.conversation_history[channel_id][0]["role"] == "system":
                self.conversation_history[channel_id] = [self.conversation_history[channel_id][0]] + self.conversation_history[channel_id][-(config.MAX_HISTORY-1):]
            else:
                self.conversation_history[channel_id] = self.conversation_history[channel_id][-config.MAX_HISTORY:]
        
        # Only respond to messages that mention the bot or are direct messages
        if not is_directed_at_bot:
            return
        
        # Remove the bot mention from the message content for processing
        clean_content = re.sub(f'<@!?{self.user.id}>', '', content).strip()
        
        # If the message is empty after removing the mention, don't process it
        if not clean_content:
            return
        
        try:
            await self._handle_streaming_response(message, channel_id, clean_content)
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            await message.reply(f"Sorry, I encountered an error: {str(e)}")
    
    def _clean_message_content(self, message):
        """
        Clean up message content by replacing Discord mentions with usernames.
        
        Args:
            message: The Discord message object
            
        Returns:
            Cleaned message content
        """
        content = message.content.strip()
        
        # Replace user mentions with usernames
        for user_id in message.mentions:
            mention_pattern = f'<@!?{user_id.id}>'
            username = user_id.display_name
            content = re.sub(mention_pattern, f'@{username}', content)
        
        # Replace channel mentions
        for channel_id in message.channel_mentions:
            channel_pattern = f'<#{channel_id.id}>'
            channel_name = channel_id.name
            content = re.sub(channel_pattern, f'#{channel_name}', content)
        
        # Replace role mentions
        if hasattr(message, 'role_mentions'):
            for role in message.role_mentions:
                role_pattern = f'<@&{role.id}>'
                role_name = role.name
                content = re.sub(role_pattern, f'@{role_name}', content)
        
        return content
    
    async def _handle_streaming_response(self, message, channel_id, clean_content):
        """Handle a streaming response."""
        # Start typing indicator
        async with message.channel.typing():
            try:
                logger.info(f"Starting streaming response for channel {channel_id}")
                logger.info(f"History length: {len(self.conversation_history[channel_id])} messages")
                logger.info(f"Max tokens setting: {config.MAX_TOKENS}")
                
                # Log the full request payload
                request_payload = {
                    "messages": self.conversation_history[channel_id],
                    "model": config.MODEL_NAME,
                    "temperature": config.TEMPERATURE,
                    "max_tokens": config.MAX_TOKENS if config.MAX_TOKENS != -1 else None,
                    "top_p": config.TOP_P,
                    "presence_penalty": config.PRESENCE_PENALTY,
                    "frequency_penalty": config.FREQUENCY_PENALTY,
                    "seed": config.SEED if config.SEED != -1 else None,
                    "stream": True
                }
                logger.info(f"API Request payload: {json.dumps(request_payload, indent=2)}")
                
                sse_client = self.api_client.chat_completion(**request_payload)
                
                # Variables to track streaming state
                accumulated_text = ""  # Current chunk being built
                full_response = ""    # Complete response for history
                chunk_count = 0
                is_final = False
                
                # Process streaming response
                for event in sse_client.events():
                    try:
                        chunk_count += 1
                        
                        # Handle [DONE] message specially
                        if event.data == "[DONE]":
                            is_final = True
                            continue
                            
                        data = json.loads(event.data)
                        logger.debug(f"Received chunk {chunk_count}: {data}")
                        
                        if "choices" in data and len(data["choices"]) > 0:
                            # Check if this is the final chunk
                            if data["choices"][0].get("finish_reason") is not None:
                                is_final = True
                            
                            if "delta" in data["choices"][0] and "content" in data["choices"][0]["delta"]:
                                content = data["choices"][0]["delta"]["content"]
                                accumulated_text += content
                                full_response += content
                                
                                # Log every 100 chunks
                                if chunk_count % 100 == 0:
                                    logger.info(f"Processed {chunk_count} chunks, current length: {len(full_response)}")
                                
                                # Check for newlines in accumulated text
                                while '\n' in accumulated_text:
                                    # Split at first newline
                                    to_send, accumulated_text = accumulated_text.split('\n', 1)
                                    
                                    # Format the text (for think tags)
                                    formatted_text = self._format_streaming_text(to_send)
                                    
                                    # Only send if we have non-empty content
                                    if formatted_text.strip():
                                        try:
                                            await message.channel.send(formatted_text)
                                            # Start new typing indicator if not the final chunk
                                            if not is_final:
                                                async with message.channel.typing():
                                                    pass  # Just to restart the typing indicator
                                        except discord.errors.HTTPException as e:
                                            logger.warning(f"Failed to send message chunk: {str(e)}")
                                    
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to decode chunk: {e}: {repr(event.data)}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")
                        continue
                
                logger.info(f"Stream completed. Total chunks: {chunk_count}")
                logger.info(f"Final response length: {len(full_response)} characters")
                
                # Send any remaining accumulated text
                if accumulated_text and accumulated_text.strip():
                    formatted_text = self._format_streaming_text(accumulated_text)
                    if formatted_text.strip():
                        try:
                            await message.channel.send(formatted_text)
                        except discord.errors.HTTPException as e:
                            logger.error(f"Failed to send final chunk: {str(e)}")
                
                # Add the complete response to conversation history WITH the think tags
                self.conversation_history[channel_id].append({
                    "role": "assistant",
                    "content": full_response
                })
                    
            except Exception as e:
                error_message = f"Error in streaming response: {str(e)}"
                logger.error(error_message)
                logger.error(f"Full error details: {repr(e)}")
                await message.channel.send(error_message)

def run_bot():
    """Run the bot."""
    bot = DeepBot()
    bot.run(config.DISCORD_TOKEN)

if __name__ == "__main__":
    run_bot()
