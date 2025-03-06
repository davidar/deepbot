# DeepBot - Discord Bot with Local LLM Integration

A Discord bot that uses a local LLM API to generate streaming responses based on conversation history.

## Features

- Integrates with local LLM API (compatible with OpenAI API format)
- Maintains conversation history for context-aware responses
- Responds to messages in Discord channels
- Customizable bot behavior through environment variables
- Shows responses as they're being generated (streaming mode)
- Includes an echo backend for testing without an LLM
- Automatically sets up a virtual environment
- Commands are triggered by mentioning the bot (no prefix needed)
- View conversation history with a simple command
- Tracks all channel messages for better context awareness
- Preserves context across restarts by fetching channel history
- Supports both text channels and direct messages
- Includes thread support for better conversation organization

## Message History

The bot maintains a conversation history for each channel it's in, which includes:

1. **All messages in the channel** - The bot tracks all messages in the channel, not just those directed at it. This provides better context for responses.

2. **System messages** - Each conversation starts with a system message that explains the bot's purpose and behavior.

3. **User messages** - Messages from users are stored with their usernames and content. Messages directed at the bot are specially marked.

4. **Assistant messages** - The bot's own responses are stored in the history.

The conversation history is used to provide context to the LLM, allowing it to generate more relevant and coherent responses. You can view the current conversation history with the `@BotName history` command.

The maximum number of messages stored in history is controlled by the `MAX_HISTORY` environment variable (default: 10). You may want to increase this value for busier channels to provide more context.

### Persistent Context Across Restarts

When the bot starts up, it automatically fetches recent message history from each channel it has access to. This means that even if you restart the bot, it will still have context from previous conversations. The bot will:

1. Fetch up to `HISTORY_FETCH_LIMIT` recent messages from each channel (default: 50)
2. Include up to `MAX_HISTORY` messages in its conversation history (default: 10)
3. Identify which messages were directed at the bot
4. Include its own responses to those messages
5. Sort all messages by timestamp to maintain the correct conversation flow

This feature ensures continuity in conversations even after bot restarts, without requiring any external storage or database.

You can also manually refresh the conversation history at any time using the `@BotName refresh` command. This will clear the current history and fetch recent messages from the channel again.

## Setup

1. Run the setup script to create a virtual environment and install dependencies:
   ```
   python setup.py
   ```

2. Create a `.env` file with the following variables:
   ```
   DISCORD_TOKEN=your_discord_bot_token
   ```

3. Run the bot:
   ```
   python main.py
   ```

## Usage

- The bot will respond to messages in channels it has access to when mentioned
- The bot will also respond to direct messages
- Use `@BotName reset` to clear the conversation history
- Use `@BotName refresh` to refresh the conversation history from channel messages
- Use `@BotName info` to see the current bot configuration
- Use `@BotName history` to view the current conversation history
- Use `@BotName raw` to see the raw conversation history
- Use `@BotName wipe` to clear all conversation history
- Use `@BotName prompt` to view or modify the system prompt
- The bot tracks all messages in the channel, not just those directed at it, for better context

## Echo Backend

The echo backend allows you to test the Discord bot functionality without connecting to the LLM API. This is useful for:

- Testing the bot's Discord integration
- Developing new features without running the LLM
- Debugging conversation history management

To use the echo backend:
```
python main.py --echo
```

## Streaming Mode

The bot shows responses as they're being generated, similar to how ChatGPT shows responses in real-time. This provides a more interactive experience for users. The streaming implementation:

- Accumulates complete lines before sending them
- Handles partial lines gracefully
- Provides smooth, natural-looking responses
- Maintains proper formatting and line breaks

## Configuration

The bot's behavior can be customized by modifying the following variables in `config.py`:

- `API_URL`: URL of the LLM API (default: http://127.0.0.1:1234/v1)
- `MODEL_NAME`: Name of the model to use (default: mistral-small-24b-instruct-2501)
- `MAX_HISTORY`: Maximum number of messages to keep in history (default: 10)
- `HISTORY_FETCH_LIMIT`: Maximum number of messages to fetch from channel history on startup (default: 50)
- `TEMPERATURE`: Sampling temperature (default: 0.7)
- `MAX_TOKENS`: Maximum tokens to generate (default: -1 for unlimited)
- `TOP_P`: Top-p sampling parameter (default: 0.9)
- `PRESENCE_PENALTY`: Presence penalty for generation (default: 0.0)
- `FREQUENCY_PENALTY`: Frequency penalty for generation (default: 0.0)
- `SEED`: Random seed for generation (default: -1 for random)

## Creating a Discord Bot

1. Go to the [Discord Developer Portal](https://discord.com/developers/applications)
2. Click "New Application" and give it a name
3. Go to the "Bot" tab and click "Add Bot"
4. Under "Privileged Gateway Intents", enable "Message Content Intent"
5. Copy the token and add it to your `.env` file
6. Go to the "OAuth2" tab, then "URL Generator"
7. Select the "bot" scope and the following permissions:
   - Read Messages/View Channels
   - Send Messages
   - Read Message History
8. Use the generated URL to invite the bot to your server

## Command Line Arguments

The `main.py` script accepts the following command line arguments:

- `--echo`: Use the echo backend instead of connecting to the LLM API
- `--setup`: Run the setup script before starting the bot
- `--skip-venv-check`: Skip the virtual environment check

## Troubleshooting

### Common Issues

1. **"The command help is already an existing command or alias"**
   - This error occurs because discord.py already has a built-in help command
   - Solution: Use `@BotName info` to see available commands

2. **"Error: DISCORD_TOKEN not set in .env file"**
   - This error occurs when the Discord token is missing or invalid
   - Solution: Make sure you've added your Discord bot token to the `.env` file

3. **"Error generating response: API request failed"**
   - This error occurs when the bot can't connect to the LLM API
   - Solution: Make sure your LLM API is running and accessible at the configured URL
   - Alternative: Use the echo backend for testing (`python main.py --echo`)

4. **"Not running in a virtual environment"**
   - This warning appears if you run the bot outside the virtual environment
   - Solution: Activate the virtual environment before running the bot
   - Alternative: Use `--skip-venv-check` to bypass this check

### Getting Help

If you encounter issues not covered in this troubleshooting section, please:
1. Check the bot.log file for detailed error messages
2. Try running with the echo backend to isolate Discord-related issues
3. Make sure all dependencies are installed correctly
