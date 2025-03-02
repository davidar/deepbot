# DeepBot - Discord Bot with Local LLM Integration

A Discord bot that uses the Oobabooga Text Generation WebUI API to generate streaming responses based on conversation history.

## Features

- Integrates with local Oobabooga Text Generation WebUI API
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

If you're experiencing issues with the history initialization, you can use the `@BotName debug` command to see detailed information about the messages in the channel and the current state of the conversation history.

## Setup

1. Run the setup script to create a virtual environment and install dependencies:
   ```
   python setup.py
   ```

2. Activate the virtual environment:
   - Windows:
     ```
     activate.bat
     ```
   - Unix/Linux/Mac:
     ```
     source activate.sh
     ```

3. Create a `.env` file with the following variables:
   ```
   DISCORD_TOKEN=your_discord_bot_token
   API_URL=http://127.0.0.1:5000
   API_KEY=your_api_key_if_needed
   BOT_PREFIX=!
   ```

4. Run the bot:
   ```
   python run.py
   ```
   
   For testing without an LLM connection:
   ```
   python run.py --echo
   ```

## Usage

- The bot will respond to messages in channels it has access to when mentioned
- The bot will also respond to direct messages
- Use `@BotName commands` to see available commands
- Use `@BotName reset` to clear the conversation history
- Use `@BotName refresh` to refresh the conversation history from channel messages
- Use `@BotName info` to see the current bot configuration
- Use `@BotName history` to view the current conversation history
- Use `@BotName debug` to see detailed information about the channel messages and history
- The bot tracks all messages in the channel, not just those directed at it, for better context

## Echo Backend

The echo backend allows you to test the Discord bot functionality without connecting to the Oobabooga API. This is useful for:

- Testing the bot's Discord integration
- Developing new features without running the LLM
- Debugging conversation history management

To use the echo backend:
```
python run.py --echo
```

## Streaming Mode

The bot shows responses as they're being generated, similar to how ChatGPT shows responses in real-time. This provides a more interactive experience for users.

## Configuration

You can customize the bot's behavior by modifying the following environment variables in the `.env` file:

- `DISCORD_TOKEN`: Your Discord bot token
- `API_URL`: URL of the Oobabooga API (default: http://127.0.0.1:5000)
- `API_KEY`: API key for authentication (if enabled)
- `BOT_PREFIX`: Command prefix for the bot (default: !)
- `MAX_HISTORY`: Maximum number of messages to keep in history (default: 10)
- `HISTORY_FETCH_LIMIT`: Maximum number of messages to fetch from channel history on startup (default: 50)
- `CHARACTER`: Character to use for responses (default: none)
- `MODE`: API mode to use (default: chat)
- `TEMPERATURE`: Sampling temperature (default: 0.7)
- `MAX_TOKENS`: Maximum tokens to generate (default: 200)
- `TOP_P`: Top-p sampling parameter (default: 0.9)

Note: The bot tracks all messages in the channel, not just those directed at it. This provides better context for responses, but you may want to adjust `MAX_HISTORY` based on how busy your channels are.

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

The `run.py` script accepts the following command line arguments:

- `--echo`: Use the echo backend instead of connecting to the LLM API
- `--setup`: Run the setup script before starting the bot
- `--skip-venv-check`: Skip the virtual environment check

## Troubleshooting

### Common Issues

1. **"The command help is already an existing command or alias"**
   - This error occurs because discord.py already has a built-in help command
   - Solution: Use `@BotName commands` instead of `@BotName help` to see available commands

2. **"PyNaCl is not installed, voice will NOT be supported"**
   - This warning appears if PyNaCl is not installed, which is needed for voice support
   - Solution: Run `pip install PyNaCl` in your virtual environment
   - Note: Voice support is not required for the bot to function properly

3. **"Error: DISCORD_TOKEN not set in .env file"**
   - This error occurs when the Discord token is missing or invalid
   - Solution: Make sure you've added your Discord bot token to the `.env` file

4. **"Error generating response: API request failed"**
   - This error occurs when the bot can't connect to the Oobabooga API
   - Solution: Make sure Oobabooga is running with the API enabled (`python server.py --api`)
   - Alternative: Use the echo backend for testing (`python run.py --echo`)

5. **"Not running in a virtual environment"**
   - This warning appears if you run the bot outside the virtual environment
   - Solution: Activate the virtual environment before running the bot
   - Alternative: Use `--skip-venv-check` to bypass this check

6. **"Commands not working with ! prefix"**
   - This is expected behavior as the bot now uses mention-based commands
   - Solution: Mention the bot with your command, like `@BotName reset` instead of `!reset`

### Getting Help

If you encounter issues not covered in this troubleshooting section, please:
1. Check the bot.log file for detailed error messages
2. Try running with the echo backend to isolate Discord-related issues
3. Make sure all dependencies are installed correctly
