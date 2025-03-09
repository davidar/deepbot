# DeepBot - Discord Bot with Local LLM Integration

A Discord bot that uses a local LLM API to generate streaming responses based on conversation history.

*Written by AI (Claude/Cursor), for AI (Mistral Small 3).*

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
   python bot.py
   ```

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
