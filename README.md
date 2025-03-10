# DeepBot - Discord Bot with Local LLM Integration

A Discord bot that uses a local LLM API to generate streaming responses based on conversation history.

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

## Service Management (Linux)

On Linux systems, the setup script automatically configures a systemd user service. Here's how to manage it:

### Enable Service to Run Without Login
```bash
# Enable lingering for your user to allow the service to run without being logged in
sudo loginctl enable-linger $USER

# Verify lingering is enabled
loginctl show-user $USER | grep Linger
```

### Basic Service Commands
```bash
# Enable the service to start on boot
systemctl --user enable deepbot.service

# Start the service
systemctl --user start deepbot.service

# Stop the service
systemctl --user stop deepbot.service

# Restart the service
systemctl --user restart deepbot.service

# Check service status
systemctl --user status deepbot.service
```

### Viewing Logs
```bash
# View recent logs
journalctl --user-unit deepbot.service

# Follow logs in real-time
journalctl --user-unit deepbot.service -f

# View logs since last boot
journalctl --user-unit deepbot.service -b

# View logs with timestamps
journalctl --user-unit deepbot.service --output=short-precise
```

### Troubleshooting
- If the service fails to start, check logs using the commands above
- Ensure the `.env` file is properly configured
- Verify Ollama is running and accessible
- Check permissions on the bot directory and files
- Run `systemctl --user daemon-reload` after making changes to the service file

## Configuration

The bot's configuration is split between environment variables and a JSON configuration file:

### Environment Variables (`.env`)
- `DISCORD_TOKEN`: Your Discord bot token

### API Configuration (`config.py`)
- `API_URL`: URL of the Ollama API (default: http://localhost:11434)
- `MODEL_NAME`: Name of the model to use (default: mistral-small)

### Model Options (`model_options.json`)
The bot's behavior can be customized by modifying the options in `model_options.json`:

- `temperature`: Controls randomness in responses (0.0 to 1.0)
- `max_tokens`: Maximum tokens to generate (-1 for unlimited)
- `top_p`: Top-p sampling parameter (0.0 to 1.0)
- `presence_penalty`: Penalty for token presence (0.0 to 1.0)
- `frequency_penalty`: Penalty for token frequency (0.0 to 1.0)
- `seed`: Random seed for generation (-1 for random)
- `max_history`: Maximum number of messages to keep in conversation history
- `history_fetch_limit`: Maximum number of messages to fetch from channel history on startup
- `max_response_lines`: Maximum number of lines in bot responses
- `max_prompt_lines`: Maximum number of lines in user prompts

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

---

*Created [by AI](https://docs.cursor.com/agent) — [for AI](https://mistral.ai/en/news/mistral-small-3).*
