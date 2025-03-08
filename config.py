import json
import os

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Discord bot token
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is not set")

# API configuration
API_URL = "http://localhost:11434"
MODEL_NAME = "mistral-small"

# Bot configuration
MAX_HISTORY = 10
HISTORY_FETCH_LIMIT = 50

# Generation parameters
TEMPERATURE = 0.7
MAX_TOKENS = -1
TOP_P = 0.9
PRESENCE_PENALTY = 0.0
FREQUENCY_PENALTY = 0.0
SEED = -1

# Example conversation for initializing new channels
EXAMPLE_CONVERSATION = json.load(open("example_conversation.json"))
