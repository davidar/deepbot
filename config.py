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
# https://lmstudio.ai/docs/app/api/endpoints/openai
API_URL = "http://127.0.0.1:1234/v1"

# https://huggingface.co/lmstudio-community/Mistral-Small-24B-Instruct-2501-GGUF
# Mistral-Small-24B-Instruct-2501-GGUF/Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf
MODEL_NAME = "mistral-small-24b-instruct-2501"

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
