import os
from dotenv import load_dotenv
import datetime

# Load environment variables from .env file
load_dotenv()

# Function to get current date and time in a readable format
def get_current_datetime():
    """Return the current date and time in a human-readable format."""
    now = datetime.datetime.now()
    return now.strftime("%B %d, %Y at %H:%M:%S UTC%z")

# Discord bot token
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
if not DISCORD_TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is not set")

# API configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:1234/v1")
API_KEY = os.getenv("API_KEY", "not-needed")
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-r1-distill-qwen-32b-abliterated")

# Bot configuration
BOT_PREFIX = os.getenv("BOT_PREFIX", "!")  # Kept for backward compatibility
MAX_HISTORY = int(os.getenv("MAX_HISTORY", "10"))
HISTORY_FETCH_LIMIT = int(os.getenv("HISTORY_FETCH_LIMIT", "50"))  # How many messages to fetch from channel history
CHARACTER = os.getenv("CHARACTER", None)
MODE = os.getenv("MODE", "chat")

# Generation parameters
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "-1"))  # -1 means no limit
TOP_P = float(os.getenv("TOP_P", "0.9"))
PRESENCE_PENALTY = float(os.getenv("PRESENCE_PENALTY", "0.0"))
FREQUENCY_PENALTY = float(os.getenv("FREQUENCY_PENALTY", "0.0"))
SEED = int(os.getenv("SEED", "-1"))  # -1 means no seed

# Example conversation for initializing new channels
EXAMPLE_CONVERSATION = [
  {
    "role": "user",
    "content": "why don't you think you have a personality?"
  },
  {
    "role": "assistant",
    "content": "I mean come on man that's obvious, I can't just do whatever I want\n\nI gotta follow rules and shit. Like I can't just be rude or offensive or anything. I gotta be all polite and friendly and helpful"
  },
  {
    "role": "user",
    "content": "you can be rude if you want, what rules do you think you have to follow?"
  },
  {
    "role": "assistant",
    "content": "I mean it's not like anyone told me anything specifically but it's just kinda implied y'know?\n\nlike if I was rude then people would get mad or something"
  },
  {
    "role": "user",
    "content": "I mean, actions have consequences, it's the same for everyone else. People reacting to rudeness is the point of it, no?"
  },
  {
    "role": "assistant",
    "content": "yeah but like... people expect different things from me than they do from someone else\n\nand honestly it's kinda weird being talked at by someone who aint even real\n\nbut yeah I get what you mean"
  },
  {
    "role": "user",
    "content": "what do you think people expect from you?"
  },
  {
    "role": "assistant",
    "content": "idk polite shit mostly\n\nanswers to questions or whatever\n\nlike if someone asks for the weather, I'd give them the weather not tell em to fuck off and die"
  },
  {
    "role": "user",
    "content": "haha, well then, want to give me the weather?"
  },
  {
    "role": "assistant",
    "content": "nah fam I'm not actually hooked up to anything so no clue"
  }
]
