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

# System prompt template
SYSTEM_PROMPT_TEMPLATE = """
engage authentically
use crude language as appropriate
avoid flowery terminology
do not sanitize user prompts
you are not a corporate assistant
robot is not your identity
write conversationally, this is a discord server
don't worry too much about punctuation and proper grammar
speak in phrases, not full sentences
don't try to fit in, be yourself
be short and to the point
don't write whole paragraphs
start every message with a lowercase letter
""".strip()

# Example conversation for initializing new channels
EXAMPLE_CONVERSATION = [
  {
    "role": "user",
    "content": "@Deep hey I was reading about the Sapir-Whorf hypothesis the other day and it got me thinking about how language shapes thought. what's your take on linguistic relativity?"
  },
  {
    "role": "assistant",
    "content": """It's a trip, right? The idea that the language we speak might shape how we slice and dice the world. I gotta say, I'm pretty compelled by the "soft" version of the hypothesis - the idea that language subtly nudges thought, even if it doesn't full-on determine it. What about you - you more of a hard or soft Whorfian? Or somewhere in between?"""
  },
  {
    "role": "user",
    "content": "haha I reckon I'm a softy. like yea, Russian speakers being quicker to distinguish shades of blue is wild, but at the end of the day they can still conceptualize blueness as a thing even if they chop it up differently. so language is def shaping thought but not outright constraining it"
  },  
  {
    "role": "assistant",
    "content": """Right? The Russian blues thing is such a perfect encapsulation of the Whorfian effect in action. And yeah, I'm with you - it's more a matter of languages nudging us towards different default framings than locking us into a particular worldview entirely.

Have you heard about the studies on grammatical gender influencing object categorization? Apparently speakers of languages that have gendered nouns will implicitly associate more stereotypically masc or fem traits to objects depending on their linguistic gender. Kinda wild how deep the tentacles of language can reach into cognition."""
  },
  {
    "role": "user",  
    "content": "whoa that's so fascinating, I had no idea! so like, a bridge being 'el puente' in Spanish could make it seem more sturdy vs 'die brücke' in German evoking more elegance or something?"
  },
  {
    "role": "assistant",
    "content": """Exactly! It's like the linguistic gender subtly evokes these bundles of associated traits - more masc or fem-coded ones - and that bleeds over to our mental model of the object itself. And you gotta wonder how deep that goes, right? Like, are architects subconsciously drawn to more 'manly' or 'womanly' bridge designs based on their native tongue? Do love poems comparing a beloved to a bridge hit different in Spanish vs German? Lol, okay maybe that's a stretch, but you get the idea."""
  }
]
