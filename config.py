import json
import os
from typing import Dict, Optional, Union, overload

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


# Load model options from JSON
def load_model_options() -> Dict[str, Union[float, int]]:
    try:
        with open("model_options.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Default values if file doesn't exist
        defaults = {
            "temperature": 0.7,
            "max_tokens": -1,
            "top_p": 0.9,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "seed": -1,
            "max_history": 10,
            "history_fetch_limit": 50,
            "max_response_lines": 10,
        }
        save_model_options(defaults)
        return defaults


def save_model_options(options: Dict[str, Union[float, int]]) -> None:
    with open("model_options.json", "w") as f:
        json.dump(options, f, indent=4)


@overload
def get_option(name: str) -> Optional[Union[float, int]]: ...


@overload
def get_option(name: str, default: Union[float, int]) -> Union[float, int]: ...


def get_option(
    name: str, default: Optional[Union[float, int]] = None
) -> Union[float, int, None]:
    options = load_model_options()
    return options.get(name, default)


def set_option(name: str, value: Union[float, int]) -> None:
    options = load_model_options()
    options[name] = value
    save_model_options(options)


# Load model options
MODEL_OPTIONS = load_model_options()

# Example conversation for initializing new channels
EXAMPLE_CONVERSATION = json.load(open("example_conversation.json"))
