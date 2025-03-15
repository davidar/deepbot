"""Configuration management for DeepBot."""

import json
import os
from typing import Dict, FrozenSet, TypedDict, Union, cast

from dotenv import load_dotenv

# Set of application-specific options that should not be passed to Ollama
APP_SPECIFIC_OPTIONS: FrozenSet[str] = frozenset(
    [
        "max_history",
        "history_fetch_limit",
        "max_response_lines",
        "max_prompt_lines",
    ]
)


class ModelOptions(TypedDict):
    """Model options for LLM configuration."""

    # Ollama-specific options (any option not in APP_SPECIFIC_OPTIONS)
    temperature: float
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    seed: int
    num_ctx: int

    # Application-specific options (listed in APP_SPECIFIC_OPTIONS)
    max_history: int  # Number of messages to keep in conversation history
    history_fetch_limit: int  # Maximum number of messages to fetch from Discord history
    max_response_lines: int  # Maximum number of lines in a response
    max_prompt_lines: int  # Maximum number of lines in a prompt


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
def load_model_options() -> ModelOptions:
    with open("model_options.json", "r") as f:
        return cast(ModelOptions, json.load(f))


def save_model_options(options: ModelOptions) -> None:
    with open("model_options.json", "w") as f:
        json.dump(options, f, indent=4)


def get_ollama_options() -> Dict[str, Union[float, int]]:
    """Get options to pass to Ollama by excluding application-specific options."""
    options = load_model_options()
    ollama_options: Dict[str, Union[float, int]] = {
        k: cast(Union[float, int], v)
        for k, v in options.items()
        if k not in APP_SPECIFIC_OPTIONS
    }
    return ollama_options
