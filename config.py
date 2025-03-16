"""Configuration management for DeepBot."""

import json
import os
from typing import Dict, FrozenSet, Type, TypedDict, Union, cast, get_type_hints

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
API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "mistral-small")

# Embedding model configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")

# Storage configuration
MESSAGE_STORE_DIR = os.getenv("MESSAGE_STORE_DIR", "message_store")
SEARCH_INDEX_PATH = os.getenv("SEARCH_INDEX_PATH", "./chroma_db")


# Load model options from JSON
def load_model_options() -> ModelOptions:
    """Load model options from JSON file.

    Returns:
        ModelOptions: The loaded model options
    """
    with open("model_options.json", "r") as f:
        return cast(ModelOptions, json.load(f))


def save_model_options(options: ModelOptions) -> None:
    """Save model options to JSON file.

    Args:
        options: The model options to save
    """
    with open("model_options.json", "w") as f:
        json.dump(options, f, indent=4)


def get_ollama_options() -> Dict[str, Union[float, int]]:
    """Get options to pass to Ollama by excluding application-specific options.

    Returns:
        Dict[str, Union[float, int]]: The Ollama-specific options
    """
    options = load_model_options()
    ollama_options: Dict[str, Union[float, int]] = {
        k: cast(Union[float, int], v)
        for k, v in options.items()
        if k not in APP_SPECIFIC_OPTIONS
    }
    return ollama_options


def get_model_option_types() -> Dict[str, Type[Union[float, int]]]:
    """Get the types of model options.

    Returns:
        Dict[str, Type[Union[float, int]]]: A dictionary mapping option names to their types
    """
    type_hints = get_type_hints(ModelOptions)
    return {k: v for k, v in type_hints.items()}
