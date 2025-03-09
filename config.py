"""Configuration management for DeepBot."""

import json
import os
from typing import TypedDict, cast

from dotenv import load_dotenv
from ollama import Message as LLMMessage


class ModelOptions(TypedDict):
    """Model options for LLM configuration."""

    temperature: float
    max_tokens: int
    top_p: float
    presence_penalty: float
    frequency_penalty: float
    seed: int
    max_history: int
    history_fetch_limit: int
    max_response_lines: int
    max_prompt_lines: int


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


# Example conversation for initializing new channels
EXAMPLE_CONVERSATION: list[LLMMessage] = [
    LLMMessage(**msg) for msg in json.load(open("example_conversation.json"))
]
