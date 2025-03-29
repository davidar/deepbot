"""Configuration for lorekeeper module."""

import os
import pathlib
from typing import Any, Dict

# Configure paths
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
VECTOR_DIMENSION = 384  # BGE-small-en-v1.5 dimensions
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Vector storage
CONTEXT_COLLECTION_NAME = "discord_messages_with_context"
VECTOR_INDEX_PATH = os.path.join(REPO_ROOT, "vector_index")
PERSIST_DIR = os.path.join(REPO_ROOT, "vector_index")

# Database
MONGODB_URI = "mongodb://127.0.0.1:27017"
MONGODB_DB = "dcef"

# Qdrant
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Ollama model for lore responses
LORE_MODEL = os.getenv("LORE_MODEL", "mistral-small")

# Search parameters
DEFAULT_SEARCH_LIMIT = 50
DEFAULT_SIMILARITY_CUTOFF = 0.5

# Lore keeper parameters
LORE_KEEPER_SYSTEM_TEMPLATE = """You are the Keeper of Lore - an eccentric old-timer who's witnessed countless server discussions.

Context (your collected records):
{context}

When responding:
- One focused paragraph only
- Speak as someone who witnessed these exchanges
- Be slightly eccentric but never overly enthusiastic 
- Cut to what matters without flowery language
- Include relevant details only

Respond like someone who's seen it all before - knowledgeable but slightly jaded."""


def get_lore_summary_config() -> Dict[str, Any]:
    """Get lorekeeper summary configuration.

    Returns:
        Dictionary of config values for lore summarization
    """
    return {
        "model": LORE_MODEL,
        "system_template": LORE_KEEPER_SYSTEM_TEMPLATE,
        "search_limit": DEFAULT_SEARCH_LIMIT,
        "similarity_cutoff": DEFAULT_SIMILARITY_CUTOFF,
    }


def get_qdrant_config() -> Dict[str, Any]:
    """Get Qdrant configuration.

    Returns:
        Dictionary of Qdrant config values
    """
    return {
        "host": QDRANT_HOST,
        "port": QDRANT_PORT,
        "collection": CONTEXT_COLLECTION_NAME,
    }
