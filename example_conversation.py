"""Example conversation management for DeepBot."""

import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ollama import Message as LLMMessage

from tools import tool_registry

# File to store the example conversation
EXAMPLE_CONVERSATION_FILE = "example_conversation.json"

logger = logging.getLogger("deepbot")


@dataclass
class MessagePair:
    """A pair of user and assistant messages."""

    user: str
    assistant: str


def load_example_conversation() -> List[LLMMessage]:
    """Load the example conversation from file and convert to LLM messages."""
    try:
        with open(EXAMPLE_CONVERSATION_FILE, "r") as f:
            data = json.load(f)
            messages: List[LLMMessage] = []
            for pair in data:
                messages.append(LLMMessage(role="user", content=pair["user"]))
                messages.append(LLMMessage(role="assistant", content=pair["assistant"]))
            logger.debug(f"Loaded {len(messages)} messages from example conversation")

            # Append tool examples from the registry
            tool_examples = tool_registry.get_examples()
            for tool_name, examples in tool_examples.items():
                for example in examples:
                    messages.append(
                        LLMMessage(role="user", content=example["user_query"])
                    )
                    messages.append(
                        LLMMessage(
                            role="assistant",
                            tool_calls=[
                                LLMMessage.ToolCall(
                                    function=LLMMessage.ToolCall.Function(
                                        name=tool_name,
                                        arguments=example["tool_args"],
                                    )
                                )
                            ],
                        )
                    )
                    messages.append(
                        LLMMessage(role="tool", content=example["response"])
                    )

            logger.debug(
                f"Added tool examples from {len(tool_examples)} tools to conversation"
            )

            return messages
    except Exception as e:
        logger.error(f"Error loading example conversation: {e}")
        return []


def save_example_conversation(pairs: List[MessagePair]) -> None:
    """Save the example conversation to file.

    Args:
        pairs: List of message pairs to save
    """
    try:
        with open(EXAMPLE_CONVERSATION_FILE, "w") as f:
            json.dump(
                [{"user": p.user, "assistant": p.assistant} for p in pairs], f, indent=4
            )
            logger.debug(
                f"Saved {len(pairs)} message pairs to example conversation file"
            )
    except Exception as e:
        logger.error(f"Error saving example conversation: {e}")


def load_pairs() -> List[MessagePair]:
    """Load the raw message pairs from file."""
    try:
        with open(EXAMPLE_CONVERSATION_FILE, "r") as f:
            data = json.load(f)
            return [MessagePair(**pair) for pair in data]
    except Exception as e:
        logger.error(f"Error loading example conversation pairs: {e}")
        return []


def add_pair(user_msg: str, assistant_msg: str) -> List[MessagePair]:
    """Add a message pair to the example conversation.

    Args:
        user_msg: The user's message
        assistant_msg: The assistant's response

    Returns:
        Updated list of message pairs
    """
    pairs = load_pairs()
    new_pair = MessagePair(user=user_msg, assistant=assistant_msg)
    pairs.append(new_pair)
    save_example_conversation(pairs)
    logger.info("Added new message pair to example conversation")
    return pairs


def remove_pair(index: int) -> Tuple[List[MessagePair], Optional[MessagePair]]:
    """Remove a message pair from the example conversation by index.

    Args:
        index: The index of the pair to remove

    Returns:
        Tuple of (updated pairs, removed pair)
    """
    pairs = load_pairs()
    if 0 <= index < len(pairs):
        removed = pairs.pop(index)
        save_example_conversation(pairs)
        logger.info(f"Removed message pair at index {index}")
        return pairs, removed
    return pairs, None


def edit_pair(
    index: int,
    user_msg: Optional[str] = None,
    assistant_msg: Optional[str] = None,
) -> Tuple[List[MessagePair], Optional[MessagePair]]:
    """Edit a message pair in the example conversation.

    Args:
        index: The index of the pair to edit
        user_msg: Optional new user message
        assistant_msg: Optional new assistant message

    Returns:
        Tuple of (updated pairs, edited pair)
    """
    pairs = load_pairs()
    if 0 <= index < len(pairs):
        pair = pairs[index]
        if user_msg is not None:
            pair.user = user_msg
        if assistant_msg is not None:
            pair.assistant = assistant_msg
        save_example_conversation(pairs)
        logger.info(f"Edited message pair at index {index}")
        return pairs, pair
    return pairs, None
