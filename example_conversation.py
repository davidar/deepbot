"""Example conversation management for DeepBot."""

import json
import logging
from typing import List, Literal, Optional, Tuple

from ollama import Message as LLMMessage

# File to store the example conversation
EXAMPLE_CONVERSATION_FILE = "example_conversation.json"

logger = logging.getLogger("deepbot")


def load_example_conversation() -> List[LLMMessage]:
    """Load the example conversation from file."""
    try:
        with open(EXAMPLE_CONVERSATION_FILE, "r") as f:
            data = json.load(f)
            messages = [LLMMessage(**msg) for msg in data]
            logger.debug(f"Loaded {len(messages)} messages from example conversation")
            return messages
    except Exception as e:
        logger.error(f"Error loading example conversation: {e}")
        return []


def save_example_conversation(messages: List[LLMMessage]) -> None:
    """Save the example conversation to file."""
    try:
        with open(EXAMPLE_CONVERSATION_FILE, "w") as f:
            json.dump([msg.model_dump() for msg in messages], f, indent=4)
            logger.debug(f"Saved {len(messages)} messages to example conversation file")
    except Exception as e:
        logger.error(f"Error saving example conversation: {e}")


def add_message(
    role: Literal["user", "assistant", "system", "tool"], content: str
) -> List[LLMMessage]:
    """Add a message to the example conversation.

    Args:
        role: The role of the message ("user" or "assistant")
        content: The content of the message

    Returns:
        Updated list of messages
    """
    messages = load_example_conversation()
    new_message = LLMMessage(role=role, content=content)
    messages.append(new_message)
    save_example_conversation(messages)
    logger.info(f"Added new message to example conversation: {role}")
    return messages


def remove_message(index: int) -> Tuple[List[LLMMessage], Optional[LLMMessage]]:
    """Remove a message from the example conversation by index.

    Args:
        index: The index of the message to remove

    Returns:
        Tuple of (updated messages, removed message)
    """
    messages = load_example_conversation()
    if 0 <= index < len(messages):
        removed = messages.pop(index)
        save_example_conversation(messages)
        logger.info(f"Removed message at index {index}")
        return messages, removed
    return messages, None


def edit_message(
    index: int,
    role: Optional[Literal["user", "assistant", "system", "tool"]] = None,
    content: Optional[str] = None,
) -> Tuple[List[LLMMessage], Optional[LLMMessage]]:
    """Edit a message in the example conversation.

    Args:
        index: The index of the message to edit
        role: Optional new role for the message
        content: Optional new content for the message

    Returns:
        Tuple of (updated messages, edited message)
    """
    messages = load_example_conversation()
    if 0 <= index < len(messages):
        message = messages[index]
        if role is not None:
            message.role = role
        if content is not None:
            message.content = content
        save_example_conversation(messages)
        logger.info(f"Edited message at index {index}")
        return messages, message
    return messages, None
