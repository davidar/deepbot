import logging
import random
from typing import List, Optional, Tuple

import config

# File to store the system prompt
SYSTEM_PROMPT_FILE = "system_prompt.txt"

logger = logging.getLogger("deepbot")


def load_system_prompt() -> List[str]:
    """Load the system prompt from file, or create with initial prompt if it doesn't exist."""
    try:
        with open(SYSTEM_PROMPT_FILE, "r") as f:
            data = f.read()
            lines = data.strip().split("\n")
            logger.debug(f"Loaded {len(lines)} lines from system prompt file")
            return lines
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}")
        return []


def save_system_prompt(lines: List[str]) -> None:
    """Save the system prompt lines to file."""
    try:
        with open(SYSTEM_PROMPT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")
            logger.debug(f"Saved {len(lines)} lines to system prompt file")
    except Exception as e:
        logger.error(f"Error saving system prompt: {e}")


def add_line(line: str) -> Tuple[List[str], List[str]]:
    """Add a line to the system prompt and return the updated lines and any removed lines.

    Args:
        line: The line to add

    Returns:
        Tuple of (current_lines, removed_lines)
    """
    lines = load_system_prompt()
    removed_lines: List[str] = []
    logger.info(f"Adding line: {line}")
    logger.info(f"Current number of lines: {len(lines)}")

    if line not in lines:  # Avoid duplicates
        lines.append(line)
        logger.info(f"Line added, new total: {len(lines)}")
        # Save the file with the new line before trimming
        save_system_prompt(lines)

        # Check if we need to trim
        max_lines = config.load_model_options()["max_prompt_lines"]
        logger.info(f"Max lines allowed: {max_lines}")
        if len(lines) > max_lines:
            logger.info("Need to trim, calling trim_prompt")
            lines, removed_lines = trim_prompt(max_lines, lines)
            logger.info(
                f"After trimming: {len(lines)} lines remain, removed {len(removed_lines)} lines"
            )
            logger.info(f"Removed lines: {removed_lines}")

    return lines, removed_lines


def remove_line(line: str) -> List[str]:
    """Remove a line from the system prompt and return the updated lines."""
    lines = load_system_prompt()
    if line in lines:
        lines.remove(line)
        save_system_prompt(lines)
        logger.info(f"Removed line: {line}")
    return lines


def trim_prompt(
    max_lines: int, current_lines: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """Trim the system prompt to the specified maximum number of lines.

    Args:
        max_lines: Maximum number of lines to keep
        current_lines: Optional list of lines to trim. If not provided, loads from file.

    Returns:
        Tuple of (current_lines, removed_lines)
    """
    lines = current_lines if current_lines is not None else load_system_prompt()
    logger.info(
        f"Trimming prompt. Current lines: {len(lines)}, max allowed: {max_lines}"
    )

    if len(lines) <= max_lines:
        logger.info("No trimming needed")
        return lines, []

    # Keep track of removed lines
    num_to_remove = len(lines) - max_lines
    removed_lines: List[str] = []
    logger.info(f"Need to remove {num_to_remove} lines")

    # Randomly select lines to remove
    indices_to_remove = random.sample(range(len(lines)), num_to_remove)
    indices_to_remove.sort(reverse=True)  # Sort in reverse to remove from end first
    logger.info(f"Selected indices to remove: {indices_to_remove}")

    # Remove the selected lines
    for idx in indices_to_remove:
        line_to_remove = lines[idx]
        removed_lines.append(line_to_remove)
        lines.pop(idx)
        logger.info(f"Removed line at index {idx}: {line_to_remove}")

    # Save the trimmed prompt
    save_system_prompt(lines)
    logger.info(f"Final line count: {len(lines)}")
    logger.info(f"Removed lines: {removed_lines}")

    return lines, removed_lines
