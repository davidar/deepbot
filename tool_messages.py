"""Tool message formatting and parsing utilities.

This module centralizes the formatting and parsing of tool-related messages
to ensure consistency across different components of the bot.
"""

import logging
from typing import Any, Dict, Optional, Tuple

# Set up logging
logger = logging.getLogger("deepbot.tool_messages")


def format_tool_call_and_response(
    tool_name: str, tool_args: Dict[str, Any], response: str
) -> str:
    """Format a tool call and its response as a Python REPL code block.

    Args:
        tool_name: The name of the tool
        tool_args: The arguments for the tool
        response: The response from the tool

    Returns:
        A formatted code block with the tool call and response
    """
    formatted_args = ", ".join([f"{k}={repr(v)}" for k, v in tool_args.items()])
    return f"```\n>>> {tool_name}({formatted_args})\n{response}\n```"


def parse_repl_tool_message(content: str) -> Optional[Tuple[str, Dict[str, Any], str]]:
    """Parse a Python REPL-style tool message.

    Args:
        content: The message content to parse

    Returns:
        A tuple of (tool_name, tool_args, response) or None if parsing fails
    """
    if not content.startswith("```") or ">>>" not in content:
        return None

    try:
        # Remove the code block markers
        code_content = content.strip("`").strip()

        # Split into command and response
        parts = code_content.split("\n", 1)
        if len(parts) < 2:
            return None

        command = parts[0].strip(">>> ").strip()
        response = parts[1].strip()

        # Parse the function call
        if "(" not in command or ")" not in command:
            return None

        tool_name = command.split("(")[0].strip()
        args_str = command.split("(", 1)[1].rsplit(")", 1)[0].strip()

        # Parse the args
        tool_args: Dict[str, Any] = {}
        if args_str:
            # This is a simplified parser and may not handle all Python syntax correctly
            for arg_pair in args_str.split(","):
                if "=" in arg_pair:
                    key, value = arg_pair.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Try to convert to appropriate types
                    try:
                        # Try to eval the value (handles strings, numbers, booleans, etc.)
                        tool_args[key] = eval(value)
                    except (SyntaxError, NameError):
                        # Keep as string if eval fails due to syntax or undefined names
                        tool_args[key] = value

        return tool_name, tool_args, response
    except Exception as e:
        logger.warning(f"Error parsing REPL tool message: {str(e)}")

    return None


def is_tool_message(content: str) -> bool:
    """Check if a message is a tool-related message.

    Args:
        content: The message content to check

    Returns:
        True if the message is tool-related, False otherwise
    """
    return content.startswith("```") and ">>>" in content
