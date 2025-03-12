"""Tools for the bot to use in responses."""

import json
import logging
import random
from typing import Any, Dict, List, Optional

# Set up logging
logger = logging.getLogger("deepbot.tools")


class ToolRegistry:
    """Registry for tools that can be used by the bot."""

    def __init__(self) -> None:
        """Initialize the tool registry."""
        self.tools: List[Dict[str, Any]] = []
        self.handlers: Dict[str, Any] = {}

        # Register default tools
        self.register_default_tools()

    def register_tool(
        self, name: str, description: str, parameters: Dict[str, Any], handler: Any
    ) -> None:
        """Register a tool.

        Args:
            name: The name of the tool
            description: The description of the tool
            parameters: The parameters for the tool
            handler: The function to handle tool calls
        """
        tool_def = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }

        self.tools.append(tool_def)
        self.handlers[name] = handler

        logger.info(f"Registered tool: {name}")

    def get_tools(self) -> List[Dict[str, Any]]:
        """Get all registered tools.

        Returns:
            The list of registered tools
        """
        return self.tools

    def get_handler(self, name: str) -> Optional[Any]:
        """Get the handler for a tool.

        Args:
            name: The name of the tool

        Returns:
            The handler function or None if not found
        """
        return self.handlers.get(name)

    def register_default_tools(self) -> None:
        """Register default tools."""
        # Dice roll tool
        self.register_tool(
            name="dice_roll",
            description="Roll dice and get the total",
            parameters={
                "type": "object",
                "properties": {
                    "dice": {
                        "type": "integer",
                        "description": "Number of dice to roll",
                    },
                    "sides": {
                        "type": "integer",
                        "description": "Number of sides on each die",
                    },
                },
                "required": ["dice", "sides"],
            },
            handler=self._handle_dice_roll_tool,
        )

        logger.info("Registered dice_roll tool")

    def _handle_dice_roll_tool(self, args: Dict[str, Any]) -> str:
        """Handle dice roll tool calls.

        Args:
            args: The tool arguments

        Returns:
            The tool response
        """
        logger.info(f"Handling dice roll with args: {args}")

        num_dice = args.get("dice", 1)
        num_sides = args.get("sides", 6)

        logger.info(f"Initial dice parameters: dice={num_dice}, sides={num_sides}")

        # Ensure parameters are integers
        try:
            num_dice = int(num_dice)
            num_sides = int(num_sides)
            logger.info(
                f"Converted parameters to integers: dice={num_dice}, sides={num_sides}"
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error converting parameters to integers: {str(e)}")
            error_response = json.dumps(
                {
                    "error": "Dice and sides must be integers",
                    "discord_message": "-# 🎲 Error: Dice and sides must be integers",
                }
            )
            logger.info(f"Returning error response: {error_response}")
            return error_response

        # Validate input
        if num_dice < 1:
            logger.info(f"Adjusting num_dice from {num_dice} to 1 (minimum)")
            num_dice = 1
        if num_sides < 2:
            logger.info(f"Adjusting num_sides from {num_sides} to 2 (minimum)")
            num_sides = 2

        # Cap at reasonable values to prevent abuse
        if num_dice > 100:
            logger.info(f"Capping num_dice from {num_dice} to 100 (maximum)")
            num_dice = 100
        if num_sides > 1000:
            logger.info(f"Capping num_sides from {num_sides} to 1000 (maximum)")
            num_sides = 1000

        # Roll the dice
        logger.info(f"Rolling {num_dice}d{num_sides}")
        rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
        total = sum(rolls)
        logger.info(f"Rolled: {rolls}, total: {total}")

        # Format rolls for display
        if len(rolls) > 10:
            # If there are too many rolls, just show the first few and the count
            display_rolls = f"{rolls[:10]} + {len(rolls) - 10} more"
        else:
            display_rolls = str(rolls)

        # Create response
        discord_message = (
            f"-# 🎲 Rolled {num_dice}d{num_sides}: {display_rolls} = {total}"
        )

        response = {
            "dice": num_dice,
            "sides": num_sides,
            "rolls": rolls,
            "total": total,
            "discord_message": discord_message,
        }

        logger.info(f"Dice roll response: {response}")
        response_json = json.dumps(response)
        logger.info(f"Serialized response: {response_json}")
        return response_json


# Create a global instance of the tool registry
tool_registry = ToolRegistry()
