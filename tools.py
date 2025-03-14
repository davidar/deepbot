"""Tools for the bot to use in responses."""

import logging
import random
from typing import Any, Dict, List, Optional, TypedDict


class ToolExample(TypedDict):
    """Type for a tool usage example."""

    user_query: str
    tool_args: Dict[str, Any]
    response: str


# Set up logging
logger = logging.getLogger("deepbot.tools")


class ToolRegistry:
    """Registry for tools that can be used by the bot."""

    def __init__(self) -> None:
        """Initialize the tool registry."""
        self.tools: List[Dict[str, Any]] = []
        self.handlers: Dict[str, Any] = {}
        self.examples: Dict[str, List[ToolExample]] = {}

        # Register default tools
        self.register_default_tools()

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        handler: Any,
        examples: Optional[List[ToolExample]] = None,
    ) -> None:
        """Register a tool.

        Args:
            name: The name of the tool
            description: The description of the tool
            parameters: The parameters for the tool
            handler: The function to handle tool calls
            examples: Optional list of example usages, each containing 'user_query', 'tool_args', and 'response'
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

        # Store examples if provided
        if examples:
            self.examples[name] = examples
            logger.info(f"Registered {len(examples)} examples for tool: {name}")

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

    def get_examples(self, name: Optional[str] = None) -> Dict[str, List[ToolExample]]:
        """Get examples for a specific tool or all tools.

        Args:
            name: Optional name of the tool to get examples for

        Returns:
            Dictionary of tool examples or examples for the specified tool
        """
        if name:
            return {name: self.examples.get(name, [])} if name in self.examples else {}
        return self.examples

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
            examples=[
                ToolExample(
                    user_query="Can you roll a pair of dice for me?",
                    tool_args={"dice": 2, "sides": 6},
                    response="Rolled 2d6: [3, 5] = 8",
                ),
                ToolExample(
                    user_query="Roll a d20 for my attack roll",
                    tool_args={"dice": 1, "sides": 20},
                    response="Rolled 1d20: [18] = 18",
                ),
            ],
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

        # Extract dice and sides from args
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
            error_response = "Error: Dice and sides must be integers"
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

        # Create a human-readable response
        response = f"Rolled {num_dice}d{num_sides}: {rolls} = {total}"
        logger.info(f"Dice roll response: {response}")

        return response


# Create a global instance of the tool registry
tool_registry = ToolRegistry()
