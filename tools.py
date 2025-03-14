"""Tools for the bot to use in responses."""

import logging
import random
from typing import Any, Dict, List, Optional, TypedDict

from discord import Message


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

    async def call_tool(self, name: str, args: Dict[str, Any], message: Message) -> str:
        """Call a tool.

        Args:
            name: The name of the tool
            args: The arguments for the tool
        """
        handler = self.handlers.get(name)
        if not handler:
            logger.warning(f"No handler found for tool: {name}")
            return "Error: Tool not found"

        return await handler(args, message)

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

        # Discord reaction tool
        self.register_tool(
            name="discord_reaction",
            description="Add a reaction emoji to the Discord message being replied to",
            parameters={
                "type": "object",
                "properties": {
                    "emoji": {
                        "type": "string",
                        "description": "The emoji to react with. Can be a Unicode emoji or a Discord custom emoji ID/name.",
                    }
                },
                "required": ["emoji"],
            },
            handler=self._handle_discord_reaction_tool,
            examples=[
                ToolExample(
                    user_query="Please react with a thumbs up to my message",
                    tool_args={"emoji": "👍"},
                    response="Added reaction 👍 to the message",
                ),
            ],
        )
        logger.info("Registered discord_reaction tool")

    async def _handle_dice_roll_tool(
        self, args: Dict[str, Any], message: Message
    ) -> str:
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

    async def _handle_discord_reaction_tool(
        self, args: Dict[str, Any], message: Message
    ) -> str:
        """Handle Discord reaction tool calls.

        This function adds a reaction to the Discord message being replied to.

        Args:
            args: The tool arguments containing the emoji to react with

        Returns:
            A confirmation message indicating the reaction was added
        """
        logger.info(f"Handling Discord reaction with args: {args}")

        # Extract emoji from args
        emoji = args.get("emoji", "")

        if not emoji:
            error_response = "Error: No emoji specified for reaction"
            logger.error(error_response)
            return error_response

        logger.info(f"Adding reaction with emoji: {emoji}")

        # Process custom emoji format if needed
        # Discord custom emojis can be in formats like:
        # - <:emoji_name:emoji_id>
        # - emoji_name:emoji_id
        if ":" in emoji and not emoji.startswith("<"):
            # If it's a custom emoji without proper formatting, add the brackets
            emoji = f"<:{emoji}>"
            logger.info(f"Formatted custom emoji: {emoji}")

        # Add the reaction
        try:
            await message.add_reaction(emoji)
            logger.info(f"Successfully added reaction {emoji} to message {message.id}")
            response = f"Added reaction {emoji} to the message"
            logger.info(f"Discord reaction response: {response}")
            return response

        except Exception as e:
            error_response = f"Error adding reaction: {str(e)}"
            logger.error(error_response)
            return error_response


# Create a global instance of the tool registry
tool_registry = ToolRegistry()
