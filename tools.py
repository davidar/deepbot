"""Tools for the bot to use in responses."""

import logging
import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Type, TypedDict

from discord import Message


class ToolExample(TypedDict):
    """Type for a tool usage example."""

    user_query: str
    tool_args: Dict[str, Any]
    response: str


# Type definition for tool function definition
class ToolFunctionDef(TypedDict):
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolDefinition(TypedDict):
    type: str
    function: ToolFunctionDef


# Set up logging
logger = logging.getLogger("deepbot.tools")


class BaseTool(ABC):
    """Base class for all tools."""

    name: ClassVar[str]
    description: ClassVar[str]
    parameters: ClassVar[Dict[str, Any]]
    examples: ClassVar[List[ToolExample]] = []

    @abstractmethod
    async def execute(self, args: Dict[str, Any], message: Message) -> str:
        """Execute the tool with the given arguments.

        Args:
            args: The arguments for the tool
            message: The Discord message

        Returns:
            The tool response
        """
        pass


class ToolRegistry:
    """Registry for tools that can be used by the bot."""

    _instance = None

    def __new__(cls):
        """Ensure singleton pattern for ToolRegistry."""
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize the tool registry."""
        if getattr(self, "_initialized", False):
            return

        self.tool_classes: Dict[str, Type[BaseTool]] = {}
        self._initialized = True

        logger.info("Tool registry initialized")

    def register_tool_class(self, tool_class: Type[BaseTool]) -> Type[BaseTool]:
        """Register a tool class.

        Args:
            tool_class: The tool class to register

        Returns:
            The registered tool class
        """
        name = tool_class.name
        if not name:
            logger.error(f"Tool class {tool_class.__name__} has empty name attribute")
            return tool_class

        self.tool_classes[name] = tool_class

        # Log registration
        example_count = (
            len(tool_class.examples) if hasattr(tool_class, "examples") else 0
        )
        if example_count > 0:
            logger.info(f"Registered {example_count} examples for tool: {name}")

        logger.info(f"Registered tool: {name}")
        return tool_class

    def get_tools(self) -> List[ToolDefinition]:
        """Get all registered tools in the format expected by the LLM.

        Returns:
            The list of registered tools
        """
        tools: List[ToolDefinition] = []
        for tool_class in self.tool_classes.values():
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool_class.name,
                        "description": tool_class.description,
                        "parameters": tool_class.parameters,
                    },
                }
            )
        return tools

    async def call_tool(self, name: str, args: Dict[str, Any], message: Message) -> str:
        """Call a tool.

        Args:
            name: The name of the tool
            args: The arguments for the tool
            message: The Discord message

        Returns:
            The tool response
        """
        tool_class = self.tool_classes.get(name)
        if not tool_class:
            logger.warning(f"No handler found for tool: {name}")
            return "Error: Tool not found"

        tool_instance = tool_class()
        return await tool_instance.execute(args, message)

    def get_examples(self, name: Optional[str] = None) -> Dict[str, List[ToolExample]]:
        """Get examples for a specific tool or all tools.

        Args:
            name: Optional name of the tool to get examples for

        Returns:
            Dictionary of tool examples or examples for the specified tool
        """
        if name:
            tool_class = self.tool_classes.get(name)
            if tool_class and hasattr(tool_class, "examples"):
                return {name: tool_class.examples}
            return {}

        # Return examples for all tools
        examples: Dict[str, List[ToolExample]] = {}
        for name, tool_class in self.tool_classes.items():
            examples[name] = tool_class.examples
        return examples


# Create a global instance of the tool registry
tool_registry = ToolRegistry()


@tool_registry.register_tool_class
class DiceRollTool(BaseTool):
    """Tool for rolling dice."""

    name: ClassVar[str] = "dice_roll"
    description: ClassVar[str] = "Roll dice and get the total"
    parameters: ClassVar[Dict[str, Any]] = {
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
    }
    examples: ClassVar[List[ToolExample]] = [
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
    ]

    async def execute(self, args: Dict[str, Any], message: Message) -> str:
        """Execute the dice roll tool.

        Args:
            args: The tool arguments
            message: The Discord message

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


@tool_registry.register_tool_class
class DiscordReactionTool(BaseTool):
    """Tool for adding reactions to Discord messages."""

    name: ClassVar[str] = "discord_reaction"
    description: ClassVar[str] = (
        "Add a reaction emoji to the Discord message being replied to"
    )
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "emoji": {
                "type": "string",
                "description": "The emoji to react with. Can be a Unicode emoji or a Discord custom emoji ID/name.",
            }
        },
        "required": ["emoji"],
    }
    examples: ClassVar[List[ToolExample]] = [
        ToolExample(
            user_query="Please react with a thumbs up to my message",
            tool_args={"emoji": "👍"},
            response="Added reaction 👍 to the message",
        ),
    ]

    async def execute(self, args: Dict[str, Any], message: Message) -> str:
        """Execute the Discord reaction tool.

        Args:
            args: The tool arguments
            message: The Discord message

        Returns:
            The tool response
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
