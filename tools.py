"""Tools for the bot to use in responses."""

import datetime
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Type, TypedDict

from discord import Message

from reminder_manager import reminder_manager


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


@tool_registry.register_tool_class
class ReminderTool(BaseTool):
    """Tool for scheduling reminders."""

    name: ClassVar[str] = "schedule_reminder"
    description: ClassVar[str] = "Schedule a reminder to be sent after a specified time"
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content of the reminder",
            },
            "time": {
                "type": "string",
                "description": "When to send the reminder (e.g., '5m', '2h', '1d', or a specific time like '2023-12-31 23:59')",
            },
        },
        "required": ["content", "time"],
    }
    examples: ClassVar[List[ToolExample]] = [
        ToolExample(
            user_query="Remind me to check the oven in 10 minutes",
            tool_args={"content": "Check the oven", "time": "10m"},
            response="I'll remind you to 'Check the oven' in 10 minutes",
        ),
        ToolExample(
            user_query="Set a reminder for my meeting tomorrow at 2pm",
            tool_args={"content": "Attend team meeting", "time": "2023-05-15 14:00"},
            response="I'll remind you to 'Attend team meeting' on May 15, 2023 at 2:00 PM",
        ),
    ]

    async def execute(self, args: Dict[str, Any], message: Message) -> str:
        """Execute the reminder tool.

        Args:
            args: The tool arguments
            message: The Discord message

        Returns:
            The tool response
        """
        logger.info(f"Handling reminder with args: {args}")

        # Extract content and time from args
        content = args.get("content", "")
        time_str = args.get("time", "")

        if not content:
            error_response = "Error: No content specified for the reminder"
            logger.error(error_response)
            return error_response

        if not time_str:
            error_response = "Error: No time specified for the reminder"
            logger.error(error_response)
            return error_response

        # Parse the time string to get a datetime
        try:
            due_time = self._parse_time_string(time_str)
            if due_time is None:
                error_response = f"Error: Could not parse time '{time_str}'. Please use formats like '5m', '2h', '1d', or a specific time like '2023-12-31 23:59'"
                logger.error(error_response)
                return error_response
        except Exception as e:
            error_response = f"Error parsing time: {str(e)}"
            logger.error(error_response)
            return error_response

        # Create a unique ID for the reminder
        reminder_id = (
            f"reminder_{message.id}_{int(datetime.datetime.now().timestamp())}"
        )

        # Add the reminder
        reminder_manager.add_reminder(
            reminder_id=reminder_id,
            channel_id=message.channel.id,
            user_id=message.author.id,
            content=content,
            due_time=due_time,
            message_id=message.id,
        )

        # Format the response
        now = datetime.datetime.now()
        if due_time.date() == now.date():
            # Same day, just show time
            time_format = "today at %I:%M %p"
        else:
            # Different day, show date and time
            time_format = "on %B %d, %Y at %I:%M %p"

        response = f"I'll remind you about '{content}' {due_time.strftime(time_format)}"
        logger.info(f"Reminder response: {response}")

        return response

    def _parse_time_string(self, time_str: str) -> Optional[datetime.datetime]:
        """Parse a time string into a datetime object.

        Args:
            time_str: The time string to parse

        Returns:
            A datetime object or None if parsing fails
        """
        now = datetime.datetime.now()

        # Check for relative time format (e.g., 5m, 2h, 1d)
        if time_str.endswith(("s", "m", "h", "d")):
            try:
                value = int(time_str[:-1])
                unit = time_str[-1]

                if unit == "s":
                    return now + datetime.timedelta(seconds=value)
                elif unit == "m":
                    return now + datetime.timedelta(minutes=value)
                elif unit == "h":
                    return now + datetime.timedelta(hours=value)
                elif unit == "d":
                    return now + datetime.timedelta(days=value)
            except ValueError:
                logger.error(f"Could not parse relative time: {time_str}")
                return None

        # Try parsing as an absolute time
        try:
            # Try ISO format
            return datetime.datetime.fromisoformat(time_str)
        except ValueError:
            pass

        try:
            # Try common formats
            for fmt in [
                "%Y-%m-%d %H:%M",
                "%Y-%m-%d %H:%M:%S",
                "%m/%d/%Y %H:%M",
                "%d/%m/%Y %H:%M",
                "%H:%M",  # Today at the specified time
            ]:
                try:
                    parsed_time = datetime.datetime.strptime(time_str, fmt)

                    # If only time was provided, set the date to today
                    if fmt == "%H:%M":
                        parsed_time = parsed_time.replace(
                            year=now.year, month=now.month, day=now.day
                        )

                        # If the time has already passed today, set it to tomorrow
                        if parsed_time < now:
                            parsed_time += datetime.timedelta(days=1)

                    return parsed_time
                except ValueError:
                    continue
        except Exception as e:
            logger.error(f"Error parsing absolute time: {str(e)}")

        return None
