"""Tools for the bot to use in responses."""

import datetime
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Dict, List, Optional, Type, TypedDict

from discord import Message

import example_conversation
import system_prompt
from reminder_manager import reminder_manager


class ToolExample(TypedDict, total=False):
    """Type for a tool usage example."""

    user_query: str
    bot_message: str
    tool_args: Dict[str, Any]
    response: str


class ToolExampleRequired(TypedDict):
    """Required fields for a tool usage example."""

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

    @classmethod
    def __new__(cls) -> "ToolRegistry":
        """Ensure singleton pattern for ToolRegistry.

        Returns:
            The singleton instance of ToolRegistry
        """
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
                "description": (
                    "When to send the reminder (e.g., '5m', '2h', '1d', "
                    "or a specific time like '2023-12-31 23:59')"
                ),
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
                error_response = (
                    f"Error: Could not parse time '{time_str}'. "
                    "Please use formats like '5m', '2h', '1d', "
                    "or a specific time like '2023-12-31 23:59'"
                )
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

    def _parse_relative_time(self, time_str: str) -> Optional[datetime.datetime]:
        """Parse a relative time string (e.g., 5m, 2h).

        Args:
            time_str: The time string to parse (e.g., "5m", "2h")

        Returns:
            A datetime object or None if parsing fails
        """
        if not time_str.endswith(("s", "m", "h", "d")):
            return None

        try:
            value = int(time_str[:-1])
            unit = time_str[-1]
            now = datetime.datetime.now()

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

        return None

    def _parse_absolute_time(self, time_str: str) -> Optional[datetime.datetime]:
        """Parse an absolute time string.

        Args:
            time_str: The time string to parse

        Returns:
            A datetime object or None if parsing fails
        """
        now = datetime.datetime.now()

        # Try ISO format first
        try:
            return datetime.datetime.fromisoformat(time_str)
        except ValueError:
            pass

        # Try common formats
        formats = [
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%m/%d/%Y %H:%M",
            "%d/%m/%Y %H:%M",
            "%H:%M",  # Today at the specified time
        ]

        for fmt in formats:
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

        return None

    def _parse_time_string(self, time_str: str) -> Optional[datetime.datetime]:
        """Parse a time string into a datetime object.

        Args:
            time_str: The time string to parse

        Returns:
            A datetime object or None if parsing fails
        """
        # Try parsing as relative time first
        result = self._parse_relative_time(time_str)
        if result:
            return result

        # Try parsing as absolute time
        result = self._parse_absolute_time(time_str)
        if result:
            return result

        return None


@tool_registry.register_tool_class
class SystemPromptTool(BaseTool):
    """Tool for the LLM to persist important behavioral patterns in its system prompt."""

    name: ClassVar[str] = "system_prompt"
    description: ClassVar[str] = (
        "Add important behavioral patterns or traits to remember long-term"
    )
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": (
                    "The behavioral pattern or trait to remember (e.g., "
                    "'be more empathetic with new users', "
                    "'use more technical terms with experienced users')"
                ),
            },
            "reason": {
                "type": "string",
                "description": "Why this pattern should be remembered (used for logging, not stored)",
            },
        },
        "required": ["pattern", "reason"],
    }
    examples: ClassVar[List[ToolExample]] = [
        ToolExample(
            bot_message=(
                "Your questions about data structures have been really insightful. "
                "I notice you grasp technical concepts quickly, so let me adjust my responses accordingly."
            ),
            tool_args={
                "pattern": "use technical terminology when the context allows",
                "reason": "user demonstrates strong technical understanding and prefers detailed explanations",
            },
            response=(
                "I'll remember to *use technical terminology when appropriate* because "
                "*user demonstrates strong technical understanding and prefers detailed explanations*"
            ),
        ),
        ToolExample(
            bot_message="I see I've been a bit too verbose in my explanations. Let me make my responses more focused.",
            tool_args={
                "pattern": "keep responses brief and to the point",
                "reason": "noticed user engagement drops with longer responses",
            },
            response=(
                "I'll remember to *keep responses brief and to the point* because "
                "*noticed user engagement drops with longer responses*"
            ),
        ),
        ToolExample(
            user_query="You're being too formal, can you be more casual?",
            bot_message=(
                "You're right, I should loosen up a bit. "
                "Let me adjust my communication style."
            ),
            tool_args={
                "pattern": "use casual, conversational language",
                "reason": "direct user feedback requesting more casual communication style",
            },
            response=(
                "I'll remember to *use casual, conversational language* because "
                "*direct user feedback requesting more casual communication style*"
            ),
        ),
    ]

    async def execute(self, args: Dict[str, Any], message: Message) -> str:
        """Execute the system prompt tool.

        Args:
            args: The tool arguments
            message: The Discord message

        Returns:
            The tool response
        """
        logger.info(f"Handling system prompt update with args: {args}")

        pattern = args.get("pattern")
        reason = args.get("reason", "No reason provided")

        if not pattern:
            return "Error: Pattern is required"

        # Log the reasoning
        logger.info(f"Adding pattern '{pattern}' because: {reason}")

        # Add the pattern to system prompt
        lines, removed_lines = system_prompt.add_line(pattern)

        response = [f"I'll remember to *{pattern}* because *{reason}*"]
        if removed_lines:
            response.append(
                f"To make room, I've forgotten some older patterns that seem less relevant now: {removed_lines}"
            )
        response.append(f"System prompt now contains {len(lines)} lines")

        return "\n".join(response)


@tool_registry.register_tool_class
class ExampleConversationTool(BaseTool):
    """Tool for the LLM to store exemplary conversation patterns."""

    name: ClassVar[str] = "example_conversation"
    description: ClassVar[str] = (
        "Store an exemplary conversation exchange that demonstrates a good interaction pattern"
    )
    parameters: ClassVar[Dict[str, Any]] = {
        "type": "object",
        "properties": {
            "user_message": {
                "type": "string",
                "description": "A representative user message that demonstrates the pattern",
            },
            "bot_message": {
                "type": "string",
                "description": "Your response that worked particularly well",
            },
            "reason": {
                "type": "string",
                "description": "Why this exchange is worth remembering (used for logging, not stored)",
            },
        },
        "required": ["user_message", "bot_message", "reason"],
    }
    examples: ClassVar[List[ToolExample]] = [
        ToolExample(
            bot_message=(
                "That restaurant analogy for async/await really clicked! "
                "I should save this approach for explaining similar concepts."
            ),
            tool_args={
                "user_message": "Can you explain how async/await works?",
                "bot_message": (
                    "think of it like a restaurant. async functions are like placing an order - "
                    "you don't wait around, you get a promise (your receipt) that food will come. "
                    "await is when you actually need the food to continue eating"
                ),
                "reason": "Restaurant analogy effectively bridges abstract programming concept with everyday experience",
            },
            response="I'll remember this interaction pattern for similar situations in the future.",
        ),
        ToolExample(
            user_query=(
                "That pencil vs pen analogy for let/const was perfect! Could you remember that?"
            ),
            bot_message=(
                "Thanks! You're right, that analogy worked really well. "
                "I'll save it for future explanations."
            ),
            tool_args={
                "user_message": "What's the difference between let and const in JavaScript?",
                "bot_message": (
                    "think of let like a pencil note - you can erase and rewrite it. "
                    "const is like using a pen - once you write it, that's it, no changing it"
                ),
                "reason": "Simple everyday object comparison that clearly illustrates mutability concept",
            },
            response="I'll remember this interaction pattern for similar situations in the future.",
        ),
        ToolExample(
            bot_message=(
                "The way I handled that frustrated user seemed to work well - "
                "using humor to redirect to problem-solving. I should remember this strategy."
            ),
            tool_args={
                "user_message": "This code is completely broken! Who wrote this garbage?",
                "bot_message": (
                    "hey, even Shakespeare wrote some rough drafts! "
                    "let's look at what's actually going on and fix it together"
                ),
                "reason": "Using humor to defuse tension while redirecting to constructive problem-solving",
            },
            response="I'll remember this interaction pattern for similar situations in the future.",
        ),
    ]

    async def execute(self, args: Dict[str, Any], message: Message) -> str:
        """Execute the example conversation tool.

        Args:
            args: The tool arguments
            message: The Discord message

        Returns:
            The tool response
        """
        logger.info(f"Handling example conversation with args: {args}")

        user_msg = args.get("user_message")
        bot_msg = args.get("bot_message")
        reason = args.get("reason", "No reason provided")

        if not user_msg or not bot_msg:
            return "Error: Both user_message and bot_message are required"

        # Log the reasoning
        logger.info(f"Adding example conversation because: {reason}")

        # Add the conversation pair
        example_conversation.add_pair(user_msg, bot_msg)

        return "I'll remember this interaction pattern for similar situations in the future."
