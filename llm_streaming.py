"""LLM streaming response handling."""

import asyncio
import json
import logging
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Tuple

import ollama
from discord import ClientUser, Message
from ollama import Message as LLMMessage

import config
from context_builder import ContextBuilder
from tools import tool_registry

# Set up logging
logger = logging.getLogger("deepbot.llm")


class LineStatus(Enum):
    """Status of a line being streamed."""

    ACCUMULATING = "accumulating"  # Line is still being built
    COMPLETE = "complete"  # Line is complete and ready to send
    TOOL_CALL = "tool_call"  # Line is a tool call

    def __str__(self) -> str:
        return self.value


class LLMResponseHandler:
    """Handles streaming responses from the LLM."""

    def __init__(self, api_client: ollama.Client, bot_user: ClientUser) -> None:
        """Initialize the LLM response handler.

        Args:
            api_client: The Ollama API client to use for responses
            bot_user: The bot's Discord user
        """
        self.api_client = api_client
        self.bot_user = bot_user
        self.response_queues: Dict[int, asyncio.Queue[Message]] = {}
        self.response_tasks: Dict[int, Optional[asyncio.Task[None]]] = {}
        self.shutup_tasks: Set[asyncio.Task[None]] = set()

        # Get the tool registry
        self.tool_registry = tool_registry

    async def _stream_response_lines(
        self,
        context: List[LLMMessage],
    ) -> AsyncGenerator[Tuple[LineStatus, str], None]:
        """Generator that yields line status and content as they stream in from the API.

        Args:
            context: The conversation context to send to the LLM

        Yields:
            Tuples of (LineStatus, content) where:
            - LineStatus.ACCUMULATING indicates the start of a new line
            - LineStatus.COMPLETE indicates a complete line ready to send
            - LineStatus.TOOL_CALL indicates a tool call
        """
        try:
            logger.info(
                f"Starting streaming response with {len(context)} context messages"
            )
            logger.info(
                f"Context: {json.dumps([{'role': m.role, 'content': m.content} for m in context], indent=2)}"
            )

            # Get tools from the registry
            tools = self.tool_registry.get_tools()
            logger.info(
                f"Available tools for this response: {[tool['function']['name'] for tool in tools]}"
            )

            stream = self.api_client.chat(  # pyright: ignore
                model=str(config.MODEL_NAME),
                messages=context,
                stream=True,
                keep_alive=-1,
                options=config.load_model_options(),
                tools=tools,
            )

            # Variables to track streaming state
            current_line = ""  # Current line being built
            has_non_whitespace = False

            # Process streaming response
            for chunk in stream:
                try:
                    # Log chunk safely
                    try:
                        logger.debug(f"Received chunk type: {type(chunk)}")
                        logger.debug(f"Received chunk: {chunk}")
                    except Exception as e:
                        logger.debug(f"Could not log chunk: {str(e)}")

                    # Check for tool calls in the chunk directly
                    if (
                        hasattr(chunk, "message")
                        and hasattr(chunk.message, "tool_calls")
                        and chunk.message.tool_calls
                    ):
                        for tool_call in chunk.message.tool_calls:
                            try:
                                if hasattr(tool_call, "function"):
                                    function_name = tool_call.function.name
                                    function_args = tool_call.function.arguments

                                    # Log the tool call
                                    logger.info(
                                        f"Tool call detected: {function_name}({function_args})"
                                    )

                                    # Store tool call data
                                    tool_call_data = {
                                        "name": function_name,
                                        "args": (
                                            function_args
                                            if isinstance(function_args, dict)
                                            else json.loads(function_args)
                                        ),
                                    }

                                    # Yield the tool call
                                    logger.info(
                                        f"Yielding tool call: {json.dumps(tool_call_data)}"
                                    )
                                    yield (
                                        LineStatus.TOOL_CALL,
                                        json.dumps(tool_call_data),
                                    )

                                    # Handle the tool call
                                    handler = self.tool_registry.get_handler(
                                        function_name
                                    )
                                    if handler:
                                        logger.info(
                                            f"Found handler for tool: {function_name}"
                                        )
                                        try:
                                            # Get the tool response
                                            tool_response = handler(
                                                tool_call_data["args"]
                                            )
                                            logger.info(
                                                f"Tool response: {tool_response}"
                                            )

                                            # Add the tool response to the context
                                            context.append(
                                                LLMMessage(
                                                    role="tool",
                                                    content=tool_response,
                                                )
                                            )
                                            logger.info(
                                                f"Added tool response to context"
                                            )

                                            # Restart the stream with the updated context
                                            logger.info(
                                                f"Restarting stream with updated context"
                                            )
                                            stream = (
                                                self.api_client.chat(  # pyright: ignore
                                                    model=str(config.MODEL_NAME),
                                                    messages=context,
                                                    stream=True,
                                                    keep_alive=-1,
                                                    options=config.load_model_options(),
                                                    tools=tools,
                                                )
                                            )

                                            # Reset streaming state
                                            current_line = ""
                                            has_non_whitespace = False
                                            continue

                                        except Exception as e:
                                            logger.error(
                                                f"Error handling tool call: {str(e)}"
                                            )
                                    else:
                                        logger.warning(
                                            f"No handler found for tool: {function_name}"
                                        )

                            except Exception as e:
                                logger.error(f"Error processing tool call: {str(e)}")
                                continue

                    # Process regular content
                    if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                        content = chunk.message.content
                        if content:
                            current_line += content

                            # Check if we've received non-whitespace content
                            if not has_non_whitespace and content.strip():
                                has_non_whitespace = True
                                # Signal start of new line with content
                                yield (LineStatus.ACCUMULATING, "")

                            # Check for newlines in current line
                            while "\n" in current_line:
                                # Split at first newline
                                to_send, current_line = current_line.split("\n", 1)
                                has_non_whitespace = False  # Reset for next line

                                # Only yield if we have non-empty content
                                if to_send.strip():
                                    yield (LineStatus.COMPLETE, to_send)

                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")
                    continue

            # Handle any remaining text
            if current_line.strip():
                yield (LineStatus.COMPLETE, current_line)

        except Exception as e:
            error_message = f"-# Error in streaming response: {str(e)}"
            logger.error(error_message)
            logger.error(f"Full error details: {repr(e)}")
            yield (LineStatus.COMPLETE, error_message)

    async def process_response_queue(
        self,
        channel_id: int,
        context_builder: ContextBuilder,
        message_history: List[Message],
    ) -> None:
        """Process the response queue for a channel.

        Args:
            channel_id: The Discord channel ID
            context_builder: The context builder to use
            message_history: The message history to build context from
        """
        while True:
            try:
                # Get the next message to respond to
                message = await self.response_queues[channel_id].get()

                # Process the response
                try:
                    await self._handle_streaming_response(
                        message, context_builder, message_history
                    )
                except Exception as e:
                    logger.error(f"Error processing response: {str(e)}")
                    await message.reply(f"-# Sorry, I encountered an error: {str(e)}")
                finally:
                    # Mark the task as done
                    self.response_queues[channel_id].task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in response queue processor: {str(e)}")
                continue

    async def ensure_queue_processor(
        self,
        channel_id: int,
        context_builder: ContextBuilder,
        message_history: List[Message],
    ) -> None:
        """Ensure there's an active queue processor for the channel.

        Args:
            channel_id: The Discord channel ID
            context_builder: The context builder to use
            message_history: The message history to build context from
        """
        task = self.response_tasks.get(channel_id)
        if task is None or task.done():
            # Cancel existing task if it exists
            if task is not None and not task.done():
                task.cancel()
            # Create a new processor task
            self.response_tasks[channel_id] = asyncio.create_task(
                self.process_response_queue(
                    channel_id, context_builder, message_history
                )
            )

    async def _handle_streaming_response(
        self,
        message: Message,
        context_builder: ContextBuilder,
        message_history: List[Message],
    ) -> None:
        """Handle a streaming response by combining streaming and sending logic.

        Args:
            message: The Discord message to respond to
            context_builder: The context builder to use
            message_history: The message history to build context from
        """
        # Get the current task
        current_task = asyncio.current_task()
        if current_task is None:
            logger.error("No current task found")
            return

        async with message.channel.typing():
            try:
                # Add typing reaction
                await message.add_reaction("⌨️")
                # Remove the thinking reaction
                try:
                    await message.remove_reaction("💭", self.bot_user)
                except:
                    # Reaction might not exist, that's okay
                    pass

                # Build context for LLM
                context = context_builder.build_context(
                    message_history, message.channel, message
                )

                first_message = True
                line_count = 0

                async for status, line in self._stream_response_lines(context):
                    # Check if this specific task has been told to shut up
                    if current_task in self.shutup_tasks:
                        logger.info(f"Task was told to shut up, stopping response")
                        break

                    if status == LineStatus.ACCUMULATING:
                        logger.info(f"Accumulating line")
                        # Start typing indicator
                        async with message.channel.typing():
                            pass
                    elif status == LineStatus.TOOL_CALL:
                        logger.info(
                            f"Processing tool call in _handle_streaming_response: {line}"
                        )
                        try:
                            # Parse the tool call data
                            tool_data = json.loads(line)
                            tool_name = tool_data.get("name", "unknown")
                            tool_args = tool_data.get("args", {})

                            logger.info(
                                f"Tool call data in _handle_streaming_response: name={tool_name}, args={tool_args}"
                            )

                            # Send a message indicating tool usage
                            tool_message = None
                            if first_message:
                                tool_message = await message.reply(
                                    f"-# 🔧 Using tool: `{tool_name}`..."
                                )
                                first_message = False
                            else:
                                tool_message = await message.channel.send(
                                    f"-# 🔧 Using tool: `{tool_name}`..."
                                )

                            logger.info(
                                f"Sent tool usage message: {tool_message.id if tool_message else 'None'}"
                            )

                            line_count += 1

                            # Handle the tool response for dice_roll
                            if tool_name == "dice_roll":
                                logger.info(
                                    f"Processing dice_roll tool call in _handle_streaming_response"
                                )
                                # Get the handler
                                handler = self.tool_registry.get_handler(tool_name)
                                if handler:
                                    logger.info(
                                        f"Found handler for dice_roll in _handle_streaming_response"
                                    )
                                    # Get the tool response
                                    tool_response_json = handler(tool_args)
                                    logger.info(
                                        f"Dice roll response in _handle_streaming_response: {tool_response_json}"
                                    )
                                    try:
                                        # Parse the response
                                        tool_response = json.loads(tool_response_json)
                                        # Check if there's a discord_message
                                        if "discord_message" in tool_response:
                                            # Send the discord message
                                            discord_message = tool_response[
                                                "discord_message"
                                            ]
                                            logger.info(
                                                f"Sending dice roll result in _handle_streaming_response: {discord_message}"
                                            )
                                            result_message = await message.channel.send(
                                                discord_message
                                            )
                                            logger.info(
                                                f"Sent dice roll result message: {result_message.id}"
                                            )
                                        else:
                                            logger.warning(
                                                f"No discord_message in tool response in _handle_streaming_response"
                                            )
                                    except json.JSONDecodeError:
                                        logger.warning(
                                            f"Failed to parse tool response in _handle_streaming_response: {tool_response_json}"
                                        )
                                else:
                                    logger.warning(
                                        f"No handler found for dice_roll in _handle_streaming_response"
                                    )

                        except Exception as e:
                            logger.warning(
                                f"Failed to process tool call in _handle_streaming_response: {str(e)}"
                            )
                            logger.exception(e)  # Log the full exception with traceback
                    elif status == LineStatus.COMPLETE and line.strip():
                        logger.info(f"Sending line: {line}")
                        try:
                            if first_message:
                                # First message should be a reply
                                await message.reply(line)
                                first_message = False
                            else:
                                await message.channel.send(line)

                            line_count += 1

                            max_lines = config.load_model_options()[
                                "max_response_lines"
                            ]
                            if line_count >= max_lines:
                                logger.info(
                                    "Reached maximum line limit, stopping response"
                                )
                                await message.channel.send(
                                    "-# Response truncated due to length limit"
                                )
                                break

                        except Exception as e:
                            logger.warning(f"Failed to send message chunk: {str(e)}")
            except Exception as e:
                logger.error(f"Error sending messages: {str(e)}")
                await message.reply(f"-# Error sending messages: {str(e)}")
            finally:
                # Remove the typing reaction
                try:
                    await message.remove_reaction("⌨️", self.bot_user)
                except:
                    # Reaction might not exist, that's okay
                    pass
                # Remove this task from the shutup set
                self.shutup_tasks.discard(current_task)

    def add_to_queue(
        self,
        channel_id: int,
        message: Message,
    ) -> None:
        """Add a message to the response queue.

        Args:
            channel_id: The Discord channel ID
            message: The message to respond to
        """
        if channel_id not in self.response_queues:
            self.response_queues[channel_id] = asyncio.Queue()
        self.response_queues[channel_id].put_nowait(message)

    def stop_responses(self, channel_id: int) -> None:
        """Stop all responses in a channel.

        Args:
            channel_id: The Discord channel ID
        """
        # Cancel the response task if it exists
        task = self.response_tasks.get(channel_id)
        if task is not None and not task.done():
            # Mark this task as shut up
            self.shutup_tasks.add(task)
            task.cancel()
            self.response_tasks[channel_id] = None

        # Clear the response queue
        if channel_id in self.response_queues:
            while not self.response_queues[channel_id].empty():
                try:
                    self.response_queues[channel_id].get_nowait()
                except asyncio.QueueEmpty:
                    break
