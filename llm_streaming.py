"""LLM streaming response handling."""

import asyncio
import json
import logging
from enum import Enum
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
)

import ollama
from discord import ClientUser, Message
from ollama import ChatResponse
from ollama import Message as LLMMessage

import config
from context_builder import ContextBuilder
from tool_messages import format_tool_call_and_response
from tools import tool_registry

# Set up logging
logger = logging.getLogger("deepbot.llm")


class LineStatus(Enum):
    """Status of a line being streamed."""

    ACCUMULATING = "accumulating"  # Line is still being built
    COMPLETE = "complete"  # Line is complete and ready to send
    # TOOL_CALL = "tool_call"  # Line is a tool call

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
        self.tool_registry = tool_registry

    async def _process_content_chunk(
        self,
        content: str,
        current_line: str,
        has_non_whitespace: bool,
    ) -> Tuple[str, bool, List[Tuple[LineStatus, str]]]:
        """Process a content chunk and generate appropriate yields.

        Args:
            content: The content to process
            current_line: The current line being built
            has_non_whitespace: Whether non-whitespace content has been seen

        Returns:
            Tuple of (updated current_line, updated has_non_whitespace, yields)
        """
        yields: List[Tuple[LineStatus, str]] = []
        if content:
            current_line += content

            # Check if we've received non-whitespace content
            if not has_non_whitespace and content.strip():
                has_non_whitespace = True
                yields.append((LineStatus.ACCUMULATING, ""))

            # Check for newlines in current line
            while "\n" in current_line:
                # Split at first newline
                to_send, current_line = current_line.split("\n", 1)
                has_non_whitespace = False  # Reset for next line

                # Only yield if we have non-empty content
                if to_send.strip():
                    yields.append((LineStatus.COMPLETE, to_send))

        return current_line, has_non_whitespace, yields

    def _create_tool_call_data(self, tool_call: Any) -> Dict[str, Any]:
        """Create tool call data from a tool call object.

        Args:
            tool_call: The tool call object

        Returns:
            The tool call data dictionary
        """
        function_name = tool_call.function.name
        function_args = tool_call.function.arguments

        return {
            "name": function_name,
            "args": (
                function_args
                if isinstance(function_args, dict)
                else json.loads(function_args)
            ),
        }

    def _create_tool_response_message(self, response: str) -> LLMMessage:
        """Create a tool response message.

        Args:
            response: The tool response content

        Returns:
            The tool response message
        """
        # The Ollama API expects tool responses to have the role "tool"
        # The content should be the raw response string, not a JSON object
        # This will be sent directly to the API
        return LLMMessage(role="tool", content=response)

    async def _handle_tool_response(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        context: List[LLMMessage],
        tools: List[Dict[str, Any]],
        message: Message,
    ) -> AsyncGenerator[Tuple[LineStatus, str], None]:
        """Handle a tool response and generate a new stream.

        Args:
            tool_name: The name of the tool
            tool_args: The tool arguments
            context: The conversation context
            tools: The available tools
            message: The Discord message to respond to

        Yields:
            Status and content pairs from the new stream
        """
        handler = self.tool_registry.get_handler(tool_name)
        if not handler:
            logger.warning(f"No handler found for tool: {tool_name}")
            return

        try:
            # Get and log tool response
            tool_response = handler(tool_args)
            logger.info(f"Tool response: {tool_response}")

            # Format the tool call and response in a Python REPL code block
            combined_message = format_tool_call_and_response(
                tool_name, tool_args, tool_response
            )
            await message.channel.send(combined_message)

            # Add tool call to context first
            tool_call_message = LLMMessage(
                role="assistant",
                content="",  # Empty content for tool calls
                tool_calls=[
                    LLMMessage.ToolCall(
                        function=LLMMessage.ToolCall.Function(
                            name=tool_name,
                            arguments=tool_args,
                        ),
                    ),
                ],
            )
            context.append(tool_call_message)
            logger.info(f"Added tool call to context: {tool_call_message}")

            # Add response to context
            response_message = self._create_tool_response_message(tool_response)
            context.append(response_message)
            logger.info(f"Added tool response to context: {response_message}")

            # Create new stream
            new_stream = await self._create_chat_stream(context, tools)

            # Trigger typing indicator before starting new stream
            async with message.channel.typing():
                # Process the new stream
                async for status, content in self._process_stream(
                    new_stream, context, tools, message
                ):
                    yield status, content

        except Exception as e:
            logger.error(f"Error handling tool response: {str(e)}")

    async def _create_chat_stream(
        self,
        context: List[LLMMessage],
        tools: List[Dict[str, Any]],
    ) -> Iterator[ChatResponse]:
        """Create a new chat stream.

        Args:
            context: The conversation context
            tools: The available tools

        Returns:
            An async generator that yields chunks from the LLM
        """
        logger.info("Creating new chat stream")
        return self.api_client.chat(  # pyright: ignore
            model=str(config.MODEL_NAME),
            messages=context,
            stream=True,
            keep_alive=-1,
            options=config.load_model_options(),
            tools=tools,
        )

    async def _handle_tool_call(
        self,
        tool_call: Any,
        context: List[LLMMessage],
        tools: List[Dict[str, Any]],
        current_line: str,
        has_non_whitespace: bool,
        message: Message,
    ) -> AsyncGenerator[Tuple[LineStatus, str], None]:
        """Handle a tool call and yield appropriate status/content pairs.

        Args:
            tool_call: The tool call to handle
            context: The conversation context
            tools: The available tools
            current_line: The current line being built
            has_non_whitespace: Whether non-whitespace content has been seen
            message: The Discord message to respond to

        Yields:
            Tuples of (LineStatus, content)
        """
        if not hasattr(tool_call, "function"):
            return

        try:
            # Create and yield tool call data
            tool_data = self._create_tool_call_data(tool_call)
            # yield (LineStatus.TOOL_CALL, json.dumps(tool_data))

            # Handle tool response
            async for status, content in self._handle_tool_response(
                tool_data["name"],
                tool_data["args"],
                context,
                tools,
                message,
            ):
                yield status, content

        except Exception as e:
            logger.error(f"Error processing tool call: {str(e)}")

    async def _process_chunk(
        self,
        chunk: ChatResponse,
        context: List[LLMMessage],
        tools: List[Dict[str, Any]],
        current_line: str,
        has_non_whitespace: bool,
        message: Message,
    ) -> AsyncGenerator[Tuple[LineStatus, str, str, bool], None]:
        """Process a single chunk from the stream.

        Args:
            chunk: The chunk to process
            context: The conversation context
            tools: The available tools
            current_line: The current line being built
            has_non_whitespace: Whether non-whitespace content has been seen
            message: The Discord message to respond to

        Yields:
            Tuples of (status, content, new_current_line, new_has_non_whitespace)
        """
        # Handle tool calls
        if chunk.message.tool_calls:
            for tool_call in chunk.message.tool_calls:
                async for status, content in self._handle_tool_call(
                    tool_call, context, tools, current_line, has_non_whitespace, message
                ):
                    yield status, content, "", False

        # Handle regular content
        elif chunk.message.content:
            current_line, has_non_whitespace, yields = (
                await self._process_content_chunk(
                    chunk.message.content, current_line, has_non_whitespace
                )
            )
            for status, content in yields:
                yield status, content, current_line, has_non_whitespace

    async def _process_stream(
        self,
        stream: Iterator[ChatResponse],
        context: List[LLMMessage],
        tools: List[Dict[str, Any]],
        message: Message,
        current_line: str = "",
        has_non_whitespace: bool = False,
    ) -> AsyncIterator[Tuple[LineStatus, str]]:
        """Process a stream and yield appropriate status/content pairs.

        Args:
            stream: The stream to process
            context: The conversation context
            tools: The available tools
            message: The Discord message to respond to
            current_line: The current line being built
            has_non_whitespace: Whether non-whitespace content has been seen

        Yields:
            Tuples of (LineStatus, content)
        """
        try:
            for chunk in stream:
                try:
                    # Log chunk
                    logger.debug(f"Received chunk type: {type(chunk)}")
                    logger.debug(f"Received chunk: {chunk}")

                    # Process chunk
                    async for (
                        status,
                        content,
                        new_line,
                        new_whitespace,
                    ) in self._process_chunk(
                        chunk, context, tools, current_line, has_non_whitespace, message
                    ):
                        yield status, content
                        current_line = new_line
                        has_non_whitespace = new_whitespace

                except Exception as e:
                    logger.error(f"Error processing chunk: {str(e)}")

            # Handle any remaining text
            if current_line.strip():
                yield (LineStatus.COMPLETE, current_line)

        except Exception as e:
            logger.error(f"Error processing stream: {str(e)}")

    async def _stream_response_lines(
        self,
        context: List[LLMMessage],
        message: Message,
    ) -> AsyncGenerator[Tuple[LineStatus, str], None]:
        """Generator that yields line status and content as they stream in from the API.

        Args:
            context: The conversation context to send to the LLM
            message: The Discord message to respond to

        Yields:
            Tuples of (LineStatus, content) where:
            - LineStatus.ACCUMULATING indicates the start of a new line
            - LineStatus.COMPLETE indicates a complete line ready to send
            - LineStatus.TOOL_CALL indicates a tool call
        """
        try:
            # Log context and tools
            logger.info(
                f"Starting streaming response with {len(context)} context messages"
            )
            logger.info(
                f"Context: {json.dumps([m.model_dump(exclude_none=True) for m in context], indent=2)}"
            )

            # Get tools
            tools = self.tool_registry.get_tools()
            logger.info(
                f"Available tools: {[tool['function']['name'] for tool in tools]}"
            )

            # Create and process stream
            stream = await self._create_chat_stream(context, tools)
            async for status, content in self._process_stream(
                stream, context, tools, message
            ):
                yield status, content

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
                # Update message reactions
                await self._update_message_reactions(message)

                # Build context for LLM
                context = context_builder.build_context(
                    message_history, message.channel, message
                )

                # Process the response
                first_message = True
                line_count = 0

                async for status, line in self._stream_response_lines(context, message):
                    # Check if this specific task has been told to shut up
                    if current_task in self.shutup_tasks:
                        logger.info(f"Task was told to shut up, stopping response")
                        break

                    if status == LineStatus.ACCUMULATING:
                        logger.info(f"Accumulating line")
                        # Start typing indicator
                        async with message.channel.typing():
                            pass
                    elif status == LineStatus.COMPLETE and line.strip():
                        await self._handle_complete_message(
                            message, line, first_message, line_count
                        )
                        first_message = False
                        line_count += 1

                    # Check line limit
                    max_lines = config.load_model_options()["max_response_lines"]
                    if line_count >= max_lines:
                        logger.info("Reached maximum line limit, stopping response")
                        await message.channel.send(
                            "-# Response truncated due to length limit"
                        )
                        break

            except Exception as e:
                logger.error(f"Error sending messages: {str(e)}")
                await message.reply(f"-# Error sending messages: {str(e)}")
            finally:
                # Cleanup
                await self._cleanup_response(message, current_task)

    async def _update_message_reactions(self, message: Message) -> None:
        """Update message reactions at the start of processing.

        Args:
            message: The message to update reactions for
        """
        # Add typing reaction
        await message.add_reaction("⌨️")
        # Remove the thinking reaction
        try:
            await message.remove_reaction("💭", self.bot_user)
        except:
            # Reaction might not exist, that's okay
            pass

    async def _handle_complete_message(
        self,
        message: Message,
        line: str,
        first_message: bool,
        line_count: int,
    ) -> None:
        """Handle a complete message.

        Args:
            message: The message to respond to
            line: The message line
            first_message: Whether this is the first message
            line_count: The current line count
        """
        logger.info(f"Sending line: {line}")
        try:
            if first_message:
                # First message should be a reply
                await message.reply(line)
            else:
                await message.channel.send(line)

        except Exception as e:
            logger.warning(f"Failed to send message chunk: {str(e)}")

    async def _cleanup_response(
        self,
        message: Message,
        current_task: asyncio.Task[None],
    ) -> None:
        """Clean up after response processing.

        Args:
            message: The message that was processed
            current_task: The current task
        """
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
