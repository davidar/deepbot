"""LLM streaming response handling."""

import asyncio
import json
import logging
import traceback
from typing import Any, Dict, List, Optional, Set

import ollama
import pendulum
from discord import ClientUser, Message
from discord.errors import Forbidden, NotFound
from ollama import Message as LLMMessage

import config
from context_builder import ContextBuilder
from tool_messages import format_tool_call_and_response
from tools import ToolDefinition, tool_registry

# Set up logging
logger = logging.getLogger("deepbot.llm")


class LLMResponseHandler:
    """Handles responses from the LLM."""

    def __init__(self, api_client: ollama.AsyncClient, bot_user: ClientUser) -> None:
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
        # Dictionary to store reminder metadata keyed by message ID
        self.reminder_metadata: Dict[int, Dict[str, Any]] = {}

    async def _handle_tool_call(
        self,
        tool_call: LLMMessage.ToolCall,
        context: List[LLMMessage],
        tools: List[ToolDefinition],
        message: Message,
    ) -> Optional[str]:
        """Handle a tool call and return the response.

        Args:
            tool_call: The tool call to handle
            context: The conversation context
            tools: The available tools
            message: The Discord message to respond to

        Returns:
            The tool response
        """
        if not hasattr(tool_call, "function"):
            return None

        try:
            # Get tool data
            function_name = tool_call.function.name
            function_args = tool_call.function.arguments
            tool_args = (
                function_args
                if isinstance(function_args, dict)
                else json.loads(str(function_args))
            )

            # Get and log tool response
            tool_response = await self.tool_registry.call_tool(
                function_name, tool_args, message
            )
            logger.info(f"Tool response: {tool_response}")

            # Format and send tool call/response
            combined_message = format_tool_call_and_response(
                function_name, tool_args, tool_response
            )
            await message.channel.send(combined_message)

            # Add tool call and response to context
            tool_call_message = LLMMessage(
                role="assistant",
                content="",
                tool_calls=[tool_call],
            )
            context.append(tool_call_message)
            context.append(LLMMessage(role="tool", content=tool_response))

            # Get new response after tool call
            response = await self.api_client.chat(  # pyright: ignore
                model=str(config.MODEL_NAME),
                messages=context,
                stream=False,
                keep_alive=-1,
                options=config.get_ollama_options(),
            )
            return response.message.content

        except Exception as e:
            logger.error(f"Error handling tool call: {str(e)}")
            return str(e)

    async def _get_response(
        self,
        context: List[LLMMessage],
        message: Message,
    ) -> Optional[str]:
        """Get a response from the LLM.

        Args:
            context: The conversation context to send to the LLM
            message: The Discord message to respond to

        Returns:
            The LLM's response
        """
        try:
            # Log context and tools
            logger.info(f"Starting response with {len(context)} context messages")
            tools = self.tool_registry.get_tools()
            logger.info(
                f"Available tools: {[tool['function']['name'] for tool in tools]}"
            )

            # Get response
            response = await self.api_client.chat(  # pyright: ignore
                model=str(config.MODEL_NAME),
                messages=context,
                stream=False,
                keep_alive=-1,
                options=config.get_ollama_options(),
                tools=tools,
            )

            # Handle tool calls if any
            if response.message.tool_calls:
                for tool_call in response.message.tool_calls:
                    return await self._handle_tool_call(
                        tool_call, context, tools, message
                    )

            return response.message.content

        except Exception as e:
            error_message = f"-# Error getting response: {str(e)}"
            logger.error(error_message)
            logger.error(f"Full error details: {repr(e)}")
            logger.error(f"Stack trace:\n{traceback.format_exc()}")
            return error_message

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
        """Handle a response by getting the complete response and sending it.

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
                context = await context_builder.build_context(
                    message_history, message.channel, message
                )

                # Inject reminder context if this is a reminder
                context = self._inject_reminder_context(context, message)

                # Get response
                response = await self._get_response(context, message)

                # Check if this specific task has been told to shut up
                if current_task in self.shutup_tasks:
                    logger.info("Task was told to shut up, stopping response")
                    return

                # Send response
                if response is not None and response.strip():
                    await message.reply(response)

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
        except (NotFound, Forbidden):
            # Reaction might not exist or we might not have permission
            pass

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
        except (NotFound, Forbidden):
            # Reaction might not exist or we might not have permission
            pass

        # Clean up any reminder metadata that might still exist
        try:
            if message.id in self.reminder_metadata:
                del self.reminder_metadata[message.id]
                logger.info(f"Cleaned up reminder metadata for message {message.id}")
        except Exception as e:
            logger.warning(f"Error cleaning up reminder metadata: {str(e)}")

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

    def add_reminder_to_queue(
        self,
        channel_id: int,
        original_message: Message,
        reminder_content: str,
    ) -> None:
        """Add a reminder message to the response queue.

        Args:
            channel_id: The Discord channel ID
            original_message: The original message that set the reminder
            reminder_content: The content of the reminder
        """
        if channel_id not in self.response_queues:
            self.response_queues[channel_id] = asyncio.Queue()

        # Store reminder metadata in our dictionary
        self.reminder_metadata[original_message.id] = {
            "is_reminder": True,
            "content": reminder_content,
            "triggered_at": pendulum.now("UTC").isoformat(),
            "user_id": original_message.author.id,
            "user_name": original_message.author.display_name,
        }

        # Add to queue
        self.response_queues[channel_id].put_nowait(original_message)
        logger.info(f"Added reminder for message {original_message.id} to LLM queue")

    def _inject_reminder_context(
        self,
        context: List[LLMMessage],
        message: Message,
    ) -> List[LLMMessage]:
        """Inject reminder context into the message list.

        This adds a tool call and response pair to the context to indicate
        that a reminder has been triggered, which helps the LLM understand
        how to respond appropriately.

        Args:
            context: The existing context list
            message: The message with reminder metadata

        Returns:
            The updated context list with reminder information
        """
        # Check if this message has reminder metadata
        reminder_metadata = self.reminder_metadata.get(message.id)
        if not reminder_metadata:
            return context

        # Get reminder details
        reminder_content = reminder_metadata.get("content", "")
        triggered_at = reminder_metadata.get(
            "triggered_at", pendulum.now("UTC").isoformat()
        )

        # Create a tool call message for the reminder
        tool_name = "schedule_reminder"
        tool_args = {
            "content": reminder_content,
            "time": triggered_at,
        }

        # Add tool call to context
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
        logger.info(f"Added reminder tool call to context: {tool_call_message}")

        # Create tool response message
        tool_response = "\n".join(
            [
                f"I'll remind you about '{reminder_content}' at {triggered_at}.",
                f"Waiting until {triggered_at} ...",
                f"Waiting complete. The time is now {triggered_at}.",
                (
                    f"Reminder triggered. You should now respond to the user's original message, "
                    f"to inform them that it is now time to '{reminder_content}'."
                ),
            ]
        )

        # Add response to context
        response_message = LLMMessage(role="tool", content=tool_response)
        context.append(response_message)
        logger.info(f"Added reminder tool response to context: {response_message}")

        # Clean up the metadata after using it
        try:
            del self.reminder_metadata[message.id]
        except KeyError:
            # The metadata might have already been removed
            logger.warning(
                f"Reminder metadata for message {message.id} already removed"
            )

        return context

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

        # Clear the response queue and clean up any associated reminder metadata
        if channel_id in self.response_queues:
            # Get all messages in the queue and clean up their metadata
            messages_to_clean: List[Message] = []
            while not self.response_queues[channel_id].empty():
                try:
                    message = self.response_queues[channel_id].get_nowait()
                    messages_to_clean.append(message)
                except asyncio.QueueEmpty:
                    break

            # Clean up reminder metadata for all messages
            for message in messages_to_clean:
                try:
                    if message.id in self.reminder_metadata:
                        del self.reminder_metadata[message.id]
                        logger.info(
                            f"Cleaned up reminder metadata for message {message.id} during stop"
                        )
                except Exception as e:
                    logger.warning(
                        f"Error cleaning up reminder metadata during stop: {str(e)}"
                    )
