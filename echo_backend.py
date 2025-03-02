import logging
import json
import time
from typing import List, Dict, Optional, Union, Any, Generator

class SSEEvent:
    """Simulated Server-Sent Event for streaming responses."""
    
    def __init__(self, data: str):
        self.data = data

class SSEClient:
    """Simulated SSE client for streaming responses."""
    
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.chunks = self._create_chunks()
    
    def _create_chunks(self) -> List[Dict[str, Any]]:
        """Split the response into chunks for streaming."""
        chunks = []
        # Split the response into words
        words = self.response_text.split()
        
        # Create chunks of 1-3 words
        current_chunk = []
        for word in words:
            current_chunk.append(word)
            
            # Randomly decide to send the chunk (simulate streaming)
            if len(current_chunk) >= 3 or (len(current_chunk) > 0 and len(chunks) % 2 == 0):
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "choices": [
                        {
                            "delta": {
                                "content": chunk_text + " "
                            }
                        }
                    ]
                })
                current_chunk = []
        
        # Add any remaining words
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append({
                "choices": [
                    {
                        "delta": {
                            "content": chunk_text
                        }
                    }
                ]
            })
        
        return chunks
    
    def events(self) -> Generator[SSEEvent, None, None]:
        """Generate SSE events from the chunks."""
        for chunk in self.chunks:
            # Simulate network delay
            time.sleep(0.2)
            yield SSEEvent(json.dumps(chunk))

class EchoClient:
    """Echo client that mimics the OpenAI API but just echoes back messages."""
    
    def __init__(self, base_url: str, api_key: str = "not-needed"):
        """Initialize the echo client."""
        self.base_url = base_url
        self.api_key = api_key
        self.logger = logging.getLogger("echo_client")
        self.logger.info("Echo client initialized (for testing)")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        seed: Optional[int] = None
    ) -> Union[str, SSEClient]:
        """
        Echo the last user message with some additional context.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model identifier to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate (optional)
            top_p: Top-p sampling parameter
            presence_penalty: Presence penalty parameter
            frequency_penalty: Frequency penalty parameter
            stop: Stop sequences
            stream: Whether to stream the response
            seed: Random seed for reproducibility
            
        Returns:
            Either a string response or an SSEClient for streaming
        """
        self.logger.info(f"Echo client received {len(messages)} messages")
        
        # Get the last user message directed to the bot
        last_user_message = None
        last_user_name = "User"
        
        # First, try to find a message that was explicitly directed to the bot
        for message in reversed(messages):
            if message["role"] == "user" and "[directed at bot]" in message["content"]:
                content = message["content"].replace(" [directed at bot]", "")
                if ":" in content:
                    parts = content.split(":", 1)
                    last_user_name = parts[0].strip()
                    last_user_message = parts[1].strip()
                else:
                    last_user_message = content
                break
        
        # If we didn't find a direct message, just use the last user message
        if last_user_message is None and len(messages) > 0:
            for message in reversed(messages):
                if message["role"] == "user":
                    content = message["content"].replace(" [directed at bot]", "")
                    if ":" in content:
                        parts = content.split(":", 1)
                        last_user_name = parts[0].strip()
                        last_user_message = parts[1].strip()
                    else:
                        last_user_message = content
                    break
        
        if not last_user_message:
            response = "I didn't receive any message to echo back."
        else:
            # Create a simple echo response
            response = f"ECHO: {last_user_message}\n\n(This is the echo backend for testing. No LLM is being used.)"
            
            # Count messages by type
            system_messages = sum(1 for m in messages if m["role"] == "system")
            user_messages = sum(1 for m in messages if m["role"] == "user")
            assistant_messages = sum(1 for m in messages if m["role"] == "assistant")
            directed_messages = sum(1 for m in messages if m["role"] == "user" and "[directed at bot]" in m["content"])
            
            # Add history info
            response += f"\n\nI have {len(messages)} messages in my history for this conversation:"
            response += f"\n- {system_messages} system messages"
            response += f"\n- {user_messages} user messages ({directed_messages} directed at me)"
            response += f"\n- {assistant_messages} assistant messages"
            
            # Add recent speakers
            recent_speakers = set()
            for message in messages:
                if message["role"] == "user" and ":" in message["content"]:
                    content = message["content"].replace(" [directed at bot]", "")
                    speaker = content.split(":", 1)[0].strip()
                    recent_speakers.add(speaker)
            
            if recent_speakers:
                response += f"\n\nRecent speakers: {', '.join(recent_speakers)}"
        
        if stream:
            return SSEClient(response)
        else:
            return response
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List fake models."""
        return {
            "data": [
                {
                    "id": "echo-model-small",
                    "object": "model",
                    "owned_by": "echo",
                    "permission": []
                },
                {
                    "id": "echo-model-medium",
                    "object": "model",
                    "owned_by": "echo",
                    "permission": []
                },
                {
                    "id": "echo-model-large",
                    "object": "model",
                    "owned_by": "echo",
                    "permission": []
                }
            ]
        } 