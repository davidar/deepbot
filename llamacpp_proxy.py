# llamacpp_openai_proxy.py
import json
import logging
import os
import sys
from typing import Any, AsyncGenerator, Union

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from starlette.responses import StreamingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Configuration
LLAMACPP_SERVER_URL = os.environ.get("LLAMACPP_SERVER_URL")
HF_TOKEN = os.environ.get("HF_TOKEN")
DEFAULT_REPETITION_PENALTY = 1.1

# HTTP client for requests to llama.cpp server
http_client = httpx.AsyncClient(timeout=60.0)


@app.post("/v1/completions", response_model=None)
async def proxy_completions(
    request: Request,
) -> Union[StreamingResponse, dict[str, Any]]:
    """Proxy for the /v1/completions endpoint that transforms OpenAI API requests to llama.cpp format."""
    # Get request body
    body = await request.json()

    # Extract OpenAI params
    prompt = body.get("prompt", "")
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 0.9)
    max_tokens = body.get("max_tokens", 1000)
    stream = body.get("stream", False)

    # Get repetition_penalty from either:
    # 1. Custom field in request body
    # 2. Default config value
    repetition_penalty = body.get("repetition_penalty", DEFAULT_REPETITION_PENALTY)

    # Handle based on streaming preference
    if stream:
        return await handle_streaming(
            prompt, temperature, top_p, max_tokens, repetition_penalty
        )
    else:
        return await handle_regular(
            prompt, temperature, top_p, max_tokens, repetition_penalty
        )


async def handle_regular(
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
) -> dict[str, Any]:
    """Handle regular (non-streaming) completions requests."""
    # Prepare headers with authentication
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {HF_TOKEN}",
    }

    # Prepare payload for llama.cpp server
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,  # llama.cpp parameter for max_tokens
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repetition_penalty,  # Native llama.cpp parameter
        "stream": False,
    }

    # Log the request parameters
    logger.info("Sending regular completion request to llama.cpp server:")
    logger.info(f"Prompt: {prompt}")
    logger.info(
        f"Parameters: temperature={temperature}, top_p={top_p}, "
        f"n_predict={max_tokens}, repeat_penalty={repetition_penalty}"
    )

    try:
        # Call llama.cpp server API directly
        async with http_client.stream(
            "POST", f"{LLAMACPP_SERVER_URL}/completion", json=payload, headers=headers
        ) as response:
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response headers: {response.headers}")

            if response.status_code != 200:
                error_text = await response.aread()
                error_str = error_text.decode()
                logger.error(f"Error response from llama.cpp server: {error_str}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Error from llama.cpp server: {error_str}",
                )

            # Read the full response content first
            content = await response.aread()
            logger.debug(f"Raw response content: {content.decode()}")

            # Parse the JSON response
            response_data = json.loads(content.decode())
            logger.info(f"Parsed response data: {response_data}")
            generated_text = response_data.get("content", "")

            # Log the response
            logger.info(f"Received response from llama.cpp server: {generated_text}")

            # Transform response to OpenAI format
            return {
                "id": "cmpl-llamacpp",
                "object": "text_completion",
                "created": 0,
                "model": "llamacpp-proxy",
                "choices": [
                    {
                        "text": generated_text,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": len(str(prompt)) // 4,  # Rough approximation
                    "completion_tokens": len(str(generated_text))
                    // 4,  # Rough approximation
                    "total_tokens": (len(str(prompt)) + len(str(generated_text)))
                    // 4,  # Rough approximation
                },
            }
    except Exception as e:
        logger.error(f"Exception during request handling: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error from llama.cpp server: {str(e)}"
        )


async def handle_streaming(
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
) -> StreamingResponse:
    """Handle streaming completions requests."""

    async def stream_generator() -> AsyncGenerator[str, None]:
        # Prepare headers with authentication
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HF_TOKEN}",
        }

        # Prepare payload for llama.cpp server
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,  # llama.cpp parameter for max_tokens
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repetition_penalty,  # Native llama.cpp parameter
            "stream": True,
        }

        # Log the request parameters
        logger.info("Sending streaming completion request to llama.cpp server:")
        logger.info(f"Prompt: {prompt}")
        logger.info(
            f"Parameters: temperature={temperature}, top_p={top_p}, "
            f"n_predict={max_tokens}, repeat_penalty={repetition_penalty}"
        )

        try:
            # Stream from the llama.cpp server
            async with http_client.stream(
                "POST",
                f"{LLAMACPP_SERVER_URL}/completion",
                json=payload,
                headers=headers,
            ) as response:
                logger.info(f"Stream response status code: {response.status_code}")
                logger.info(f"Stream response headers: {response.headers}")

                if response.status_code != 200:
                    error_text = await response.aread()
                    error_str = error_text.decode()
                    logger.error(
                        f"Error response from llama.cpp server during streaming: {error_str}"
                    )
                    error_data = {"error": f"Error from llama.cpp server: {error_str}"}
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                # Process the streaming response
                async for line in response.aiter_lines():
                    logger.debug(f"Received raw line: {line}")
                    if line.startswith("data: "):
                        json_str = line[6:]  # Remove "data: " prefix

                        # Handle [DONE] marker from llama.cpp
                        if json_str.strip() == "[DONE]":
                            logger.debug("Received [DONE] marker")
                            yield "data: [DONE]\n\n"
                            break

                        try:
                            data = json.loads(json_str)
                            logger.debug(f"Parsed JSON data: {data}")
                            chunk = data.get("content", "")

                            # Log each chunk
                            logger.debug(
                                f"Received chunk from llama.cpp server: {chunk}"
                            )

                            # Format chunk in OpenAI SSE format
                            openai_data = {
                                "id": "cmpl-llamacpp",
                                "object": "text_completion",
                                "created": 0,
                                "model": "llamacpp-proxy",
                                "choices": [
                                    {
                                        "text": chunk,
                                        "index": 0,
                                        "logprobs": None,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(openai_data)}\n\n"
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON from line: {line}")

            # Send final "done" message if not already sent
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Exception during streaming: {str(e)}", exc_info=True)
            error_data = {"error": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


# Additional endpoints that may be needed for OpenAI compatibility
@app.get("/v1/models")
async def list_models() -> dict[str, Any]:
    """Return a simple model list for compatibility."""
    return {
        "object": "list",
        "data": [
            {
                "id": "llamacpp-proxy",
                "object": "model",
                "created": 0,
                "owned_by": "proxy",
            }
        ],
    }


if __name__ == "__main__":
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set.")
        print("Please set your Hugging Face API token in the .env file or environment.")
        sys.exit(1)

    print(
        f"Starting proxy server. Forwarding requests to llama.cpp server at {LLAMACPP_SERVER_URL}"
    )
    print(f"Default repetition_penalty: {DEFAULT_REPETITION_PENALTY}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
