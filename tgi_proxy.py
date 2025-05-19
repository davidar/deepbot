# tgi_completions_proxy.py
import json
import logging
import os
import sys
from typing import Any, AsyncGenerator, Union

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from huggingface_hub import InferenceClient
from starlette.responses import StreamingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# Configuration
TGI_API_URL = os.environ.get("TGI_API_URL")
HF_TOKEN = os.environ.get("HF_TOKEN")
DEFAULT_REPETITION_PENALTY = 1.1

# HTTP client for non-HF requests
http_client = httpx.AsyncClient(timeout=60.0)


@app.post("/v1/completions", response_model=None)
async def proxy_completions(
    request: Request,
) -> Union[StreamingResponse, dict[str, Any]]:
    """Proxy for the /v1/completions endpoint that Loomsidian uses."""
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
    # Create HF Inference Client
    inference_client = InferenceClient(model=TGI_API_URL, token=HF_TOKEN)

    # Log the request parameters
    logger.info("Sending regular completion request to TGI:")
    logger.info(f"Prompt: {prompt}")
    logger.info(
        f"Parameters: temperature={temperature}, top_p={top_p}, "
        f"max_tokens={max_tokens}, repetition_penalty={repetition_penalty}, "
        f"do_sample={bool(temperature > 0)}"
    )

    try:
        # Call TGI API using the HF client
        response: str = inference_client.text_generation(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=bool(temperature > 0),
        )

        # Log the response
        logger.info(f"Received response from TGI: {response}")

        # Transform response to OpenAI format
        return {
            "id": "cmpl-proxy",
            "object": "text_completion",
            "created": 0,
            "model": "tgi-proxy",
            "choices": [
                {
                    "text": response,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(str(prompt)) // 4,  # Rough approximation
                "completion_tokens": len(str(response)) // 4,  # Rough approximation
                "total_tokens": (len(str(prompt)) + len(str(response)))
                // 4,  # Rough approximation
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error from TGI: {str(e)}")


async def handle_streaming(
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    repetition_penalty: float,
) -> StreamingResponse:
    """Handle streaming completions requests."""

    async def stream_generator() -> AsyncGenerator[str, None]:
        # Create HF Inference Client
        inference_client = InferenceClient(model=TGI_API_URL, token=HF_TOKEN)

        # Log the request parameters
        logger.info("Sending streaming completion request to TGI:")
        logger.info(f"Prompt: {prompt}")
        logger.info(
            f"Parameters: temperature={temperature}, top_p={top_p}, "
            f"max_tokens={max_tokens}, repetition_penalty={repetition_penalty}, "
            f"do_sample={bool(temperature > 0)}"
        )

        try:
            # Use the generator from the HF client
            for chunk in inference_client.text_generation(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=bool(temperature > 0),
                stream=True,
            ):
                # Log each chunk
                logger.debug(f"Received chunk from TGI: {chunk}")
                # Format chunk in OpenAI SSE format
                data = {
                    "id": "cmpl-proxy",
                    "object": "text_completion",
                    "created": 0,
                    "model": "tgi-proxy",
                    "choices": [
                        {
                            "text": chunk,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
                yield f"data: {json.dumps(data)}\n\n"

            # Send final "done" message
            yield "data: [DONE]\n\n"

        except Exception as e:
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
            {"id": "tgi-proxy", "object": "model", "created": 0, "owned_by": "proxy"}
        ],
    }


if __name__ == "__main__":
    if not HF_TOKEN:
        print("Error: HF_TOKEN environment variable not set.")
        print("Please set your Hugging Face API token in the .env file or environment.")
        sys.exit(1)

    print(f"Starting proxy server. Forwarding requests to TGI at {TGI_API_URL}")
    print(f"Default repetition_penalty: {DEFAULT_REPETITION_PENALTY}")
    uvicorn.run(app, host="0.0.0.0", port=8000)
