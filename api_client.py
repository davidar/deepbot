import json
import logging
from typing import (
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Optional,
    Protocol,
    TypeVar,
    Union,
    cast,
)

import requests
import sseclient  # type: ignore[import-untyped]
from requests import Response


# Protocol for SSE client
class SSEClient(Protocol):
    def events(self) -> Iterator[Any]: ...
    def close(self) -> None: ...


# Type variable for response data
ResponseData = TypeVar("ResponseData", bound=Dict[str, Any])

# Type for API payload
PayloadDict = Dict[
    str,
    Union[
        str,
        List[Dict[str, str]],
        float,
        bool,
        Optional[int],
        Optional[Union[str, List[str]]],
    ],
]

# Type for SSE response
SSEResponse = Union[str, SSEClient]


# Helper function to create SSE client
def create_sse_client(response: Response) -> SSEClient:
    """Create an SSE client from a response object."""

    # Create a generator that yields bytes from the response
    def generate_bytes() -> Generator[bytes, None, None]:
        for chunk in response.iter_content(chunk_size=None):
            if chunk:
                yield chunk

    # Create SSE client from the generator
    return cast(SSEClient, sseclient.SSEClient(generate_bytes()))


class OpenAICompatibleClient:
    """Client for interacting with OpenAI-compatible API endpoints."""

    def __init__(self, base_url: str, api_key: str = "not-needed"):
        """
        Initialize the API client.

        Args:
            base_url: Base URL of the API (e.g. http://localhost:1234/v1)
            api_key: API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.logger = logging.getLogger("openai_compatible_client")
        self.logger.info(f"Client initialized with base URL: {base_url}")

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
        seed: Optional[int] = None,
    ) -> SSEResponse:
        """
        Generate a chat completion using the API.

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
        # Prepare the request payload
        payload: PayloadDict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stream": stream,
        }

        # Add optional parameters only if they are provided
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Log the request details
        self.logger.info("=== API Request ===")
        self.logger.info(f"URL: {self.base_url}/chat/completions")
        self.logger.info(f"Headers: {headers}")
        self.logger.info(f"API Request payload: {json.dumps(payload, indent=2)}")

        try:
            if stream:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    stream=True,
                )
                response.raise_for_status()
                return create_sse_client(response)
            else:
                response = requests.post(
                    f"{self.base_url}/chat/completions", json=payload, headers=headers
                )
                response.raise_for_status()

                response_data = response.json()
                result = str(response_data["choices"][0]["message"]["content"])

                self.logger.debug("=== API Response ===")
                self.logger.debug(f"Response: {result[:200]}...")

                return result

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise Exception(f"API request failed: {str(e)}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to parse API response: {str(e)}")
            raise Exception(f"Failed to parse API response: {str(e)}")

    def list_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List available models from the API.

        Returns:
            Dictionary containing list of model information dictionaries
        """
        response = requests.get(
            f"{self.base_url}/models",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status code {response.status_code}: {response.text}"
            )

        data = cast(Dict[str, List[Dict[str, Any]]], response.json())
        return data

    def create_embeddings(
        self, input: Union[str, List[str]], model: str = "default"
    ) -> List[List[float]]:
        """
        Generate embeddings for the given text(s).

        Args:
            input: Text or list of texts to generate embeddings for
            model: Model to use for embeddings

        Returns:
            List of embedding vectors
        """
        if isinstance(input, str):
            input = [input]

        response = requests.post(
            f"{self.base_url}/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            json={"input": input, "model": model},
        )

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status code {response.status_code}: {response.text}"
            )

        data = response.json()
        return [item["embedding"] for item in data["data"]]

    def create_completion(
        self,
        prompt: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 0.9,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        stop: Optional[Union[str, List[str]]] = None,
        stream: bool = False,
        seed: Optional[int] = None,
    ) -> SSEResponse:
        """
        Generate a text completion using the API.

        Args:
            prompt: Text prompt to complete
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
        payload: PayloadDict = {
            "model": model,
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "stream": stream,
        }

        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            if stream:
                response = requests.post(
                    f"{self.base_url}/completions",
                    json=payload,
                    headers=headers,
                    stream=True,
                )
                response.raise_for_status()
                return create_sse_client(response)
            else:
                response = requests.post(
                    f"{self.base_url}/completions", json=payload, headers=headers
                )
                response.raise_for_status()

                response_data = response.json()
                result = str(response_data["choices"][0]["text"])

                return result

        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {str(e)}")
            raise Exception(f"API request failed: {str(e)}")
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            self.logger.error(f"Failed to parse API response: {str(e)}")
            raise Exception(f"Failed to parse API response: {str(e)}")
