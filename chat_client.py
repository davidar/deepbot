#!/usr/bin/env python3

import sys
from typing import Dict, Iterator, List, Optional, TypedDict, cast

from llama_cpp import (
    ChatCompletionRequestMessage,
    CreateChatCompletionStreamResponse,
    Llama,
)
from rich.console import Console


class ChatChoice(TypedDict):
    delta: Dict[str, Optional[str]]


class ChatResponse(TypedDict):
    choices: List[ChatChoice]


def main() -> None:
    console = Console()
    console.print("[bold blue]Welcome to the Local Llama Chat Client![/bold blue]")
    console.print("Type your messages and press Enter to chat. Press Ctrl+C to exit.\n")

    repo_id = "bartowski/Mistral-Small-24B-Instruct-2501-GGUF"
    filename = "*Q4_K_M.gguf"

    # Initialize the model
    try:
        llm = Llama.from_pretrained(repo_id, filename)  # pyright: ignore
        console.print("[green]Model loaded successfully![/green]\n")
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        return

    messages: List[ChatCompletionRequestMessage] = []
    try:
        while True:
            # Get user input
            user_message = console.input("[bold green]You:[/bold green] ")

            # Add user message to history
            messages.append({"role": "user", "content": user_message})

            # Print assistant prefix
            console.print("[bold purple]Assistant:[/bold purple] ", end="")

            # Get and display response
            full_response = ""

            try:
                completion = cast(
                    Iterator[CreateChatCompletionStreamResponse],
                    llm.create_chat_completion(
                        messages=messages,
                        stream=True,
                        top_k=40,
                        top_p=0.95,
                        min_p=0.05,
                        temperature=0.8,
                        repeat_penalty=1.1,
                        max_tokens=2048,
                    ),
                )

                for chunk in completion:
                    content = chunk["choices"][0]["delta"].get("content")
                    if content:
                        content = str(content)
                        full_response += content
                        console.print(content, end="")
                        sys.stdout.flush()

            except KeyboardInterrupt:
                print("\nStreaming interrupted by user")
            finally:
                print()  # Add a newline at the end

            # Add assistant's response to message history
            messages.append({"role": "assistant", "content": full_response})

    except KeyboardInterrupt:
        console.print("\n[bold blue]Goodbye![/bold blue]")


if __name__ == "__main__":
    main()
