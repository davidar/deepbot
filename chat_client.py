#!/usr/bin/env python3

import sys
from typing import Dict, List

import ollama
from rich.console import Console


def main() -> None:
    console = Console()
    console.print("[bold blue]Welcome to the Ollama Chat Client![/bold blue]")
    console.print("Type your messages and press Enter to chat. Press Ctrl+C to exit.\n")

    model_name = "mistral-small"

    # Initialize the client
    client = ollama.Client(host="http://localhost:11434")

    try:
        # Test connection by getting model info
        client.show(model_name)
        console.print("[green]Connected to Ollama successfully![/green]\n")
    except Exception as e:
        console.print(f"[red]Failed to connect to Ollama: {e}[/red]")
        console.print(
            "[yellow]Make sure Ollama is running and the model is pulled using 'ollama pull {model_name}'[/yellow]"
        )
        return

    messages: List[Dict[str, str]] = []
    try:
        while True:
            # Get user input
            user_message = console.input("[bold green]You:[/bold green] ")

            # Add user message to history
            messages.append({"role": "user", "content": user_message})

            # Print assistant prefix
            console.print("[bold purple]Assistant:[/bold purple] ", end="")

            full_response = ""
            try:
                # Stream the response
                stream = client.chat(  # pyright: ignore
                    model=model_name,
                    messages=messages,
                    stream=True,
                )

                for chunk in stream:
                    content = chunk["message"]["content"]
                    if content:
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
