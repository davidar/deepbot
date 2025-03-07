import datetime
from typing import List

# File to store the system prompt
SYSTEM_PROMPT_FILE = "system_prompt.txt"


def load_system_prompt() -> List[str]:
    """Load the system prompt from file, or create with initial prompt if it doesn't exist."""
    try:
        with open(SYSTEM_PROMPT_FILE, "r") as f:
            data = f.read()
            return data.strip().split("\n")
    except Exception as e:
        print(f"Error loading system prompt: {e}")
        return []


def save_system_prompt(lines: List[str]) -> None:
    """Save the system prompt lines to file."""
    try:
        with open(SYSTEM_PROMPT_FILE, "w") as f:
            f.write("\n".join(lines) + "\n")
    except Exception as e:
        print(f"Error saving system prompt: {e}")


def add_line(line: str) -> List[str]:
    """Add a line to the system prompt and return the updated lines."""
    lines = load_system_prompt()
    if line not in lines:  # Avoid duplicates
        lines.append(line)
        save_system_prompt(lines)
    return lines


def remove_line(line: str) -> List[str]:
    """Remove a line from the system prompt and return the updated lines."""
    lines = load_system_prompt()
    if line in lines:
        lines.remove(line)
        save_system_prompt(lines)
    return lines


def get_system_prompt(server_name: str) -> str:
    """Get the complete system prompt as a string, formatted with the given kwargs."""
    prompt = "\n".join(load_system_prompt())
    current_time = datetime.datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")
    return f"# Discord Server: {server_name}\n# Current Time: {current_time}\n\n{prompt}"
