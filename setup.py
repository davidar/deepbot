import os
import platform
import subprocess
import sys
import venv
from typing import Literal

# Type hint for platform.system()
PlatformSystem = Literal["Windows", "Linux", "Darwin", "Java"]


def check_python_version(min_version: tuple[int, int, int] = (3, 8, 0)) -> bool:
    """Check if Python version meets minimum requirements."""
    current_version = sys.version_info[:3]
    return current_version >= min_version


def create_virtual_environment(venv_path: str) -> None:
    """Create a virtual environment at the specified path."""
    if os.path.exists(venv_path):
        print(f"Virtual environment already exists at {venv_path}")
        return

    print(f"Creating virtual environment at {venv_path}...")
    venv.create(venv_path, with_pip=True)
    print("Virtual environment created successfully.")


def get_venv_python_path(venv_path: str) -> str:
    """Get the path to the Python executable in the virtual environment."""
    system: PlatformSystem = platform.system()  # type: ignore[assignment]
    if system == "Windows":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:
        return os.path.join(venv_path, "bin", "python")


def install_dependencies(
    venv_python: str, requirements_file: str = "requirements.txt"
) -> None:
    """Install required dependencies in the virtual environment."""
    print("Installing dependencies in virtual environment...")
    try:
        subprocess.check_call(
            [venv_python, "-m", "pip", "install", "-r", requirements_file]
        )

        # Install PyNaCl for voice support (optional)
        print("Installing PyNaCl for voice support (optional)...")
        try:
            subprocess.check_call([venv_python, "-m", "pip", "install", "PyNaCl"])
            print("PyNaCl installed successfully.")
        except subprocess.CalledProcessError:
            print(
                "Warning: Failed to install PyNaCl. Voice support will not be available."
            )

        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print("Error: Failed to install dependencies.")
        sys.exit(1)


def setup_env_file(env_template: str = ".env.template", env_file: str = ".env") -> None:
    """Set up the environment file from template if it doesn't exist."""
    if not os.path.exists(env_file) and os.path.exists(env_template):
        print(f"Creating {env_file} from {env_template}...")
        with open(env_template, "r") as template, open(env_file, "w") as env:
            env.write(template.read())


def setup_systemd_service() -> None:
    """Set up the systemd service for the bot."""
    if platform.system() != "Linux":
        print("Skipping systemd service setup - not on Linux")
        return

    # Create service file content with the current directory
    current_dir = os.path.abspath(os.path.dirname(__file__))
    service_content = f"""[Unit]
Description=DeepBot Discord Bot with Local LLM Integration
After=network.target

[Service]
Type=simple
WorkingDirectory={current_dir}
Environment=PYTHONUNBUFFERED=1
ExecStart={current_dir}/venv/bin/python bot.py
Restart=on-failure
RestartSec=5s

[Install]
WantedBy=default.target
"""

    # Ensure systemd user directory exists
    systemd_dir = os.path.expanduser("~/.config/systemd/user")
    os.makedirs(systemd_dir, exist_ok=True)

    # Write service file
    service_path = os.path.join(systemd_dir, "deepbot.service")
    with open(service_path, "w") as f:
        f.write(service_content)

    print(f"Created systemd service file at {service_path}")
    print("To manage the service, use:")
    print("  systemctl --user enable deepbot.service  # Enable on startup")
    print("  systemctl --user start deepbot.service   # Start the service")
    print("  systemctl --user stop deepbot.service    # Stop the service")
    print("  systemctl --user restart deepbot.service # Restart the service")
    print("  systemctl --user status deepbot.service  # Check service status")


def main() -> None:
    """Main setup function."""
    print("Setting up DeepBot...")

    # Check Python version
    if not check_python_version():
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

    # Create virtual environment
    venv_path = "venv"
    create_virtual_environment(venv_path)
    venv_python = get_venv_python_path(venv_path)

    # Install dependencies in virtual environment
    install_dependencies(venv_python)

    # Set up environment file
    setup_env_file()

    # Set up systemd service
    setup_systemd_service()

    print("\nSetup complete!")
    print("Next steps:")
    print("1. Edit the .env file with your Discord bot token")
    print("2. Activate the virtual environment")
    print("3. Make sure ollama is running with the API enabled")
    print("4. Run the bot with: python bot.py")
    print("   Or use systemd to manage the service (see instructions above)")


if __name__ == "__main__":
    main()
