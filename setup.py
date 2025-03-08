import os
import platform
import subprocess
import sys
import venv
from pathlib import Path
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


def create_activation_scripts(venv_path: str) -> None:
    """Create activation scripts for different shells."""
    print("Creating activation scripts...")
    scripts_dir = Path(venv_path) / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    # Create activate.sh for bash/zsh
    activate_sh = scripts_dir / "activate.sh"
    activate_sh.write_text(f'source "{venv_path}/bin/activate"\n')

    # Create activate.ps1 for PowerShell
    activate_ps1 = scripts_dir / "activate.ps1"
    activate_ps1.write_text(f'& "{venv_path}\\Scripts\\Activate.ps1"\n')

    # Create activate.bat for Windows Command Prompt
    activate_bat = scripts_dir / "activate.bat"
    activate_bat.write_text(f'@echo off\ncall "{venv_path}\\Scripts\\activate.bat"\n')

    # Make the shell script executable on Unix-like systems
    system: PlatformSystem = platform.system()  # type: ignore[assignment]
    if system != "Windows":
        os.chmod("activate.sh", 0o755)

    print("Created activation scripts: activate.bat and activate.sh")


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

    # Create activation scripts
    create_activation_scripts(venv_path)

    # Set up environment file
    setup_env_file()

    print("\nSetup complete!")
    print("Next steps:")
    print("1. Edit the .env file with your Discord bot token")
    print("2. Activate the virtual environment:")
    system: PlatformSystem = platform.system()  # type: ignore[assignment]
    if system == "Windows":
        print("   - Run: activate.bat")
    else:
        print("   - Run: source activate.sh")
    print("3. Make sure LM Studio is running with the API enabled")
    print("4. Run the bot with: python main.py")


if __name__ == "__main__":
    main()
