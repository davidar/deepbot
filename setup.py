import os
import sys
import subprocess
import shutil
import venv
import platform

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        sys.exit(1)

def create_virtual_environment():
    """Create a virtual environment for the bot."""
    venv_dir = "venv"
    
    if os.path.exists(venv_dir):
        print(f"Virtual environment already exists at {venv_dir}")
        return venv_dir
    
    print(f"Creating virtual environment at {venv_dir}...")
    venv.create(venv_dir, with_pip=True)
    print("Virtual environment created successfully.")
    
    return venv_dir

def get_venv_python_path(venv_dir):
    """Get the path to the Python executable in the virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        return os.path.join(venv_dir, "bin", "python")

def install_dependencies(venv_python):
    """Install required dependencies in the virtual environment."""
    print("Installing dependencies in virtual environment...")
    try:
        subprocess.check_call([venv_python, "-m", "pip", "install", "-r", "requirements.txt"])
        
        # Install PyNaCl for voice support (optional)
        print("Installing PyNaCl for voice support (optional)...")
        try:
            subprocess.check_call([venv_python, "-m", "pip", "install", "PyNaCl"])
            print("PyNaCl installed successfully.")
        except subprocess.CalledProcessError:
            print("Warning: Failed to install PyNaCl. Voice support will not be available.")
        
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print("Error: Failed to install dependencies.")
        sys.exit(1)

def setup_env_file():
    """Set up the .env file if it doesn't exist."""
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            shutil.copy(".env.example", ".env")
            print("Created .env file from .env.example. Please edit it with your Discord token.")
        else:
            print("Error: .env.example file not found.")
            sys.exit(1)
    else:
        print(".env file already exists.")

def create_activation_scripts(venv_dir):
    """Create activation scripts for different platforms."""
    # Create activate.bat for Windows
    with open("activate.bat", "w") as f:
        f.write(f"@echo off\n")
        f.write(f"call {venv_dir}\\Scripts\\activate.bat\n")
    
    # Create activate.sh for Unix-like systems
    with open("activate.sh", "w") as f:
        f.write(f"#!/bin/bash\n")
        f.write(f"source {venv_dir}/bin/activate\n")
    
    # Make the shell script executable on Unix-like systems
    if platform.system() != "Windows":
        os.chmod("activate.sh", 0o755)
    
    print("Created activation scripts: activate.bat and activate.sh")

def main():
    """Main setup function."""
    print("Setting up DeepBot...")
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    venv_dir = create_virtual_environment()
    venv_python = get_venv_python_path(venv_dir)
    
    # Install dependencies in virtual environment
    install_dependencies(venv_python)
    
    # Create activation scripts
    create_activation_scripts(venv_dir)
    
    # Set up .env file
    setup_env_file()
    
    print("\nSetup complete!")
    print("Next steps:")
    print("1. Edit the .env file with your Discord bot token")
    print("2. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   - Run: activate.bat")
    else:
        print("   - Run: source activate.sh")
    print("3. Make sure Oobabooga Text Generation WebUI is running with the API enabled")
    print("   - Or use the echo backend for testing: python run.py --echo")
    print("4. Run the bot with: python run.py")

if __name__ == "__main__":
    main() 