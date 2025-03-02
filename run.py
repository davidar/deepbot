import os
import sys
import subprocess
import argparse

def check_env_file():
    """Check if .env file exists and has required variables."""
    if not os.path.exists(".env"):
        print("Error: .env file not found. Please run setup.py first.")
        sys.exit(1)
    
    with open(".env", "r") as f:
        env_content = f.read()
    
    if "DISCORD_TOKEN=" not in env_content or "DISCORD_TOKEN=your_discord_bot_token_here" in env_content:
        print("Error: DISCORD_TOKEN not set in .env file. Please edit the .env file.")
        sys.exit(1)

def check_venv():
    """Check if running in a virtual environment."""
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("Warning: Not running in a virtual environment. It's recommended to activate the virtual environment first.")
        print("- Windows: activate.bat")
        print("- Unix/Linux/Mac: source activate.sh")
        
        # Ask user if they want to continue
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)

def run_bot(echo=False):
    """Run the bot."""
    # Set environment variable for echo mode
    if echo:
        os.environ["USE_ECHO_BACKEND"] = "1"
        print("Using echo backend for testing (no LLM connection required)")
    else:
        os.environ.pop("USE_ECHO_BACKEND", None)
    
    try:
        # Use the unified bot script
        cmd = [sys.executable, "unified_bot.py"]
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except subprocess.CalledProcessError as e:
        print(f"Error running bot: {e}")
        sys.exit(1)

def main():
    """Main function to parse arguments and run the bot."""
    parser = argparse.ArgumentParser(description="Run the DeepBot Discord bot")
    parser.add_argument("--setup", action="store_true", help="Run the setup script first")
    parser.add_argument("--echo", action="store_true", help="Use echo backend instead of LLM API")
    parser.add_argument("--skip-venv-check", action="store_true", help="Skip virtual environment check")
    
    args = parser.parse_args()
    
    if args.setup:
        print("Running setup script...")
        subprocess.run([sys.executable, "setup.py"], check=True)
    
    check_env_file()
    
    if not args.skip_venv_check:
        check_venv()
    
    backend = "echo" if args.echo else "LLM API"
    print(f"Starting DeepBot using {backend} backend...")
    
    run_bot(echo=args.echo)

if __name__ == "__main__":
    main() 