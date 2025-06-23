"""
Development Setup Script

This script helps you set up the development environment.
Professional developers automate environment setup.
"""

import subprocess
import sys
import platform
from pathlib import Path


def check_python_version():
    """Ensure we have Python 3.8+"""
    if sys.version_info < (3, 8):
        print("ERROR: Python 3.8 or higher is required")
        return False
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True


def check_windows():
    """Ensure we're on Windows (required for system integration)"""
    if platform.system() != "Windows":
        print("WARNING: This system is designed for Windows")
        return False
    print(f"✓ Windows {platform.release()}")
    return True


def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✓ Dependencies installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "models", 
        "temp"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
    print("✓ Directories created")


def main():
    """Main setup function"""
    print("=== Voice Activator Development Setup ===\n")
    
    # Check requirements
    if not check_python_version():
        return 1
    
    if not check_windows():
        print("Continuing anyway, but some features may not work...")
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        return 1
    
    print("\n=== Setup Complete! ===")
    print("\nNext steps:")
    print("1. Run: python main.py")
    print("2. Follow the voice training prompts")
    print("3. Test the system")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
