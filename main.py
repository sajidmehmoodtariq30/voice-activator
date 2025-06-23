"""
Voice Activator System
A secure voice-authenticated system controller

This is the main entry point for the Voice Activator System.
Professional developers always start with clear entry points and documentation.
"""

import sys
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from core.engine import VoiceActivatorEngine
from core.config import ConfigManager
from core.logger import setup_logging


def main():
    """
    Main entry point for the Voice Activator System
    
    This function demonstrates professional application structure:
    1. Configuration loading
    2. Logging setup  
    3. Error handling
    4. Clean shutdown
    """
    try:
        # Load configuration
        config = ConfigManager()
        
        # Setup logging
        setup_logging(config.get('system.log_level', 'INFO'))
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Voice Activator System...")
        
        # Initialize and start the main engine
        engine = VoiceActivatorEngine(config)
        engine.start()
        
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
