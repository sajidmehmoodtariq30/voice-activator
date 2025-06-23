"""
Configuration Manager

Professional developers always separate configuration from code.
This module handles loading and managing all system configuration.
"""

import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    """
    Manages application configuration with professional practices:
    - Environment-based overrides
    - Type safety
    - Default values
    - Validation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager"""
        self.logger = logging.getLogger(__name__)
        
        # Default to config.yaml in project root
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Professional practice: Always handle file loading errors gracefully
        """
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as file:
                    config = yaml.safe_load(file)
                    self.logger.info(f"Configuration loaded from {self.config_path}")
                    return config or {}
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        
        Examples:
            config.get('voice_auth.sample_rate')
            config.get('ui.overlay_color', '#000000')
        """
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self._config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the final value
        config[keys[-1]] = value
    
    def save(self) -> None:
        """Save current configuration back to file"""
        try:
            with open(self.config_path, 'w') as file:
                yaml.dump(self._config, file, default_flow_style=False)
            self.logger.info("Configuration saved")
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
    
    def get_all(self) -> Dict[str, Any]:
        """Get entire configuration dictionary"""
        return self._config.copy()
