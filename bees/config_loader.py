"""
Configuration loading and management module for bee spore analysis.

This module provides functionality for loading and managing configuration files
in YAML format with validation and default value handling.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, TypeVar, Type, List
from contextlib import contextmanager

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic types
T = TypeVar('T')


class ConfigurationError(Exception):
    """Raised when there's an error with configuration loading or validation."""
    pass


class ConfigurationValidator:
    """Validates configuration data and structure."""
    
    REQUIRED_SECTIONS = {'data_dir', 'results_dir'}
    OPTIONAL_SECTIONS = {
        'min_contour_area', 'max_contour_area', 'min_ellipse_area', 'max_ellipse_area',
        'canny_threshold1', 'canny_threshold2', 'min_spore_contour_length', 'intensity_threshold'
    }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> List[str]:
        """
        Validate configuration data.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        
        # Check for required sections
        missing_sections = cls.REQUIRED_SECTIONS - set(config.keys())
        if missing_sections:
            errors.append(f"Missing required configuration sections: {', '.join(missing_sections)}")
        
        # Validate parameter types and ranges
        errors.extend(cls._validate_parameters(config))
        
        return errors
    
    @classmethod
    def _validate_parameters(cls, config: Dict[str, Any]) -> List[str]:
        """Validate individual parameters."""
        errors = []
        
        # Validate area parameters
        area_params = ['min_contour_area', 'max_contour_area', 'min_ellipse_area', 'max_ellipse_area']
        for param in area_params:
            if param in config:
                value = config[param]
                if not isinstance(value, (int, float)) or value <= 0:
                    errors.append(f"Parameter '{param}' must be a positive number, got: {value}")
        
        # Validate threshold parameters
        threshold_params = ['canny_threshold1', 'canny_threshold2', 'intensity_threshold']
        for param in threshold_params:
            if param in config:
                value = config[param]
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(f"Parameter '{param}' must be a non-negative number, got: {value}")
        
        # Validate contour length parameter
        if 'min_spore_contour_length' in config:
            value = config['min_spore_contour_length']
            if not isinstance(value, int) or value < 3:
                errors.append(f"Parameter 'min_spore_contour_length' must be an integer >= 3, got: {value}")
        
        return errors


class ConfigurationLoader:
    """Handles loading and parsing of configuration files."""
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize the configuration loader.
        
        Args:
            config_path: Path to the configuration file
            
        Raises:
            ConfigurationError: If YAML is not available or file doesn't exist
        """
        if not YAML_AVAILABLE:
            raise ConfigurationError(
                "PyYAML is required. Please install with 'pip install pyyaml'."
            )
        
        self.config_path = Path(config_path)
        self._config_cache: Optional[Dict[str, Any]] = None
        
        if not self.config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationError: If configuration is invalid or cannot be loaded
        """
        if self._config_cache is not None:
            return self._config_cache
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
            
            logger.debug(f"Loaded configuration from {self.config_path}")
            
            # Validate configuration
            validation_errors = ConfigurationValidator.validate_config(config_data)
            if validation_errors:
                error_msg = f"Configuration validation failed:\n" + "\n".join(validation_errors)
                logger.error(error_msg)
                raise ConfigurationError(error_msg)
            
            self._config_cache = config_data
            return config_data
            
        except yaml.YAMLError as e:
            error_msg = f"Failed to parse YAML configuration {self.config_path}: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        except Exception as e:
            error_msg = f"Failed to load configuration {self.config_path}: {e}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
    
    def reload_config(self) -> Dict[str, Any]:
        """Reload configuration from file, bypassing cache."""
        self._config_cache = None
        return self.load_config()
    
    def get_config_path(self) -> Path:
        """Get the path to the configuration file."""
        return self.config_path
    
    def is_valid(self) -> bool:
        """Check if the configuration file is valid."""
        try:
            self.load_config()
            return True
        except ConfigurationError:
            return False


class ConfigurationManager:
    """Manages configuration with default values and type conversion."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the configuration manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self._defaults = self._get_defaults()
    
    def _get_defaults(self) -> Dict[str, Any]:
        """Get default configuration values."""
        return {
            'data_dir': 'dataset2',
            'results_dir': 'results2',
            'min_contour_area': 25,
            'max_contour_area': 500,
            'min_ellipse_area': 25,
            'max_ellipse_area': 500,
            'canny_threshold1': 40,
            'canny_threshold2': 125,
            'min_spore_contour_length': 5,
            'intensity_threshold': 50,
        }
    
    def get_param(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a configuration parameter with optional default value.
        
        Args:
            key: Parameter key
            default: Default value if key not found
            
        Returns:
            Parameter value or default
        """
        if default is None:
            default = self._defaults.get(key)
        
        value = self.config.get(key, default)
        logger.debug(f"Config parameter '{key}' = {value}")
        return value
    
    def get_param_typed(self, key: str, param_type: Type[T], default: Optional[T] = None) -> T:
        """
        Get a configuration parameter with type conversion.
        
        Args:
            key: Parameter key
            param_type: Expected type
            default: Default value if key not found
            
        Returns:
            Parameter value converted to specified type
        """
        value = self.get_param(key, default)
        
        try:
            if value is not None:
                return param_type(value)
            else:
                return value
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert '{key}' value '{value}' to {param_type.__name__}: {e}")
            return default
    
    def get_int_param(self, key: str, default: Optional[int] = None) -> int:
        """Get an integer parameter."""
        return self.get_param_typed(key, int, default)
    
    def get_float_param(self, key: str, default: Optional[float] = None) -> float:
        """Get a float parameter."""
        return self.get_param_typed(key, float, default)
    
    def get_string_param(self, key: str, default: Optional[str] = None) -> str:
        """Get a string parameter."""
        return self.get_param_typed(key, str, default)
    
    def get_bool_param(self, key: str, default: Optional[bool] = None) -> bool:
        """Get a boolean parameter."""
        return self.get_param_typed(key, bool, default)
    
    def has_param(self, key: str) -> bool:
        """Check if a parameter exists in configuration."""
        return key in self.config
    
    def get_all_params(self) -> Dict[str, Any]:
        """Get all configuration parameters with defaults applied."""
        result = self._defaults.copy()
        result.update(self.config)
        return result
    
    def validate_required_params(self, required_keys: list) -> List[str]:
        """
        Validate that required parameters are present.
        
        Args:
            required_keys: List of required parameter keys
            
        Returns:
            List of missing parameter keys
        """
        missing = []
        for key in required_keys:
            if not self.has_param(key):
                missing.append(key)
        return missing


@contextmanager
def load_config_context(config_path: Union[str, Path]):
    """
    Context manager for loading configuration.
    
    Args:
        config_path: Path to configuration file
        
    Yields:
        ConfigurationManager instance
        
    Example:
        >>> with load_config_context("config.yaml") as config:
        >>>     data_dir = config.get_param("data_dir")
        >>>     threshold = config.get_int_param("intensity_threshold")
    """
    loader = ConfigurationLoader(config_path)
    config_data = loader.load_config()
    manager = ConfigurationManager(config_data)
    
    try:
        yield manager
    finally:
        # Cleanup if needed
        pass


def create_config_manager(config_path: Union[str, Path]) -> ConfigurationManager:
    """
    Create a ConfigurationManager for the given config file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        ConfigurationManager instance
        
    Example:
        >>> config = create_config_manager("config.yaml")
        >>> data_dir = config.get_param("data_dir")
        >>> threshold = config.get_int_param("intensity_threshold")
    """
    loader = ConfigurationLoader(config_path)
    config_data = loader.load_config()
    return ConfigurationManager(config_data)


