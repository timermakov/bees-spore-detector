"""
Configuration loading and management module for bee spore analysis.

This module provides functionality for loading and managing configuration files
in YAML format. Required values come from the file; missing or mistyped
values raise ConfigurationError.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from contextlib import contextmanager

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

# Configure logging
logger = logging.getLogger(__name__)

_UNSET = object()

_TILEY_REQUIRED_KEYS: Dict[str, tuple[str, ...]] = {
    "export": (
        "input",
        "cvat_xml",
        "out",
        "tile_size",
        "overlap_height_ratio",
        "overlap_width_ratio",
    ),
    "predict": (
        "input",
        "out",
        "tile_size",
        "overlap_height_ratio",
        "overlap_width_ratio",
        "conf",
        "write_previews",
        "weights",
        "device",
        "model_type",
    ),
}

_TILEY_INT_KEYS: Dict[str, tuple[str, ...]] = {
    "export": ("tile_size",),
    "predict": ("tile_size",),
}

_TILEY_FLOAT_KEYS: Dict[str, tuple[str, ...]] = {
    "export": ("overlap_height_ratio", "overlap_width_ratio"),
    "predict": ("overlap_height_ratio", "overlap_width_ratio", "conf"),
}

_TILEY_BOOL_KEYS: Dict[str, tuple[str, ...]] = {
    "predict": ("write_previews",),
}

_TILEY_TEXT_KEYS: Dict[str, tuple[str, ...]] = {
    "export": ("input", "out"),
    "predict": ("input", "out", "device", "model_type"),
}

_TILEY_NULLABLE_TEXT_KEYS: Dict[str, tuple[str, ...]] = {
    "export": ("cvat_xml",),
    "predict": ("weights",),
}


def _as_int(value: Any, label: str) -> int:
    """Return an integer, rejecting null, booleans, and non-numeric values."""
    if value is None:
        raise ConfigurationError(f"config.yaml '{label}' is null; an integer is required")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"config.yaml '{label}' must be an integer, got {value!r}")
    if isinstance(value, float) and not value.is_integer():
        raise ConfigurationError(f"config.yaml '{label}' must be an integer, got {value!r}")
    return int(value)


def _as_float(value: Any, label: str) -> float:
    """Return a float, rejecting null, booleans, and non-numeric values."""
    if value is None:
        raise ConfigurationError(f"config.yaml '{label}' is null; a number is required")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigurationError(f"config.yaml '{label}' must be a number, got {value!r}")
    return float(value)


def _as_bool(value: Any, label: str) -> bool:
    """Return a boolean. Only YAML true/false are accepted."""
    if not isinstance(value, bool):
        raise ConfigurationError(f"config.yaml '{label}' must be true or false, got {value!r}")
    return value


def _as_text(value: Any, label: str, allow_null: bool = False) -> Optional[str]:
    """Return a non-empty string. Null is accepted only when allow_null is set."""
    if value is None and allow_null:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"config.yaml '{label}' must be a non-empty string")
    return value


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
    """Reads configuration from config.yaml without substituting hidden defaults."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the configuration manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
    
    def get_param(self, key: str, default: Any = _UNSET) -> Any:
        """
        Get a configuration parameter.

        Args:
            key: Parameter key
            default: Value used only when the key is absent

        Returns:
            Parameter value, including null when the key is present

        Raises:
            ConfigurationError: If the key is absent and no default was given
        """
        if key in self.config:
            value = self.config[key]
        elif default is not _UNSET:
            value = default
        else:
            raise ConfigurationError(f"config.yaml is missing '{key}'")
        logger.debug(f"Config parameter '{key}' = {value}")
        return value
    
    def get_int_param(self, key: str, default: Any = _UNSET) -> int:
        """Get an integer parameter. Null and non-numeric values are rejected."""
        if key not in self.config:
            if default is _UNSET:
                raise ConfigurationError(f"config.yaml is missing '{key}'")
            return _as_int(default, key)
        return _as_int(self.config[key], key)
    
    def get_float_param(self, key: str, default: Any = _UNSET) -> float:
        """Get a float parameter. Null and non-numeric values are rejected."""
        if key not in self.config:
            if default is _UNSET:
                raise ConfigurationError(f"config.yaml is missing '{key}'")
            return _as_float(default, key)
        return _as_float(self.config[key], key)
    
    def get_string_param(self, key: str, default: Any = _UNSET) -> str:
        """Get a non-empty string parameter."""
        if key not in self.config:
            if default is _UNSET:
                raise ConfigurationError(f"config.yaml is missing '{key}'")
            value = default
        else:
            value = self.config[key]
        text = _as_text(value, key)
        if text is None:
            raise ConfigurationError(f"config.yaml '{key}' must be a non-empty string")
        return text
    
    def get_bool_param(self, key: str, default: Any = _UNSET) -> bool:
        """Get a boolean parameter. Only real true/false values are accepted."""
        if key not in self.config:
            if default is _UNSET:
                raise ConfigurationError(f"config.yaml is missing '{key}'")
            return _as_bool(default, key)
        return _as_bool(self.config[key], key)
    
    def has_param(self, key: str) -> bool:
        """Check if a parameter exists in configuration."""
        return key in self.config

    def get_tiley(self) -> Dict[str, Dict[str, Any]]:
        """
        Return required ``tiley`` settings from config.yaml.

        Top-level keys are ``export`` and ``predict``; each maps to a flat
        dictionary. Values are intentionally not defaulted in Python.
        """
        raw = self.config.get("tiley")
        if not isinstance(raw, dict):
            raise ConfigurationError("config.yaml must contain a tiley mapping")

        resolved: Dict[str, Dict[str, Any]] = {}
        for section, required_keys in _TILEY_REQUIRED_KEYS.items():
            block = raw.get(section)
            if not isinstance(block, dict):
                raise ConfigurationError(f"config.yaml must contain tiley.{section}")
            missing_keys = [key for key in required_keys if key not in block]
            if missing_keys:
                raise ConfigurationError(
                    f"config.yaml is missing tiley.{section} keys: " + ", ".join(missing_keys)
                )
            resolved_block = dict(block)
            for key in _TILEY_INT_KEYS.get(section, ()):
                resolved_block[key] = _as_int(resolved_block[key], f"tiley.{section}.{key}")
            for key in _TILEY_FLOAT_KEYS.get(section, ()):
                resolved_block[key] = _as_float(resolved_block[key], f"tiley.{section}.{key}")
            for key in _TILEY_BOOL_KEYS.get(section, ()):
                resolved_block[key] = _as_bool(resolved_block[key], f"tiley.{section}.{key}")
            for key in _TILEY_TEXT_KEYS.get(section, ()):
                resolved_block[key] = _as_text(resolved_block[key], f"tiley.{section}.{key}")
            for key in _TILEY_NULLABLE_TEXT_KEYS.get(section, ()):
                resolved_block[key] = _as_text(
                    resolved_block[key], f"tiley.{section}.{key}", allow_null=True
                )
            resolved[section] = resolved_block
        return resolved
    
    def get_all_params(self) -> Dict[str, Any]:
        """Get the configuration values loaded from YAML."""
        return dict(self.config)
    
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


