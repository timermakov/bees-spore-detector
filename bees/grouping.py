"""
Image grouping module for bee spore analysis.

This module provides functionality for grouping images into sets of three
for analysis according to the Goryaev chamber method.
"""

import os
import re
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Constants
GROUP_SUFFIXES = {"_1", "_2", "_3"}
VALID_SUFFIX_PATTERN = re.compile(r"^(?P<prefix>.+)_(?P<idx>[123])\.jpg$", re.IGNORECASE)


class ImageGroupValidator:
    """Validates image groups according to naming conventions."""
    
    @staticmethod
    def validate_group_structure(prefix: str, parts: Dict[str, str]) -> Tuple[bool, List[str]]:
        """
        Validate if a group has the correct structure.
        
        Args:
            prefix: Group prefix name
            parts: Dictionary mapping suffixes to file paths
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check for missing suffixes
        missing = sorted(list(GROUP_SUFFIXES - set(parts.keys())))
        if missing:
            errors.append(
                f"Group '{prefix}' missing files: {', '.join(missing)}. "
                f"Required: {prefix}_1.jpg, {prefix}_2.jpg, {prefix}_3.jpg"
            )
        
        # Check for extra suffixes
        extra = sorted([k for k in parts.keys() if k not in GROUP_SUFFIXES])
        if extra:
            errors.append(
                f"Group '{prefix}' has extra suffixes: {', '.join(extra)}. "
                f"Only _1, _2, _3 are allowed"
            )
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def validate_file_exists(file_path: str) -> bool:
        """Check if a file exists and is accessible."""
        return os.path.isfile(file_path) and os.access(file_path, os.R_OK)


class ImageGrouper:
    """Groups images into sets of three for analysis."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the image grouper.
        
        Args:
            data_dir: Directory containing image files
        """
        self.data_dir = Path(data_dir)
        self.errors: List[str] = []
        self.groups: Dict[str, List[str]] = {}
    
    def group_images(self) -> Tuple[Dict[str, List[str]], List[str]]:
        """
        Group images according to naming conventions.
        
        Returns:
            Tuple of (groups, errors) where:
            - groups: Dictionary mapping prefixes to lists of image paths
            - errors: List of error messages
        """
        if not self.data_dir.is_dir():
            self.errors.append(f"Directory not found: {self.data_dir}")
            return {}, self.errors
        
        # Scan directory for image files
        candidates = self._scan_directory()
        
        # Validate and group candidates
        self._validate_and_group(candidates)
        
        logger.info(f"Successfully grouped {len(self.groups)} image groups from {self.data_dir}")
        return self.groups, self.errors
    
    def _scan_directory(self) -> Dict[str, Dict[str, str]]:
        """Scan directory for image files and group by prefix."""
        candidates: Dict[str, Dict[str, str]] = defaultdict(dict)
        
        for file_path in self.data_dir.glob("*.jpg"):
            match = VALID_SUFFIX_PATTERN.match(file_path.name)
            if not match:
                self.errors.append(
                    f"Invalid filename: {file_path.name}. "
                    f"Expected format: <name>_1.jpg, <name>_2.jpg, <name>_3.jpg"
                )
                continue
            
            prefix = match.group('prefix')
            idx = match.group('idx')
            suffix = f"_{idx}"
            
            if suffix in candidates[prefix]:
                self.errors.append(
                    f"Duplicate file for {prefix}{suffix}.jpg. "
                    f"Only one file allowed per suffix"
                )
                continue
            
            candidates[prefix][suffix] = str(file_path)
        
        return candidates
    
    def _validate_and_group(self, candidates: Dict[str, Dict[str, str]]) -> None:
        """Validate candidates and create valid groups."""
        for prefix, parts in candidates.items():
            is_valid, group_errors = ImageGroupValidator.validate_group_structure(prefix, parts)
            
            if not is_valid:
                self.errors.extend(group_errors)
                continue
            
            # Validate file accessibility
            all_files_accessible = all(
                ImageGroupValidator.validate_file_exists(parts[suffix])
                for suffix in GROUP_SUFFIXES
            )
            
            if not all_files_accessible:
                self.errors.append(f"Group '{prefix}' has inaccessible files")
                continue
            
            # Create ordered list of paths
            ordered_paths = [parts[f"_{i}"] for i in range(1, 4)]
            self.groups[prefix] = ordered_paths
            
            logger.debug(f"Created group '{prefix}' with {len(ordered_paths)} images")


class GroupedImageManager:
    """Manages grouped images and provides utility methods."""
    
    def __init__(self, groups: Dict[str, List[str]]):
        """
        Initialize the manager with grouped images.
        
        Args:
            groups: Dictionary of grouped images
        """
        self.groups = groups
        self._validate_groups()
    
    def _validate_groups(self) -> None:
        """Validate that all groups have exactly 3 images."""
        for prefix, paths in self.groups.items():
            if len(paths) != 3:
                logger.warning(f"Group '{prefix}' has {len(paths)} images, expected 3")
    
    def get_group(self, prefix: str) -> Optional[List[str]]:
        """
        Get image paths for a specific group.
        
        Args:
            prefix: Group prefix name
            
        Returns:
            List of image paths or None if group doesn't exist
        """
        return self.groups.get(prefix)
    
    def get_all_groups(self) -> Dict[str, List[str]]:
        """Get all grouped images."""
        return self.groups.copy()
    
    def get_group_count(self) -> int:
        """Get the total number of groups."""
        return len(self.groups)
    
    def get_total_image_count(self) -> int:
        """Get the total number of images across all groups."""
        return sum(len(paths) for paths in self.groups.values())
    
    def list_group_prefixes(self) -> List[str]:
        """Get a list of all group prefixes."""
        return list(self.groups.keys())
    
    def has_group(self, prefix: str) -> bool:
        """Check if a group exists."""
        return prefix in self.groups
    
    def get_group_info(self, prefix: str) -> Optional[dict]:
        """
        Get detailed information about a group.
        
        Args:
            prefix: Group prefix name
            
        Returns:
            Dictionary with group information or None if group doesn't exist
        """
        if prefix not in self.groups:
            return None
        
        paths = self.groups[prefix]
        return {
            'prefix': prefix,
            'image_count': len(paths),
            'image_paths': paths,
            'image_names': [Path(p).name for p in paths],
            'all_files_exist': all(Path(p).exists() for p in paths)
        }


def list_grouped_images(data_dir: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """
    Legacy function for listing grouped images.
    
    Args:
        data_dir: Directory containing image files
        
    Returns:
        Tuple of (groups, errors) where:
        - groups: Dictionary mapping prefixes to lists of image paths
        - errors: List of error messages
        
    Example:
        >>> groups, errors = list_grouped_images("dataset/")
        >>> if errors:
        >>>     print("Errors found:", errors)
        >>> else:
        >>>     print(f"Found {len(groups)} groups")
    """
    grouper = ImageGrouper(data_dir)
    return grouper.group_images()


def create_group_manager(data_dir: str) -> Optional[GroupedImageManager]:
    """
    Create a GroupedImageManager for the given directory.
    
    Args:
        data_dir: Directory containing image files
        
    Returns:
        GroupedImageManager instance or None if grouping failed
        
    Example:
        >>> manager = create_group_manager("dataset/")
        >>> if manager:
        >>>     print(f"Found {manager.get_group_count()} groups")
        >>>     for prefix in manager.list_group_prefixes():
        >>>         info = manager.get_group_info(prefix)
        >>>         print(f"Group {prefix}: {info['image_count']} images")
    """
    groups, errors = list_grouped_images(data_dir)
    
    if errors:
        logger.error(f"Failed to group images: {errors}")
        return None
    
    return GroupedImageManager(groups)


