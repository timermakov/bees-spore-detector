"""
Titer calculation module for bee spore analysis.

This module provides functionality for calculating titer values (million spores per ml)
using the Goryaev chamber method.
"""

from typing import Union, List, Tuple
import logging

# Configure logging
logger = logging.getLogger(__name__)


class TiterCalculator:
    """Calculator for spore titer values using the Goryaev chamber method."""
    
    def __init__(self, volume_factor: float = 12.0):
        """
        Initialize the titer calculator.
        
        Args:
            volume_factor: Divisor according to the Goryaev chamber method (default 12.0)
            
        Raises:
            ValueError: If volume_factor is zero or negative
        """
        if volume_factor <= 0:
            raise ValueError("volume_factor must be positive")
        
        self.volume_factor = float(volume_factor)
        logger.debug(f"Initialized TiterCalculator with volume_factor={self.volume_factor}")
    
    def calculate_titer(self, spores: Union[int, List[int], Tuple[int, ...]]) -> float:
        """
        Calculate titer (million spores per ml) for spore counts.
        
        Args:
            spores: Spore count(s) - can be a single integer or iterable of counts
            
        Returns:
            Titer value in million spores per ml
            
        Raises:
            ValueError: If spores is negative or empty
            TypeError: If spores is not a valid type
            
        Example:
            >>> calculator = TiterCalculator()
            >>> # Single count
            >>> titer = calculator.calculate_titer(150)
            >>> print(f"Titer: {titer:.2f} million spores/ml")
            >>> # Multiple counts (summed)
            >>> titer = calculator.calculate_titer([120, 135, 145])
            >>> print(f"Group titer: {titer:.2f} million spores/ml")
        """
        if spores is None:
            logger.warning("Received None for spores, returning 0.0")
            return 0.0
        
        try:
            # Handle iterable of counts (x, y, z, ...)
            if hasattr(spores, '__iter__') and not isinstance(spores, (str, bytes)):
                total_spores = sum(spores)
                logger.debug(f"Summed {len(spores)} counts: {spores} = {total_spores}")
            else:
                # Single integer
                total_spores = int(spores)
                logger.debug(f"Single count: {total_spores}")
            
            # Validate count
            if total_spores < 0:
                raise ValueError(f"Spore count cannot be negative: {total_spores}")
            
            # Calculate titer
            titer = float(total_spores) / self.volume_factor
            logger.debug(f"Calculated titer: {total_spores} / {self.volume_factor} = {titer:.2f}")
            
            return titer
            
        except (TypeError, ValueError) as e:
            logger.error(f"Error calculating titer for {spores}: {e}")
            raise
    
    def calculate_group_titer(self, group_counts: List[int]) -> float:
        """
        Calculate titer for a group of spore counts.
        
        Args:
            group_counts: List of spore counts for the group
            
        Returns:
            Group titer value
            
        Raises:
            ValueError: If group_counts is empty or contains negative values
            
        Example:
            >>> calculator = TiterCalculator()
            >>> group_titer = calculator.calculate_group_titer([120, 135, 145])
            >>> print(f"Group titer: {group_titer:.2f} million spores/ml")
        """
        if not group_counts:
            raise ValueError("group_counts cannot be empty")
        
        if any(count < 0 for count in group_counts):
            raise ValueError("All counts must be non-negative")
        
        return self.calculate_titer(group_counts)
    
    def get_volume_factor(self) -> float:
        """Get the current volume factor."""
        return self.volume_factor
    
    def set_volume_factor(self, volume_factor: float) -> None:
        """
        Set a new volume factor.
        
        Args:
            volume_factor: New volume factor value
            
        Raises:
            ValueError: If volume_factor is zero or negative
        """
        if volume_factor <= 0:
            raise ValueError("volume_factor must be positive")
        
        old_factor = self.volume_factor
        self.volume_factor = float(volume_factor)
        logger.info(f"Updated volume factor from {old_factor} to {self.volume_factor}")



# Constants for common volume factors
STANDARD_VOLUME_FACTOR = 12.0
ALTERNATIVE_VOLUME_FACTOR = 10.0  # Alternative method
CUSTOM_VOLUME_FACTOR = 15.0  # Custom calibration


def create_standard_calculator() -> TiterCalculator:
    """Create a standard titer calculator with default volume factor."""
    return TiterCalculator(STANDARD_VOLUME_FACTOR)


def create_custom_calculator(volume_factor: float) -> TiterCalculator:
    """
    Create a custom titer calculator with specified volume factor.
    
    Args:
        volume_factor: Custom volume factor
        
    Returns:
        Configured TiterCalculator instance
    """
    return TiterCalculator(volume_factor)