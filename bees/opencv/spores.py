"""
Spore counting and analysis module.

This module provides functionality for counting and analyzing detected spores.
"""

from typing import List, Union
import numpy as np


def count_spores(spore_objects: List[np.ndarray]) -> int:
    """
    Count the number of detected spores.
    
    Args:
        spore_objects: List of spore contours (numpy arrays)
        
    Returns:
        Total count of detected spores
        
    Example:
        >>> contours = [np.array([[x1, y1], [x2, y2], ...]), ...]
        >>> count = count_spores(contours)
        >>> print(f"Found {count} spores")
    """
    if not isinstance(spore_objects, list):
        raise TypeError("spore_objects must be a list")
    
    return len(spore_objects)


def analyze_spore_distribution(spore_objects: List[np.ndarray]) -> dict:
    """
    Analyze the distribution of detected spores.
    
    Args:
        spore_objects: List of spore contours
        
    Returns:
        Dictionary containing analysis results:
        - total_count: Total number of spores
        - areas: List of contour areas
        - mean_area: Average contour area
        - std_area: Standard deviation of areas
        - min_area: Minimum contour area
        - max_area: Maximum contour area
        
    Example:
        >>> analysis = analyze_spore_distribution(contours)
        >>> print(f"Average spore area: {analysis['mean_area']:.2f}")
    """
    import cv2
    
    if not spore_objects:
        return {
            'total_count': 0,
            'areas': [],
            'mean_area': 0.0,
            'std_area': 0.0,
            'min_area': 0.0,
            'max_area': 0.0
        }
    
    # Calculate areas for all contours
    areas = [cv2.contourArea(contour) for contour in spore_objects]
    
    # Calculate statistics
    areas_array = np.array(areas)
    mean_area = float(np.mean(areas_array))
    std_area = float(np.std(areas_array))
    min_area = float(np.min(areas_array))
    max_area = float(np.max(areas_array))
    
    return {
        'total_count': len(spore_objects),
        'areas': areas,
        'mean_area': mean_area,
        'std_area': std_area,
        'min_area': min_area,
        'max_area': max_area
    }


def filter_spores_by_area(spore_objects: List[np.ndarray], 
                         min_area: float, 
                         max_area: float) -> List[np.ndarray]:
    """
    Filter spores by contour area.
    
    Args:
        spore_objects: List of spore contours
        min_area: Minimum allowed area
        max_area: Maximum allowed area
        
    Returns:
        Filtered list of spore contours
        
    Example:
        >>> filtered = filter_spores_by_area(contours, 25.0, 500.0)
        >>> print(f"Kept {len(filtered)} out of {len(contours)} spores")
    """
    import cv2
    
    if min_area >= max_area:
        raise ValueError("min_area must be less than max_area")
    
    filtered = []
    for contour in spore_objects:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            filtered.append(contour)
    
    return filtered


def validate_spore_contours(spore_objects: List[np.ndarray]) -> dict:
    """
    Validate spore contours for quality assessment.
    
    Args:
        spore_objects: List of spore contours
        
    Returns:
        Dictionary containing validation results:
        - valid_count: Number of valid contours
        - invalid_count: Number of invalid contours
        - validation_errors: List of error descriptions
        
    Example:
        >>> validation = validate_spore_contours(contours)
        >>> print(f"Valid: {validation['valid_count']}, Invalid: {validation['invalid_count']}")
    """
    import cv2
    
    valid_count = 0
    invalid_count = 0
    validation_errors = []
    
    for i, contour in enumerate(spore_objects):
        try:
            # Check if contour has enough points for ellipse fitting
            if len(contour) < 5:
                invalid_count += 1
                validation_errors.append(f"Contour {i}: Insufficient points ({len(contour)} < 5)")
                continue
            
            # Check if contour area is positive
            area = cv2.contourArea(contour)
            if area <= 0:
                invalid_count += 1
                validation_errors.append(f"Contour {i}: Invalid area ({area})")
                continue
            
            # Try to fit ellipse
            ellipse = cv2.fitEllipse(contour)
            valid_count += 1
            
        except Exception as e:
            invalid_count += 1
            validation_errors.append(f"Contour {i}: Ellipse fitting failed - {str(e)}")
    
    return {
        'valid_count': valid_count,
        'invalid_count': invalid_count,
        'validation_errors': validation_errors
    } 