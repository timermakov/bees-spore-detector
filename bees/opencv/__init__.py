"""
OpenCV-based image processing and spore detection module.

This module provides computer vision functionality for detecting and counting
bee spores in microscopic images using OpenCV techniques.
"""

from .image_proc import (
    ImagePreprocessor,
    EdgeDetector,
    SporeDetector,
    SporeDetectionPipeline
)

from .spores import (
    count_spores,
    analyze_spore_distribution,
    filter_spores_by_area,
    validate_spore_contours
)

__all__ = [
    # Classes from image_proc
    'ImagePreprocessor',
    'EdgeDetector',
    'SporeDetector',
    'SporeDetectionPipeline',
    # Functions from spores
    'count_spores',
    'analyze_spore_distribution',
    'filter_spores_by_area',
    'validate_spore_contours',
]
