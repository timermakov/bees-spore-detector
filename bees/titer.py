"""
Titer calculation module for bee spore analysis.

New formula: Titer = Σ_spores / (4 * (S_photo / square_size²) * N_photos)
Old formula (legacy): Titer = count / 12.0 (single image)
"""

import logging
from typing import Union, List, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class TiterCalculator:
    """Calculator for spore titer values."""

    ANALYSIS_SQUARE_SIZE = 780
    VOLUME_FACTOR = 4.0

    def __init__(self, square_size: int = ANALYSIS_SQUARE_SIZE, volume_factor: float = VOLUME_FACTOR):
        if square_size <= 0:
            raise ValueError("square_size must be positive")
        if volume_factor <= 0:
            raise ValueError("volume_factor must be positive")

        self.square_size = int(square_size)
        self.volume_factor = float(volume_factor)
        self.square_area = self.square_size ** 2

        logger.debug(f"TiterCalculator: square={square_size}, volume_factor={volume_factor}")

    # ==================== NEW METHODS (hierarchical) ====================

    def calculate_sample_titer(self, photo_data: List[Tuple[int, int, int]]) -> float:
        """
        Calculate titer for a complete sample folder (NEW).

        Formula: total_spores / (volume_factor * (photo_area / square_area) * n_photos)

        Args:
            photo_data: List of (count, width, height) for each photo.
                       All photos should be the same size.

        Returns:
            Titer value in million spores/ml
        """
        if not photo_data:
            logger.warning("Empty photo_data, returning 0.0")
            return 0.0

        total_spores = sum(count for count, _, _ in photo_data)
        n_photos = len(photo_data)

        # Dimensions of first photo (all photos same size)
        _, first_width, first_height = photo_data[0]
        photo_area = first_width * first_height

        # How many analysis squares fit in ONE photo
        squares_per_photo = photo_area / self.square_area

        # Total effective squares across ALL photos
        total_squares = squares_per_photo * n_photos

        # Final calculation
        denominator = self.volume_factor * total_squares
        titer = total_spores / denominator

        logger.debug(
            f"Sample titer: {total_spores} spores / ({self.volume_factor} * "
            f"{squares_per_photo:.4f} grids * {n_photos} photos) = {titer:.6f}"
        )

        return titer

    def calculate_sample_titer_safe(self, photo_data: List[Tuple[int, int, int]]) -> Tuple[float, dict]:
        """Safe version with detailed breakdown for debugging."""
        if not photo_data:
            return 0.0, {
                'total_spores': 0, 'n_photos': 0, 'photo_area': 0,
                'squares_per_photo': 0.0, 'total_squares': 0.0,
                'denominator': 0.0, 'titer': 0.0, 'error': 'No photos'
            }

        total_spores = sum(count for count, _, _ in photo_data)
        n_photos = len(photo_data)
        _, w, h = photo_data[0]
        photo_area = w * h
        squares_per_photo = photo_area / self.square_area
        total_squares = squares_per_photo * n_photos
        denominator = self.volume_factor * total_squares
        titer = total_spores / denominator

        breakdown = {
            'total_spores': total_spores,
            'n_photos': n_photos,
            'photo_width': w,
            'photo_height': h,
            'photo_area': photo_area,
            'square_size': self.square_size,
            'square_area': self.square_area,
            'squares_per_photo': squares_per_photo,
            'total_squares': total_squares,
            'volume_factor': self.volume_factor,
            'denominator': denominator,
            'titer': titer
        }

        return titer, breakdown

    def one_sample_ttest(self, sample_titers: List[float], control_titer: float) -> Tuple[float, float]:
        """One-sample t-test: H0 = mean(sample_titers) == control_titer."""
        if len(sample_titers) < 2:
            logger.warning(f"Need >= 2 samples for t-test, got {len(sample_titers)}")
            return 0.0, 1.0

        t_stat, p_value = stats.ttest_1samp(sample_titers, control_titer)
        logger.debug(f"One-sample t-test: mean={np.mean(sample_titers):.4f}, control={control_titer:.4f}, p={p_value:.6f}")
        return t_stat, p_value

    def two_sample_ttest(self, group_titers: List[float], control_titers: List[float]) -> Tuple[float, float]:
        """Two-sample Welch's t-test: group vs control."""
        if len(group_titers) < 2 or len(control_titers) < 2:
            logger.warning(f"Need >= 2 in each group: group={len(group_titers)}, control={len(control_titers)}")
            return 0.0, 1.0

        t_stat, p_value = stats.ttest_ind(group_titers, control_titers, equal_var=False)
        logger.debug(
            f"Two-sample t-test: group_mean={np.mean(group_titers):.4f}, "
            f"control_mean={np.mean(control_titers):.4f}, p={p_value:.6f}"
        )
        return t_stat, p_value

    # ==================== UTILITY ====================

    def get_square_size(self) -> int:
        return self.square_size

    def set_square_size(self, square_size: int) -> None:
        if square_size <= 0:
            raise ValueError("square_size must be positive")
        self.square_size = int(square_size)
        self.square_area = self.square_size ** 2

def create_calculator_from_config(config_manager) -> TiterCalculator:
    """Factory: create TiterCalculator from ConfigurationManager."""
    square_size = config_manager.get_int_param('analysis_square_size', TiterCalculator.ANALYSIS_SQUARE_SIZE)
    volume_factor = config_manager.get_float_param('volume_factor', TiterCalculator.VOLUME_FACTOR)
    return TiterCalculator(square_size=square_size, volume_factor=volume_factor)