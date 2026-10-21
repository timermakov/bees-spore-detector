"""
Titer calculation module for bee spore analysis.

Whole frame: titer = Σ spores / (volume_factor * Σ (S_photo / square_size²))
Inside square: titer = Σ spores / (volume_factor * Σ N_sq), N_sq = 1 when the square fits.
Square size 0 does not produce a Goryaev titer.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

from bees.analysis_zone import goryaev_squares

logger = logging.getLogger(__name__)


class TiterCalculator:
    """Calculator for spore titer values."""

    ANALYSIS_SQUARE_SIZE = 780
    VOLUME_FACTOR = 4.0

    def __init__(
        self,
        square_size: int = ANALYSIS_SQUARE_SIZE,
        volume_factor: float = VOLUME_FACTOR,
        count_inside_square: bool = False,
    ):
        if volume_factor <= 0:
            raise ValueError("volume_factor must be positive")

        self.square_size = int(square_size)
        self.volume_factor = float(volume_factor)
        self.count_inside_square = bool(count_inside_square) and self.square_size > 0
        self.square_area = None if self.square_size <= 0 else self.square_size ** 2

        logger.debug(
            "TiterCalculator: square=%s, volume_factor=%s, inside_square=%s",
            square_size,
            volume_factor,
            self.count_inside_square,
        )

    # ==================== NEW METHODS (hierarchical) ====================

    def calculate_sample_titer(self, photo_data: List[Tuple[int, int, int]]) -> Optional[float]:
        """
        Titer in million spores/ml.

        The count in photo_data is the population for this mode: every spore on
        the frame, or only spores inside the square. A square size of 0 returns
        None because the Goryaev conversion has no reference area.
        """
        if not photo_data:
            logger.warning("Empty photo_data, returning 0.0")
            return 0.0

        total_spores, total_squares = self._totals(photo_data)
        if total_squares is None:
            return None
        denominator = self.volume_factor * total_squares
        titer = total_spores / denominator

        logger.debug(
            f"Sample titer: {total_spores} spores / ({self.volume_factor} * "
            f"{total_squares:.4f} grids) = {titer:.6f}"
        )

        return titer

    def calculate_sample_titer_safe(self, photo_data: List[Tuple[int, int, int]]) -> Tuple[Optional[float], dict]:
        """Safe version with detailed breakdown for debugging."""
        if not photo_data:
            return 0.0, {
                'total_spores': 0, 'n_photos': 0, 'photo_area': 0,
                'squares_per_photo': 0.0, 'total_squares': 0.0,
                'denominator': 0.0, 'titer': 0.0, 'error': 'No photos'
            }

        total_spores, total_squares = self._totals(photo_data)
        n_photos = len(photo_data)
        _, w, h = photo_data[0]
        photo_area = w * h
        if total_squares is None:
            return None, {
                'total_spores': total_spores,
                'n_photos': n_photos,
                'photo_width': w,
                'photo_height': h,
                'photo_area': photo_area,
                'square_size': self.square_size,
                'square_area': None,
                'squares_per_photo': None,
                'total_squares': None,
                'volume_factor': self.volume_factor,
                'denominator': None,
                'titer': None,
                'error': 'No analysis square',
            }
        squares_per_photo = total_squares / n_photos
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

    def _totals(self, photo_data: List[Tuple[int, int, int]]) -> Tuple[int, Optional[float]]:
        """Sum spores and the Goryaev squares contributed by each photo."""
        total_spores = 0
        total_squares = 0.0
        has_square = self.square_size > 0
        for count, width, height in photo_data:
            if width <= 0 or height <= 0:
                raise ValueError(f"Photo size must be positive, got {width}x{height}")
            total_spores += count
            if not has_square:
                continue
            squares = goryaev_squares(width, height, self.square_size, self.count_inside_square)
            total_squares += squares
        if not has_square:
            return total_spores, None
        return total_spores, total_squares

    def get_square_size(self) -> int:
        return self.square_size

    def set_square_size(self, square_size: int) -> None:
        self.square_size = int(square_size)
        self.square_area = None if self.square_size <= 0 else self.square_size ** 2
        if self.square_size <= 0:
            self.count_inside_square = False

def create_calculator_from_config(config_manager, count_inside_square: Optional[bool] = None) -> TiterCalculator:
    """Factory: create TiterCalculator from config.yaml values."""
    square_size = config_manager.get_int_param('analysis_square_size')
    volume_factor = config_manager.get_float_param('volume_factor')
    if count_inside_square is None:
        count_inside_square = config_manager.get_bool_param('count_inside_square')
    return TiterCalculator(
        square_size=square_size,
        volume_factor=volume_factor,
        count_inside_square=bool(count_inside_square),
    )