"""Centered analysis square used by the titer count."""

from typing import Sequence, Tuple

Point = Tuple[float, float]
Bounds = Tuple[int, int, int, int]


def analysis_square_bounds(width: int, height: int, square_size: int) -> Bounds:
    """Return x1, y1, x2, y2 of the centered square, clipped to the image."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Image width and height must be positive, got {width}x{height}")
    if square_size <= 0:
        return 0, 0, width - 1, height - 1

    center_x, center_y = width // 2, height // 2
    half = square_size // 2
    return (
        max(0, center_x - half),
        max(0, center_y - half),
        min(width - 1, center_x + half),
        min(height - 1, center_y + half),
    )


def point_inside_analysis_square(
    x: float,
    y: float,
    width: int,
    height: int,
    square_size: int,
) -> bool:
    """True when the point lies inside the centered analysis square."""
    x1, y1, x2, y2 = analysis_square_bounds(width, height, square_size)
    return x1 <= x <= x2 and y1 <= y <= y2


def split_by_analysis_square(
    items: Sequence,
    centers: Sequence[Point],
    width: int,
    height: int,
    square_size: int,
):
    """Split items into those whose center is inside the square and those outside."""
    if len(items) != len(centers):
        raise ValueError(
            f"Expected one center per item, got {len(items)} items and {len(centers)} centers"
        )
    inside = []
    outside = []
    for item, (x, y) in zip(items, centers):
        if point_inside_analysis_square(x, y, width, height, square_size):
            inside.append(item)
        else:
            outside.append(item)
    return inside, outside
