"""Shared CVAT XML shapes, parsing, and box geometry."""

import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EllipseAnnotation:
    """One ellipse or box from a CVAT image. Boxes are stored as an axis-aligned ellipse."""

    cx: float
    cy: float
    rx: float
    ry: float
    rotation: float = 0.0
    label: str = "spore"


@dataclass
class ImageAnnotation:
    """Every shape on one CVAT image."""

    image_id: int
    name: str
    width: int
    height: int
    ellipses: List[EllipseAnnotation]


def _ellipse_half_sizes(ellipse: EllipseAnnotation) -> Tuple[float, float]:
    """Half-width and half-height of the box around a rotated ellipse."""
    theta = math.radians(ellipse.rotation)
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))
    half_w = ellipse.rx * cos_t + ellipse.ry * sin_t
    half_h = ellipse.rx * sin_t + ellipse.ry * cos_t
    return half_w, half_h


def ellipse_to_xyxy_pixels(ellipse: EllipseAnnotation) -> Tuple[float, float, float, float]:
    """Axis-aligned box in pixels: x1, y1, x2, y2."""
    half_w, half_h = _ellipse_half_sizes(ellipse)
    return (
        ellipse.cx - half_w,
        ellipse.cy - half_h,
        ellipse.cx + half_w,
        ellipse.cy + half_h,
    )


def clip_xyxy_to_image(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_width: int,
    img_height: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Clip box corners to the image. None when no area remains inside."""
    if img_width <= 0 or img_height <= 0:
        raise ValueError(
            f"Image width and height must be positive, got {img_width}x{img_height}"
        )

    width = float(img_width)
    height = float(img_height)
    left = max(0.0, min(width, min(x1, x2)))
    top = max(0.0, min(height, min(y1, y2)))
    right = max(0.0, min(width, max(x1, x2)))
    bottom = max(0.0, min(height, max(y1, y2)))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def clipped_ellipse_box(
    ellipse: EllipseAnnotation,
    img_width: int,
    img_height: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Ellipse or box corners clipped to the image, as x1, y1, x2, y2."""
    x1, y1, x2, y2 = ellipse_to_xyxy_pixels(ellipse)
    return clip_xyxy_to_image(x1, y1, x2, y2, img_width, img_height)


def normalized_yolo_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_width: int,
    img_height: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Clip corners, then return YOLO center and size in 0..1."""
    clipped = clip_xyxy_to_image(x1, y1, x2, y2, img_width, img_height)
    if clipped is None:
        return None

    left, top, right, bottom = clipped
    return (
        ((left + right) / 2.0) / img_width,
        ((top + bottom) / 2.0) / img_height,
        (right - left) / img_width,
        (bottom - top) / img_height,
    )


def xyxy_to_yolo_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_width: int,
    img_height: int,
    class_id: int,
) -> Optional[str]:
    """Clip corners, then format one YOLO label line."""
    box = normalized_yolo_box(x1, y1, x2, y2, img_width, img_height)
    if box is None:
        return None
    x_center, y_center, width, height = box
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def require_positive_image_size(name: str, width: int, height: int) -> None:
    """Reject an image that cannot be used as a label coordinate space."""
    if width <= 0 or height <= 0:
        raise ValueError(f"Image '{name}' has non-positive size {width}x{height}")


def require_known_label(label: str, class_names: List[str], class_to_id: dict) -> int:
    """Return the class id, or raise when the label is not in the configured list."""
    if label not in class_to_id:
        raise ValueError(f"Unknown label '{label}'. Known classes: {class_names}")
    return class_to_id[label]


def find_annotated_image(annotation: ImageAnnotation, images_dir: Path) -> Optional[Path]:
    """Find the image file named by a CVAT image entry."""
    image_name = Path(annotation.name).name
    candidates = (
        images_dir / image_name,
        images_dir / annotation.name,
        images_dir.parent / annotation.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _ellipse_from_xml(element: ET.Element) -> EllipseAnnotation:
    return EllipseAnnotation(
        cx=float(element.get("cx", 0)),
        cy=float(element.get("cy", 0)),
        rx=float(element.get("rx", 0)),
        ry=float(element.get("ry", 0)),
        rotation=float(element.get("rotation", 0)),
        label=element.get("label", "spore"),
    )


def _box_from_xml(element: ET.Element) -> EllipseAnnotation:
    xtl = float(element.get("xtl", 0))
    ytl = float(element.get("ytl", 0))
    xbr = float(element.get("xbr", 0))
    ybr = float(element.get("ybr", 0))
    return EllipseAnnotation(
        cx=(xtl + xbr) / 2,
        cy=(ytl + ybr) / 2,
        rx=(xbr - xtl) / 2,
        ry=(ybr - ytl) / 2,
        rotation=0.0,
        label=element.get("label", "spore"),
    )


def read_cvat_xml(xml_path: Path) -> List[ImageAnnotation]:
    """Read ellipses and boxes from a CVAT XML file."""
    root = ET.parse(xml_path).getroot()
    annotations: List[ImageAnnotation] = []

    for image_elem in root.findall(".//image"):
        shapes = [_ellipse_from_xml(element) for element in image_elem.findall("ellipse")]
        shapes.extend(_box_from_xml(element) for element in image_elem.findall("box"))
        annotations.append(
            ImageAnnotation(
                image_id=int(image_elem.get("id", 0)),
                name=image_elem.get("name", ""),
                width=int(image_elem.get("width", 0)),
                height=int(image_elem.get("height", 0)),
                ellipses=shapes,
            )
        )

    logger.info(
        "Parsed %d images with %d total annotations",
        len(annotations),
        sum(len(annotation.ellipses) for annotation in annotations),
    )
    return annotations
