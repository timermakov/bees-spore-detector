"""
CVAT XML to YOLO format converter.

Converts ellipse annotations from CVAT XML to YOLO bounding box format.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math
import logging
import shutil

logger = logging.getLogger(__name__)


@dataclass
class EllipseAnnotation:
    """Represents a single ellipse annotation."""
    cx: float
    cy: float
    rx: float
    ry: float
    rotation: float = 0.0
    label: str = "spore"


def _ellipse_aabb_half_sizes(ellipse: EllipseAnnotation) -> Tuple[float, float]:
    """Half-width and half-height of the axis-aligned box containing the ellipse (supports rotation)."""
    theta = math.radians(ellipse.rotation)
    cos_t = abs(math.cos(theta))
    sin_t = abs(math.sin(theta))
    rx, ry = ellipse.rx, ellipse.ry
    half_w = rx * cos_t + ry * sin_t
    half_h = rx * sin_t + ry * cos_t
    return half_w, half_h


def ellipse_to_xyxy_pixels(ellipse: EllipseAnnotation) -> Tuple[float, float, float, float]:
    """Axis-aligned bounding box in pixel coordinates (x1, y1, x2, y2)."""
    half_w, half_h = _ellipse_aabb_half_sizes(ellipse)
    cx, cy = ellipse.cx, ellipse.cy
    return (cx - half_w, cy - half_h, cx + half_w, cy + half_h)


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
    x_center, y_center, w_norm, h_norm = box
    return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


@dataclass
class ImageAnnotation:
    """Represents all annotations for a single image."""
    image_id: int
    name: str
    width: int
    height: int
    ellipses: List[EllipseAnnotation]


class CVATToYOLOConverter:
    """Converts CVAT XML annotations to YOLO format."""
    
    def __init__(self, class_names: List[str] = None):
        """
        Initialize converter.
        
        Args:
            class_names: List of class names. Default: ["spore"]
        """
        self.class_names = class_names or ["spore"]
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}

    def parse_cvat_xml(self, xml_path: Path) -> List[ImageAnnotation]:
        """
        Parse CVAT XML file and extract all annotations (ellipses and boxes).

        Args:
            xml_path: Path to CVAT XML file

        Returns:
            List of ImageAnnotation objects
        """
        tree = ET.parse(xml_path)
        root = tree.getroot()

        annotations = []

        for image_elem in root.findall('.//image'):
            img_id = int(image_elem.get('id', 0))
            name = image_elem.get('name', '')
            width = int(image_elem.get('width', 0))
            height = int(image_elem.get('height', 0))

            ellipses = []

            # 1. Читаем эллипсы (как и раньше)
            for ellipse_elem in image_elem.findall('ellipse'):
                ellipse = EllipseAnnotation(
                    cx=float(ellipse_elem.get('cx', 0)),
                    cy=float(ellipse_elem.get('cy', 0)),
                    rx=float(ellipse_elem.get('rx', 0)),
                    ry=float(ellipse_elem.get('ry', 0)),
                    rotation=float(ellipse_elem.get('rotation', 0)),
                    label=ellipse_elem.get('label', 'spore')
                )
                ellipses.append(ellipse)

            # 2. ДОБАВЛЯЕМ: Читаем боксы из твоего нового XML
            for box_elem in image_elem.findall('box'):
                xtl = float(box_elem.get('xtl', 0))
                ytl = float(box_elem.get('ytl', 0))
                xbr = float(box_elem.get('xbr', 0))
                ybr = float(box_elem.get('ybr', 0))

                # Конвертируем координаты box (угол-угол) в формат ellipse (центр-радиус)
                # Это позволит остальному коду работать без ошибок
                ellipse = EllipseAnnotation(
                    cx=(xtl + xbr) / 2,  # Центр X
                    cy=(ytl + ybr) / 2,  # Центр Y
                    rx=(xbr - xtl) / 2,  # "Радиус" X
                    ry=(ybr - ytl) / 2,  # "Радиус" Y
                    rotation=0.0,
                    label=box_elem.get('label', 'spore')
                )
                ellipses.append(ellipse)

            annotations.append(ImageAnnotation(
                image_id=img_id,
                name=name,
                width=width,
                height=height,
                ellipses=ellipses
            ))

        logger.info(f"Parsed {len(annotations)} images with "
                    f"{sum(len(a.ellipses) for a in annotations)} total annotations")
        return annotations
    
    def ellipse_to_bbox(self, ellipse: EllipseAnnotation, 
                        img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """
        Convert ellipse to axis-aligned bounding box in YOLO format.
        
        Handles rotation by computing the AABB of the rotated ellipse.
        
        Args:
            ellipse: EllipseAnnotation object
            img_width: Image width in pixels
            img_height: Image height in pixels
            
        Returns:
            Tuple of (x_center, y_center, width, height) normalized to [0, 1]
        """
        x1, y1, x2, y2 = ellipse_to_xyxy_pixels(ellipse)
        box = normalized_yolo_box(x1, y1, x2, y2, img_width, img_height)
        if box is None:
            raise ValueError("Bounding box is outside the image")
        return box

    def convert_annotation(self, annotation: ImageAnnotation) -> List[str]:
        """
        Convert ImageAnnotation to YOLO format lines.

        Unknown labels are rejected. Box corners are clipped to the image
        before the center and size are calculated.
        """
        if annotation.width <= 0 or annotation.height <= 0:
            raise ValueError(
                f"Image '{annotation.name}' has non-positive size "
                f"{annotation.width}x{annotation.height}"
            )

        lines = []
        for ellipse in annotation.ellipses:
            if ellipse.label not in self.class_to_id:
                raise ValueError(
                    f"Unknown label '{ellipse.label}'. Known classes: {self.class_names}"
                )
            x1, y1, x2, y2 = ellipse_to_xyxy_pixels(ellipse)
            line = xyxy_to_yolo_line(
                x1,
                y1,
                x2,
                y2,
                annotation.width,
                annotation.height,
                self.class_to_id[ellipse.label],
            )
            if line is None:
                logger.warning(
                    "Skipping box outside image %s for label %s",
                    annotation.name,
                    ellipse.label,
                )
                continue
            lines.append(line)

        return lines
    
    def convert_and_save(self, 
                         xml_path: Path,
                         images_root: Path,
                         output_dir: Path,
                         copy_images: bool = True) -> Dict[str, int]:
        """
        Convert CVAT annotations and save to YOLO format.
        
        Args:
            xml_path: Path to CVAT XML file
            images_root: Root directory for source images
            output_dir: Output directory for YOLO dataset
            copy_images: Whether to copy images to output directory
            
        Returns:
            Dict with conversion statistics
        """
        annotations = self.parse_cvat_xml(xml_path)
        
        output_dir = Path(output_dir)
        images_out = output_dir / "images"
        labels_out = output_dir / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'images': 0,
            'annotations': 0,
            'skipped_images': 0
        }
        
        for annotation in annotations:
            # Get image filename from path in annotation
            img_name = Path(annotation.name).name
            src_image = images_root / img_name
            
            # Handle case where path in annotation includes subdirectory
            if not src_image.exists():
                src_image = images_root / annotation.name
            if not src_image.exists():
                # Try without subdirectory prefix
                src_image = images_root.parent / annotation.name
            
            if not src_image.exists():
                logger.warning(f"Image not found: {annotation.name}")
                stats['skipped_images'] += 1
                continue
            
            # Copy image
            if copy_images:
                dst_image = images_out / img_name
                shutil.copy2(src_image, dst_image)
            
            # Save labels
            label_file = labels_out / (Path(img_name).stem + ".txt")
            lines = self.convert_annotation(annotation)
            
            with open(label_file, 'w') as f:
                f.write('\n'.join(lines))
            
            stats['images'] += 1
            stats['annotations'] += len(lines)
            
            logger.debug(f"Converted {img_name}: {len(lines)} annotations")
        
        logger.info(f"Conversion complete: {stats['images']} images, "
                   f"{stats['annotations']} annotations")
        
        return stats


def convert_cvat_to_yolo(xml_path: str, 
                         images_root: str, 
                         output_dir: str) -> Dict[str, int]:
    """
    Convenience function to convert CVAT to YOLO format.
    
    Args:
        xml_path: Path to CVAT XML file
        images_root: Root directory for source images
        output_dir: Output directory for YOLO dataset
        
    Returns:
        Conversion statistics
    """
    converter = CVATToYOLOConverter()
    return converter.convert_and_save(
        Path(xml_path), 
        Path(images_root), 
        Path(output_dir)
    )
