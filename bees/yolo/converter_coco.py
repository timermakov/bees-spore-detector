"""
CVAT XML to COCO format converter.

Converts ellipse annotations from CVAT XML to COCO bounding box format.
"""

import logging
import xml.etree.ElementTree as ET
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from sahi.utils.coco import Coco, CocoImage, CocoAnnotation, CocoCategory
    from sahi.utils.file import save_json
except ImportError:
    raise ImportError("sahi package required. Install with: pip install sahi")

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


@dataclass
class ImageAnnotation:
    """Represents all annotations for a single image."""
    image_id: int
    name: str
    width: int
    height: int
    ellipses: List[EllipseAnnotation]


class CVATToCocoConverter:
    """Converts CVAT XML annotations to COCO format."""
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize converter.
        
        Args:
            class_names: List of class names. Default: ["spore"]
        """
        self.class_names = class_names or ["spore"]
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        self.coco = None

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

            # Parse ellipses
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

            # Parse boxes
            for box_elem in image_elem.findall('box'):
                xtl = float(box_elem.get('xtl', 0))
                ytl = float(box_elem.get('ytl', 0))
                xbr = float(box_elem.get('xbr', 0))
                ybr = float(box_elem.get('ybr', 0))

                # Convert box coordinates to ellipse format
                ellipse = EllipseAnnotation(
                    cx=(xtl + xbr) / 2,
                    cy=(ytl + ybr) / 2,
                    rx=(xbr - xtl) / 2,
                    ry=(ybr - ytl) / 2,
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

    def _initialize_coco(self) -> Coco:
        """Initialize COCO dataset with categories."""
        coco = Coco()
        for idx, class_name in enumerate(self.class_names):
            coco.add_category(CocoCategory(id=idx, name=class_name))
        return coco

    def parse_cvat_to_coco(self, xml_path: Path, images_dir: Optional[Path] = None) -> Coco:
        """
        Parse CVAT XML and build COCO dataset.

        Args:
            xml_path: Path to CVAT XML file
            images_dir: Optional directory to use for resolving image file names

        Returns:
            Coco dataset object
        """
        xml_path = Path(xml_path)
        images_dir = Path(images_dir) if images_dir else xml_path.parent

        self.coco = self._initialize_coco()
        annotations = self.parse_cvat_xml(xml_path)

        for ann in annotations:
            # Resolve image path
            img_path = self._resolve_image_path(ann, images_dir)
            if img_path is None or not img_path.is_file():
                logger.warning(f"Skipping missing image: {ann.name}")
                continue

            # Create COCO image entry
            # Use relative path for file_name
            file_name = img_path.name if images_dir is None else str(img_path.relative_to(images_dir))
            coco_image = CocoImage(
                file_name=file_name,
                height=ann.height,
                width=ann.width,
            )

            # Add annotations (convert ellipses to bboxes in COCO format [x, y, w, h])
            for ellipse in ann.ellipses:
                x1, y1, x2, y2 = ellipse_to_xyxy_pixels(ellipse)
                width = x2 - x1
                height = y2 - y1

                # Get class ID
                class_id = self.class_to_id.get(ellipse.label, 0)
                class_name = ellipse.label if ellipse.label in self.class_names else self.class_names[0]

                coco_image.add_annotation(
                    CocoAnnotation(
                        bbox=[x1, y1, width, height],  # COCO format: [x, y, w, h]
                        category_id=class_id,
                        category_name=class_name,
                        area=width * height,
                    )
                )

            self.coco.add_image(coco_image)

        logger.info(f"Converted to COCO: {len(self.coco.image_list)} images, "
                    f"{sum(len(img.annotations) for img in self.coco.image_list)} annotations")
        return self.coco

    def _resolve_image_path(self, ann: ImageAnnotation, images_dir: Path) -> Optional[Path]:
        """Resolve image path from annotation name."""
        name = Path(ann.name).name
        candidates = [
            images_dir / name,
            images_dir / ann.name,
            images_dir.parent / ann.name,
        ]
        for p in candidates:
            if p.is_file():
                return p
        return None

    def export_to_coco_json(self, output_path: Path) -> Path:
        """
        Export current COCO dataset to JSON file.

        Args:
            output_path: Path to save COCO JSON

        Returns:
            Path to saved JSON file
        """
        if self.coco is None:
            raise RuntimeError("No COCO dataset loaded. Call parse_cvat_to_coco first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_json(self.coco.json, output_path)
        logger.info(f"Exported COCO dataset to {output_path}")
        return output_path
