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
        Parse CVAT XML file and extract all annotations.
        
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
        cx, cy = ellipse.cx, ellipse.cy
        rx, ry = ellipse.rx, ellipse.ry
        rotation_deg = ellipse.rotation
        
        # Convert rotation to radians
        theta = math.radians(rotation_deg)
        
        # Calculate AABB for rotated ellipse
        # Width and height of AABB containing rotated ellipse
        cos_t = abs(math.cos(theta))
        sin_t = abs(math.sin(theta))
        
        bbox_half_w = rx * cos_t + ry * sin_t
        bbox_half_h = rx * sin_t + ry * cos_t
        
        # Full bbox dimensions
        bbox_w = 2 * bbox_half_w
        bbox_h = 2 * bbox_half_h
        
        # Normalize to [0, 1]
        x_center_norm = cx / img_width
        y_center_norm = cy / img_height
        w_norm = bbox_w / img_width
        h_norm = bbox_h / img_height
        
        # Clamp to valid range
        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        w_norm = max(0.001, min(1.0, w_norm))
        h_norm = max(0.001, min(1.0, h_norm))
        
        return x_center_norm, y_center_norm, w_norm, h_norm
    
    def convert_annotation(self, annotation: ImageAnnotation) -> List[str]:
        """
        Convert ImageAnnotation to YOLO format lines.
        
        Args:
            annotation: ImageAnnotation object
            
        Returns:
            List of YOLO format strings (one per object)
        """
        lines = []
        
        for ellipse in annotation.ellipses:
            class_id = self.class_to_id.get(ellipse.label, 0)
            x, y, w, h = self.ellipse_to_bbox(
                ellipse, annotation.width, annotation.height
            )
            lines.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        
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
