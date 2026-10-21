"""Write CVAT XML annotations as YOLO label files."""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .cvat import (
    EllipseAnnotation,
    ImageAnnotation,
    clipped_ellipse_box,
    ellipse_to_xyxy_pixels,
    find_annotated_image,
    normalized_yolo_box,
    read_cvat_xml,
    require_known_label,
    require_positive_image_size,
    xyxy_to_yolo_line,
)

logger = logging.getLogger(__name__)


class CVATToYOLOConverter:
    """Converts CVAT XML annotations to YOLO format."""

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ["spore"]
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}

    def parse_cvat_xml(self, xml_path: Path) -> List[ImageAnnotation]:
        """Read ellipses and boxes from a CVAT XML file."""
        return read_cvat_xml(xml_path)

    def ellipse_to_bbox(
        self,
        ellipse: EllipseAnnotation,
        img_width: int,
        img_height: int,
    ) -> Tuple[float, float, float, float]:
        """Normalized YOLO center and size. Raises when the box is not visible."""
        x1, y1, x2, y2 = ellipse_to_xyxy_pixels(ellipse)
        box = normalized_yolo_box(x1, y1, x2, y2, img_width, img_height)
        if box is None:
            raise ValueError("Bounding box is outside the image")
        return box

    def convert_annotation(self, annotation: ImageAnnotation) -> List[str]:
        """YOLO lines for one image. Unknown labels are rejected."""
        require_positive_image_size(annotation.name, annotation.width, annotation.height)

        lines = []
        for ellipse in annotation.ellipses:
            class_id = require_known_label(ellipse.label, self.class_names, self.class_to_id)
            corners = clipped_ellipse_box(ellipse, annotation.width, annotation.height)
            if corners is None:
                logger.warning(
                    "Skipping box outside image %s for label %s",
                    annotation.name,
                    ellipse.label,
                )
                continue
            left, top, right, bottom = corners
            line = xyxy_to_yolo_line(
                left,
                top,
                right,
                bottom,
                annotation.width,
                annotation.height,
                class_id,
            )
            if line is not None:
                lines.append(line)
        return lines

    def convert_and_save(
        self,
        xml_path: Path,
        images_root: Path,
        output_dir: Path,
        copy_images: bool = True,
    ) -> Dict[str, int]:
        """Convert one CVAT file and write YOLO images and labels."""
        annotations = self.parse_cvat_xml(xml_path)
        output_dir = Path(output_dir)
        images_out = output_dir / "images"
        labels_out = output_dir / "labels"
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        stats = {"images": 0, "annotations": 0, "skipped_images": 0}
        for annotation in annotations:
            src_image = find_annotated_image(annotation, Path(images_root))
            if src_image is None:
                logger.warning("Image not found: %s", annotation.name)
                stats["skipped_images"] += 1
                continue

            img_name = Path(annotation.name).name
            if copy_images:
                shutil.copy2(src_image, images_out / img_name)

            lines = self.convert_annotation(annotation)
            label_file = labels_out / f"{Path(img_name).stem}.txt"
            label_file.write_text("\n".join(lines), encoding="utf-8")
            stats["images"] += 1
            stats["annotations"] += len(lines)
            logger.debug("Converted %s: %d annotations", img_name, len(lines))

        logger.info(
            "Conversion complete: %d images, %d annotations",
            stats["images"],
            stats["annotations"],
        )
        return stats


def convert_cvat_to_yolo(xml_path: str, images_root: str, output_dir: str) -> Dict[str, int]:
    """Convert one CVAT XML file to a YOLO images/labels folder."""
    converter = CVATToYOLOConverter()
    return converter.convert_and_save(Path(xml_path), Path(images_root), Path(output_dir))
