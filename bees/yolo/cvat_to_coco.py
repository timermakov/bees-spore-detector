"""Write CVAT XML annotations as a COCO dataset."""

import logging
from pathlib import Path
from typing import List, Optional

from .cvat import (
    ImageAnnotation,
    clipped_ellipse_box,
    find_annotated_image,
    read_cvat_xml,
    require_known_label,
    require_positive_image_size,
)

logger = logging.getLogger(__name__)


def _sahi_coco():
    """Import SAHI only when a COCO dataset is built or saved."""
    try:
        from sahi.utils.coco import Coco, CocoAnnotation, CocoCategory, CocoImage
        from sahi.utils.file import save_json
    except ImportError:
        raise ImportError("sahi package required. Install with: pip install sahi")
    return Coco, CocoAnnotation, CocoCategory, CocoImage, save_json


class CVATToCocoConverter:
    """Converts CVAT XML annotations to COCO format."""

    def __init__(self, class_names: Optional[List[str]] = None):
        self.class_names = class_names or ["spore"]
        self.class_to_id = {name: idx for idx, name in enumerate(self.class_names)}
        self.coco = None

    def parse_cvat_xml(self, xml_path: Path) -> List[ImageAnnotation]:
        """Read ellipses and boxes from a CVAT XML file."""
        return read_cvat_xml(xml_path)

    def _initialize_coco(self):
        Coco, _, CocoCategory, _, _ = _sahi_coco()
        coco = Coco()
        for idx, class_name in enumerate(self.class_names):
            coco.add_category(CocoCategory(id=idx, name=class_name))
        return coco

    def parse_cvat_to_coco(self, xml_path: Path, images_dir: Optional[Path] = None):
        """Parse CVAT XML and build a COCO dataset."""
        xml_path = Path(xml_path)
        images_dir = Path(images_dir) if images_dir else xml_path.parent
        self.coco = self._initialize_coco()
        converted_images = 0
        converted_annotations = 0

        for annotation in self.parse_cvat_xml(xml_path):
            image_path = find_annotated_image(annotation, images_dir)
            if image_path is None:
                logger.warning("Skipping missing image: %s", annotation.name)
                continue

            require_positive_image_size(annotation.name, annotation.width, annotation.height)
            try:
                file_name = str(image_path.relative_to(images_dir))
            except ValueError:
                file_name = image_path.name
            _, _, _, CocoImage, _ = _sahi_coco()
            coco_image = CocoImage(
                file_name=file_name,
                height=annotation.height,
                width=annotation.width,
            )
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
                width = right - left
                height = bottom - top
                coco_annotation = _coco_annotation(
                    bbox=[left, top, width, height],
                    category_id=class_id,
                    category_name=ellipse.label,
                    area=width * height,
                )
                coco_image.add_annotation(coco_annotation)
                converted_annotations += 1

            self.coco.add_image(coco_image)
            converted_images += 1

        logger.info("Converted to COCO: %s images, %s annotations", converted_images, converted_annotations)
        return self.coco

    def export_to_coco_json(self, output_path: Path) -> Path:
        """Write the current COCO dataset to JSON."""
        if self.coco is None:
            raise RuntimeError("No COCO dataset loaded. Call parse_cvat_to_coco first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _, _, _, _, save_json = _sahi_coco()
        save_json(self.coco.json, output_path)
        logger.info("Exported COCO dataset to %s", output_path)
        return output_path


def _coco_annotation(bbox, category_id: int, category_name: str, area: float):
    _, CocoAnnotation, _, _, _ = _sahi_coco()
    try:
        return CocoAnnotation(
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
            area=area,
        )
    except TypeError:
        return CocoAnnotation(
            bbox=bbox,
            category_id=category_id,
            category_name=category_name,
        )
