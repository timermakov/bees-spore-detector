"""
Export CVAT annotations to fixed-size tiled images with Pascal VOC XML per tile.

Used to build training crops from full-resolution microscopy frames.
"""

from __future__ import annotations

import logging
import random
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from xml.dom import minidom

import cv2

from .converter import CVATToYOLOConverter, ImageAnnotation, ellipse_to_xyxy_pixels
from .tiling import crop_tile_to_square, tile_axis_origins

logger = logging.getLogger(__name__)


@dataclass
class TiledDatasetExportStats:
    """Counts after exporting a tiled Pascal VOC dataset."""

    total_tiles: int
    tiles_with_objects: int
    empty_tiles: int
    box_count: int
    tile_size: int
    seed: int


def _build_pascal_annotation_root(filename: str, boxes_xyxy: List[List[float]], size: Tuple[int, int, int]) -> ET.Element:
    width, height, depth = size
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = filename
    size_el = ET.SubElement(annotation, "size")
    ET.SubElement(size_el, "width").text = str(width)
    ET.SubElement(size_el, "height").text = str(height)
    ET.SubElement(size_el, "depth").text = str(depth)
    for box in boxes_xyxy:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = "spore"
        bnd = ET.SubElement(obj, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(int(box[0]))
        ET.SubElement(bnd, "ymin").text = str(int(box[1]))
        ET.SubElement(bnd, "xmax").text = str(int(box[2]))
        ET.SubElement(bnd, "ymax").text = str(int(box[3]))
    return annotation


class CvatTiledPascalExporter:
    """Slices images listed in a CVAT 1.1 XML into tiles with Pascal VOC labels."""

    def __init__(self, class_name: str = "spore") -> None:
        self._converter = CVATToYOLOConverter([class_name])

    def export(
        self,
        cvat_xml: Path,
        images_dir: Path,
        output_dir: Path,
        tile_size: int = 512,
        overlap: float = 0.25,
        negative_ratio: float = 0.1,
        seed: Optional[int] = None,
        replace_output: bool = True,
    ) -> TiledDatasetExportStats:
        """
        Write ``output_dir/images``, ``output_dir/labels_xml``, and ``annotations_all.xml``.

        Empty tiles are subsampled: each empty tile is kept with probability ``negative_ratio``.
        """
        cvat_xml = Path(cvat_xml)
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)

        current_seed = int(seed) if seed is not None else int(time.time())
        random.seed(current_seed)

        if replace_output and output_dir.exists():
            shutil.rmtree(output_dir)
        (output_dir / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / "labels_xml").mkdir(parents=True, exist_ok=True)

        annotations = self._converter.parse_cvat_xml(cvat_xml)
        tree = ET.parse(cvat_xml)
        root = tree.getroot()

        common_root = ET.Element("annotations")
        meta_node = root.find("meta")
        if meta_node is not None:
            common_root.append(meta_node)

        total = empty_kept = with_objs = box_count = 0
        img_id = 0

        for ann in annotations:
            boxes_xyxy = [list(ellipse_to_xyxy_pixels(e)) for e in ann.ellipses]
            img_path = self._resolve_image_path(ann, images_dir)
            if img_path is None or not img_path.is_file():
                logger.warning("Skipping missing image: %s", ann.name)
                continue

            image_bgr = cv2.imread(str(img_path))
            if image_bgr is None:
                logger.warning("Could not read image: %s", img_path)
                continue

            h, w = image_bgr.shape[:2]
            depth = image_bgr.shape[2]
            y_origins = tile_axis_origins(h, tile_size, overlap)
            x_origins = tile_axis_origins(w, tile_size, overlap)

            for y1 in y_origins:
                for x1 in x_origins:
                    y2 = min(y1 + tile_size, h)
                    x2 = min(x1 + tile_size, w)
                    crop_h, crop_w = y2 - y1, x2 - x1

                    tile_boxes: List[List[float]] = []
                    for box in boxes_xyxy:
                        cx = (box[0] + box[2]) / 2.0
                        cy = (box[1] + box[3]) / 2.0
                        if x1 <= cx < x1 + tile_size and y1 <= cy < y1 + tile_size:
                            tile_boxes.append(
                                [
                                    max(0.0, box[0] - x1),
                                    max(0.0, box[1] - y1),
                                    min(float(crop_w), box[2] - x1),
                                    min(float(crop_h), box[3] - y1),
                                ]
                            )

                    if not tile_boxes:
                        if random.random() > negative_ratio:
                            continue
                        empty_kept += 1
                    else:
                        with_objs += 1
                        box_count += len(tile_boxes)

                    total += 1
                    stem = Path(ann.name).stem
                    t_name = f"{stem}_tile_{y1}_{x1}.jpg"
                    tile_img, _, _ = crop_tile_to_square(image_bgr, y1, x1, tile_size)
                    cv2.imwrite(str(output_dir / "images" / t_name), tile_img)

                    xml_el = _build_pascal_annotation_root(t_name, tile_boxes, (tile_size, tile_size, depth))
                    label_path = output_dir / "labels_xml" / f"{Path(t_name).stem}.xml"
                    label_path.write_text(
                        minidom.parseString(ET.tostring(xml_el)).toprettyxml(indent="   "),
                        encoding="utf-8",
                    )

                    img_el = ET.SubElement(
                        common_root,
                        "image",
                        {
                            "id": str(img_id),
                            "name": t_name,
                            "width": str(tile_size),
                            "height": str(tile_size),
                        },
                    )
                    for b in tile_boxes:
                        ET.SubElement(
                            img_el,
                            "box",
                            {
                                "label": "spore",
                                "xtl": str(b[0]),
                                "ytl": str(b[1]),
                                "xbr": str(b[2]),
                                "ybr": str(b[3]),
                            },
                        )
                    img_id += 1

        all_xml = output_dir / "annotations_all.xml"
        all_xml.write_text(
            minidom.parseString(ET.tostring(common_root)).toprettyxml(indent="  "),
            encoding="utf-8",
        )

        logger.info(
            "Tiled export: %s tiles (%s with objects, %s empty kept), seed=%s",
            total,
            with_objs,
            empty_kept,
            current_seed,
        )
        return TiledDatasetExportStats(
            total_tiles=total,
            tiles_with_objects=with_objs,
            empty_tiles=empty_kept,
            box_count=box_count,
            tile_size=tile_size,
            seed=current_seed,
        )

    def _resolve_image_path(self, ann: ImageAnnotation, images_dir: Path) -> Optional[Path]:
        name = Path(ann.name).name
        candidates = [images_dir / name, images_dir / ann.name, images_dir.parent / ann.name]
        for p in candidates:
            if p.is_file():
                return p
        return None
