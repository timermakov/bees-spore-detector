"""
Batch tiled inference: Pascal VOC XML per image and optional preview overlays.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Set
from xml.dom import minidom

import cv2

from .detector import SporeDetector

logger = logging.getLogger(__name__)


def _write_pascal_voc(filename: str, img_shape: tuple, detections, output_path: Path) -> None:
    height, width, depth = img_shape
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = filename
    size_el = ET.SubElement(annotation, "size")
    ET.SubElement(size_el, "width").text = str(width)
    ET.SubElement(size_el, "height").text = str(height)
    ET.SubElement(size_el, "depth").text = str(depth)
    for det in detections:
        x1, y1, x2, y2 = (int(det.x1), int(det.y1), int(det.x2), int(det.y2))
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = det.class_name
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(max(0, x1))
        ET.SubElement(bbox, "ymin").text = str(max(0, y1))
        ET.SubElement(bbox, "xmax").text = str(min(width, x2))
        ET.SubElement(bbox, "ymax").text = str(min(height, y2))
    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    output_path.write_text(xml_str, encoding="utf-8")


def run_tiled_prediction_folder(
    detector: SporeDetector,
    source_dir: Path,
    output_dir: Path,
    *,
    tile_size: Optional[int] = None,
    overlap: float = 0.2,
    merge_iou: float = 0.15,
    confidence: Optional[float] = None,
    imgsz: Optional[int] = None,
    use_clahe: bool = True,
    write_previews: bool = True,
) -> Dict[str, int]:
    """
    Run tiled detection on every image in ``source_dir`` and write XML (+ optional previews).

    Returns:
        ``{"images": n, "detections": m}`` counts.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed: Set[str] = {e.lower() for e in detector.config.image_extensions}
    paths = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() in allowed)

    total_det = 0
    processed = 0
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            logger.warning("Skip (unreadable): %s", path.name)
            continue

        detections = detector.detect_tiled(
            bgr,
            tile_size=tile_size,
            overlap=overlap,
            merge_iou=merge_iou,
            confidence=confidence,
            imgsz=imgsz,
            use_clahe=use_clahe,
        )
        total_det += len(detections)

        h, w = bgr.shape[:2]
        depth = bgr.shape[2]
        _write_pascal_voc(path.name, (h, w, depth), detections, output_dir / f"{path.stem}.xml")

        if write_previews and detections:
            vis = bgr.copy()
            for det in detections:
                cv2.rectangle(vis, (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2)), (0, 255, 0), 2)
            cv2.imwrite(str(output_dir / f"pred_{path.name}"), vis)

        logger.info("%s: %s detections", path.name, len(detections))
        processed += 1

    return {"images": processed, "detections": total_det}
