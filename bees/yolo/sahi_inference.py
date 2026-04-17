"""
SAHI-based sliced inference for large image detection.

Replaces custom tiling and detection merging with SAHI's framework-agnostic approach.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, List
from xml.dom import minidom

import cv2
import numpy as np

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction, get_prediction
    from sahi.utils.coco import Coco
    from sahi.utils.file import save_json
except ImportError:
    raise ImportError("sahi package required. Install with: pip install sahi")

logger = logging.getLogger(__name__)


class SAHIDetector:
    """SAHI-based detector wrapper for sliced inference."""

    def __init__(
        self,
        model_path: Path | str,
        model_type: str = "ultralytics",
        confidence_threshold: float = 0.25,
        device: str = "cuda:0",
    ):
        """
        Initialize SAHI detector.

        Args:
            model_path: Path to model weights file
            model_type: Model type ('ultralytics', 'mmdet', 'huggingface', 'torchvision')
            confidence_threshold: Confidence threshold for detections
            device: Device to use ('cpu', 'cuda:0', etc.)
        """
        self.model_path = Path(model_path)
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.detection_model = None
        self._load_model()

    def _load_model(self):
        """Load SAHI detection model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Loading {self.model_type} model from {self.model_path}")
        
        try:
            self.detection_model = AutoDetectionModel.from_pretrained(
                model_type=self.model_type,
                model_path=str(self.model_path),
                confidence_threshold=self.confidence_threshold,
                device=self.device,
            )
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def detect(
        self,
        image: np.ndarray,
        confidence: Optional[float] = None,
    ) -> List[Dict]:
        """
        Run standard (non-sliced) inference on an image.

        Args:
            image: BGR image as numpy array
            confidence: Optional confidence threshold override

        Returns:
            List of detection dicts with keys: bbox, confidence, class_id, class_name
        """
        if self.detection_model is None:
            raise RuntimeError("Model not loaded")

        conf = confidence if confidence is not None else self.confidence_threshold

        # Convert BGR to RGB for SAHI
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = get_prediction(image=image_rgb, detection_model=self.detection_model)

        detections = []
        for pred in result.object_prediction_list:
            bbox_xyxy = pred.bbox.to_xyxy()
            score = pred.score.value
            
            if score < conf:
                continue

            detections.append({
                "bbox": bbox_xyxy,  # [x1, y1, x2, y2]
                "confidence": score,
                "class_id": pred.category.id,
                "class_name": pred.category.name,
            })

        return detections

    def detect_sliced(
        self,
        image: np.ndarray,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        confidence: Optional[float] = None,
        progress_bar: bool = False,
    ) -> List[Dict]:
        """
        Run sliced inference on an image.

        Args:
            image: BGR image as numpy array
            slice_height: Height of each slice
            slice_width: Width of each slice
            overlap_height_ratio: Vertical overlap ratio (0-1)
            overlap_width_ratio: Horizontal overlap ratio (0-1)
            confidence: Optional confidence threshold override
            progress_bar: Whether to show tqdm progress bar

        Returns:
            List of detection dicts with keys: bbox, confidence, class_id, class_name
        """
        if self.detection_model is None:
            raise RuntimeError("Model not loaded")

        conf = confidence if confidence is not None else self.confidence_threshold

        # Convert BGR to RGB for SAHI
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = get_sliced_prediction(
            image=image_rgb,
            detection_model=self.detection_model,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            perform_standard_pred=False,  # Only sliced
            progress_bar=progress_bar,
        )

        detections = []
        for pred in result.object_prediction_list:
            bbox_xyxy = pred.bbox.to_xyxy()
            score = pred.score.value
            
            if score < conf:
                continue

            detections.append({
                "bbox": bbox_xyxy,  # [x1, y1, x2, y2]
                "confidence": score,
                "class_id": pred.category.id,
                "class_name": pred.category.name,
            })

        return detections


def _write_pascal_voc_xml(
    filename: str,
    img_shape: tuple,
    detections: List[Dict],
    output_path: Path,
) -> None:
    """Write detections to Pascal VOC XML format."""
    height, width, depth = img_shape
    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = filename
    
    size_el = ET.SubElement(annotation, "size")
    ET.SubElement(size_el, "width").text = str(width)
    ET.SubElement(size_el, "height").text = str(height)
    ET.SubElement(size_el, "depth").text = str(depth)

    for det in detections:
        x1, y1, x2, y2 = (int(det["bbox"][0]), int(det["bbox"][1]),
                          int(det["bbox"][2]), int(det["bbox"][3]))
        
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = det["class_name"]
        ET.SubElement(obj, "confidence").text = str(det["confidence"])
        
        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(max(0, x1))
        ET.SubElement(bbox, "ymin").text = str(max(0, y1))
        ET.SubElement(bbox, "xmax").text = str(min(width, x2))
        ET.SubElement(bbox, "ymax").text = str(min(height, y2))

    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    output_path.write_text(xml_str, encoding="utf-8")


def run_sliced_inference_folder(
    detector: SAHIDetector,
    source_dir: Path,
    output_dir: Path,
    *,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
    confidence: Optional[float] = None,
    image_extensions: Optional[List[str]] = None,
    write_previews: bool = True,
    progress_bar: bool = False,
) -> Dict[str, int]:
    """
    Run sliced inference on all images in a folder.

    Args:
        detector: SAHIDetector instance
        source_dir: Directory containing images
        output_dir: Directory to save results
        slice_height: Height of each slice
        slice_width: Width of each slice
        overlap_height_ratio: Vertical overlap ratio
        overlap_width_ratio: Horizontal overlap ratio
        confidence: Optional confidence threshold override
        image_extensions: List of image extensions to process (default: jpg, png, tiff)
        write_previews: Whether to write preview images with bounding boxes
        progress_bar: Whether to show progress bar

    Returns:
        Dict with "images" and "detections" counts
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if image_extensions is None:
        image_extensions = [".jpg", ".jpeg", ".png", ".tiff", ".tif"]
    
    image_extensions = {ext.lower() for ext in image_extensions}

    # Find all images
    image_paths = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    )

    total_detections = 0
    processed = 0

    for image_path in image_paths:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            logger.warning(f"Skip (unreadable): {image_path.name}")
            continue

        try:
            detections = detector.detect_sliced(
                bgr,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_height_ratio,
                overlap_width_ratio=overlap_width_ratio,
                confidence=confidence,
                progress_bar=progress_bar,
            )
        except Exception as e:
            logger.error(f"Error processing {image_path.name}: {e}")
            continue

        total_detections += len(detections)

        # Write Pascal VOC XML
        h, w = bgr.shape[:2]
        depth = bgr.shape[2]
        xml_output = output_dir / f"{image_path.stem}.xml"
        _write_pascal_voc_xml(image_path.name, (h, w, depth), detections, xml_output)

        # Write preview if requested
        if write_previews and detections:
            vis = bgr.copy()
            for det in detections:
                x1, y1, x2, y2 = int(det["bbox"][0]), int(det["bbox"][1]), int(det["bbox"][2]), int(det["bbox"][3])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            preview_path = output_dir / f"pred_{image_path.name}"
            cv2.imwrite(str(preview_path), vis)

        logger.info(f"{image_path.name}: {len(detections)} detections")
        processed += 1

    logger.info(f"Processed {processed} images, total {total_detections} detections")
    return {"images": processed, "detections": total_detections}


def predictions_to_coco(
    detections_per_image: List[tuple],
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Convert detection results to COCO format.

    Args:
        detections_per_image: List of (image_id, detections) tuples
        class_names: List of class names for category mapping

    Returns:
        COCO-format dictionary suitable for evaluation
    """
    if class_names is None:
        class_names = ["spore"]

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": idx, "name": name} for idx, name in enumerate(class_names)],
    }

    annotation_id = 0
    for image_id, (height, width, detections) in enumerate(detections_per_image):
        coco_data["images"].append({
            "id": image_id,
            "height": height,
            "width": width,
        })

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bbox_w = x2 - x1
            bbox_h = y2 - y1

            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": det["class_id"],
                "bbox": [x1, y1, bbox_w, bbox_h],  # COCO format
                "area": bbox_w * bbox_h,
                "iscrowd": 0,
                "score": det["confidence"],
            })
            annotation_id += 1

    return coco_data
