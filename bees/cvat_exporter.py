"""
CVAT export utilities for pipeline and YOLO predictions.
"""

from pathlib import Path
from typing import List, Optional, Literal, TYPE_CHECKING
import logging
import shutil
import zipfile

from bees import io_utils

if TYPE_CHECKING:
    from bees.yolo.detector import SporeDetector

logger = logging.getLogger(__name__)


class CVATExporter:
    """Handles export of results to CVAT format."""

    def __init__(self, output_dir: str):
        """
        Initialize the CVAT exporter.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.exporter = io_utils.CVATExporter()

    def export_task(self,
                    task_name: str,
                    image_files: List[str],
                    spore_objects_list: List[List]) -> str:
        """
        Export analysis results to CVAT format.

        Args:
            task_name: Name for the CVAT task
            image_files: List of image file paths
            spore_objects_list: List of spore object lists for each image

        Returns:
            Path to the generated ZIP file
        """
        export_dir = self.output_dir / task_name
        images_dir = export_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        for image_path in image_files:
            shutil.copy(image_path, images_dir)

        # Build XML annotations
        root = self._build_annotations_xml(image_files, spore_objects_list)

        # Save XML
        xml_path = export_dir / "annotations.xml"

        # Pretty-print the XML
        io_utils.XMLFormatter.indent_xml(root)

        # Convert to string and save
        import xml.etree.ElementTree as ET
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

        # Create ZIP file
        zip_path = self.output_dir / f"{task_name}.zip"
        self._create_zip(export_dir, zip_path)

        logger.info(f"CVAT export completed: {zip_path}")
        return str(zip_path)

    def _build_annotations_xml(self,
                               image_files: List[str],
                               spore_objects_list: List[List]):
        """Build the annotations XML structure."""
        import xml.etree.ElementTree as ET

        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"

        # Add meta and image elements
        for i, (image_path, spore_objects) in enumerate(zip(image_files, spore_objects_list)):
            meta, image_elem = self.exporter.export_image_elements(
                image_path, spore_objects, i
            )
            if i == 0:  # Use first meta
                root.insert(1, meta)
            root.append(image_elem)

        return root

    def _create_zip(self, source_dir: Path, zip_path: Path) -> None:
        """Create a ZIP file from the source directory."""
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_dir.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir)
                    zipf.write(file_path, arcname)

    def _list_image_files(self, source_dir: Path, extensions: List[str]) -> List[Path]:
        """List image files recursively from source directory."""
        valid_extensions = {ext.lower() for ext in extensions}
        image_files: List[Path] = []
        for file_path in source_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in valid_extensions:
                image_files.append(file_path)
        return sorted(image_files)

    def _build_yolo_meta(self, task_name: str, image_count: int, shape_type: str):
        """Build CVAT meta section for YOLO-generated annotations."""
        import xml.etree.ElementTree as ET

        meta = ET.Element("meta")
        task = ET.SubElement(meta, "task")
        ET.SubElement(task, "name").text = task_name
        ET.SubElement(task, "size").text = str(image_count)
        ET.SubElement(task, "mode").text = "annotation"
        ET.SubElement(task, "overlap").text = "0"
        labels = ET.SubElement(task, "labels")
        label = ET.SubElement(labels, "label")
        ET.SubElement(label, "name").text = "spore"
        ET.SubElement(label, "type").text = "ellipse" if shape_type == "ellipse" else "rectangle"
        ET.SubElement(label, "attributes")
        return meta

    def export_yolo_predictions(self,
                                detector: "SporeDetector",
                                source_dir: Path,
                                task_name: str,
                                confidence: Optional[float] = None,
                                max_det: int = 1000,
                                copy_images: bool = True,
                                shape_type: Literal["box", "ellipse"] = "box") -> str:
        """
        Run YOLO inference and export predictions to CVAT XML + ZIP.

        shape_type:
        - box: export detections as CVAT <box>
        - ellipse: export detections as CVAT <ellipse> derived from bbox geometry
        """
        import xml.etree.ElementTree as ET
        from PIL import Image

        source_dir = Path(source_dir)
        if not source_dir.exists() or not source_dir.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_dir}")

        image_files = self._list_image_files(source_dir, detector.config.image_extensions)
        if not image_files:
            raise FileNotFoundError(f"No images found in {source_dir}")

        export_dir = self.output_dir / task_name
        export_dir.mkdir(parents=True, exist_ok=True)
        images_dir = export_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        root = ET.Element("annotations")
        ET.SubElement(root, "version").text = "1.1"
        root.append(self._build_yolo_meta(task_name, len(image_files), shape_type))

        total_detections = 0
        for image_id, image_path in enumerate(image_files):
            with Image.open(image_path) as img:
                width, height = img.size

            detections = detector.detect(
                image=image_path,
                confidence=confidence,
                max_det=max_det
            )

            image_elem = ET.SubElement(root, "image", {
                "id": str(image_id),
                "name": image_path.name,
                "width": str(width),
                "height": str(height),
            })

            for det in detections:
                x1, y1, x2, y2 = float(det.x1), float(det.y1), float(det.x2), float(det.y2)
                if shape_type == "ellipse":
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    rx = max((x2 - x1) / 2, 0.5)
                    ry = max((y2 - y1) / 2, 0.5)
                    ET.SubElement(image_elem, "ellipse", {
                        "label": "spore",
                        "source": "auto",
                        "occluded": "0",
                        "z_order": "0",
                        "cx": f"{cx:.2f}",
                        "cy": f"{cy:.2f}",
                        "rx": f"{rx:.2f}",
                        "ry": f"{ry:.2f}",
                        "rotation": "0.00",
                    })
                else:
                    ET.SubElement(image_elem, "box", {
                        "label": "spore",
                        "source": "auto",
                        "occluded": "0",
                        "z_order": "0",
                        "xtl": f"{x1:.2f}",
                        "ytl": f"{y1:.2f}",
                        "xbr": f"{x2:.2f}",
                        "ybr": f"{y2:.2f}",
                    })
                total_detections += 1

            if copy_images:
                shutil.copy2(image_path, images_dir / image_path.name)

        io_utils.XMLFormatter.indent_xml(root)
        xml_path = export_dir / "annotations.xml"
        ET.ElementTree(root).write(xml_path, encoding="utf-8", xml_declaration=True)

        zip_path = self.output_dir / f"{task_name}.zip"
        self._create_zip(export_dir, zip_path)
        logger.info(
            "YOLO -> CVAT export completed: %s images, %s objects, zip=%s",
            len(image_files),
            total_detections,
            zip_path
        )
        return str(zip_path)
