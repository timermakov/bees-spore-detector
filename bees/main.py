"""
Main module for bee spore analysis.

This module provides the command-line interface and main processing pipeline
for analyzing bee spore images using computer vision techniques.
"""

import os
import argparse
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal

from bees import io_utils, image_proc, spores, titer
from bees.config_loader import create_config_manager, ConfigurationError
from bees.grouping import create_group_manager, GroupedImageManager
from bees.reporting import ReportManager
from bees.image_proc import SporeDetectionPipeline
from bees.titer import TiterCalculator

# YOLO imports (optional)
try:
    from bees.yolo import YOLOConfig, SporeDetector, SporeCounter, SporeTrainer, DatasetPreparer, PseudoLabeler
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SporeAnalysisPipeline:
    """Main pipeline for spore analysis."""
    
    def __init__(self, config_manager, data_dir: str, results_dir: str, use_yolo: bool = False):
        """
        Initialize the analysis pipeline.
        
        Args:
            config_manager: Configuration manager instance
            data_dir: Directory containing input images
            results_dir: Directory for output results
            use_yolo: Whether to use YOLO-based detection
        """
        self.config_manager = config_manager
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.use_yolo = use_yolo and YOLO_AVAILABLE
        self.detection_pipeline = SporeDetectionPipeline()
        self.titer_calculator = TiterCalculator()
        
        # Initialize YOLO components if requested
        self.yolo_counter = None
        if self.use_yolo:
            yolo_config = YOLOConfig.from_config_manager(config_manager)
            self.yolo_counter = SporeCounter(yolo_config)
            logger.info("Using YOLO-based spore detection")
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_parameters(self) -> Dict[str, any]:
        """Get detection parameters from configuration."""
        return {
            'min_contour_area': self.config_manager.get_int_param('min_contour_area'),
            'max_contour_area': self.config_manager.get_int_param('max_contour_area'),
            'min_ellipse_area': self.config_manager.get_int_param('min_ellipse_area'),
            'max_ellipse_area': self.config_manager.get_int_param('max_ellipse_area'),
            'canny_threshold1': self.config_manager.get_int_param('canny_threshold1'),
            'canny_threshold2': self.config_manager.get_int_param('canny_threshold2'),
            'min_spore_contour_length': self.config_manager.get_int_param('min_spore_contour_length'),
            'intensity_threshold': self.config_manager.get_int_param('intensity_threshold'),
            'analysis_square_size': self.config_manager.get_int_param('analysis_square_size'),
            'analysis_square_line_width': self.config_manager.get_int_param('analysis_square_line_width'),
        }
    
    def process_image(self, 
                     image_path: str, 
                     xml_path: str, 
                     debug_prefix: Optional[str] = None) -> Dict[str, any]:
        """
        Process a single image for spore detection.
        
        Args:
            image_path: Path to the image file
            xml_path: Path to the metadata XML file
            debug_prefix: Optional prefix for debug output
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load image and metadata
            image = io_utils.ImageLoader.load_image(image_path)
            metadata = io_utils.MetadataLoader.load_metadata(xml_path)
            
            debug_base = Path(self.results_dir) / Path(image_path).stem
            
            # Use YOLO or OpenCV pipeline based on configuration
            if self.use_yolo and self.yolo_counter:
                return self._process_image_yolo(image_path, image, metadata, debug_base)
            else:
                return self._process_image_opencv(image_path, image, metadata, debug_base, debug_prefix)
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            raise
    
    def _process_image_yolo(self, image_path: str, image, metadata, debug_base: Path) -> Dict[str, any]:
        """Process image using YOLO detector."""
        counts, inside_dets, outside_dets = self.yolo_counter.count_with_detections(image_path)
        
        count = counts['inside']
        titer_value = self.titer_calculator.calculate_titer(count)
        
        # Save visualization
        vis_path = str(debug_base) + "_yolo_detections.jpg"
        self.yolo_counter.visualize(image_path, vis_path)
        
        # Convert detections to contour-like format for compatibility
        spore_objects = self._detections_to_contours(inside_dets + outside_dets)
        
        return {
            'image': image,
            'spore_objects': spore_objects,
            'count': count,
            'titer': titer_value,
            'metadata': metadata,
            'image_path': image_path,
            'debug_images': [vis_path] if Path(vis_path).exists() else [],
            'count_inside_square': counts['inside'],
            'count_outside_square': counts['outside'],
            'method': 'yolo'
        }
    
    def _process_image_opencv(self, image_path: str, image, metadata, 
                               debug_base: Path, debug_prefix: Optional[str]) -> Dict[str, any]:
        """Process image using OpenCV pipeline."""
        params = self.get_parameters()
        detection_pipeline = SporeDetectionPipeline(debug_path=str(debug_base))
        spore_objects = detection_pipeline.detect_spores(image, **params)
        
        try:
            square_size = int(self.config_manager.get_int_param('analysis_square_size'))
        except Exception:
            square_size = None
    
        if square_size and square_size > 0:
            img_width, img_height = image.size
            inside, outside = image_proc.count_spores_inside_outside(
                spore_objects, (img_width, img_height), square_size
            )
            count = inside
        else:
            count = spores.count_spores(spore_objects)
        inside, outside = count, 0
    
        titer_value = self.titer_calculator.calculate_titer(count)
        
        if debug_prefix:
            debug_path = str(debug_prefix) + '_debug'
            image_proc.save_debug_image(image, spore_objects, debug_path)

        debug_candidates = [
            f"{debug_base}_blur.jpg",
            f"{debug_base}_clahe.jpg",
            f"{debug_base}_edges.jpg",
            f"{debug_base}_edges_morph.jpg",
            f"{debug_base}_edges_nolines.jpg",
            f"{debug_base}_edges_close.jpg",
            f"{debug_base}_ellipses.jpg",
            f"{debug_base}_analysis_overlay.jpg",
            f"{debug_base}_debug.jpg",
        ]
        debug_images = [str(p) for p in map(Path, debug_candidates) if Path(p).exists()]
        
        return {
            'image': image,
            'spore_objects': spore_objects,
            'count': count,
            'titer': titer_value,
            'metadata': metadata,
            'image_path': image_path,
            'debug_images': debug_images,
        'count_inside_square': inside,
        'count_outside_square': outside,
        'method': 'opencv'
    }
    
    def _detections_to_contours(self, detections) -> List:
        """Convert YOLO detections to contour format for compatibility."""
        import numpy as np
        contours = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            contour = np.array([
                [[int(x1), int(y1)]],
                [[int(cx), int(y1)]],
                [[int(x2), int(y1)]],
                [[int(x2), int(cy)]],
                [[int(x2), int(y2)]],
                [[int(cx), int(y2)]],
                [[int(x1), int(y2)]],
                [[int(x1), int(cy)]]
            ], dtype=np.int32)
            contours.append(contour)
        return contours
    
    def process_group(self, 
                     group_prefix: str, 
                     image_paths: List[str]) -> Tuple[List[int], float, List[Dict]]:
        """
        Process a group of three images.
        
        Args:
            group_prefix: Group prefix name
            image_paths: List of three image paths
            
        Returns:
            Tuple of (counts, group_titer, results)
        """
        counts = []
        results = []
        
        for idx, image_path in enumerate(image_paths, 1):
            xml_path = image_path + '_meta.xml'
            
            if not os.path.exists(xml_path):
                logger.warning(f"Metadata file not found: {xml_path}")
                continue
            
            debug_prefix = Path(self.results_dir) / f"{Path(image_path).stem}"
            result = self.process_image(image_path, xml_path, debug_prefix)
            
            counts.append(result['count'])
            results.append(result)
            
            logger.info(f"Processed {group_prefix} sample {idx}: {result['count']} spores")
        
        if not counts:
            raise ValueError(f"No valid images processed for group {group_prefix}")
        
        group_titer = self.titer_calculator.calculate_titer(counts)
        logger.info(f"Group {group_prefix} titer: {group_titer:.2f} million spores/ml")
        
        return counts, group_titer, results
    
    def run_analysis(self) -> Dict[str, List[Tuple[int, float]]]:
        """
        Run the complete analysis pipeline.
        
        Returns:
            Dictionary mapping group prefixes to lists of (count, titer) tuples
        """
        # Create group manager
        group_manager = create_group_manager(str(self.data_dir))
        if not group_manager:
            raise RuntimeError("Failed to create group manager")
        
        logger.info(f"Found {group_manager.get_group_count()} image groups")
        
        # Process each group
        groups_results = {}
        self.group_results = {}  # Store detailed results for CVAT export
        
        for prefix in group_manager.list_group_prefixes():
            image_paths = group_manager.get_group(prefix)
            if image_paths:
                counts, group_titer, results = self.process_group(prefix, image_paths)
                # Store as (count, group_titer) tuples for each sample
                groups_results[prefix] = [(count, group_titer) for count in counts]
                # Store detailed results for CVAT export
                self.group_results[prefix] = results
        
        return groups_results


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
        images_dir = export_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        for image_path in image_files:
            shutil.copy(image_path, images_dir)
        
        # Build XML annotations
        root = self._build_annotations_xml(image_files, spore_objects_list)
        
        # Save XML
        xml_path = export_dir / 'annotations.xml'
        
        # Pretty-print the XML
        io_utils.XMLFormatter.indent_xml(root)
        
        # Convert to string and save
        import xml.etree.ElementTree as ET
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
        # Create ZIP file
        zip_path = self.output_dir / f'{task_name}.zip'
        self._create_zip(export_dir, zip_path)
        
        logger.info(f"CVAT export completed: {zip_path}")
        return str(zip_path)
    
    def _build_annotations_xml(self, 
                              image_files: List[str], 
                              spore_objects_list: List[List]) -> any:
        """Build the annotations XML structure."""
        import xml.etree.ElementTree as ET
        
        root = ET.Element('annotations')
        ET.SubElement(root, 'version').text = '1.1'
        
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
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_dir.rglob('*'):
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


class AnalysisRunner:
    """Main runner class for the analysis pipeline."""
    
    def __init__(self, config_path: str, data_dir: Optional[str] = None, 
                 results_dir: Optional[str] = None, use_yolo: bool = False):
        """
        Initialize the analysis runner.
        
        Args:
            config_path: Path to configuration file
            data_dir: Optional override for data directory
            results_dir: Optional override for results directory
            use_yolo: Whether to use YOLO-based detection
        """
        self.config_path = Path(config_path)
        self.config_manager = create_config_manager(config_path)
        self.use_yolo = use_yolo
        
        # Get directories
        self.data_dir = data_dir or self.config_manager.get_param('data_dir')
        self.results_dir = results_dir or self.config_manager.get_param('results_dir')
        
        # Create pipeline
        self.pipeline = SporeAnalysisPipeline(
            self.config_manager, self.data_dir, self.results_dir, use_yolo=use_yolo
        )
        
        # Create exporters
        self.cvat_exporter = CVATExporter(str(self.results_dir))
        self.report_manager = ReportManager(str(self.results_dir))
    
    def run(self, export_cvat: bool = False) -> Dict[str, any]:
        """
        Run the complete analysis.
        
        Args:
            export_cvat: Whether to export CVAT format
            
        Returns:
            Dictionary containing analysis results and report paths
        """
        logger.info("Starting bee spore analysis")
        
        try:
            # Run analysis
            groups_results = self.pipeline.run_analysis()
            
            # Generate reports
            reports = self.report_manager.generate_all_reports(groups_results)
            
            # Export CVAT if requested
            if export_cvat:
                # Collect all image files and spore objects
                image_files = []
                spore_objects_list = []
                
                for prefix, rows in groups_results.items():
                    group_paths = self._get_group_image_paths(prefix)
                    group_objects = self._get_group_spore_objects(prefix, rows)
                    
                    image_files.extend(group_paths)
                    spore_objects_list.extend(group_objects)
                
                if image_files:
                    cvat_path = self.cvat_exporter.export_task(
                        'bees_task', image_files, spore_objects_list
                    )
                    reports['cvat'] = cvat_path
            
            # Save run parameters
            self._save_run_parameters()
            
            logger.info("Analysis completed successfully")
            return {
                'groups_results': groups_results,
                'reports': reports
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _get_group_image_paths(self, prefix: str) -> List[str]:
        """Get image paths for a group."""
        group_manager = create_group_manager(str(self.data_dir))
        if group_manager and group_manager.has_group(prefix):
            return group_manager.get_group(prefix)
        return []
    
    def _get_group_spore_objects(self, prefix: str, rows: List[Tuple[int, float]]) -> List[List]:
        """Get spore objects for a group from stored results."""
        if hasattr(self.pipeline, 'group_results') and prefix in self.pipeline.group_results:
            return [result['spore_objects'] for result in self.pipeline.group_results[prefix]]
        # Fallback to empty lists if no stored results
        return [[] for _ in rows]
    
    def _save_run_parameters(self) -> None:
        """Save run parameters for reproducibility."""
        params_txt = Path(self.results_dir) / 'params_used.txt'
        
        log_lines = [
            "--------------------------------",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Config file: {self.config_path.absolute()}",
            f"Data dir: {Path(self.data_dir).absolute()}",
            f"Results dir: {Path(self.results_dir).absolute()}",
            "Parameters:"
        ]
        
        # Add all parameters
        all_params = self.config_manager.get_all_params()
        for key, value in all_params.items():
            log_lines.append(f"  {key}: {value}")
        
        log_lines.append("--------------------------------")
        
        try:
            with open(params_txt, 'a', encoding='utf-8') as f:
                f.write('\n'.join(log_lines) + '\n')
        except Exception as e:
            logger.warning(f"Could not write parameters log: {e}")


def train_yolo_model(config_path: str, quick_test: bool = False) -> int:
    """
    Train YOLO model for spore detection.
    
    Args:
        config_path: Path to config file
        quick_test: If True, use reduced settings for fast testing (~15 min)
    """
    if not YOLO_AVAILABLE:
        logger.error("YOLO module not available. Install ultralytics: pip install ultralytics")
        return 1
    
    try:
        config_manager = create_config_manager(config_path)
        yolo_config = YOLOConfig.from_config_manager(config_manager)
        
        if quick_test:
            logger.info("Starting YOLO QUICK TEST training (30 epochs, 640px)...")
        else:
            logger.info("Starting YOLO training pipeline...")
        
        # Prepare dataset with 75/25 split
        preparer = DatasetPreparer(yolo_config)
        data_yaml = preparer.prepare_dataset(val_split=0.4)  # 60/40 split for better validation
        
        # Train model
        trainer = SporeTrainer(yolo_config)
        metrics = trainer.train(data_yaml, quick_test=quick_test)
        
        logger.info(f"Training complete. Metrics: {metrics}")
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def pseudo_label(config_path: str, source_dir: str, output_dir: str = "pseudo_labels", confidence: float = 0.5, max_det: int = 1000) -> int:
    """Generate pseudo-labels for unlabeled images using trained model."""
    if not YOLO_AVAILABLE:
        logger.error("YOLO module not available")
        return 1
    
    try:
        config_manager = create_config_manager(config_path)
        yolo_config = YOLOConfig.from_config_manager(config_manager)
        
        logger.info(f"Generating pseudo-labels from {source_dir}")
        logger.info(f"Confidence threshold: {confidence}, max_det: {max_det}")
        
        labeler = PseudoLabeler(yolo_config)
        stats = labeler.generate_labels(
            Path(source_dir),
            Path(output_dir),
            confidence=confidence,
            max_det=max_det
        )
        
        logger.info(f"Pseudo-labeling complete: {stats['images']} images, {stats['detections']} detections")
        logger.info(f"Output saved to: {output_dir}")
        logger.info("Review the labels, then merge with: --merge-pseudo")
        return 0
        
    except Exception as e:
        logger.error(f"Pseudo-labeling failed: {e}")
        return 1


def merge_pseudo_labels(config_path: str, pseudo_dir: str = "pseudo_labels") -> int:
    """Merge verified pseudo-labels into training set."""
    if not YOLO_AVAILABLE:
        logger.error("YOLO module not available")
        return 1
    
    try:
        config_manager = create_config_manager(config_path)
        yolo_config = YOLOConfig.from_config_manager(config_manager)
        
        logger.info(f"Merging pseudo-labels from {pseudo_dir}")
        
        labeler = PseudoLabeler(yolo_config)
        stats = labeler.merge_with_training(Path(pseudo_dir))
        
        logger.info(f"Merge complete: {stats['added']} images added to training set")
        return 0
        
    except Exception as e:
        logger.error(f"Merge failed: {e}")
        return 1


def main():
    """Main entry point for the command-line interface."""
    parser = argparse.ArgumentParser(description='Bees Spore Counter CLI')
    parser.add_argument('-c', '--config', required=False, default='config.yaml', 
                       help='Path to YAML config')
    parser.add_argument('-d', '--data', required=False, 
                       help='Override data directory (images)')
    parser.add_argument('-o', '--output', required=False, 
                       help='Override results directory')
    parser.add_argument('--export-cvat-zip', action='store_true', 
                       help='Export CVAT 1.1 ZIP with ellipses')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    
    # YOLO arguments
    parser.add_argument('--use-yolo', action='store_true',
                       help='Use YOLO-based detection instead of OpenCV')
    parser.add_argument('--train-yolo', action='store_true',
                       help='Train YOLO model on annotated data')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick training mode: 30 epochs, 640px images (~15 min on CPU)')
    
    # Pseudo-labeling arguments
    parser.add_argument('--pseudo-label', action='store_true',
                       help='Generate pseudo-labels for unlabeled images')
    parser.add_argument('--pseudo-source', type=str, default='dataset_test',
                       help='Source directory for pseudo-labeling (default: dataset_test)')
    parser.add_argument('--pseudo-output', type=str, default='pseudo_labels',
                       help='Output directory for pseudo-labels (default: pseudo_labels)')
    parser.add_argument('--pseudo-conf', type=float, default=0.5,
                       help='Confidence threshold for pseudo-labeling (default: 0.5)')
    parser.add_argument('--pseudo-max-det', type=int, default=1000,
                       help='Max detections per image for pseudo-labeling (default: 1000)')
    parser.add_argument('--merge-pseudo', action='store_true',
                       help='Merge pseudo-labels into training set')
    parser.add_argument('--export-yolo-cvat', action='store_true',
                       help='Run YOLO on a folder and export predictions as CVAT ZIP (box format)')
    parser.add_argument('--yolo-source', type=str, default='dataset_test',
                       help='Source directory for YOLO->CVAT export (default: dataset_test)')
    parser.add_argument('--yolo-cvat-output', type=str, default='results',
                       help='Output directory for YOLO->CVAT export (default: results)')
    parser.add_argument('--yolo-cvat-task', type=str, default='yolo_auto_annotations',
                       help='Task name for YOLO->CVAT export ZIP/XML (default: yolo_auto_annotations)')
    parser.add_argument('--yolo-cvat-conf', type=float, default=None,
                       help='Confidence threshold override for YOLO->CVAT export (default: from config)')
    parser.add_argument('--yolo-cvat-max-det', type=int, default=1000,
                       help='Max detections per image for YOLO->CVAT export (default: 1000)')
    parser.add_argument('--yolo-cvat-shape', choices=['box', 'ellipse'], default='box',
                       help='Shape for YOLO->CVAT export (box or ellipse, default: box)')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle pseudo-labeling
    if args.pseudo_label:
        return pseudo_label(args.config, args.pseudo_source, args.pseudo_output, args.pseudo_conf, args.pseudo_max_det)
    
    if args.merge_pseudo:
        return merge_pseudo_labels(args.config, args.pseudo_output)

    if args.export_yolo_cvat:
        if not YOLO_AVAILABLE:
            logger.error("YOLO module not available. Install ultralytics: pip install ultralytics")
            return 1
        try:
            config_manager = create_config_manager(args.config)
            yolo_config = YOLOConfig.from_config_manager(config_manager)
            detector = SporeDetector(yolo_config)
            exporter = CVATExporter(args.yolo_cvat_output)
            zip_path = exporter.export_yolo_predictions(
                detector=detector,
                source_dir=Path(args.yolo_source),
                task_name=args.yolo_cvat_task,
                confidence=args.yolo_cvat_conf,
                max_det=args.yolo_cvat_max_det,
                shape_type=args.yolo_cvat_shape
            )
            print(f"YOLO CVAT export: {zip_path}")
            return 0
        except Exception as e:
            logger.error(f"YOLO -> CVAT export failed: {e}")
            return 1
    
    # Handle YOLO training mode
    if args.train_yolo:
        return train_yolo_model(args.config, quick_test=args.quick_test)
    
    # Check YOLO availability
    if args.use_yolo and not YOLO_AVAILABLE:
        logger.warning("YOLO not available, falling back to OpenCV. Install: pip install ultralytics")
        args.use_yolo = False
    
    try:
        # Create and run analysis
        runner = AnalysisRunner(args.config, args.data, args.output, use_yolo=args.use_yolo)
        results = runner.run(export_cvat=args.export_cvat_zip)
        
        # Print summary
        method = "YOLO" if args.use_yolo else "OpenCV"
        print(f"\n===== Analysis Summary ({method}) =====")
        print(f"Processed {len(results['groups_results'])} groups")
        print(f"Reports generated: {len(results['reports'])}")
        if 'cvat' in results['reports']:
            print(f"CVAT export: {results['reports']['cvat']}")
        print("============================\n")
        
    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main()) 