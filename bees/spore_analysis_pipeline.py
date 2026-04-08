"""
Main processing pipeline for bee spore analysis.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from bees import io_utils, image_proc, spores
from bees.grouping import create_group_manager
from bees.image_proc import SporeDetectionPipeline
from bees.titer import TiterCalculator

# YOLO imports (optional)
try:
    from bees.yolo import YOLOConfig, SporeCounter
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

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

    def get_parameters(self) -> Dict[str, Any]:
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
                      debug_prefix: Optional[str] = None) -> Dict[str, Any]:
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
            return self._process_image_opencv(image_path, image, metadata, debug_base, debug_prefix)

        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            raise

    def _process_image_yolo(self, image_path: str, image, metadata, debug_base: Path) -> Dict[str, Any]:
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
                              debug_base: Path, debug_prefix: Optional[str]) -> Dict[str, Any]:
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
            xml_path = image_path + "_meta.xml"

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
