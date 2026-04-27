"""
Main processing pipeline for bee spore analysis.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from bees import io_utils, image_proc, spores
from bees.image_proc import SporeDetectionPipeline
from bees.titer import create_calculator_from_config

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
        self.titer_calculator = create_calculator_from_config(config_manager)

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
        detections = self.yolo_counter.detector.detect(image_path)  # получаем все детекции
        count = len(detections)

        # Save visualization
        vis_path = str(debug_base) + "_yolo_detections.jpg"
        self.yolo_counter.visualize(image_path, vis_path)

        # Convert detections to contour-like format for compatibility
        spore_objects = self._detections_to_contours(detections)

        return {
            'image': image,
            'spore_objects': spore_objects,
            'count': count,
            'metadata': metadata,
            'image_path': image_path,
            'debug_images': [vis_path] if Path(vis_path).exists() else [],
            'count_outside_square': 0,
            'method': 'yolo'
        }

    def _process_image_opencv(self, image_path: str, image, metadata,
                              debug_base: Path, debug_prefix: Optional[str]) -> Dict[str, Any]:
        """Process image using OpenCV pipeline."""
        params = self.get_parameters()
        detection_pipeline = SporeDetectionPipeline(debug_path=str(debug_base))
        spore_objects = detection_pipeline.detect_spores(image, **params)

        count = len(spore_objects)

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
            'metadata': metadata,
            'image_path': image_path,
            'debug_images': debug_images,
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



    def run_analysis(self) -> Dict[str, List[Tuple[int, float]]]:
        """
        Run the complete analysis pipeline on all images in data_dir (no grouping).

        Returns:
            Dictionary with one key 'all_images' containing list of (count, titer) tuples.
        """
        from bees.grouping import find_images  # вспомогательная функция для поиска изображений

        # Находим все изображения в data_dir (рекурсивно или только в корне? оставим только корень)
        image_paths = find_images(self.data_dir)
        if not image_paths:
            raise RuntimeError(f"No images found in {self.data_dir}")

        logger.info(f"Found {len(image_paths)} images for processing")

        all_results = []
        photo_data = []  # (count, width, height)

        for img_path in image_paths:
            # Ищем соответствующий XML (имя_файла.jpg_meta.xml)
            xml_path = img_path.with_suffix(img_path.suffix + "_meta.xml")
            if not xml_path.exists():
                logger.warning(f"Metadata not found: {xml_path}, skipping {img_path.name}")
                continue

            debug_prefix = self.results_dir / img_path.stem
            result = self.process_image(str(img_path), str(xml_path), str(debug_prefix))
            all_results.append(result)

            # Собираем данные для титра
            photo_data.append((result['count'], result['image'].size[0], result['image'].size[1]))
            logger.info(f"Processed {img_path.name}: {result['count']} spores")

        if not photo_data:
            raise RuntimeError("No images were successfully processed")

        # Вычисляем общий титр по формуле с учётом площади
        titer = self.titer_calculator.calculate_sample_titer(photo_data)

        # Формируем результат в старом формате (для совместимости с отчётами)
        groups_results = {
            'all_images': [(count, titer) for count, _, _ in photo_data]
        }
        # Сохраняем детальные результаты для CVAT (если нужно)
        self.group_results = {'all_images': all_results}

        return groups_results