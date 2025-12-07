"""
Spore counter with analysis zone support.

Counts spores inside/outside the analysis square zone.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
import logging
import numpy as np
from PIL import Image
import cv2

from .config import YOLOConfig
from .detector import SporeDetector, Detection

logger = logging.getLogger(__name__)


class SporeCounter:
    """Counts spores with analysis zone support."""
    
    def __init__(self, config: YOLOConfig, detector: Optional[SporeDetector] = None):
        """
        Initialize counter.
        
        Args:
            config: YOLOConfig instance
            detector: Optional SporeDetector (created if not provided)
        """
        self.config = config
        self.detector = detector or SporeDetector(config)
    
    def count_in_image(self,
                       image: Union[str, Path, Image.Image, np.ndarray],
                       confidence: Optional[float] = None) -> Dict[str, int]:
        """
        Count spores in image with analysis zone breakdown.
        
        Args:
            image: Image input
            confidence: Confidence threshold
            
        Returns:
            Dict with 'total', 'inside', 'outside' counts
        """
        # Get image dimensions
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                img_width, img_height = img.size
        elif isinstance(image, Image.Image):
            img_width, img_height = image.size
        else:
            img_height, img_width = image.shape[:2]
        
        # Detect spores
        detections = self.detector.detect(image, confidence)
        
        # Count inside/outside analysis zone
        square_size = self.config.analysis_square_size
        
        if square_size <= 0:
            return {
                'total': len(detections),
                'inside': len(detections),
                'outside': 0
            }
        
        bounds = self._compute_zone_bounds(img_width, img_height, square_size)
        
        inside = 0
        outside = 0
        
        for det in detections:
            cx, cy = det.center
            if self._is_inside_zone(bounds, cx, cy):
                inside += 1
            else:
                outside += 1
        
        return {
            'total': len(detections),
            'inside': inside,
            'outside': outside
        }
    
    def count_with_detections(self,
                              image: Union[str, Path, Image.Image, np.ndarray],
                              confidence: Optional[float] = None
                              ) -> Tuple[Dict[str, int], List[Detection], List[Detection]]:
        """
        Count spores and return detections split by zone.
        
        Args:
            image: Image input
            confidence: Confidence threshold
            
        Returns:
            Tuple of (counts_dict, inside_detections, outside_detections)
        """
        # Get image dimensions
        if isinstance(image, (str, Path)):
            with Image.open(image) as img:
                img_width, img_height = img.size
        elif isinstance(image, Image.Image):
            img_width, img_height = image.size
        else:
            img_height, img_width = image.shape[:2]
        
        detections = self.detector.detect(image, confidence)
        
        square_size = self.config.analysis_square_size
        
        if square_size <= 0:
            return {
                'total': len(detections),
                'inside': len(detections),
                'outside': 0
            }, detections, []
        
        bounds = self._compute_zone_bounds(img_width, img_height, square_size)
        
        inside_dets = []
        outside_dets = []
        
        for det in detections:
            cx, cy = det.center
            if self._is_inside_zone(bounds, cx, cy):
                inside_dets.append(det)
            else:
                outside_dets.append(det)
        
        counts = {
            'total': len(detections),
            'inside': len(inside_dets),
            'outside': len(outside_dets)
        }
        
        return counts, inside_dets, outside_dets
    
    def visualize(self,
                  image: Union[str, Path, Image.Image, np.ndarray],
                  output_path: Optional[str] = None,
                  confidence: Optional[float] = None,
                  line_width: int = 1,
                  show_labels: bool = False) -> np.ndarray:
        """
        Visualize detections with analysis zone.
        
        Args:
            image: Image input
            output_path: Optional path to save visualization
            confidence: Confidence threshold
            line_width: Line width for drawing
            show_labels: Whether to show per-detection labels (disabled by default)
            
        Returns:
            Visualization as numpy array (BGR)
        """
        # Load image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        img_height, img_width = img.shape[:2]
        
        # Get detections
        counts, inside_dets, outside_dets = self.count_with_detections(image, confidence)
        
        # Draw analysis zone
        square_size = self.config.analysis_square_size
        if square_size > 0:
            x1, y1, x2, y2 = self._compute_zone_bounds(img_width, img_height, square_size)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), line_width + 1)
        
        # Draw detections - red for inside, blue for outside (boxes only, no text)
        for det in inside_dets:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), line_width)
        
        for det in outside_dets:
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), line_width)
        
        # Add summary count text at top
        text = f"Inside: {counts['inside']} | Outside: {counts['outside']} | Total: {counts['total']}"
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save if path provided
        if output_path:
            cv2.imwrite(output_path, img)
            logger.info(f"Visualization saved to {output_path}")
        
        return img
    
    def _compute_zone_bounds(self, 
                              img_width: int, 
                              img_height: int, 
                              square_size: int) -> Tuple[int, int, int, int]:
        """Compute bounds of centered analysis zone."""
        cx, cy = img_width // 2, img_height // 2
        half = square_size // 2
        
        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(img_width - 1, cx + half)
        y2 = min(img_height - 1, cy + half)
        
        return x1, y1, x2, y2
    
    def _is_inside_zone(self, 
                        bounds: Tuple[int, int, int, int], 
                        x: float, 
                        y: float) -> bool:
        """Check if point is inside zone bounds."""
        x1, y1, x2, y2 = bounds
        return x1 <= x <= x2 and y1 <= y <= y2


def count_spores_yolo(image_path: str, 
                      config: YOLOConfig) -> Dict[str, int]:
    """
    Convenience function to count spores using YOLO.
    
    Args:
        image_path: Path to image
        config: YOLOConfig instance
        
    Returns:
        Count dict with 'total', 'inside', 'outside'
    """
    counter = SporeCounter(config)
    return counter.count_in_image(image_path)
