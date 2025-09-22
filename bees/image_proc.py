"""
Image processing module for bee spore detection.

This module provides classes and functions for preprocessing images and detecting
spores using computer vision techniques.
"""

import logging
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2

# Configure logging
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Handles image preprocessing operations for spore detection."""
    
    def __init__(self, debug_path: Optional[str] = None):
        """
        Initialize the preprocessor.
        
        Args:
            debug_path: Optional path for saving debug images
        """
        self.debug_path = debug_path
    
    def preprocess(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess an image for spore detection.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Preprocessed grayscale image as numpy array
        """
        # Convert to grayscale
        gray = image.convert('L')
        arr = np.array(gray, dtype=np.uint8)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(arr, (5, 5), 2)
        if self.debug_path:
            self._save_debug_image(blurred, f"{self.debug_path}_blur", is_mask=True)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        if self.debug_path:
            self._save_debug_image(enhanced, f"{self.debug_path}_clahe", is_mask=True)
        
        return enhanced
    
    def _save_debug_image(self, image: np.ndarray, path: str, is_mask: bool = False) -> None:
        """Save debug image to disk."""
        try:
            cv2.imwrite(f"{path}.jpg", image)
        except Exception as e:
            logger.warning(f"Failed to save debug image {path}: {e}")


class EdgeDetector:
    """Handles edge detection and morphological operations."""
    
    def __init__(self, debug_path: Optional[str] = None):
        """
        Initialize the edge detector.
        
        Args:
            debug_path: Optional path for saving debug images
        """
        self.debug_path = debug_path
    
    def detect_edges(self, image: np.ndarray, 
                    canny_threshold1: int, 
                    canny_threshold2: int) -> np.ndarray:
        """
        Detect edges using Canny algorithm with dual thresholds.
        
        Args:
            image: Input grayscale image
            canny_threshold1: Lower Canny threshold
            canny_threshold2: Upper Canny threshold
            
        Returns:
            Binary edge image
        """
        # Apply bilateral filter to reduce noise while preserving edges
        denoised = cv2.bilateralFilter(image, d=5, sigmaColor=20, sigmaSpace=5)
        
        # Detect edges with two threshold levels for better coverage
        edges_strong = cv2.Canny(denoised, canny_threshold1, canny_threshold2)
        edges_soft = cv2.Canny(
            denoised, 
            max(0, int(canny_threshold1 * 0.8)), 
            max(0, int(canny_threshold2 * 0.8))
        )
        
        # Combine both edge maps
        edges = cv2.bitwise_or(edges_strong, edges_soft)
        
        if self.debug_path:
            self._save_debug_image(edges, f"{self.debug_path}_edges", is_mask=True)
        
        return edges, edges_strong
    
    def clean_edges(self, edges: np.ndarray) -> np.ndarray:
        """
        Clean edges using morphological operations and line removal.
        
        Args:
            edges: Binary edge image
            
        Returns:
            Cleaned edge image
        """
        # Remove noise with morphological opening
        kernel = np.ones((2, 2), np.uint8)
        edges_morph = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        if self.debug_path:
            self._save_debug_image(edges_morph, f"{self.debug_path}_edges_morph", is_mask=True)
        
        # Check if morphology was too aggressive
        if self._is_morphology_too_aggressive(edges, edges_morph):
            logger.debug("Morphology too aggressive, using raw edges")
            edges_working = edges
        else:
            edges_working = edges_morph
        
        # Remove long lines (hairs/boundaries)
        edges_no_lines = self._remove_long_lines(edges_working)
        if self.debug_path:
            self._save_debug_image(edges_no_lines, f"{self.debug_path}_edges_nolines", is_mask=True)
        
        # Close small gaps in contours
        edges_final = self._close_gaps(edges_no_lines)
        if self.debug_path:
            self._save_debug_image(edges_final, f"{self.debug_path}_edges_close", is_mask=True)
        
        return edges_final
    
    def _is_morphology_too_aggressive(self, original: np.ndarray, morphed: np.ndarray) -> bool:
        """Check if morphological operation removed too many edges."""
        sum_original = float(np.count_nonzero(original)) + 1e-9
        sum_morphed = float(np.count_nonzero(morphed))
        return sum_morphed < 0.1 * sum_original
    
    def _remove_long_lines(self, edges: np.ndarray) -> np.ndarray:
        """Remove long straight lines from edge image."""
        lines = cv2.HoughLinesP(
            edges, 1, np.pi/180, 150, 
            minLineLength=80, maxLineGap=5
        )
        
        if lines is not None:
            edges_copy = edges.copy()
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(edges_copy, (x1, y1), (x2, y2), 0, 2)
            return edges_copy
        
        return edges
    
    def _close_gaps(self, edges: np.ndarray) -> np.ndarray:
        """Close small gaps in contours using morphological closing."""
        kernel = np.ones((2, 2), np.uint8)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Check if closing introduced too much noise
        if float(np.count_nonzero(edges_closed)) > 1.5 * float(np.count_nonzero(edges)):
            return edges
        
        return edges_closed
    
    def _save_debug_image(self, image: np.ndarray, path: str, is_mask: bool = False) -> None:
        """Save debug image to disk."""
        try:
            cv2.imwrite(f"{path}.jpg", image)
        except Exception as e:
            logger.warning(f"Failed to save debug image {path}: {e}")


class SporeDetector:
    """Handles spore detection and filtering."""
    
    def __init__(self, debug_path: Optional[str] = None):
        """
        Initialize the spore detector.
        
        Args:
            debug_path: Optional path for saving debug images
        """
        self.debug_path = debug_path
        self.min_center_distance = 6.0
    
    def detect_spores(self, 
                     image: np.ndarray,
                     edges: np.ndarray,
                     edges_strong: np.ndarray,
                     params: dict) -> List[np.ndarray]:
        """
        Detect spores in the preprocessed image.
        
        Args:
            image: Preprocessed grayscale image
            edges: Cleaned edge image
            edges_strong: Strong edge image for support validation
            params: Detection parameters
            
        Returns:
            List of detected spore contours
        """
        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        
        logger.debug(f"Found {len(contours)} total contours")
        
        spores = []
        accepted_centers = []
        processed_count = 0
        
        for contour in contours:
            processed_count += 1
            
            if self._is_valid_spore_contour(contour, image, edges_strong, params, accepted_centers):
                # Extract ellipse parameters
                ellipse = cv2.fitEllipse(contour)
                center = (ellipse[0][0], ellipse[0][1])
                accepted_centers.append(center)
                spores.append(contour)
        
        logger.debug(f"Processed {processed_count} contours, kept {len(spores)} spores")
        
        # Save debug visualization
        if self.debug_path:
            self._save_debug_ellipses(image, spores)
        
        return spores
    
    def _is_valid_spore_contour(self, 
                               contour: np.ndarray,
                               image: np.ndarray,
                               edges_strong: np.ndarray,
                               params: dict,
                               accepted_centers: List[Tuple[float, float]]) -> bool:
        """
        Check if a contour represents a valid spore.
        
        Args:
            contour: Contour to validate
            image: Original image for intensity analysis
            edges_strong: Strong edges for support validation
            params: Detection parameters
            accepted_centers: Already accepted spore centers
            
        Returns:
            True if contour is a valid spore
        """
        # Check contour length
        if len(contour) < params.get('min_spore_contour_length', 5):
            return False
        
        # Check contour area
        area = cv2.contourArea(contour)
        if not (params.get('min_contour_area', 25) < area < params.get('max_contour_area', 500)):
            return False
        
        # Check if contour can fit an ellipse
        if len(contour) < 5:
            return False
        
        try:
            ellipse = cv2.fitEllipse(contour)
        except Exception:
            return False
        
        # Extract ellipse parameters
        (x, y), (major_axis, minor_axis), angle = ellipse
        
        # Check ellipse area
        ellipse_area = np.pi * (major_axis / 2.0) * (minor_axis / 2.0)
        if not (params.get('min_ellipse_area', 25) < ellipse_area < params.get('max_ellipse_area', 500)):
            return False
        
        # Check axis ratio and eccentricity
        if not self._is_valid_ellipse_geometry(major_axis, minor_axis):
            return False
        
        # Check intensity characteristics
        if not self._is_valid_intensity(image, ellipse, params.get('intensity_threshold', 50)):
            return False
        
        # Check edge support
        if not self._has_sufficient_edge_support(edges_strong, ellipse):
            return False
        
        # Check for duplicates
        if self._is_duplicate_center((x, y), accepted_centers):
            return False
        
        return True
    
    def _is_valid_ellipse_geometry(self, major_axis: float, minor_axis: float) -> bool:
        """Check if ellipse has valid geometric properties."""
        ratio = min(major_axis, minor_axis) / max(major_axis, minor_axis)
        if ratio < 0.45 or ratio > 0.92:
            return False
        
        eccentricity = np.sqrt(1 - (min(major_axis, minor_axis) / max(major_axis, minor_axis))**2)
        if not (0.45 < eccentricity < 0.95):
            return False
        
        return True
    
    def _is_valid_intensity(self, 
                           image: np.ndarray, 
                           ellipse: Tuple, 
                           intensity_threshold: int) -> bool:
        """Check if ellipse has valid intensity characteristics."""
        if intensity_threshold < 0:
            return True
        
        (x, y), (major_axis, minor_axis), angle = ellipse
        
        # Create mask for ellipse interior
        mask = np.zeros(image.shape, dtype=np.uint8)
        cv2.ellipse(
            mask, 
            (int(x), int(y)), 
            (int(major_axis/2), int(minor_axis/2)), 
            angle, 0, 360, 255, -1
        )
        
        # Calculate mean intensity inside and outside ellipse
        mean_inside = cv2.mean(image, mask=mask)[0]
        mean_total = np.mean(image)
        intensity_diff = abs(mean_inside - mean_total)
        
        return intensity_diff <= intensity_threshold
    
    def _has_sufficient_edge_support(self, 
                                   edges: np.ndarray, 
                                   ellipse: Tuple) -> bool:
        """Check if ellipse has sufficient edge support."""
        (x, y), (major_axis, minor_axis), angle = ellipse
        
        # Dilate edges slightly for support calculation
        edges_dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        
        # Create perimeter mask
        perimeter_mask = np.zeros_like(edges_dilated)
        cv2.ellipse(
            perimeter_mask, 
            (int(x), int(y)), 
            (int(major_axis/2), int(minor_axis/2)), 
            angle, 0, 360, 255, 1
        )
        
        # Calculate support ratio
        support_pixels = cv2.countNonZero(
            cv2.bitwise_and(edges_dilated, perimeter_mask)
        )
        perimeter_pixels = cv2.countNonZero(perimeter_mask)
        support_ratio = float(support_pixels) / float(perimeter_pixels + 1e-9)
        
        return support_ratio >= 0.25
    
    def _is_duplicate_center(self, 
                            center: Tuple[float, float], 
                            accepted_centers: List[Tuple[float, float]]) -> bool:
        """Check if center is too close to already accepted centers."""
        for accepted_center in accepted_centers:
            distance_squared = (center[0] - accepted_center[0])**2 + (center[1] - accepted_center[1])**2
            if distance_squared < self.min_center_distance**2:
                return True
        return False
    
    def _save_debug_ellipses(self, image: np.ndarray, spores: List[np.ndarray]) -> None:
        """Save debug image with detected ellipses."""
        try:
            debug_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for contour in spores:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(debug_img, ellipse, (0, 0, 255), 2)
            cv2.imwrite(f"{self.debug_path}_ellipses.jpg", debug_img)
        except Exception as e:
            logger.warning(f"Failed to save debug ellipses: {e}")


class SporeDetectionPipeline:
    """Main pipeline for spore detection."""
    
    def __init__(self, debug_path: Optional[str] = None):
        """
        Initialize the detection pipeline.
        
        Args:
            debug_path: Optional path for saving debug images
        """
        self.preprocessor = ImagePreprocessor(debug_path)
        self.edge_detector = EdgeDetector(debug_path)
        self.spore_detector = SporeDetector(debug_path)
        self.debug_path = debug_path
    
    def detect_spores(self, 
                     image: Image.Image,
                     **params) -> List[np.ndarray]:
        """
        Complete spore detection pipeline.
        
        Args:
            image: Input PIL Image
            **params: Detection parameters
            
        Returns:
            List of detected spore contours
        """
        # Preprocess image
        preprocessed = self.preprocessor.preprocess(image)
        
        # Detect and clean edges
        edges, edges_strong = self.edge_detector.detect_edges(
            preprocessed, 
            params.get('canny_threshold1', 40),
            params.get('canny_threshold2', 125)
        )
        cleaned_edges = self.edge_detector.clean_edges(edges)
        
        # Detect spores
        spores = self.spore_detector.detect_spores(
            preprocessed, cleaned_edges, edges_strong, params
        )
        
        return spores

 


def save_debug_image(image: Union[Image.Image, np.ndarray], 
                    spores: List[np.ndarray], 
                    out_path: str, 
                    is_mask: bool = False) -> None:
    """
    Save debug image with contours or as mask.
    
    Args:
        image: PIL Image or numpy array
        spores: List of spore contours
        out_path: Output file path
        is_mask: Whether to save as mask
    """
    try:
        if is_mask:
            if isinstance(image, Image.Image):
                arr = np.array(image)
            else:
                arr = image
            cv2.imwrite(f"{out_path}.jpg", arr)
            return
        
        # Convert to BGR for OpenCV
        if isinstance(image, Image.Image):
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            if len(image.shape) == 2:
                img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = image
        
        # Draw contours
        cv2.drawContours(img_bgr, spores, -1, (0, 0, 255), 2)
        cv2.imwrite(f"{out_path}.jpg", img_bgr)
        
    except Exception as e:
        logger.warning(f"Failed to save debug image {out_path}: {e}") 