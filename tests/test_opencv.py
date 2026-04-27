"""
Unit tests for OpenCV-based image processing and spore detection.

Tests cover:
- ImagePreprocessor: preprocessing operations
- EdgeDetector: edge detection and morphological operations  
- SporeDetector: spore detection pipeline
- SporeDetectionPipeline: end-to-end detection
- spores module: counting and analysis functions
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO

# Ensure bees package is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bees.opencv import (
    ImagePreprocessor,
    EdgeDetector,
    SporeDetector,
    SporeDetectionPipeline,
    count_spores,
    analyze_spore_distribution,
    filter_spores_by_area,
    validate_spore_contours
)


class TestImagePreprocessor:
    """Tests for ImagePreprocessor class."""

    def test_preprocessor_creation(self):
        """Test creating preprocessor with and without debug path."""
        prep_no_debug = ImagePreprocessor()
        prep_with_debug = ImagePreprocessor(debug_path="/tmp/debug")

        assert prep_no_debug.debug_path is None
        assert prep_with_debug.debug_path == "/tmp/debug"

    def test_preprocess_grayscale_conversion(self):
        """Test that preprocessing converts image to grayscale."""
        preprocessor = ImagePreprocessor()

        # Create a test RGB image
        img = Image.new('RGB', (100, 100), color='red')
        result = preprocessor.preprocess(img)

        # Result should be 2D (grayscale)
        assert len(result.shape) == 2
        assert result.dtype == np.uint8

    def test_preprocess_output_shape(self):
        """Test that output maintains input dimensions."""
        preprocessor = ImagePreprocessor()

        sizes = [(100, 100), (200, 150), (512, 512)]
        for width, height in sizes:
            img = Image.new('L', (width, height))
            result = preprocessor.preprocess(img)
            assert result.shape == (height, width)

    def test_preprocess_enhances_contrast(self):
        """Test that CLAHE enhances contrast."""
        preprocessor = ImagePreprocessor()

        # Create low contrast image
        img_array = np.full((100, 100), 128, dtype=np.uint8)
        img = Image.fromarray(img_array)

        result = preprocessor.preprocess(img)

        # Result should be different from input (contrast enhanced)
        assert not np.array_equal(result, img_array)


class TestEdgeDetector:
    """Tests for EdgeDetector class."""

    def test_detector_creation(self):
        """Test creating edge detector."""
        detector = EdgeDetector()
        assert detector.debug_path is None

        detector_with_debug = EdgeDetector(debug_path="/tmp/debug")
        assert detector_with_debug.debug_path == "/tmp/debug"

    def test_detect_edges_output_shape(self):
        """Test edge detection output shape."""
        detector = EdgeDetector()

        # Create test image
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        edges, edges_strong = detector.detect_edges(img, canny_threshold1=40, canny_threshold2=120)

        assert edges.shape == (100, 100)
        assert edges.dtype == np.uint8
        assert edges_strong.shape == (100, 100)
        assert edges_strong.dtype == np.uint8

    def test_detect_edges_returns_binary(self):
        """Test that edge detection returns binary image."""
        detector = EdgeDetector()

        # Create test image with clear edges
        img = np.zeros((100, 100), dtype=np.uint8)
        img[40:60, 40:60] = 255  # White square

        edges, edges_strong = detector.detect_edges(img, canny_threshold1=50, canny_threshold2=150)

        # Edges should be binary (0 or 255)
        unique_values = np.unique(edges)
        assert set(unique_values).issubset({0, 255})

    def test_clean_edges(self):
        """Test edge cleaning operation."""
        detector = EdgeDetector()

        # Create binary edge image
        edges = np.zeros((50, 50), dtype=np.uint8)
        edges[20:30, 10:20] = 255

        # clean_edges should work without errors
        cleaned = detector.clean_edges(edges)

        assert cleaned.shape == edges.shape
        assert cleaned.dtype == np.uint8


class TestSporeDetector:
    """Tests for SporeDetector class."""

    def test_detector_creation(self):
        """Test creating spore detector."""
        detector = SporeDetector()

        assert detector.min_center_distance == 6.0
        assert detector.debug_path is None

    def test_detector_creation_with_debug_path(self):
        """Test creating detector with debug path."""
        detector = SporeDetector(debug_path="/tmp/debug")

        assert detector.debug_path == "/tmp/debug"
        assert detector.min_center_distance == 6.0

    def test_detect_spores_empty(self):
        """Test detection on empty image returns empty list."""
        detector = SporeDetector()

        # Create blank images
        img = np.zeros((100, 100), dtype=np.uint8)
        edges = np.zeros((100, 100), dtype=np.uint8)
        edges_strong = np.zeros((100, 100), dtype=np.uint8)

        params = {
            'min_contour_area': 25,
            'max_contour_area': 500,
            'min_ellipse_area': 25,
            'max_ellipse_area': 500,
            'min_spore_contour_length': 5,
            'intensity_threshold': 50
        }

        spores = detector.detect_spores(img, edges, edges_strong, params)
        assert spores == []


class TestSporeDetectionPipeline:
    """Tests for SporeDetectionPipeline class."""

    def test_pipeline_creation(self):
        """Test creating detection pipeline."""
        pipeline = SporeDetectionPipeline()
        assert pipeline.preprocessor is not None
        assert pipeline.edge_detector is not None

    def test_detect_pil_image(self):
        """Test pipeline with PIL image input."""
        pipeline = SporeDetectionPipeline()

        # Create test PIL image
        img = Image.new('RGB', (100, 100), color='white')

        # Should not raise exception with keyword params
        result = pipeline.detect_spores(
            img,
            canny_threshold1=40,
            canny_threshold2=120,
            min_contour_area=25,
            max_contour_area=500,
            min_ellipse_area=25,
            max_ellipse_area=500,
            min_spore_contour_length=5,
            intensity_threshold=50
        )
        assert isinstance(result, list)

    def test_detect_numpy_array(self):
        """Test pipeline with numpy array input."""
        pipeline = SporeDetectionPipeline()

        # Create test numpy array and convert to PIL
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)

        # Should not raise exception with keyword params
        result = pipeline.detect_spores(
            img,
            canny_threshold1=40,
            canny_threshold2=120,
            min_contour_area=25,
            max_contour_area=500,
            min_ellipse_area=25,
            max_ellipse_area=500,
            min_spore_contour_length=5,
            intensity_threshold=50
        )
        assert isinstance(result, list)


class TestSporesFunctions:
    """Tests for spores module functions."""

    def test_count_spores_empty(self):
        """Test counting empty list."""
        assert count_spores([]) == 0

    def test_count_spores(self):
        """Test counting spore objects."""
        # Create fake spore contours
        spores = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),
            np.array([[20, 20], [30, 20], [30, 30], [20, 30]]),
        ]
        assert count_spores(spores) == 2

    def test_count_spores_invalid_type(self):
        """Test count_spores with invalid input type."""
        with pytest.raises(TypeError):
            count_spores("not a list")

    def test_analyze_spore_distribution_empty(self):
        """Test analyzing empty spore list."""
        result = analyze_spore_distribution([])

        assert result['total_count'] == 0
        assert result['areas'] == []
        assert result['mean_area'] == 0.0
        assert result['std_area'] == 0.0
        assert result['min_area'] == 0.0
        assert result['max_area'] == 0.0

    def test_analyze_spore_distribution(self):
        """Test analyzing spore distribution."""
        # Create fake contours with known areas
        spores = [
            np.array([[0, 0], [10, 0], [10, 10], [0, 10]]),  # ~100 area
            np.array([[0, 0], [20, 0], [20, 20], [0, 20]]),  # ~400 area
        ]

        result = analyze_spore_distribution(spores)

        assert result['total_count'] == 2
        assert len(result['areas']) == 2
        assert result['min_area'] > 0
        assert result['max_area'] >= result['min_area']

    def test_filter_spores_by_area_empty(self):
        """Test filtering empty list."""
        result = filter_spores_by_area([], 0, 1000)
        assert result == []

    def test_filter_spores_by_area(self):
        """Test filtering spores by area range."""
        # Create contours with different areas
        small = np.array([[0, 0], [5, 0], [5, 5], [0, 5]])
        medium = np.array([[0, 0], [15, 0], [15, 15], [0, 15]])
        large = np.array([[0, 0], [30, 0], [30, 30], [0, 30]])

        spores = [small, medium, large]

        # Filter only medium (around 225 area)
        filtered = filter_spores_by_area(spores, 100, 300)
        assert len(filtered) == 1

    def test_validate_spore_contours_empty(self):
        """Test validating empty contours."""
        result = validate_spore_contours([])

        assert result['valid_count'] == 0
        assert result['invalid_count'] == 0
        assert result['validation_errors'] == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
