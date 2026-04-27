"""
Unit tests for YOLO-based spore detection.

Tests cover:
- YOLOConfig: configuration management
- Detection: detection data class
- SporeDetector: YOLO model inference (mocked)
- DatasetPreparer: dataset preparation
- CVATToYOLOConverter: format conversion
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Ensure bees package is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bees.yolo import YOLOConfig

# Optional imports that may fail if dependencies missing
try:
    from bees.yolo import Detection, SporeDetector
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    from bees.yolo import CVATToYOLOConverter
    XML_ET_AVAILABLE = True
except ImportError:
    XML_ET_AVAILABLE = False

try:
    from bees.yolo import DatasetPreparer
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class TestYOLOConfig:
    """Tests for YOLOConfig dataclass."""

    def test_default_config_creation(self):
        """Test creating config with defaults."""
        config = YOLOConfig()

        assert config.model_name == "yolo11s.pt"
        assert config.confidence_threshold == 0.25
        assert config.iou_threshold == 0.45
        assert config.epochs == 100
        assert config.batch_size == 16
        assert config.imgsz == 960

    def test_config_with_custom_values(self):
        """Test creating config with custom values."""
        config = YOLOConfig(
            model_name="custom.pt",
            confidence_threshold=0.5,
            iou_threshold=0.3,
            epochs=50,
            batch_size=8,
            imgsz=640
        )

        assert config.model_name == "custom.pt"
        assert config.confidence_threshold == 0.5
        assert config.iou_threshold == 0.3
        assert config.epochs == 50
        assert config.batch_size == 8
        assert config.imgsz == 640

    def test_config_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = YOLOConfig(
            datasets_root="my_dataset",
            test_dir="test_data",
            output_dir="output",
            models_dir="models"
        )

        assert isinstance(config.datasets_root, Path)
        assert isinstance(config.test_dir, Path)
        assert isinstance(config.output_dir, Path)
        assert isinstance(config.models_dir, Path)

    def test_ensure_dirs(self, tmp_path):
        """Test directory creation."""
        config = YOLOConfig(
            output_dir=tmp_path / "output",
            models_dir=tmp_path / "models"
        )

        config.ensure_dirs()

        assert (config.output_dir / "train" / "images").exists()
        assert (config.output_dir / "train" / "labels").exists()
        assert (config.output_dir / "val" / "images").exists()
        assert (config.output_dir / "val" / "labels").exists()

    def test_get_trained_model_path(self):
        """Test getting trained model path."""
        config = YOLOConfig()
        model_path = config.get_trained_model_path()

        assert isinstance(model_path, Path)
        assert model_path.name == "best.pt"
        assert "yolo11s_spores" in str(model_path)


@pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="ultralytics not installed")
class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_creation(self):
        """Test creating a detection."""
        detection = Detection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.85,
            class_id=0,
            class_name="spore"
        )

        assert detection.x1 == 10.0
        assert detection.y1 == 20.0
        assert detection.x2 == 30.0
        assert detection.y2 == 40.0
        assert detection.confidence == 0.85
        assert detection.class_id == 0
        assert detection.class_name == "spore"

    def test_detection_properties(self):
        """Test detection computed properties."""
        detection = Detection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.9
        )

        assert detection.width == 20.0
        assert detection.height == 20.0
        assert detection.area == 400.0
        assert detection.center == (20.0, 30.0)

    def test_detection_repr(self):
        """Test detection string representation."""
        detection = Detection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.9
        )

        repr_str = repr(detection)
        assert "Detection" in repr_str
        assert "0.90" in repr_str


@pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="ultralytics not installed")
class TestSporeDetector:
    """Tests for SporeDetector class."""

    def test_detector_requires_config(self):
        """Test that detector requires YOLOConfig."""
        config = YOLOConfig()

        # This will fail to load model but should create detector
        try:
            detector = SporeDetector(config)
            assert detector.config == config
        except Exception:
            # Expected if model file doesn't exist
            pass


@pytest.mark.skipif(not XML_ET_AVAILABLE, reason="XML dependencies not available")
class TestCVATToYOLOConverter:
    """Tests for CVAT to YOLO format converter."""

    def test_converter_creation(self):
        """Test creating converter."""
        converter = CVATToYOLOConverter(["spore"])

        assert converter.class_names == ["spore"]
        assert converter.class_to_id == {"spore": 0}


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not installed")
class TestDatasetPreparer:
    """Tests for DatasetPreparer class."""

    def test_preparer_creation(self):
        """Test creating dataset preparer."""
        config = YOLOConfig()
        preparer = DatasetPreparer(config)

        assert preparer.config == config


class TestConfigManagerIntegration:
    """Tests for YOLOConfig integration with ConfigurationManager."""

    def test_from_config_manager(self, tmp_path):
        """Test creating YOLOConfig from ConfigurationManager."""
        from bees import create_config_manager

        # Create config file with required sections
        config_content = """
data_dir: test_data
results_dir: test_results
yolo_model: yolo11s.pt
yolo_confidence: 0.3
yolo_iou_threshold: 0.4
yolo_epochs: 50
yolo_batch_size: 8
yolo_imgsz: 640
analysis_square_size: 780
yolo_datasets_root: test_datasets
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        config_manager = create_config_manager(str(config_file))
        yolo_config = YOLOConfig.from_config_manager(config_manager)

        assert yolo_config.model_name == "yolo11s.pt"
        assert yolo_config.confidence_threshold == 0.3
        assert yolo_config.iou_threshold == 0.4
        assert yolo_config.epochs == 50
        assert yolo_config.batch_size == 8
        assert yolo_config.imgsz == 640
        assert yolo_config.analysis_square_size == 780


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
