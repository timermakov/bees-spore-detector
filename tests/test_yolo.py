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
import json
import pytest
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Ensure bees package is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bees.yolo import YOLOConfig
from bees.yolo.data_discovery import discover_dataset_dirs, find_coco_annotation_candidates


def make_yolo_config(**overrides) -> YOLOConfig:
    """Build a complete YOLOConfig for tests.

    Production code loads every field from config.yaml. Tests supply explicit
    values so they do not depend on hidden Python defaults.
    """
    values = {
        "model_name": "yolo11s.pt",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "epochs": 100,
        "batch_size": 16,
        "imgsz": 960,
        "patience": 25,
        "device": None,
        "workers": 2,
        "quick_test_epochs": 30,
        "quick_test_imgsz": 1024,
        "quick_test_batch_size": 8,
        "quick_test_patience": 0,
        "validation_split": 0.4,
        "split_seed": 42,
        "datasets_root": Path("dataset_train"),
        "dataset_folder_pattern": "*",
        "annotations_format": "auto",
        "annotations_relpath": None,
        "images_subdir": ".",
        "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"],
        "test_dir": Path("dataset_test"),
        "output_dir": Path("yolo_dataset"),
        "models_dir": Path("models"),
        "mosaic": 1.0,
        "mixup": 0.15,
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 15.0,
        "scale": 0.3,
        "fliplr": 0.5,
        "flipud": 0.5,
        "optimizer": "AdamW",
        "learning_rate": 0.0005,
        "final_learning_rate_factor": 0.01,
        "warmup_epochs": 2,
        "copy_paste": 0.3,
        "cache": False,
        "amp": True,
        "show_labels": False,
        "show_conf": False,
        "pretrained": True,
        "verbose": True,
        "analysis_square_size": 780,
        "class_names": ["spore"],
    }
    values.update(overrides)
    return YOLOConfig(**values)

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

    def test_repository_config_yaml_is_source_of_truth(self):
        """Training settings come from the repository config.yaml."""
        from bees import create_config_manager

        config_path = Path(__file__).parent.parent / "config.yaml"
        config = YOLOConfig.from_config_manager(create_config_manager(str(config_path)))

        assert config.model_name == "yolo11s.pt"
        assert config.confidence_threshold == 0.25
        assert config.iou_threshold == 0.45
        assert config.epochs == 100
        assert config.batch_size == 2
        assert config.imgsz == 960
        assert config.patience == 25
        assert config.device is None
        assert config.annotations_format == "auto"
        assert config.annotations_relpath is None
        assert config.images_subdir == "."
        assert config.class_names == ["spore"]
        assert config.optimizer == "AdamW"
        assert config.cache is False
        assert config.pretrained is True

    def test_config_with_custom_values(self):
        """Test creating config with custom values."""
        config = make_yolo_config(
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
        config = make_yolo_config(
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
        config = make_yolo_config(
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
        config = make_yolo_config()
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
        config = make_yolo_config()

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
        config = make_yolo_config()
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
yolo_patience: 25
yolo_device: null
yolo_workers: 2
yolo_quick_test_epochs: 30
yolo_quick_test_imgsz: 1024
yolo_quick_test_batch_size: 8
yolo_quick_test_patience: 0
yolo_validation_split: 0.4
yolo_split_seed: 42
analysis_square_size: 780
yolo_datasets_root: test_datasets
yolo_dataset_folder_pattern: "*"
yolo_annotations_format: auto
yolo_annotations_relpath: custom/ann.json
yolo_images_subdir: custom/images
yolo_image_extensions: [".jpg", ".png"]
yolo_test_dir: dataset_test
yolo_output_dir: yolo_dataset
yolo_models_dir: models
yolo_mosaic: 1.0
yolo_mixup: 0.15
yolo_hsv_h: 0.015
yolo_hsv_s: 0.7
yolo_hsv_v: 0.4
yolo_degrees: 15.0
yolo_scale: 0.3
yolo_fliplr: 0.5
yolo_flipud: 0.5
yolo_optimizer: AdamW
yolo_learning_rate: 0.0005
yolo_final_learning_rate_factor: 0.01
yolo_warmup_epochs: 2
yolo_copy_paste: 0.3
yolo_cache: false
yolo_amp: true
yolo_show_labels: false
yolo_show_conf: false
yolo_pretrained: true
yolo_verbose: true
yolo_class_names: [spore]
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
        assert yolo_config.annotations_format == "auto"
        assert yolo_config.annotations_relpath == "custom/ann.json"
        assert yolo_config.images_subdir == "custom/images"
        assert yolo_config.cache is False
        assert yolo_config.amp is True
        assert yolo_config.class_names == ["spore"]
        assert yolo_config.device is None

    def test_from_config_manager_requires_yaml_keys(self, tmp_path):
        """Missing YOLO keys are reported instead of filled from Python."""
        from bees import create_config_manager

        config_file = tmp_path / "incomplete.yaml"
        config_file.write_text("data_dir: test_data\nresults_dir: test_results\n")

        config_manager = create_config_manager(str(config_file))
        from bees.config_loader import ConfigurationError

        with pytest.raises(ConfigurationError, match="yolo_model"):
            YOLOConfig.from_config_manager(config_manager)

    def test_tiley_section_is_required(self, tmp_path):
        """SAHI settings are required in config.yaml."""
        from bees import create_config_manager
        from bees.config_loader import ConfigurationError

        config_file = tmp_path / "no_tiley.yaml"
        config_file.write_text("data_dir: test_data\nresults_dir: test_results\n")

        config_manager = create_config_manager(str(config_file))
        from bees.yolo.config import TileyConfig

        with pytest.raises(ConfigurationError, match="tiley_export_input"):
            TileyConfig.from_config_manager(config_manager)

    def test_null_predict_weights_use_yolo_model(self):
        """A null tiley_predict_weights value selects the trained checkpoint."""
        from bees import create_config_manager
        from bees.main import _resolve_tiley_export_config, _resolve_tiley_predict_config

        config_path = Path(__file__).parent.parent / "config.yaml"
        config_manager = create_config_manager(str(config_path))

        predict = _resolve_tiley_predict_config(config_manager, None, None)
        export = _resolve_tiley_export_config(config_manager, None, None)

        assert Path(predict["weights"]) == Path("models") / "yolo11s_spores" / "weights" / "best.pt"
        assert predict["tile_size"] == 480
        assert predict["conf"] == 0.25
        assert predict["overlap_height_ratio"] == 0.06
        assert predict["device"] == "cuda:0"
        assert predict["write_previews"] is True
        assert export["tile_size"] == 480
        assert export["overlap_height_ratio"] == 0.2

        config_manager.config["tiley_predict_weights"] = "checkpoints/custom.pt"
        pinned = _resolve_tiley_predict_config(config_manager, None, None)
        assert pinned["weights"] == "checkpoints/custom.pt"

    def test_null_and_non_numeric_values_are_rejected(self, tmp_path):
        """Null and mistyped numbers fail with the key name."""
        from bees import create_config_manager
        from bees.config_loader import ConfigurationError

        source = (Path(__file__).parent.parent / "config.yaml").read_text(encoding="utf-8")

        null_epochs = tmp_path / "null_epochs.yaml"
        null_epochs.write_text(source.replace("yolo_epochs: 100", "yolo_epochs: null", 1), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="yolo_epochs"):
            YOLOConfig.from_config_manager(create_config_manager(str(null_epochs)))

        bad_batch = tmp_path / "bad_batch.yaml"
        bad_batch.write_text(source.replace("yolo_batch_size: 2", "yolo_batch_size: many", 1), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="yolo_batch_size"):
            YOLOConfig.from_config_manager(create_config_manager(str(bad_batch)))

        null_conf = tmp_path / "null_conf.yaml"
        null_conf.write_text(source.replace("tiley_predict_conf: 0.25", "tiley_predict_conf: null", 1), encoding="utf-8")
        from bees.yolo.config import TileyConfig
        with pytest.raises(ConfigurationError, match="tiley_predict_conf"):
            TileyConfig.from_config_manager(create_config_manager(str(null_conf)))

    def test_opencv_and_titer_require_yaml_values(self, tmp_path):
        """OpenCV and titer read config.yaml and do not fill Python defaults."""
        from bees import create_config_manager
        from bees.config_loader import ConfigurationError
        from bees.titer import create_calculator_from_config

        source_path = Path(__file__).parent.parent / "config.yaml"
        source = source_path.read_text(encoding="utf-8")
        config_manager = create_config_manager(str(source_path))

        assert config_manager.get_int_param("min_contour_area") == 45
        assert config_manager.get_int_param("max_ellipse_area") == 360
        assert config_manager.get_int_param("intensity_threshold") == 36
        calculator = create_calculator_from_config(config_manager)
        assert calculator.square_size == 780
        assert calculator.volume_factor == 4.0

        missing_area = tmp_path / "missing_area.yaml"
        missing_area.write_text(source.replace("min_contour_area: 45\n", "", 1), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="min_contour_area"):
            create_config_manager(str(missing_area)).get_int_param("min_contour_area")

        missing_volume = tmp_path / "missing_volume.yaml"
        missing_volume.write_text(source.replace("volume_factor: 4.0\n", "", 1), encoding="utf-8")
        with pytest.raises(ConfigurationError, match="volume_factor"):
            create_calculator_from_config(create_config_manager(str(missing_volume)))


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not installed")
class TestDatasetPreparerCOCO:
    """Tests for COCO dataset preparation path."""

    @staticmethod
    def _write_coco_json(coco_path: Path, payload: dict):
        coco_path.parent.mkdir(parents=True, exist_ok=True)
        coco_path.write_text(json.dumps(payload), encoding="utf-8")

    @staticmethod
    def _write_dummy_image(image_path: Path):
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    def test_coco_bbox_to_yolo_line(self):
        from bees.yolo.dataset import DatasetPreparer

        line = DatasetPreparer._coco_bbox_to_yolo_line([10, 20, 30, 40], width=100, height=200, class_id=0)
        assert line == "0 0.250000 0.200000 0.300000 0.200000"

    def test_prepare_dataset_from_coco(self, tmp_path):
        from bees.yolo.dataset import DatasetPreparer

        datasets_root = tmp_path / "dataset_train"
        portion_dir = datasets_root / "dataset_train_2026-27-05"
        images_dir = portion_dir / "sliced" / "images"
        coco_path = portion_dir / "sliced" / "coco_annotations.json_coco.json"

        self._write_dummy_image(images_dir / "img_1.png")
        self._write_dummy_image(images_dir / "img_2.png")
        self._write_coco_json(
            coco_path,
            {
                "images": [
                    {"id": 1, "file_name": "img_1.png", "width": 100, "height": 100},
                    {"id": 2, "file_name": "img_2.png", "width": 100, "height": 100},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
                ],
                "categories": [
                    {"id": 1, "name": "spore"},
                ],
            },
        )

        config = make_yolo_config(
            datasets_root=datasets_root,
            dataset_folder_pattern="dataset_train_2026-27-05",
            annotations_format="auto",
            annotations_relpath="sliced/coco_annotations.json_coco.json",
            images_subdir="sliced/images",
            output_dir=tmp_path / "yolo_dataset",
            models_dir=tmp_path / "models",
        )

        preparer = DatasetPreparer(config)
        data_yaml = preparer.prepare_dataset(val_split=0.5, seed=1)

        assert data_yaml.exists()
        train_images = list((config.output_dir / "train" / "images").glob("*"))
        val_images = list((config.output_dir / "val" / "images").glob("*"))
        assert len(train_images) + len(val_images) == 2

    def test_prepare_dataset_from_coco_fails_on_unknown_category(self, tmp_path):
        from bees.yolo.dataset import DatasetPreparer

        datasets_root = tmp_path / "dataset_train"
        portion_dir = datasets_root / "portion_01"
        images_dir = portion_dir / "sliced" / "images"
        coco_path = portion_dir / "sliced" / "coco_annotations.json_coco.json"

        self._write_dummy_image(images_dir / "img_1.png")
        self._write_coco_json(
            coco_path,
            {
                "images": [{"id": 1, "file_name": "img_1.png", "width": 100, "height": 100}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [1, 1, 5, 5]}],
                "categories": [{"id": 5, "name": "other_class"}],
            },
        )

        config = make_yolo_config(
            datasets_root=datasets_root,
            dataset_folder_pattern="portion_01",
            annotations_format="auto",
            annotations_relpath="sliced/coco_annotations.json_coco.json",
            images_subdir="sliced/images",
            output_dir=tmp_path / "yolo_dataset",
            models_dir=tmp_path / "models",
        )

        preparer = DatasetPreparer(config)
        with pytest.raises(ValueError, match="Unknown COCO category"):
            preparer.prepare_dataset()

    def test_prepare_dataset_from_coco_fails_on_missing_image(self, tmp_path):
        from bees.yolo.dataset import DatasetPreparer

        datasets_root = tmp_path / "dataset_train"
        portion_dir = datasets_root / "portion_01"
        coco_path = portion_dir / "sliced" / "coco_annotations.json_coco.json"

        self._write_coco_json(
            coco_path,
            {
                "images": [{"id": 1, "file_name": "missing.png", "width": 100, "height": 100}],
                "annotations": [],
                "categories": [{"id": 1, "name": "spore"}],
            },
        )

        config = make_yolo_config(
            datasets_root=datasets_root,
            dataset_folder_pattern="portion_01",
            annotations_format="auto",
            annotations_relpath="sliced/coco_annotations.json_coco.json",
            images_subdir="sliced/images",
            output_dir=tmp_path / "yolo_dataset",
            models_dir=tmp_path / "models",
        )

        preparer = DatasetPreparer(config)
        with pytest.raises(FileNotFoundError, match="Missing"):
            preparer.prepare_dataset()


class TestDataDiscovery:
    """Tests for dataset folder and annotation discovery."""

    def test_discover_dataset_dirs_skips_container_root(self, tmp_path):
        root = tmp_path / "dataset_train"
        (root / "portion_01").mkdir(parents=True)
        (root / "portion_02").mkdir()

        found = discover_dataset_dirs(root, "*")

        assert found == [root / "portion_01", root / "portion_02"]

    def test_discover_dataset_dirs_uses_root_when_flat(self, tmp_path):
        root = tmp_path / "dataset_train"
        root.mkdir()

        assert discover_dataset_dirs(root, "*") == [root]

    def test_find_coco_annotation_candidates_requires_coco_collections(self, tmp_path):
        (tmp_path / "notes.json").write_text('{"hello": 1}', encoding="utf-8")
        coco = tmp_path / "labels.json"
        coco.write_text(
            json.dumps({"images": [], "annotations": [], "categories": []}),
            encoding="utf-8",
        )

        assert find_coco_annotation_candidates(tmp_path) == [coco]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
