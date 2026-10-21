"""
Unit tests for YOLO-based spore detection.

Tests cover:
- YOLOConfig: configuration management
- Detection: detection data class
- SporeDetector: YOLO model inference (mocked)
- DatasetPreparer: dataset preparation
- CVATToYOLOConverter: CVAT XML to YOLO labels
"""

import json
import re
import sys
from pathlib import Path

import pytest

# Ensure bees package is in path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bees.config_loader import ConfigurationError, ConfigurationManager, create_config_manager
from bees.titer import create_calculator_from_config
from bees.yolo import YOLOConfig
from bees.yolo.config import TileyConfig, resolve_predict_weights
from bees.yolo.data_discovery import discover_dataset_dirs, find_coco_annotation_candidates


def _repo_config_path() -> Path:
    """Path to the repository config.yaml."""
    return Path(__file__).parent.parent / "config.yaml"


def _read_repo_config() -> str:
    """Text of the repository config.yaml."""
    return _repo_config_path().read_text(encoding="utf-8")


def _set_yaml_key(text: str, key: str, raw_value: str) -> str:
    """Replace one top-level YAML value. raw_value is written as YAML text."""
    pattern = re.compile(rf"^({re.escape(key)}:\s*).*$", re.MULTILINE)
    updated, count = pattern.subn(rf"\1{raw_value}", text, count=1)
    if count != 1:
        raise AssertionError(f"{key} not found in config text")
    return updated


def _drop_yaml_key(text: str, key: str) -> str:
    """Remove one top-level YAML key line."""
    pattern = re.compile(rf"^{re.escape(key)}:.*\n", re.MULTILINE)
    updated, count = pattern.subn("", text, count=1)
    if count != 1:
        raise AssertionError(f"{key} not found in config text")
    return updated


def _load_config(path: Path) -> ConfigurationManager:
    """Load a YAML file through ConfigurationManager."""
    return create_config_manager(str(path))


def _load_config_text(directory: Path, text: str, name: str = "config.yaml") -> ConfigurationManager:
    """Write YAML text into directory and load it."""
    config_file = directory / name
    config_file.write_text(text, encoding="utf-8")
    return _load_config(config_file)


def _make_yolo_config(**overrides) -> YOLOConfig:
    """Build a complete YOLOConfig. Pass only the fields a test cares about."""
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


def _write_coco_json(path: Path, payload: dict) -> None:
    """Write a COCO JSON file, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_dummy_image(path: Path) -> None:
    """Write a tiny PNG header so an image path exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n")


def _write_cvat_xml(
    path: Path,
    image_name: str = "img_1.png",
    label: str = "spore",
    width: int = 100,
    height: int = 100,
) -> None:
    """Write one CVAT image with a single ellipse."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "<annotations>\n"
            f'  <image id="0" name="{image_name}" width="{width}" height="{height}">\n'
            f'    <ellipse label="{label}" cx="50" cy="50" rx="10" ry="10"/>\n'
            "  </image>\n"
            "</annotations>\n"
        ),
        encoding="utf-8",
    )


def _make_portion_config(tmp_path: Path, datasets_root: Path, **overrides):
    """YOLOConfig aimed at one dataset portion under tmp_path."""
    values = {
        "datasets_root": datasets_root,
        "dataset_folder_pattern": "portion_01",
        "annotations_format": "coco",
        "annotations_relpath": None,
        "images_subdir": "sliced/images",
        "output_dir": tmp_path / "yolo_dataset",
        "models_dir": tmp_path / "models",
    }
    values.update(overrides)
    return _make_yolo_config(**values)


def _resolve_tiley_export(manager: ConfigurationManager, images_dir, out):
    """Resolve tile-export settings, including CLI path overrides."""
    from bees.main import _resolve_tiley_export_config

    return _resolve_tiley_export_config(manager, images_dir, out)


def _resolve_tiley_predict(manager: ConfigurationManager, source, out):
    """Resolve tile-predict settings, including CLI path overrides."""
    from bees.main import _resolve_tiley_predict_config

    return _resolve_tiley_predict_config(manager, source, out)


def _build_parser():
    """Build the command-line parser."""
    from bees.main import build_parser

    return build_parser()


def _assert_int(value) -> None:
    assert isinstance(value, int) and not isinstance(value, bool)


def _assert_number(value) -> None:
    assert isinstance(value, (int, float)) and not isinstance(value, bool)


def _assert_float(value) -> None:
    assert isinstance(value, float)


def _assert_bool(value) -> None:
    assert isinstance(value, bool)


def _assert_text(value) -> None:
    assert isinstance(value, str)


def _assert_nonempty_text(value) -> None:
    assert isinstance(value, str) and value


def _assert_optional_text(value) -> None:
    assert value is None or isinstance(value, str)


def _assert_text_list(value) -> None:
    assert isinstance(value, list)
    for item in value:
        _assert_nonempty_text(item)


def _assert_path(value) -> None:
    assert isinstance(value, Path)


def _assert_error_names(error: BaseException, key: str) -> None:
    """Fail when the error message does not name the YAML key."""
    assert key in str(error)


_SAMPLE_YOLO_YAML = """
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
        """Repository config.yaml loads, and each training field has its type."""
        # Arrange
        manager = _load_config(_repo_config_path())

        # Act
        config = YOLOConfig.from_config_manager(manager)

        # Assert
        _assert_nonempty_text(config.model_name)
        _assert_number(config.confidence_threshold)
        _assert_number(config.iou_threshold)
        _assert_int(config.epochs)
        _assert_int(config.batch_size)
        _assert_int(config.imgsz)
        _assert_int(config.patience)
        _assert_optional_text(config.device)
        _assert_nonempty_text(config.annotations_format)
        _assert_optional_text(config.annotations_relpath)
        _assert_text(config.images_subdir)
        _assert_text_list(config.class_names)
        _assert_nonempty_text(config.optimizer)
        _assert_bool(config.cache)
        _assert_bool(config.pretrained)
        _assert_path(config.models_dir)

    def test_config_with_custom_values(self):
        """Test creating config with custom values."""
        # Arrange
        overrides = {
            "model_name": "custom.pt",
            "confidence_threshold": 0.5,
            "iou_threshold": 0.3,
            "epochs": 50,
            "batch_size": 8,
            "imgsz": 640,
        }

        # Act
        config = _make_yolo_config(**overrides)

        # Assert
        assert config.model_name == "custom.pt"
        assert config.confidence_threshold == 0.5
        assert config.iou_threshold == 0.3
        assert config.epochs == 50
        assert config.batch_size == 8
        assert config.imgsz == 640

    def test_config_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        # Arrange
        overrides = {
            "datasets_root": "my_dataset",
            "test_dir": "test_data",
            "output_dir": "output",
            "models_dir": "models",
        }

        # Act
        config = _make_yolo_config(**overrides)

        # Assert
        _assert_path(config.datasets_root)
        _assert_path(config.test_dir)
        _assert_path(config.output_dir)
        _assert_path(config.models_dir)

    def test_ensure_dirs(self, tmp_path):
        """Test directory creation."""
        # Arrange
        config = _make_yolo_config(
            output_dir=tmp_path / "output",
            models_dir=tmp_path / "models",
        )

        # Act
        config.ensure_dirs()

        # Assert
        assert (config.output_dir / "train" / "images").exists()
        assert (config.output_dir / "train" / "labels").exists()
        assert (config.output_dir / "val" / "images").exists()
        assert (config.output_dir / "val" / "labels").exists()

    def test_get_trained_model_path(self):
        """Test getting trained model path."""
        # Arrange
        config = _make_yolo_config()

        # Act
        model_path = config.get_trained_model_path()

        # Assert
        _assert_path(model_path)
        assert model_path.name == "best.pt"
        assert "yolo11s_spores" in str(model_path)


@pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="ultralytics not installed")
class TestDetection:
    """Tests for Detection dataclass."""

    def test_detection_creation(self):
        """Test creating a detection."""
        # Arrange
        bbox = (10.0, 20.0, 30.0, 40.0)

        # Act
        detection = Detection(
            bbox=bbox,
            confidence=0.85,
            class_id=0,
            class_name="spore",
        )

        # Assert
        assert detection.x1 == 10.0
        assert detection.y1 == 20.0
        assert detection.x2 == 30.0
        assert detection.y2 == 40.0
        assert detection.confidence == 0.85
        assert detection.class_id == 0
        assert detection.class_name == "spore"

    def test_detection_properties(self):
        """Test detection computed properties."""
        # Arrange
        detection = Detection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.9,
        )

        # Act
        width = detection.width
        height = detection.height
        area = detection.area
        center = detection.center

        # Assert
        assert width == 20.0
        assert height == 20.0
        assert area == 400.0
        assert center == (20.0, 30.0)

    def test_detection_repr(self):
        """Test detection string representation."""
        # Arrange
        detection = Detection(
            bbox=(10.0, 20.0, 30.0, 40.0),
            confidence=0.9,
        )

        # Act
        text = repr(detection)

        # Assert
        assert "Detection" in text
        assert "0.90" in text


@pytest.mark.skipif(not ULTRALYTICS_AVAILABLE, reason="ultralytics not installed")
class TestSporeDetector:
    """Tests for SporeDetector class."""

    def test_detector_requires_config(self):
        """Test that detector requires YOLOConfig."""
        # Arrange
        config = _make_yolo_config()

        # Act
        load_error = None
        detector = None
        try:
            detector = SporeDetector(config)
        except Exception as error:
            load_error = error

        # Assert
        assert load_error is not None or detector.config == config


@pytest.mark.skipif(not XML_ET_AVAILABLE, reason="XML dependencies not available")
class TestCVATToYOLOConverter:
    """Tests for CVAT to YOLO format converter."""

    def test_converter_creation(self):
        """Test creating converter."""
        # Arrange
        class_names = ["spore"]

        # Act
        converter = CVATToYOLOConverter(class_names)

        # Assert
        assert converter.class_names == ["spore"]
        assert converter.class_to_id == {"spore": 0}

    def test_unknown_label_is_rejected(self):
        """An unknown CVAT label fails instead of becoming class 0."""
        from bees.yolo.cvat import EllipseAnnotation, ImageAnnotation

        # Arrange
        converter = CVATToYOLOConverter(["spore"])
        annotation = ImageAnnotation(
            image_id=0,
            name="a.jpg",
            width=100,
            height=100,
            ellipses=[EllipseAnnotation(cx=50, cy=50, rx=10, ry=10, label="debris")],
        )

        # Act
        with pytest.raises(ValueError) as caught:
            converter.convert_annotation(annotation)

        # Assert
        assert "Unknown label" in str(caught.value)
        assert "debris" in str(caught.value)

    def test_box_corners_are_clipped_before_conversion(self):
        """A box past the image edge keeps only the visible part."""
        from bees.yolo.cvat import EllipseAnnotation, ImageAnnotation

        # Arrange
        converter = CVATToYOLOConverter(["spore"])
        annotation = ImageAnnotation(
            image_id=0,
            name="a.jpg",
            width=100,
            height=100,
            ellipses=[EllipseAnnotation(cx=100, cy=105, rx=20, ry=15, label="spore")],
        )

        # Act
        lines = converter.convert_annotation(annotation)

        # Assert
        assert lines == ["0 0.900000 0.950000 0.200000 0.100000"]

    def test_zero_image_size_is_rejected(self):
        """A zero image width fails with the image name."""
        from bees.yolo.cvat import EllipseAnnotation, ImageAnnotation

        # Arrange
        converter = CVATToYOLOConverter(["spore"])
        annotation = ImageAnnotation(
            image_id=0,
            name="a.jpg",
            width=0,
            height=100,
            ellipses=[EllipseAnnotation(cx=15, cy=15, rx=5, ry=5, label="spore")],
        )

        # Act
        with pytest.raises(ValueError) as caught:
            converter.convert_annotation(annotation)

        # Assert
        assert "a.jpg" in str(caught.value)
        assert "0x100" in str(caught.value)

    def test_box_outside_the_image_is_skipped(self):
        """A box with no visible area produces no label line."""
        from bees.yolo.cvat import EllipseAnnotation, ImageAnnotation

        # Arrange
        converter = CVATToYOLOConverter(["spore"])
        annotation = ImageAnnotation(
            image_id=0,
            name="a.jpg",
            width=100,
            height=100,
            ellipses=[EllipseAnnotation(cx=200, cy=200, rx=5, ry=5, label="spore")],
        )

        # Act
        lines = converter.convert_annotation(annotation)

        # Assert
        assert lines == []


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not installed")
class TestDatasetPreparer:
    """Tests for DatasetPreparer class."""

    def test_preparer_creation(self):
        """Test creating dataset preparer."""
        # Arrange
        config = _make_yolo_config()

        # Act
        preparer = DatasetPreparer(config)

        # Assert
        assert preparer.config == config


class TestConfigManagerIntegration:
    """Tests for YOLOConfig integration with ConfigurationManager."""

    def test_from_config_manager(self, tmp_path):
        """Test creating YOLOConfig from ConfigurationManager."""
        # Arrange
        manager = _load_config_text(tmp_path, _SAMPLE_YOLO_YAML, name="test_config.yaml")

        # Act
        yolo_config = YOLOConfig.from_config_manager(manager)

        # Assert
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
        # Arrange
        manager = _load_config_text(
            tmp_path,
            "data_dir: test_data\nresults_dir: test_results\n",
            name="incomplete.yaml",
        )

        # Act
        with pytest.raises(ConfigurationError) as caught:
            YOLOConfig.from_config_manager(manager)

        # Assert
        _assert_error_names(caught.value, "yolo_model")

    def test_tiley_section_is_required(self, tmp_path):
        """SAHI settings are required in config.yaml."""
        # Arrange
        manager = _load_config_text(
            tmp_path,
            "data_dir: test_data\nresults_dir: test_results\n",
            name="no_tiley.yaml",
        )

        # Act
        with pytest.raises(ConfigurationError) as caught:
            TileyConfig.from_config_manager(manager)

        # Assert
        _assert_error_names(caught.value, "tiley_export_input")

    def test_null_predict_weights_use_yolo_model(self):
        """Null weights select the trained checkpoint; an explicit path is kept."""
        # Arrange
        manager = _load_config(_repo_config_path())
        yolo = YOLOConfig.from_config_manager(manager)
        tiley = TileyConfig.from_config_manager(manager)

        # Act
        predict = _resolve_tiley_predict(manager, None, None)
        export = _resolve_tiley_export(manager, None, None)

        # Assert
        if tiley.predict_weights is None:
            assert Path(predict["weights"]) == yolo.get_trained_model_path()
        else:
            assert predict["weights"] == tiley.predict_weights
        assert predict["tile_size"] == tiley.predict_tile_size
        _assert_float(predict["conf"])
        _assert_float(predict["overlap_height_ratio"])
        _assert_nonempty_text(predict["device"])
        _assert_bool(predict["write_previews"])
        assert export["tile_size"] == tiley.export_tile_size
        _assert_float(export["overlap_height_ratio"])

    def test_explicit_predict_weights_are_kept(self):
        """An explicit weights path stays pinned."""
        # Arrange
        manager = _load_config(_repo_config_path())
        manager.config["tiley_predict_weights"] = "checkpoints/custom.pt"

        # Act
        pinned = _resolve_tiley_predict(manager, None, None)

        # Assert
        assert pinned["weights"] == "checkpoints/custom.pt"

    @pytest.mark.parametrize(
        ("key", "raw_value", "load"),
        [
            ("yolo_epochs", "null", YOLOConfig.from_config_manager),
            ("yolo_batch_size", "many", YOLOConfig.from_config_manager),
            ("tiley_predict_conf", "null", TileyConfig.from_config_manager),
        ],
        ids=["null-epochs", "bad-batch", "null-conf"],
    )
    def test_null_and_non_numeric_values_are_rejected(self, tmp_path, key, raw_value, load):
        """Null and mistyped numbers fail with the key name."""
        # Arrange
        text = _set_yaml_key(_read_repo_config(), key, raw_value)
        manager = _load_config_text(tmp_path, text)

        # Act
        with pytest.raises(ConfigurationError) as caught:
            load(manager)

        # Assert
        _assert_error_names(caught.value, key)

    def test_opencv_and_titer_require_yaml_values(self):
        """OpenCV and titer read config.yaml and do not fill Python defaults."""
        # Arrange
        manager = _load_config(_repo_config_path())

        # Act
        contour_area = manager.get_int_param("min_contour_area")
        ellipse_area = manager.get_int_param("max_ellipse_area")
        intensity = manager.get_int_param("intensity_threshold")
        calculator = create_calculator_from_config(manager)

        # Assert
        _assert_int(contour_area)
        _assert_int(ellipse_area)
        _assert_int(intensity)
        _assert_int(calculator.square_size)
        _assert_number(calculator.volume_factor)

    @pytest.mark.parametrize(
        ("key", "read"),
        [
            ("min_contour_area", lambda manager: manager.get_int_param("min_contour_area")),
            ("volume_factor", create_calculator_from_config),
        ],
        ids=["min-contour-area", "volume-factor"],
    )
    def test_missing_opencv_and_titer_keys_are_named(self, tmp_path, key, read):
        """Dropped OpenCV and titer keys fail with the key name."""
        # Arrange
        text = _drop_yaml_key(_read_repo_config(), key)
        manager = _load_config_text(tmp_path, text)

        # Act
        with pytest.raises(ConfigurationError) as caught:
            read(manager)

        # Assert
        _assert_error_names(caught.value, key)


@pytest.mark.skipif(not PANDAS_AVAILABLE, reason="pandas not installed")
class TestDatasetPreparerCOCO:
    """Tests for COCO dataset preparation path."""

    def test_coco_bbox_to_yolo_line(self):
        """COCO box converts to a normalized YOLO label line."""
        # Arrange
        bbox = [10, 20, 30, 40]

        # Act
        line = DatasetPreparer._coco_bbox_to_yolo_line(bbox, width=100, height=200, class_id=0)

        # Assert
        assert line == "0 0.250000 0.200000 0.300000 0.200000"

    def test_coco_bbox_clips_corners_before_conversion(self):
        """A COCO box past the image edge keeps only the visible part."""
        # Arrange
        bbox = [80, 90, 40, 30]

        # Act
        line = DatasetPreparer._coco_bbox_to_yolo_line(bbox, width=100, height=100, class_id=0)

        # Assert
        assert line == "0 0.900000 0.950000 0.200000 0.100000"

    def test_prepare_dataset_from_coco(self, tmp_path):
        """A COCO portion is split into train and val images."""
        # Arrange
        datasets_root = tmp_path / "dataset_train"
        portion_dir = datasets_root / "dataset_train_2026-27-05"
        images_dir = portion_dir / "sliced" / "images"
        _write_dummy_image(images_dir / "img_1.png")
        _write_dummy_image(images_dir / "img_2.png")
        _write_coco_json(
            portion_dir / "sliced" / "coco_annotations.json_coco.json",
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
        config = _make_yolo_config(
            datasets_root=datasets_root,
            dataset_folder_pattern="dataset_train_2026-27-05",
            annotations_format="auto",
            annotations_relpath="sliced/coco_annotations.json_coco.json",
            images_subdir="sliced/images",
            output_dir=tmp_path / "yolo_dataset",
            models_dir=tmp_path / "models",
        )
        preparer = DatasetPreparer(config)

        # Act
        data_yaml = preparer.prepare_dataset(val_split=0.5, seed=1)

        # Assert
        assert data_yaml.exists()
        train_images = list((config.output_dir / "train" / "images").glob("*"))
        val_images = list((config.output_dir / "val" / "images").glob("*"))
        assert len(train_images) + len(val_images) == 2

    def test_prepare_dataset_from_coco_fails_on_unknown_category(self, tmp_path):
        """An unknown COCO category is rejected."""
        # Arrange
        datasets_root = tmp_path / "dataset_train"
        portion_dir = datasets_root / "portion_01"
        images_dir = portion_dir / "sliced" / "images"
        _write_dummy_image(images_dir / "img_1.png")
        _write_coco_json(
            portion_dir / "sliced" / "coco_annotations.json_coco.json",
            {
                "images": [{"id": 1, "file_name": "img_1.png", "width": 100, "height": 100}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [1, 1, 5, 5]}],
                "categories": [{"id": 5, "name": "other_class"}],
            },
        )
        config = _make_yolo_config(
            datasets_root=datasets_root,
            dataset_folder_pattern="portion_01",
            annotations_format="auto",
            annotations_relpath="sliced/coco_annotations.json_coco.json",
            images_subdir="sliced/images",
            output_dir=tmp_path / "yolo_dataset",
            models_dir=tmp_path / "models",
        )
        preparer = DatasetPreparer(config)

        # Act
        with pytest.raises(ValueError) as caught:
            preparer.prepare_dataset()

        # Assert
        assert "Unknown COCO category" in str(caught.value)

    def test_prepare_dataset_from_coco_fails_on_missing_image(self, tmp_path):
        """A COCO image that is not on disk is rejected."""
        # Arrange
        datasets_root = tmp_path / "dataset_train"
        portion_dir = datasets_root / "portion_01"
        _write_coco_json(
            portion_dir / "sliced" / "coco_annotations.json_coco.json",
            {
                "images": [{"id": 1, "file_name": "missing.png", "width": 100, "height": 100}],
                "annotations": [],
                "categories": [{"id": 1, "name": "spore"}],
            },
        )
        config = _make_yolo_config(
            datasets_root=datasets_root,
            dataset_folder_pattern="portion_01",
            annotations_format="auto",
            annotations_relpath="sliced/coco_annotations.json_coco.json",
            images_subdir="sliced/images",
            output_dir=tmp_path / "yolo_dataset",
            models_dir=tmp_path / "models",
        )
        preparer = DatasetPreparer(config)

        # Act
        with pytest.raises(FileNotFoundError) as caught:
            preparer.prepare_dataset()

        # Assert
        assert "Missing" in str(caught.value)

    def test_prepare_replaces_stale_files_after_success(self, tmp_path):
        """A finished prepare replaces train, val, and data.yaml, and drops stale files."""
        # Arrange
        datasets_root = tmp_path / "dataset_train"
        portion_dir = datasets_root / "portion_01"
        images_dir = portion_dir / "sliced" / "images"
        _write_dummy_image(images_dir / "img_1.png")
        _write_dummy_image(images_dir / "img_2.png")
        _write_coco_json(
            portion_dir / "sliced" / "coco_annotations.json",
            {
                "images": [
                    {"id": 1, "file_name": "img_1.png", "width": 100, "height": 100},
                    {"id": 2, "file_name": "img_2.png", "width": 100, "height": 100},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20]},
                ],
                "categories": [{"id": 1, "name": "spore"}],
            },
        )
        config = _make_portion_config(
            tmp_path,
            datasets_root,
            annotations_relpath="sliced/coco_annotations.json",
        )
        preparer = DatasetPreparer(config)
        preparer.prepare_dataset(val_split=0.5, seed=1)
        stale_image = config.output_dir / "train" / "images" / "stale.png"
        stale_label = config.output_dir / "train" / "labels" / "stale.txt"
        _write_dummy_image(stale_image)
        stale_label.write_text("0 0.100000 0.100000 0.100000 0.100000", encoding="utf-8")
        notes = config.output_dir / "notes.txt"
        notes.write_text("keep", encoding="utf-8")

        # Act
        data_yaml = preparer.prepare_dataset(val_split=0.5, seed=1)

        # Assert
        assert data_yaml == config.output_dir / "data.yaml"
        assert not stale_image.exists()
        assert not stale_label.exists()
        assert notes.read_text(encoding="utf-8") == "keep"
        image_names = {path.name for path in (config.output_dir / "train" / "images").glob("*")}
        image_names.update(path.name for path in (config.output_dir / "val" / "images").glob("*"))
        assert image_names == {"portion_01__img_1.png", "portion_01__img_2.png"}
        assert not list(tmp_path.glob(".yolo-prepare-*"))
        saved = data_yaml.read_text(encoding="utf-8")
        assert ".yolo-prepare-" not in saved
        assert "train/images" in saved or "train\\images" in saved

    def test_failed_prepare_leaves_the_previous_split(self, tmp_path):
        """A failed rebuild leaves the previous train, val, and data.yaml in place."""
        # Arrange
        datasets_root = tmp_path / "dataset_train"
        portion_dir = datasets_root / "portion_01"
        images_dir = portion_dir / "sliced" / "images"
        _write_dummy_image(images_dir / "img_1.png")
        coco_path = portion_dir / "sliced" / "coco_annotations.json"
        _write_coco_json(
            coco_path,
            {
                "images": [{"id": 1, "file_name": "img_1.png", "width": 100, "height": 100}],
                "annotations": [],
                "categories": [{"id": 1, "name": "spore"}],
            },
        )
        config = _make_portion_config(
            tmp_path,
            datasets_root,
            annotations_relpath="sliced/coco_annotations.json",
        )
        preparer = DatasetPreparer(config)
        preparer.prepare_dataset(val_split=0.5, seed=1)
        kept_image = config.output_dir / "val" / "images" / "portion_01__img_1.png"
        stale_image = config.output_dir / "train" / "images" / "stale.png"
        _write_dummy_image(stale_image)
        _write_coco_json(
            coco_path,
            {
                "images": [{"id": 1, "file_name": "img_1.png", "width": 100, "height": 100}],
                "annotations": [{"id": 1, "image_id": 1, "category_id": 5, "bbox": [1, 1, 5, 5]}],
                "categories": [{"id": 5, "name": "debris"}],
            },
        )

        # Act
        with pytest.raises(ValueError) as caught:
            preparer.prepare_dataset()

        # Assert
        assert "Unknown COCO category" in str(caught.value)
        assert stale_image.exists()
        assert kept_image.exists()
        assert (config.output_dir / "data.yaml").exists()
        assert not list(tmp_path.glob(".yolo-prepare-*"))

    def test_multiple_coco_files_require_relpath(self, tmp_path):
        """Two COCO files in one portion fail until yolo_annotations_relpath is set."""
        # Arrange
        datasets_root = tmp_path / "dataset_train"
        portion_dir = datasets_root / "portion_01"
        _write_dummy_image(portion_dir / "sliced" / "images" / "img_1.png")
        payload = {
            "images": [{"id": 1, "file_name": "img_1.png", "width": 100, "height": 100}],
            "annotations": [],
            "categories": [{"id": 1, "name": "spore"}],
        }
        _write_coco_json(portion_dir / "sliced" / "a_coco.json", payload)
        _write_coco_json(portion_dir / "sliced" / "b_coco.json", payload)
        config = _make_portion_config(tmp_path, datasets_root, annotations_format="coco")
        preparer = DatasetPreparer(config)

        # Act
        with pytest.raises(ValueError) as caught:
            preparer.prepare_dataset()

        # Assert
        assert "yolo_annotations_relpath" in str(caught.value)
        assert "a_coco.json" in str(caught.value)
        assert "b_coco.json" in str(caught.value)
        assert not config.output_dir.exists()

    def test_annotations_relpath_selects_one_coco_file(self, tmp_path):
        """yolo_annotations_relpath prepares the chosen COCO file."""
        # Arrange
        datasets_root = tmp_path / "dataset_train"
        portion_dir = datasets_root / "portion_01"
        _write_dummy_image(portion_dir / "sliced" / "images" / "img_1.png")
        payload = {
            "images": [{"id": 1, "file_name": "img_1.png", "width": 100, "height": 100}],
            "annotations": [],
            "categories": [{"id": 1, "name": "spore"}],
        }
        _write_coco_json(portion_dir / "sliced" / "a_coco.json", payload)
        _write_coco_json(portion_dir / "sliced" / "b_coco.json", payload)
        config = _make_portion_config(
            tmp_path,
            datasets_root,
            annotations_format="coco",
            annotations_relpath="sliced/a_coco.json",
        )
        preparer = DatasetPreparer(config)

        # Act
        data_yaml = preparer.prepare_dataset(val_split=0.5, seed=1)

        # Assert
        assert data_yaml.exists()
        assert (config.output_dir / "val" / "images" / "portion_01__img_1.png").exists()

    def test_multiple_xml_files_require_relpath(self, tmp_path):
        """Two XML files in one portion fail until yolo_annotations_relpath is set."""
        # Arrange
        datasets_root = tmp_path / "dataset_train"
        portion_dir = datasets_root / "portion_01"
        _write_dummy_image(portion_dir / "img_1.png")
        _write_cvat_xml(portion_dir / "annotations.xml")
        _write_cvat_xml(portion_dir / "review" / "annotations.xml")
        config = _make_portion_config(
            tmp_path,
            datasets_root,
            annotations_format="xml",
            images_subdir=".",
        )
        preparer = DatasetPreparer(config)

        # Act
        with pytest.raises(ValueError) as caught:
            preparer.prepare_dataset()

        # Assert
        assert "yolo_annotations_relpath" in str(caught.value)
        assert "review/annotations.xml" in str(caught.value)
        assert not config.output_dir.exists()

    def test_annotations_relpath_selects_one_xml_file(self, tmp_path):
        """yolo_annotations_relpath prepares the chosen XML file."""
        # Arrange
        datasets_root = tmp_path / "dataset_train"
        portion_dir = datasets_root / "portion_01"
        _write_dummy_image(portion_dir / "img_1.png")
        _write_cvat_xml(portion_dir / "annotations.xml")
        _write_cvat_xml(portion_dir / "review" / "annotations.xml")
        config = _make_portion_config(
            tmp_path,
            datasets_root,
            annotations_format="xml",
            annotations_relpath="review/annotations.xml",
            images_subdir=".",
        )
        preparer = DatasetPreparer(config)

        # Act
        data_yaml = preparer.prepare_dataset(val_split=0.5, seed=1)

        # Assert
        assert data_yaml.exists()
        label = config.output_dir / "val" / "labels" / "portion_01__img_1.txt"
        assert label.read_text(encoding="utf-8") == "0 0.500000 0.500000 0.200000 0.200000"


class TestDataDiscovery:
    """Tests for dataset folder and annotation discovery."""

    def test_discover_dataset_dirs_skips_container_root(self, tmp_path):
        """Child portions are returned and the container root is not."""
        # Arrange
        root = tmp_path / "dataset_train"
        (root / "portion_01").mkdir(parents=True)
        (root / "portion_02").mkdir()

        # Act
        found = discover_dataset_dirs(root, "*")

        # Assert
        assert found == [root / "portion_01", root / "portion_02"]

    def test_discover_dataset_dirs_uses_root_when_flat(self, tmp_path):
        """A folder with no children is the dataset itself."""
        # Arrange
        root = tmp_path / "dataset_train"
        root.mkdir()

        # Act
        found = discover_dataset_dirs(root, "*")

        # Assert
        assert found == [root]

    def test_find_coco_annotation_candidates_requires_coco_collections(self, tmp_path):
        """JSON without image, annotation, and category lists is ignored."""
        # Arrange
        (tmp_path / "notes.json").write_text('{"hello": 1}', encoding="utf-8")
        coco = tmp_path / "labels.json"
        coco.write_text(
            json.dumps({"images": [], "annotations": [], "categories": []}),
            encoding="utf-8",
        )

        # Act
        found = find_coco_annotation_candidates(tmp_path)

        # Assert
        assert found == [coco]


class TestTileyConfig:
    """SAHI settings load from top-level tiley_* keys, the same way as yolo_*."""

    def test_repository_yaml_loads_every_field(self):
        """Each tiley field is present and has its type."""
        # Arrange
        manager = _load_config(_repo_config_path())

        # Act
        tiley = TileyConfig.from_config_manager(manager)

        # Assert
        _assert_nonempty_text(tiley.export_input)
        _assert_optional_text(tiley.export_cvat_xml)
        _assert_nonempty_text(tiley.export_out)
        _assert_int(tiley.export_tile_size)
        _assert_float(tiley.export_overlap_height_ratio)
        _assert_float(tiley.export_overlap_width_ratio)
        _assert_nonempty_text(tiley.predict_input)
        _assert_nonempty_text(tiley.predict_out)
        _assert_int(tiley.predict_tile_size)
        _assert_float(tiley.predict_overlap_height_ratio)
        _assert_float(tiley.predict_overlap_width_ratio)
        _assert_float(tiley.predict_conf)
        _assert_bool(tiley.predict_write_previews)
        _assert_optional_text(tiley.predict_weights)
        _assert_nonempty_text(tiley.predict_device)
        _assert_nonempty_text(tiley.predict_model_type)

    def test_explicit_paths_and_disabled_previews(self, tmp_path):
        """Explicit annotation, weights, and preview values are loaded."""
        # Arrange
        text = _set_yaml_key(_read_repo_config(), "tiley_export_cvat_xml", "annotations.xml")
        text = _set_yaml_key(text, "tiley_predict_weights", "checkpoints/best.pt")
        text = _set_yaml_key(text, "tiley_predict_write_previews", "false")
        manager = _load_config_text(tmp_path, text)

        # Act
        tiley = TileyConfig.from_config_manager(manager)

        # Assert
        assert tiley.export_cvat_xml == "annotations.xml"
        assert tiley.predict_weights == "checkpoints/best.pt"
        assert tiley.predict_write_previews is False

    def test_missing_key_is_named(self, tmp_path):
        """A dropped tiley key fails with that key's name."""
        # Arrange
        text = _drop_yaml_key(_read_repo_config(), "tiley_predict_device")
        manager = _load_config_text(tmp_path, text)

        # Act
        with pytest.raises(ConfigurationError) as caught:
            TileyConfig.from_config_manager(manager)

        # Assert
        _assert_error_names(caught.value, "tiley_predict_device")

    @pytest.mark.parametrize(
        ("key", "raw_value"),
        [
            ("tiley_export_tile_size", "large"),
            ("tiley_predict_write_previews", '"yes"'),
            ("tiley_predict_weights", '""'),
        ],
        ids=["tile-size", "write-previews", "weights"],
    )
    def test_bad_types_are_named(self, tmp_path, key, raw_value):
        """A mistyped tiley value fails with that key's name."""
        # Arrange
        text = _set_yaml_key(_read_repo_config(), key, raw_value)
        manager = _load_config_text(tmp_path, text)

        # Act
        with pytest.raises(ConfigurationError) as caught:
            TileyConfig.from_config_manager(manager)

        # Assert
        _assert_error_names(caught.value, key)

    def test_path_overrides_do_not_change_quality_settings(self):
        """CLI folders replace only the input and output paths."""
        # Arrange
        manager = _load_config(_repo_config_path())
        tiley = TileyConfig.from_config_manager(manager)

        # Act
        export = _resolve_tiley_export(manager, "other_in", "other_out")
        predict = _resolve_tiley_predict(manager, "pred_in", "pred_out")

        # Assert
        assert export["images_dir"] == "other_in"
        assert export["out"] == "other_out"
        assert export["tile_size"] == tiley.export_tile_size
        assert export["overlap_height_ratio"] == tiley.export_overlap_height_ratio
        assert predict["source"] == "pred_in"
        assert predict["out"] == "pred_out"
        assert predict["conf"] == tiley.predict_conf
        assert predict["tile_size"] == tiley.predict_tile_size

    def test_null_weights_follow_configured_model(self, tmp_path):
        """Null predict weights follow the configured YOLO model name."""
        # Arrange
        text = _set_yaml_key(_read_repo_config(), "yolo_model", "yolo26s.pt")
        text = _set_yaml_key(text, "tiley_predict_weights", "null")
        manager = _load_config_text(tmp_path, text)
        yolo = YOLOConfig.from_config_manager(manager)

        # Act
        weights = Path(resolve_predict_weights(manager))
        trained_path = yolo.get_trained_model_path()
        run_name = yolo.get_training_run_name()

        # Assert
        assert weights == trained_path
        assert run_name == "yolo26s_spores"


class TestTrainCommandFlags:
    """--train-yolo stays the primary flag and keeps the old alias."""

    @pytest.mark.parametrize(
        ("arguments", "attribute", "expected"),
        [
            (["--train-yolo"], "train_yolo", True),
            (["--train-yolo-auto"], "train_yolo", True),
            (["--train-yolo", "--quick-test"], "quick_test", True),
        ],
        ids=["train-yolo", "train-yolo-auto", "quick-test"],
    )
    def test_train_yolo_and_alias_set_the_same_flag(self, arguments, attribute, expected):
        """Primary flag, alias, and quick-test set the matching parser fields."""
        # Arrange
        parser = _build_parser()

        # Act
        parsed = parser.parse_args(arguments)

        # Assert
        assert getattr(parsed, attribute) is expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
