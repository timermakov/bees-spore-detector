"""
Configuration for YOLO spore detection.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional, List
import logging

from ..config_loader import ConfigurationError

logger = logging.getLogger(__name__)


@dataclass
class YOLOConfig:
    """Configuration for YOLO model training and inference."""

    REQUIRED_YAML_KEYS: ClassVar[tuple[str, ...]] = (
        "yolo_model",
        "yolo_confidence",
        "yolo_iou_threshold",
        "yolo_epochs",
        "yolo_batch_size",
        "yolo_imgsz",
        "yolo_patience",
        "yolo_device",
        "yolo_workers",
        "yolo_quick_test_epochs",
        "yolo_quick_test_imgsz",
        "yolo_quick_test_batch_size",
        "yolo_quick_test_patience",
        "yolo_validation_split",
        "yolo_split_seed",
        "yolo_datasets_root",
        "yolo_dataset_folder_pattern",
        "yolo_annotations_format",
        "yolo_annotations_relpath",
        "yolo_images_subdir",
        "yolo_image_extensions",
        "yolo_test_dir",
        "yolo_output_dir",
        "yolo_models_dir",
        "yolo_mosaic",
        "yolo_mixup",
        "yolo_hsv_h",
        "yolo_hsv_s",
        "yolo_hsv_v",
        "yolo_degrees",
        "yolo_scale",
        "yolo_fliplr",
        "yolo_flipud",
        "yolo_optimizer",
        "yolo_learning_rate",
        "yolo_final_learning_rate_factor",
        "yolo_warmup_epochs",
        "yolo_copy_paste",
        "yolo_cache",
        "yolo_amp",
        "yolo_show_labels",
        "yolo_show_conf",
        "yolo_pretrained",
        "yolo_verbose",
        "yolo_class_names",
        "analysis_square_size",
        "yolo_pseudo_source",
        "yolo_pseudo_output",
        "yolo_pseudo_confidence",
        "yolo_pseudo_max_det",
        "yolo_cvat_source",
        "yolo_cvat_output",
        "yolo_cvat_task",
        "yolo_cvat_max_det",
        "yolo_cvat_shape",
    )
    
    # All values are required. ``from_config_manager`` builds this object from
    # config.yaml; there is deliberately no second configuration in Python.
    model_name: str
    confidence_threshold: float
    iou_threshold: float
    epochs: int
    batch_size: int
    imgsz: int
    patience: int
    device: Optional[str]
    workers: int
    quick_test_epochs: int
    quick_test_imgsz: int
    quick_test_batch_size: int
    quick_test_patience: int
    validation_split: float
    split_seed: int
    datasets_root: Path
    dataset_folder_pattern: str
    annotations_format: str
    annotations_relpath: Optional[str]
    images_subdir: str
    image_extensions: List[str]
    test_dir: Path
    output_dir: Path
    models_dir: Path
    mosaic: float
    mixup: float
    hsv_h: float
    hsv_s: float
    hsv_v: float
    degrees: float
    scale: float
    fliplr: float
    flipud: float
    optimizer: str
    learning_rate: float
    final_learning_rate_factor: float
    warmup_epochs: int
    copy_paste: float
    cache: bool
    amp: bool
    show_labels: bool
    show_conf: bool
    pretrained: bool
    verbose: bool
    analysis_square_size: int
    class_names: List[str]
    pseudo_source: str
    pseudo_output: str
    pseudo_confidence: float
    pseudo_max_det: int
    cvat_source: str
    cvat_output: str
    cvat_task: str
    cvat_max_det: int
    cvat_shape: str
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.datasets_root, str):
            self.datasets_root = Path(self.datasets_root)
        if isinstance(self.test_dir, str):
            self.test_dir = Path(self.test_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.models_dir, str):
            self.models_dir = Path(self.models_dir)
    
    @classmethod
    def from_config_manager(cls, config_manager) -> 'YOLOConfig':
        """Create YOLOConfig from existing ConfigurationManager."""
        missing_keys = [key for key in cls.REQUIRED_YAML_KEYS if not config_manager.has_param(key)]
        if missing_keys:
            raise ConfigurationError(
                "config.yaml is missing required YOLO keys: " + ", ".join(missing_keys)
            )

        image_extensions = config_manager.get_param('yolo_image_extensions')
        class_names = config_manager.get_param('yolo_class_names')
        if not isinstance(image_extensions, list) or not all(isinstance(item, str) for item in image_extensions):
            raise ConfigurationError("yolo_image_extensions must be a list of file extensions")
        if not isinstance(class_names, list) or not all(isinstance(item, str) and item for item in class_names):
            raise ConfigurationError("yolo_class_names must be a non-empty list of class names")
        cvat_shape = config_manager.get_string_param("yolo_cvat_shape")
        if cvat_shape not in {"box", "ellipse"}:
            raise ConfigurationError("yolo_cvat_shape must be 'box' or 'ellipse'")

        device = config_manager.get_param('yolo_device')
        if device is not None and not isinstance(device, str):
            raise ConfigurationError("yolo_device must be a string such as 'cpu' or '0', or null")
        annotations_relpath = config_manager.get_param('yolo_annotations_relpath')
        if annotations_relpath is not None and (
            not isinstance(annotations_relpath, str) or not annotations_relpath.strip()
        ):
            raise ConfigurationError("yolo_annotations_relpath must be a relative path or null")

        return cls(
            model_name=config_manager.get_string_param('yolo_model'),
            confidence_threshold=config_manager.get_float_param('yolo_confidence'),
            iou_threshold=config_manager.get_float_param('yolo_iou_threshold'),
            epochs=config_manager.get_int_param('yolo_epochs'),
            batch_size=config_manager.get_int_param('yolo_batch_size'),
            imgsz=config_manager.get_int_param('yolo_imgsz'),
            patience=config_manager.get_int_param('yolo_patience'),
            analysis_square_size=config_manager.get_int_param('analysis_square_size'),
            device=device,
            workers=config_manager.get_int_param('yolo_workers'),
            quick_test_epochs=config_manager.get_int_param('yolo_quick_test_epochs'),
            quick_test_imgsz=config_manager.get_int_param('yolo_quick_test_imgsz'),
            quick_test_batch_size=config_manager.get_int_param('yolo_quick_test_batch_size'),
            quick_test_patience=config_manager.get_int_param('yolo_quick_test_patience'),
            validation_split=config_manager.get_float_param('yolo_validation_split'),
            split_seed=config_manager.get_int_param('yolo_split_seed'),
            datasets_root=Path(config_manager.get_string_param('yolo_datasets_root')),
            dataset_folder_pattern=config_manager.get_string_param('yolo_dataset_folder_pattern'),
            annotations_format=config_manager.get_string_param('yolo_annotations_format'),
            annotations_relpath=annotations_relpath,
            images_subdir=config_manager.get_string_param('yolo_images_subdir'),
            image_extensions=image_extensions,
            test_dir=Path(config_manager.get_string_param('yolo_test_dir')),
            output_dir=Path(config_manager.get_string_param('yolo_output_dir')),
            models_dir=Path(config_manager.get_string_param('yolo_models_dir')),
            mosaic=config_manager.get_float_param('yolo_mosaic'),
            mixup=config_manager.get_float_param('yolo_mixup'),
            hsv_h=config_manager.get_float_param('yolo_hsv_h'),
            hsv_s=config_manager.get_float_param('yolo_hsv_s'),
            hsv_v=config_manager.get_float_param('yolo_hsv_v'),
            degrees=config_manager.get_float_param('yolo_degrees'),
            scale=config_manager.get_float_param('yolo_scale'),
            fliplr=config_manager.get_float_param('yolo_fliplr'),
            flipud=config_manager.get_float_param('yolo_flipud'),
            optimizer=config_manager.get_string_param('yolo_optimizer'),
            learning_rate=config_manager.get_float_param('yolo_learning_rate'),
            final_learning_rate_factor=config_manager.get_float_param('yolo_final_learning_rate_factor'),
            warmup_epochs=config_manager.get_int_param('yolo_warmup_epochs'),
            copy_paste=config_manager.get_float_param('yolo_copy_paste'),
            cache=config_manager.get_bool_param('yolo_cache'),
            amp=config_manager.get_bool_param('yolo_amp'),
            show_labels=config_manager.get_bool_param('yolo_show_labels'),
            show_conf=config_manager.get_bool_param('yolo_show_conf'),
            pretrained=config_manager.get_bool_param('yolo_pretrained'),
            verbose=config_manager.get_bool_param('yolo_verbose'),
            class_names=class_names,
            pseudo_source=config_manager.get_string_param("yolo_pseudo_source"),
            pseudo_output=config_manager.get_string_param("yolo_pseudo_output"),
            pseudo_confidence=config_manager.get_float_param("yolo_pseudo_confidence"),
            pseudo_max_det=config_manager.get_int_param("yolo_pseudo_max_det"),
            cvat_source=config_manager.get_string_param("yolo_cvat_source"),
            cvat_output=config_manager.get_string_param("yolo_cvat_output"),
            cvat_task=config_manager.get_string_param("yolo_cvat_task"),
            cvat_max_det=config_manager.get_int_param("yolo_cvat_max_det"),
            cvat_shape=cvat_shape,
        )
    
    def get_trained_model_path(self) -> Path:
        """Get path to the trained model."""
        return self.models_dir / self.get_training_run_name() / "weights" / "best.pt"

    def get_training_run_name(self) -> str:
        """Derive a stable training run name from the configured model file."""
        return f"{Path(self.model_name).stem}_spores"
    
    
    def ensure_dirs(self):
        """Create necessary directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)


@dataclass
class TileyConfig:
    """SAHI tiling settings. Built from ``tiley_*`` keys in config.yaml."""

    REQUIRED_YAML_KEYS: ClassVar[tuple[str, ...]] = (
        "tiley_export_input",
        "tiley_export_cvat_xml",
        "tiley_export_out",
        "tiley_export_tile_size",
        "tiley_export_overlap_height_ratio",
        "tiley_export_overlap_width_ratio",
        "tiley_predict_input",
        "tiley_predict_out",
        "tiley_predict_tile_size",
        "tiley_predict_overlap_height_ratio",
        "tiley_predict_overlap_width_ratio",
        "tiley_predict_conf",
        "tiley_predict_write_previews",
        "tiley_predict_weights",
        "tiley_predict_device",
        "tiley_predict_model_type",
    )

    export_input: str
    export_cvat_xml: Optional[str]
    export_out: str
    export_tile_size: int
    export_overlap_height_ratio: float
    export_overlap_width_ratio: float
    predict_input: str
    predict_out: str
    predict_tile_size: int
    predict_overlap_height_ratio: float
    predict_overlap_width_ratio: float
    predict_conf: float
    predict_write_previews: bool
    predict_weights: Optional[str]
    predict_device: str
    predict_model_type: str

    @classmethod
    def from_config_manager(cls, config_manager) -> "TileyConfig":
        """Create TileyConfig from config.yaml. Missing keys are reported by name."""
        missing_keys = [key for key in cls.REQUIRED_YAML_KEYS if not config_manager.has_param(key)]
        if missing_keys:
            raise ConfigurationError(
                "config.yaml is missing required tiley keys: " + ", ".join(missing_keys)
            )

        export_cvat_xml = _optional_yaml_text(config_manager, "tiley_export_cvat_xml")
        predict_weights = _optional_yaml_text(config_manager, "tiley_predict_weights")
        return cls(
            export_input=config_manager.get_string_param("tiley_export_input"),
            export_cvat_xml=export_cvat_xml,
            export_out=config_manager.get_string_param("tiley_export_out"),
            export_tile_size=config_manager.get_int_param("tiley_export_tile_size"),
            export_overlap_height_ratio=config_manager.get_float_param("tiley_export_overlap_height_ratio"),
            export_overlap_width_ratio=config_manager.get_float_param("tiley_export_overlap_width_ratio"),
            predict_input=config_manager.get_string_param("tiley_predict_input"),
            predict_out=config_manager.get_string_param("tiley_predict_out"),
            predict_tile_size=config_manager.get_int_param("tiley_predict_tile_size"),
            predict_overlap_height_ratio=config_manager.get_float_param("tiley_predict_overlap_height_ratio"),
            predict_overlap_width_ratio=config_manager.get_float_param("tiley_predict_overlap_width_ratio"),
            predict_conf=config_manager.get_float_param("tiley_predict_conf"),
            predict_write_previews=config_manager.get_bool_param("tiley_predict_write_previews"),
            predict_weights=predict_weights,
            predict_device=config_manager.get_string_param("tiley_predict_device"),
            predict_model_type=config_manager.get_string_param("tiley_predict_model_type"),
        )


def _optional_yaml_text(config_manager, key: str) -> Optional[str]:
    """Read a YAML string that may be null."""
    value = config_manager.get_param(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"config.yaml '{key}' must be a non-empty string or null")
    return value


def resolve_predict_weights(config_manager) -> str:
    """Resolve ``tiley_predict_weights``.

    Null selects the trained checkpoint for ``yolo_model``:
    ``<yolo_models_dir>/<model stem>_spores/weights/best.pt``.
    An explicit path pins a different checkpoint.
    """
    weights = TileyConfig.from_config_manager(config_manager).predict_weights
    if weights:
        return weights
    return str(YOLOConfig.from_config_manager(config_manager).get_trained_model_path())
