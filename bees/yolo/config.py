"""
Configuration for YOLO spore detection.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


@dataclass
class YOLOConfig:
    """Configuration for YOLO model training and inference."""

    REQUIRED_YAML_KEYS = (
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
    )
    
    # Model settings
    model_name: str = "yolo11s.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    # Training settings
    epochs: int = 100
    batch_size: int = 16  # Reduced for CPU memory (1280px images)
    imgsz: int = 960
    patience: int = 25
    device: Optional[str] = None  # Device for training (None = auto-detect Nvidia GPU, 'cpu', 'cuda:0', etc.)
    workers: int = 2
    quick_test_epochs: int = 30
    quick_test_imgsz: int = 1024
    quick_test_batch_size: int = 8
    quick_test_patience: int = 0
    validation_split: float = 0.4
    split_seed: int = 42
    
    # Data paths
    datasets_root: Path = field(default_factory=lambda: Path("dataset_train"))
    dataset_folder_pattern: str = "*"
    annotations_format: str = "auto"
    annotations_relpath: Optional[str] = None
    images_subdir: str = "."
    image_extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"])
    test_dir: Path = field(default_factory=lambda: Path("dataset_test"))
    output_dir: Path = field(default_factory=lambda: Path("yolo_dataset"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    
    # Export settings (not used - models stay in PyTorch format)
    half_precision: bool = False  # FP32 precision
    
    # Augmentation settings
    mosaic: float = 1.0
    mixup: float = 0.15
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 15.0
    scale: float = 0.3
    fliplr: float = 0.5
    flipud: float = 0.5
    optimizer: str = "AdamW"
    learning_rate: float = 0.0005
    final_learning_rate_factor: float = 0.01
    warmup_epochs: int = 2
    copy_paste: float = 0.3
    cache: bool = False
    amp: bool = True
    show_labels: bool = False
    show_conf: bool = False
    pretrained: bool = True
    verbose: bool = True
    
    # Analysis zone (from main config)
    analysis_square_size: int = 780
    
    # Class configuration
    class_names: List[str] = field(default_factory=lambda: ["spore"])
    
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
            raise ValueError(
                "config.yaml is missing required YOLO keys: " + ", ".join(missing_keys)
            )

        return cls(
            model_name=config_manager.get_param('yolo_model'),
            confidence_threshold=config_manager.get_float_param('yolo_confidence'),
            iou_threshold=config_manager.get_float_param('yolo_iou_threshold'),
            epochs=config_manager.get_int_param('yolo_epochs'),
            batch_size=config_manager.get_int_param('yolo_batch_size'),
            imgsz=config_manager.get_int_param('yolo_imgsz'),
            patience=config_manager.get_int_param('yolo_patience'),
            analysis_square_size=config_manager.get_int_param('analysis_square_size'),
            device=config_manager.get_param('yolo_device'),
            workers=config_manager.get_int_param('yolo_workers'),
            quick_test_epochs=config_manager.get_int_param('yolo_quick_test_epochs'),
            quick_test_imgsz=config_manager.get_int_param('yolo_quick_test_imgsz'),
            quick_test_batch_size=config_manager.get_int_param('yolo_quick_test_batch_size'),
            quick_test_patience=config_manager.get_int_param('yolo_quick_test_patience'),
            validation_split=config_manager.get_float_param('yolo_validation_split'),
            split_seed=config_manager.get_int_param('yolo_split_seed'),
            datasets_root=Path(config_manager.get_param('yolo_datasets_root')),
            dataset_folder_pattern=config_manager.get_param('yolo_dataset_folder_pattern'),
            annotations_format=config_manager.get_param('yolo_annotations_format'),
            annotations_relpath=config_manager.get_param('yolo_annotations_relpath'),
            images_subdir=config_manager.get_param('yolo_images_subdir'),
            image_extensions=config_manager.get_param('yolo_image_extensions'),
            test_dir=Path(config_manager.get_param('yolo_test_dir')),
            output_dir=Path(config_manager.get_param('yolo_output_dir')),
            models_dir=Path(config_manager.get_param('yolo_models_dir')),
            mosaic=config_manager.get_float_param('yolo_mosaic'),
            mixup=config_manager.get_float_param('yolo_mixup'),
            hsv_h=config_manager.get_float_param('yolo_hsv_h'),
            hsv_s=config_manager.get_float_param('yolo_hsv_s'),
            hsv_v=config_manager.get_float_param('yolo_hsv_v'),
            degrees=config_manager.get_float_param('yolo_degrees'),
            scale=config_manager.get_float_param('yolo_scale'),
            fliplr=config_manager.get_float_param('yolo_fliplr'),
            flipud=config_manager.get_float_param('yolo_flipud'),
            optimizer=config_manager.get_param('yolo_optimizer'),
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
            class_names=config_manager.get_param('yolo_class_names'),
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
