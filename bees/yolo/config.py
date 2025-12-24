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
    
    # Model settings
    model_name: str = "yolo11s.pt"
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    
    # Training settings
    epochs: int = 100
    batch_size: int = 4  # Reduced for CPU memory (1280px images)
    imgsz: int = 1280
    patience: int = 20
    
    # Data paths
    annotations_path: Path = field(default_factory=lambda: Path("annotations.xml"))
    images_dir: Path = field(default_factory=lambda: Path("dataset_train"))
    test_dir: Path = field(default_factory=lambda: Path("dataset_test"))
    output_dir: Path = field(default_factory=lambda: Path("yolo_dataset"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    
    # Export settings
    export_format: str = "openvino"
    half_precision: bool = False  # FP32 for Iris Xe
    
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
    
    # Analysis zone (from main config)
    analysis_square_size: int = 780
    
    # Class configuration
    class_names: List[str] = field(default_factory=lambda: ["spore"])
    
    def __post_init__(self):
        """Convert string paths to Path objects."""
        if isinstance(self.annotations_path, str):
            self.annotations_path = Path(self.annotations_path)
        if isinstance(self.images_dir, str):
            self.images_dir = Path(self.images_dir)
        if isinstance(self.test_dir, str):
            self.test_dir = Path(self.test_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.models_dir, str):
            self.models_dir = Path(self.models_dir)
    
    @classmethod
    def from_config_manager(cls, config_manager) -> 'YOLOConfig':
        """Create YOLOConfig from existing ConfigurationManager."""
        return cls(
            model_name=config_manager.get_param('yolo_model', 'yolo11s.pt'),
            confidence_threshold=config_manager.get_float_param('yolo_confidence', 0.25),
            iou_threshold=config_manager.get_float_param('yolo_iou_threshold', 0.45),
            epochs=config_manager.get_int_param('yolo_epochs', 100),
            batch_size=config_manager.get_int_param('yolo_batch_size', 4),
            imgsz=config_manager.get_int_param('yolo_imgsz', 1280),
            export_format=config_manager.get_param('yolo_export_format', 'openvino'),
            analysis_square_size=config_manager.get_int_param('analysis_square_size', 780),
        )
    
    def get_trained_model_path(self) -> Path:
        """Get path to the trained model."""
        return self.models_dir / "yolo11s_spores" / "weights" / "best.pt"
    
    def get_exported_model_path(self) -> Path:
        """Get path to the exported model."""
        if self.export_format == "openvino":
            return self.models_dir / "yolo11s_spores_openvino"
        return self.models_dir / f"yolo11s_spores.{self.export_format}"
    
    def ensure_dirs(self):
        """Create necessary directories."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "val" / "images").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "val" / "labels").mkdir(parents=True, exist_ok=True)
