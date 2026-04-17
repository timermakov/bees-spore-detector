"""
YOLO-based spore detection module.

Provides YOLO11-S model training, inference, and counting functionality
using PyTorch with CUDA support.

SAHI Integration (v0.11.36+):
  - converter_coco: CVAT XML → COCO format conversion
  - sahi_inference: SAHI-based sliced inference wrapper
  - dataset_slicer: COCO dataset slicing and splitting
  
Legacy modules (still available):
  - converter: CVAT XML → YOLO format
  - cvat_tiled_export: Custom dataset slicing
  - tiled_predict: Custom tiled inference
"""

from .config import YOLOConfig
from .converter import CVATToYOLOConverter
from .dataset import DatasetPreparer
from .trainer import SporeTrainer
from .detector import SporeDetector
from .counter import SporeCounter
from .pseudo_label import PseudoLabeler, pseudo_label_images
from .cvat_tiled_export import CvatTiledPascalExporter, TiledDatasetExportStats
from .tiled_predict import run_tiled_prediction_folder

# SAHI Integration
try:
    from .converter_coco import CVATToCocoConverter
    from .sahi_inference import SAHIDetector, run_sliced_inference_folder
    from .dataset_slicer import DatasetSlicer, SlicedDatasetStats
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False

__all__ = [
    'YOLOConfig',
    'CVATToYOLOConverter',
    'DatasetPreparer',
    'SporeTrainer',
    'SporeDetector',
    'SporeCounter',
    'PseudoLabeler',
    'pseudo_label_images',
    'CvatTiledPascalExporter',
    'TiledDatasetExportStats',
    'run_tiled_prediction_folder',
]

# Add SAHI exports if available
if SAHI_AVAILABLE:
    __all__.extend([
        'CVATToCocoConverter',
        'SAHIDetector',
        'DatasetSlicer',
        'SlicedDatasetStats',
        'run_sliced_inference_folder',
    ])
