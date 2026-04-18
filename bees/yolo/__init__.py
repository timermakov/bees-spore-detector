"""
YOLO-based spore detection module.

Provides YOLO11-S model training, inference, and counting functionality
using PyTorch with CUDA support.
"""

from .config import YOLOConfig
from .converter import CVATToYOLOConverter
from .dataset import DatasetPreparer
from .trainer import SporeTrainer
from .detector import SporeDetector
from .counter import SporeCounter
from .pseudo_label import PseudoLabeler, pseudo_label_images

__all__ = [
    'YOLOConfig',
    'CVATToYOLOConverter',
    'DatasetPreparer',
    'SporeTrainer',
    'SporeDetector',
    'SporeCounter',
    'PseudoLabeler',
    'pseudo_label_images',
]

try:
    from .converter_coco import CVATToCocoConverter
    from .sahi_inference import SAHIDetector, run_sliced_inference_folder
    from .dataset_slicer import DatasetSlicer, SlicedDatasetStats

    __all__.extend([
        'CVATToCocoConverter',
        'SAHIDetector',
        'run_sliced_inference_folder',
        'DatasetSlicer',
        'SlicedDatasetStats',
    ])
except ImportError:
    # SAHI dependencies are optional for non-tiled workflows.
    pass
