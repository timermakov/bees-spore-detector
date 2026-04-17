"""
YOLO-based spore detection module.

Provides YOLO11-S model training, inference, and counting functionality
using PyTorch with CUDA support.

SAHI-based inference (v0.11.36+):
  - converter_coco: CVAT XML → COCO format conversion
  - sahi_inference: SAHI-based sliced inference wrapper
  - dataset_slicer: COCO dataset slicing and splitting

Hierarchical Analysis:
  - hierarchical_analysis: Nested folder structure analysis without XML annotations
"""

from .config import YOLOConfig
from .converter import CVATToYOLOConverter
from .dataset import DatasetPreparer
from .trainer import SporeTrainer
from .detector import SporeDetector
from .counter import SporeCounter
from .pseudo_label import PseudoLabeler, pseudo_label_images

# SAHI-based modules (optional)
try:
    from .converter_coco import CVATToCocoConverter
    from .sahi_inference import SAHIDetector, run_sliced_inference_folder
    from .dataset_slicer import DatasetSlicer, SlicedDatasetStats
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False
    CVATToCocoConverter = None
    SAHIDetector = None
    run_sliced_inference_folder = None
    DatasetSlicer = None
    SlicedDatasetStats = None

# Hierarchical analysis (requires spore detection)
try:
    from .hierarchical_analysis import HierarchicalAnalyzer
    HIERARCHICAL_AVAILABLE = True
except ImportError:
    HIERARCHICAL_AVAILABLE = False
    HierarchicalAnalyzer = None

__all__ = [
    'YOLOConfig',
    'CVATToYOLOConverter',
    'DatasetPreparer',
    'SporeTrainer',
    'SporeDetector',
    'SporeCounter',
    'PseudoLabeler',
    'pseudo_label_images',
    'CVATToCocoConverter',
    'SAHIDetector',
    'DatasetSlicer',
    'SlicedDatasetStats',
    'run_sliced_inference_folder',
    'HierarchicalAnalyzer',
    'SAHI_AVAILABLE',
    'HIERARCHICAL_AVAILABLE',
]
