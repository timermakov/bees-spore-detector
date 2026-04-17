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
from .cvat_tiled_export import CvatTiledPascalExporter, TiledDatasetExportStats
from .tiled_predict import run_tiled_prediction_folder

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
