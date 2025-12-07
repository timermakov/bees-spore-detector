"""
YOLO-based spore detection module.

Provides YOLO11-S model training, inference, and counting functionality
optimized for Intel Iris Xe with OpenVINO.
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
