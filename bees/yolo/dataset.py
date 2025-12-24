"""
Dataset preparation for YOLO training.

Handles train/val split, data.yaml generation, and augmentation configuration.
"""

import random
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import logging
import yaml

from .config import YOLOConfig
from .converter import CVATToYOLOConverter

logger = logging.getLogger(__name__)


class DatasetPreparer:
    """Prepares dataset for YOLO training."""
    
    def __init__(self, config: YOLOConfig):
        """
        Initialize dataset preparer.
        
        Args:
            config: YOLOConfig instance
        """
        self.config = config
        self.converter = CVATToYOLOConverter(config.class_names)
    
    def prepare_dataset(self, 
                        val_split: float = 0.4,
                        seed: int = 42) -> Path:
        """
        Prepare complete YOLO dataset from CVAT annotations.
        
        Args:
            val_split: Fraction of data for validation (default 0.4 for 60/40 split)
            seed: Random seed for reproducibility
            
        Returns:
            Path to generated data.yaml
        """
        random.seed(seed)
        
        self.config.ensure_dirs()
        
        # Parse annotations
        annotations = self.converter.parse_cvat_xml(self.config.annotations_path)
        
        if not annotations:
            raise ValueError(f"No annotations found in {self.config.annotations_path}")
        
        # Shuffle and split
        random.shuffle(annotations)
        val_count = max(1, int(len(annotations) * val_split))
        train_count = len(annotations) - val_count
        
        train_annotations = annotations[:train_count]
        val_annotations = annotations[train_count:]
        
        logger.info(f"Split: {len(train_annotations)} train, {len(val_annotations)} val")
        
        # Process train set
        train_stats = self._process_split(
            train_annotations, 
            self.config.output_dir / "train"
        )
        
        # Process val set
        val_stats = self._process_split(
            val_annotations,
            self.config.output_dir / "val"
        )
        
        # Generate data.yaml
        data_yaml_path = self._generate_data_yaml()
        
        logger.info(f"Dataset prepared: train={train_stats}, val={val_stats}")
        logger.info(f"Data config: {data_yaml_path}")
        
        return data_yaml_path
    
    def _process_split(self, annotations: List, output_dir: Path) -> dict:
        """Process a dataset split (train or val)."""
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {'images': 0, 'annotations': 0}
        
        for annotation in annotations:
            # Find source image
            img_name = Path(annotation.name).name
            src_image = self._find_source_image(annotation.name)
            
            if src_image is None:
                logger.warning(f"Image not found: {annotation.name}")
                continue
            
            # Copy image
            dst_image = images_dir / img_name
            shutil.copy2(src_image, dst_image)
            
            # Generate and save labels
            lines = self.converter.convert_annotation(annotation)
            label_file = labels_dir / (Path(img_name).stem + ".txt")
            
            with open(label_file, 'w') as f:
                f.write('\n'.join(lines))
            
            stats['images'] += 1
            stats['annotations'] += len(lines)
        
        return stats
    
    def _find_source_image(self, annotation_path: str) -> Optional[Path]:
        """Find source image file from annotation path."""
        # Try different path combinations
        candidates = [
            self.config.images_dir / Path(annotation_path).name,
            self.config.images_dir / annotation_path,
            self.config.images_dir.parent / annotation_path,
            Path(annotation_path),
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        return None
    
    def _generate_data_yaml(self) -> Path:
        """Generate data.yaml for YOLO training."""
        data_yaml_path = self.config.output_dir / "data.yaml"
        
        # Use absolute paths for reliability
        train_path = (self.config.output_dir / "train" / "images").resolve()
        val_path = (self.config.output_dir / "val" / "images").resolve()
        
        data_config = {
            'path': str(self.config.output_dir.resolve()),
            'train': str(train_path),
            'val': str(val_path),
            'names': {i: name for i, name in enumerate(self.config.class_names)},
            'nc': len(self.config.class_names),
        }
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, allow_unicode=True)
        
        return data_yaml_path
    
    def get_augmentation_config(self) -> dict:
        """
        Get augmentation configuration for YOLO training.
        
        Returns:
            Dict of augmentation parameters
        """
        return {
            'mosaic': self.config.mosaic,
            'mixup': self.config.mixup,
            'hsv_h': self.config.hsv_h,
            'hsv_s': self.config.hsv_s,
            'hsv_v': self.config.hsv_v,
            'degrees': self.config.degrees,
            'scale': self.config.scale,
            'fliplr': self.config.fliplr,
            'flipud': self.config.flipud,
        }


def prepare_yolo_dataset(config: YOLOConfig, val_split: float = 0.4) -> Path:
    """
    Convenience function to prepare YOLO dataset.
    
    Args:
        config: YOLOConfig instance
        val_split: Validation split fraction (default 0.4 for 60/40 split)
        
    Returns:
        Path to data.yaml
    """
    preparer = DatasetPreparer(config)
    return preparer.prepare_dataset(val_split=val_split)
