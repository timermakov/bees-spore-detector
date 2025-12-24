"""
YOLO model training for spore detection.

Fine-tunes YOLO11-S on spore dataset with configurable parameters.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging

from .config import YOLOConfig
from .dataset import DatasetPreparer

logger = logging.getLogger(__name__)


class SporeTrainer:
    """Trains YOLO model for spore detection."""
    
    def __init__(self, config: YOLOConfig):
        """
        Initialize trainer.
        
        Args:
            config: YOLOConfig instance
        """
        self.config = config
        self.model = None
        self.results = None
    
    def prepare_data(self, val_split: float = 0.4) -> Path:
        """
        Prepare dataset for training.
        
        Args:
            val_split: Validation split fraction (default 0.4 for 60/40 split)
            
        Returns:
            Path to data.yaml
        """
        preparer = DatasetPreparer(self.config)
        return preparer.prepare_dataset(val_split=val_split)
    
    def train(self, 
              data_yaml: Optional[Path] = None,
              resume: bool = False,
              quick_test: bool = False) -> Dict[str, Any]:
        """
        Train YOLO model on spore dataset.
        
        Args:
            data_yaml: Path to data.yaml (if None, prepares dataset first)
            resume: Whether to resume from last checkpoint
            quick_test: If True, use reduced settings for fast testing (~15 min)
            
        Returns:
            Training results dict
        """
        import platform
        
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")
        
        # Prepare data if needed
        if data_yaml is None:
            data_yaml = self.prepare_data()
        
        # Load model
        if resume and self.config.get_trained_model_path().exists():
            logger.info(f"Resuming from {self.config.get_trained_model_path()}")
            self.model = YOLO(str(self.config.get_trained_model_path()))
        else:
            logger.info(f"Loading pretrained {self.config.model_name}")
            self.model = YOLO(self.config.model_name)
        
        # Get augmentation config
        preparer = DatasetPreparer(self.config)
        aug_config = preparer.get_augmentation_config()
        
        # Quick test mode: reduced settings for ~15 min training
        if quick_test:
            epochs = 30
            imgsz = 640  # 4x faster than 1280
            batch_size = 16
            patience = 0  # Disable early stopping - train all epochs
            logger.info("QUICK TEST MODE: 30 epochs, 640px images, NO early stopping")
        else:
            epochs = self.config.epochs
            imgsz = self.config.imgsz
            batch_size = self.config.batch_size
            patience = self.config.patience
        
        # Windows: use workers=0 to avoid multiprocessing issues
        workers = 0 if platform.system() == 'Windows' else 4
        
        # Train
        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Image size: {imgsz}, Batch size: {batch_size}, Workers: {workers}")
        
        self.results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            patience=patience,
            project=str(self.config.models_dir),
            name="yolo11s_spores",
            exist_ok=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=3,
            # Augmentations
            mosaic=aug_config['mosaic'],
            mixup=aug_config['mixup'],
            hsv_h=aug_config['hsv_h'],
            hsv_s=aug_config['hsv_s'],
            hsv_v=aug_config['hsv_v'],
            degrees=aug_config['degrees'],
            scale=aug_config['scale'],
            fliplr=aug_config['fliplr'],
            flipud=aug_config['flipud'],
            copy_paste=0.3,  # Copy-paste augmentation for small datasets
            # Device - auto-select best available
            device='0' if self._cuda_available() else 'cpu',
            workers=workers,
            verbose=True,
            # Disable text labels on visualizations (too cluttered with many detections)
            show_labels=False,
            show_conf=False,
        )
        
        logger.info("Training completed")
        return self._get_metrics()
    
    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _get_metrics(self) -> Dict[str, Any]:
        """Extract metrics from training results."""
        if self.results is None:
            return {}
        
        metrics = {
            'best_model': str(self.config.get_trained_model_path()),
        }
        
        # Extract final metrics if available
        if hasattr(self.results, 'results_dict'):
            rd = self.results.results_dict
            metrics.update({
                'mAP50': rd.get('metrics/mAP50(B)', 0),
                'mAP50-95': rd.get('metrics/mAP50-95(B)', 0),
                'precision': rd.get('metrics/precision(B)', 0),
                'recall': rd.get('metrics/recall(B)', 0),
            })
        
        return metrics
    
    def validate(self, model_path: Optional[Path] = None) -> Dict[str, float]:
        """
        Validate trained model.
        
        Args:
            model_path: Path to model (uses best.pt if None)
            
        Returns:
            Validation metrics
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package required")
        
        if model_path is None:
            model_path = self.config.get_trained_model_path()
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = YOLO(str(model_path))
        data_yaml = self.config.output_dir / "data.yaml"
        
        results = model.val(
            data=str(data_yaml),
            imgsz=self.config.imgsz,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
        )
        
        return {
            'mAP50': float(results.box.map50),
            'mAP50-95': float(results.box.map),
            'precision': float(results.box.mp),
            'recall': float(results.box.mr),
        }
    
    def export_model(self, format: Optional[str] = None) -> Path:
        """
        Export trained model to specified format.
        
        Args:
            format: Export format (openvino, onnx, etc.)
            
        Returns:
            Path to exported model
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package required")
        
        format = format or self.config.export_format
        model_path = self.config.get_trained_model_path()
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        model = YOLO(str(model_path))
        
        logger.info(f"Exporting model to {format} format")
        
        export_path = model.export(
            format=format,
            imgsz=self.config.imgsz,
            half=self.config.half_precision,
        )
        
        logger.info(f"Model exported to: {export_path}")
        return Path(export_path)


def train_spore_model(config: YOLOConfig) -> Dict[str, Any]:
    """
    Convenience function to train spore detection model.
    
    Args:
        config: YOLOConfig instance
        
    Returns:
        Training metrics
    """
    trainer = SporeTrainer(config)
    return trainer.train()
