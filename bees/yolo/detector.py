"""
YOLO spore detector with OpenVINO optimization for Intel Iris Xe.
"""

from pathlib import Path
from typing import List, Tuple, Optional, Union
import logging
import numpy as np
from PIL import Image

from .config import YOLOConfig

logger = logging.getLogger(__name__)


class Detection:
    """Represents a single detection."""
    
    def __init__(self, 
                 bbox: Tuple[float, float, float, float],
                 confidence: float,
                 class_id: int = 0,
                 class_name: str = "spore"):
        """
        Initialize detection.
        
        Args:
            bbox: (x1, y1, x2, y2) in pixel coordinates
            confidence: Detection confidence [0, 1]
            class_id: Class ID
            class_name: Class name
        """
        self.bbox = bbox
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
    
    @property
    def x1(self) -> float:
        return self.bbox[0]
    
    @property
    def y1(self) -> float:
        return self.bbox[1]
    
    @property
    def x2(self) -> float:
        return self.bbox[2]
    
    @property
    def y2(self) -> float:
        return self.bbox[3]
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center point of detection."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    def __repr__(self):
        return f"Detection({self.class_name}, conf={self.confidence:.2f}, bbox={self.bbox})"


class SporeDetector:
    """YOLO-based spore detector optimized for Intel Iris Xe."""
    
    def __init__(self, config: YOLOConfig, use_openvino: bool = True):
        """
        Initialize detector.
        
        Args:
            config: YOLOConfig instance
            use_openvino: Whether to use OpenVINO backend (recommended for Iris Xe)
        """
        self.config = config
        self.use_openvino = use_openvino
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package required. Install with: pip install ultralytics")
        
        # Try OpenVINO model first if available
        openvino_path = self.config.get_exported_model_path()
        pt_path = self.config.get_trained_model_path()
        
        if self.use_openvino and openvino_path.exists():
            logger.info(f"Loading OpenVINO model from {openvino_path}")
            self.model = YOLO(str(openvino_path))
        elif pt_path.exists():
            logger.info(f"Loading PyTorch model from {pt_path}")
            self.model = YOLO(str(pt_path))
            
            # Export to OpenVINO if requested
            if self.use_openvino:
                logger.info("Exporting to OpenVINO format...")
                self._export_openvino()
        else:
            # Use pretrained model for testing
            logger.warning(f"Trained model not found, using pretrained {self.config.model_name}")
            self.model = YOLO(self.config.model_name)
    
    def _export_openvino(self):
        """Export model to OpenVINO format."""
        if self.model is None:
            return
        
        try:
            export_path = self.model.export(
                format='openvino',
                imgsz=self.config.imgsz,
                half=self.config.half_precision,
            )
            logger.info(f"OpenVINO model exported to: {export_path}")
            
            # Reload with OpenVINO
            from ultralytics import YOLO
            self.model = YOLO(export_path)
        except Exception as e:
            logger.warning(f"OpenVINO export failed: {e}. Using PyTorch backend.")
    
    def detect(self, 
               image: Union[str, Path, Image.Image, np.ndarray],
               confidence: Optional[float] = None,
               iou_threshold: Optional[float] = None) -> List[Detection]:
        """
        Detect spores in image.
        
        Args:
            image: Image path, PIL Image, or numpy array
            confidence: Confidence threshold (uses config default if None)
            iou_threshold: IoU threshold for NMS (uses config default if None)
            
        Returns:
            List of Detection objects
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        conf = confidence or self.config.confidence_threshold
        iou = iou_threshold or self.config.iou_threshold
        
        # Run inference
        results = self.model.predict(
            source=image if isinstance(image, (str, Path)) else np.array(image) if isinstance(image, Image.Image) else image,
            conf=conf,
            iou=iou,
            imgsz=self.config.imgsz,
            verbose=False,
        )
        
        # Parse results
        detections = []
        for result in results:
            if result.boxes is None:
                continue
                
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()
                conf_val = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.config.class_names[cls_id] if cls_id < len(self.config.class_names) else "unknown"
                
                detections.append(Detection(
                    bbox=tuple(bbox),
                    confidence=conf_val,
                    class_id=cls_id,
                    class_name=cls_name,
                ))
        
        return detections
    
    def detect_batch(self, 
                     images: List[Union[str, Path]],
                     confidence: Optional[float] = None) -> List[List[Detection]]:
        """
        Detect spores in batch of images.
        
        Args:
            images: List of image paths
            confidence: Confidence threshold
            
        Returns:
            List of detection lists (one per image)
        """
        conf = confidence or self.config.confidence_threshold
        
        results = self.model.predict(
            source=images,
            conf=conf,
            iou=self.config.iou_threshold,
            imgsz=self.config.imgsz,
            verbose=False,
            stream=True,  # Memory efficient for large batches
        )
        
        all_detections = []
        for result in results:
            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    bbox = box.xyxy[0].cpu().numpy()
                    conf_val = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = self.config.class_names[cls_id] if cls_id < len(self.config.class_names) else "unknown"
                    
                    detections.append(Detection(
                        bbox=tuple(bbox),
                        confidence=conf_val,
                        class_id=cls_id,
                        class_name=cls_name,
                    ))
            all_detections.append(detections)
        
        return all_detections
    
    def count_spores(self, 
                     image: Union[str, Path, Image.Image, np.ndarray],
                     confidence: Optional[float] = None) -> int:
        """
        Count spores in image.
        
        Args:
            image: Image input
            confidence: Confidence threshold
            
        Returns:
            Number of detected spores
        """
        detections = self.detect(image, confidence)
        return len(detections)


def detect_spores(image_path: str, config: YOLOConfig) -> List[Detection]:
    """
    Convenience function to detect spores in image.
    
    Args:
        image_path: Path to image
        config: YOLOConfig instance
        
    Returns:
        List of detections
    """
    detector = SporeDetector(config)
    return detector.detect(image_path)
