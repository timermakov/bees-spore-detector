"""
Pseudo-labeling for expanding training dataset.

Uses trained model to generate labels for unlabeled images,
which can then be reviewed and added to training set.
"""

from pathlib import Path
from typing import List, Optional
import logging
import shutil

from .config import YOLOConfig
from .detector import SporeDetector

logger = logging.getLogger(__name__)


class PseudoLabeler:
    """Generates pseudo-labels using trained model."""
    
    def __init__(self, config: YOLOConfig):
        self.config = config
        self.detector = SporeDetector(config)
    
    def generate_labels(self,
                        source_dir: Path,
                        output_dir: Path,
                        confidence: float = 0.5,
                        max_det: int = 1000,
                        copy_images: bool = True) -> dict:
        """
        Generate YOLO format labels for images in source directory.
        
        Args:
            source_dir: Directory with unlabeled images
            output_dir: Output directory for images and labels
            confidence: Minimum confidence threshold (higher = more reliable)
            max_det: Maximum detections per image
            copy_images: Whether to copy images to output directory
            
        Returns:
            Stats dict with counts
        """
        source_dir = Path(source_dir)
        output_dir = Path(output_dir)
        
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in source_dir.iterdir() 
                       if f.suffix.lower() in image_extensions]
        
        logger.info(f"Found {len(image_files)} images in {source_dir}")
        logger.info(f"Using confidence threshold: {confidence}, max_det: {max_det}")
        
        stats = {'images': 0, 'detections': 0, 'skipped': 0}
        
        for img_path in image_files:
            try:
                # Get detections
                detections = self.detector.detect(str(img_path), confidence=confidence, max_det=max_det)
                
                if not detections:
                    logger.info(f"No detections in {img_path.name}, skipping")
                    stats['skipped'] += 1
                    continue
                
                # Get image dimensions
                from PIL import Image
                with Image.open(img_path) as img:
                    img_width, img_height = img.size
                
                # Convert to YOLO format
                lines = []
                for det in detections:
                    # YOLO format: class_id x_center y_center width height (normalized)
                    x_center = (det.x1 + det.x2) / 2 / img_width
                    y_center = (det.y1 + det.y2) / 2 / img_height
                    width = det.width / img_width
                    height = det.height / img_height
                    
                    lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                
                # Save label file
                label_path = labels_dir / (img_path.stem + ".txt")
                with open(label_path, 'w') as f:
                    f.write('\n'.join(lines))
                
                # Copy image
                if copy_images:
                    dst_image = images_dir / img_path.name
                    shutil.copy2(img_path, dst_image)
                
                stats['images'] += 1
                stats['detections'] += len(detections)
                logger.info(f"Labeled {img_path.name}: {len(detections)} detections")
                
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
                stats['skipped'] += 1
        
        logger.info(f"Pseudo-labeling complete: {stats}")
        return stats
    
    def merge_with_training(self, 
                            pseudo_dir: Path,
                            verified_images: Optional[List[str]] = None) -> dict:
        """
        Merge pseudo-labeled data with training set.
        
        Args:
            pseudo_dir: Directory with pseudo-labeled data
            verified_images: Optional list of verified image names to include
                            (if None, includes all)
        
        Returns:
            Stats dict
        """
        pseudo_dir = Path(pseudo_dir)
        train_images = self.config.output_dir / "train" / "images"
        train_labels = self.config.output_dir / "train" / "labels"
        
        pseudo_images = pseudo_dir / "images"
        pseudo_labels = pseudo_dir / "labels"
        
        stats = {'added': 0, 'skipped': 0}
        
        for label_file in pseudo_labels.glob("*.txt"):
            img_name = label_file.stem
            
            # Check if verified list provided
            if verified_images and img_name not in verified_images:
                stats['skipped'] += 1
                continue
            
            # Find corresponding image
            img_file = None
            for ext in ['.jpg', '.jpeg', '.png']:
                candidate = pseudo_images / f"{img_name}{ext}"
                if candidate.exists():
                    img_file = candidate
                    break
            
            if not img_file:
                logger.warning(f"Image not found for {label_file.name}")
                stats['skipped'] += 1
                continue
            
            # Copy to training set
            shutil.copy2(img_file, train_images / img_file.name)
            shutil.copy2(label_file, train_labels / label_file.name)
            stats['added'] += 1
            logger.info(f"Added {img_name} to training set")
        
        # Clear labels cache to force regeneration
        cache_file = train_labels.parent / "labels.cache"
        if cache_file.exists():
            cache_file.unlink()
        
        logger.info(f"Merge complete: {stats}")
        return stats


def pseudo_label_images(config: YOLOConfig,
                        source_dir: str,
                        output_dir: str = "pseudo_labels",
                        confidence: float = 0.5,
                        max_det: int = 1000) -> dict:
    """
    Convenience function for pseudo-labeling.
    
    Args:
        config: YOLOConfig
        source_dir: Directory with unlabeled images
        output_dir: Output directory
        confidence: Confidence threshold
        max_det: Maximum detections per image
        
    Returns:
        Stats dict
    """
    labeler = PseudoLabeler(config)
    return labeler.generate_labels(
        Path(source_dir),
        Path(output_dir),
        confidence=confidence,
        max_det=max_det
    )
