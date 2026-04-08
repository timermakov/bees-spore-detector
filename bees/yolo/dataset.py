"""
Dataset preparation for YOLO training.

Handles train/val split, data.yaml generation, and augmentation configuration.
"""

import random
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Dict
import logging
import yaml

from .config import YOLOConfig
from .converter import CVATToYOLOConverter

logger = logging.getLogger(__name__)


@dataclass
class SourceAnnotation:
    """Annotation with source dataset directory."""
    annotation: object
    source_dir: Path


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
        self._image_index_cache: Dict[Path, Dict[str, Path]] = {}
    
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

        if self.config.output_dir.exists():
            logger.info(f"Cleaning old dataset at {self.config.output_dir}")
            # shutil.rmtree(self.config.output_dir) # Раскомментируй, если хочешь полную чистку

        self.config.ensure_dirs()
        
        source_annotations = self._load_source_annotations()
        valid_annotations = self._filter_valid_annotations(source_annotations)

        if not valid_annotations:
            raise ValueError("No matching image files found for the annotations in XML!")

        # Shuffle and split
        random.shuffle(valid_annotations)
        val_count = max(1, int(len(valid_annotations) * val_split))
        train_count = len(valid_annotations) - val_count
        
        train_annotations = valid_annotations[:train_count]
        val_annotations = valid_annotations[train_count:]
        
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
    
    def _process_split(self, annotations: List[SourceAnnotation], output_dir: Path) -> dict:
        """Process a dataset split (train or val)."""
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {'images': 0, 'annotations': 0}
        used_output_stems: Dict[str, int] = {}
        
        for source_annotation in annotations:
            annotation = source_annotation.annotation
            # Find source image
            src_image = self._find_source_image(annotation.name, source_annotation.source_dir)
            
            if src_image is None:
                logger.warning(f"Image not found: {annotation.name}")
                continue
            
            output_stem = self._build_unique_output_stem(
                annotation_name=annotation.name,
                source_dir=source_annotation.source_dir,
                used_stems=used_output_stems
            )

            # Copy image
            dst_image = images_dir / f"{output_stem}{src_image.suffix}"
            shutil.copy2(src_image, dst_image)
            
            # Generate and save labels
            lines = self.converter.convert_annotation(annotation)
            label_file = labels_dir / f"{output_stem}.txt"
            
            with open(label_file, 'w') as f:
                f.write('\n'.join(lines))
            
            stats['images'] += 1
            stats['annotations'] += len(lines)
        
        return stats
    
    def _discover_dataset_dirs(self) -> List[Path]:
        """Discover dataset directories inside datasets_root."""
        if not self.config.datasets_root.exists():
            return []

        candidates = [self.config.datasets_root]
        candidates.extend(
            sorted(
                p for p in self.config.datasets_root.glob(self.config.dataset_folder_pattern)
                if p.is_dir()
            )
        )

        unique_dirs = []
        seen = set()
        for directory in candidates:
            key = str(directory.resolve())
            if key not in seen:
                seen.add(key)
                unique_dirs.append(directory)

        return unique_dirs

    def _resolve_annotation_file(self, dataset_dir: Path) -> Optional[Path]:
        """Resolve XML annotation file for one dataset directory."""
        if self.config.annotations_filename:
            candidate = dataset_dir / self.config.annotations_filename
            return candidate if candidate.exists() else None

        xml_files = sorted(dataset_dir.glob("*.xml"))
        if not xml_files:
            return None

        if len(xml_files) > 1:
            logger.warning(
                "Multiple XML files found in %s, using %s",
                dataset_dir,
                xml_files[0].name
            )

        return xml_files[0]

    def _load_source_annotations(self) -> List[SourceAnnotation]:
        """
        Load annotations from multi-dataset root.
        """
        source_annotations: List[SourceAnnotation] = []
        dataset_dirs = self._discover_dataset_dirs()

        if not dataset_dirs:
            raise FileNotFoundError(
                f"No dataset directories found in {self.config.datasets_root}. "
                "Set yolo_datasets_root to a folder that contains dataset portions."
            )

        for dataset_dir in dataset_dirs:
            xml_path = self._resolve_annotation_file(dataset_dir)
            if xml_path is None:
                continue

            annotations = self.converter.parse_cvat_xml(xml_path)
            logger.info(
                "Loaded %d annotated images from %s",
                len(annotations),
                xml_path
            )
            source_annotations.extend(
                SourceAnnotation(annotation=annotation, source_dir=dataset_dir)
                for annotation in annotations
            )

        if not source_annotations:
            if self.config.annotations_filename:
                raise FileNotFoundError(
                    f"No annotation XML found in dataset directories under {self.config.datasets_root}. "
                    f"Expected file name: {self.config.annotations_filename}"
                )
            raise FileNotFoundError(
                f"No XML annotation files found in dataset directories under {self.config.datasets_root}."
            )

        logger.info(
            "Using multi-dataset mode from %s (%d samples)",
            self.config.datasets_root,
            len(source_annotations)
        )
        return source_annotations

    def _filter_valid_annotations(self, source_annotations: List[SourceAnnotation]) -> List[SourceAnnotation]:
        """Keep only annotations that have corresponding image files."""
        valid_annotations: List[SourceAnnotation] = []
        missing_counter = 0

        for source_annotation in source_annotations:
            annotation = source_annotation.annotation
            img_path = self._find_source_image(annotation.name, source_annotation.source_dir)
            if img_path is not None and img_path.is_file():
                valid_annotations.append(source_annotation)
            else:
                missing_counter += 1
                logger.warning(
                    "File %s is in XML but not found in %s",
                    annotation.name,
                    source_annotation.source_dir
                )

        if missing_counter:
            logger.warning("Skipped %d annotations due to missing source images", missing_counter)

        return valid_annotations

    def _build_image_index(self, source_dir: Path) -> Dict[str, Path]:
        """Build lookup index for source images."""
        source_dir = source_dir.resolve()
        if source_dir in self._image_index_cache:
            return self._image_index_cache[source_dir]

        valid_ext = {ext.lower() for ext in self.config.image_extensions}
        index: Dict[str, Path] = {}

        if source_dir.exists():
            for file_path in source_dir.rglob("*"):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in valid_ext:
                    continue

                rel_key = file_path.relative_to(source_dir).as_posix()
                index.setdefault(rel_key, file_path)
                index.setdefault(file_path.name, file_path)

        self._image_index_cache[source_dir] = index
        return index

    def _find_source_image(self, annotation_path: str, source_dir: Path) -> Optional[Path]:
        """Find source image file from annotation path within one dataset."""
        index = self._build_image_index(source_dir)

        normalized = str(annotation_path).replace("\\", "/").lstrip("./")
        annotation_name = Path(normalized).name
        candidates = [normalized, annotation_name]

        for key in candidates:
            if key in index:
                return index[key]

        return None

    @staticmethod
    def _build_unique_output_stem(annotation_name: str, source_dir: Path, used_stems: Dict[str, int]) -> str:
        """Build deterministic unique output stem across multiple datasets."""
        base_stem = Path(annotation_name).stem
        source_tag = source_dir.name or "dataset"
        stem = f"{source_tag}__{base_stem}"

        if stem not in used_stems:
            used_stems[stem] = 1
            return stem

        used_stems[stem] += 1
        return f"{stem}__{used_stems[stem]}"
    
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
