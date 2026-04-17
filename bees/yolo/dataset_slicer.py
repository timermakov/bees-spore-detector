"""
Dataset slicing utilities for SAHI-based tiling.

Replaces cvat_tiled_export.py with SAHI's slice_coco functionality.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from sahi.slicing import slice_coco
    from sahi.utils.coco import Coco
    from sahi.utils.file import save_json
except ImportError:
    raise ImportError("sahi package required. Install with: pip install sahi")

logger = logging.getLogger(__name__)


@dataclass
class SlicedDatasetStats:
    """Statistics after slicing a dataset."""
    total_slices: int
    images_with_objects: int
    empty_slices: int
    total_annotations: int
    slice_height: int
    slice_width: int
    overlap_height_ratio: float
    overlap_width_ratio: float


class DatasetSlicer:
    """SAHI-based dataset slicing for creating training crops from large images."""

    @staticmethod
    def slice_coco_dataset(
        coco_json_path: Path,
        images_dir: Path,
        output_dir: Path,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_height_ratio: float = 0.25,
        overlap_width_ratio: float = 0.25,
        min_area_filter: int = 0,
        max_area_filter: Optional[int] = None,
    ) -> SlicedDatasetStats:
        """
        Slice COCO dataset and save to output directory.

        Args:
            coco_json_path: Path to COCO annotations JSON
            images_dir: Directory containing original images
            output_dir: Directory to save sliced images and annotations
            slice_height: Height of each slice
            slice_width: Width of each slice
            overlap_height_ratio: Vertical overlap ratio (0-1)
            overlap_width_ratio: Horizontal overlap ratio (0-1)
            min_area_filter: Minimum annotation area to keep
            max_area_filter: Maximum annotation area to keep

        Returns:
            SlicedDatasetStats object with statistics
        """
        coco_json_path = Path(coco_json_path)
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)

        if not coco_json_path.exists():
            raise FileNotFoundError(f"COCO JSON not found: {coco_json_path}")
        if not images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        logger.info(f"Slicing COCO dataset from {coco_json_path}")
        logger.info(f"Slice size: {slice_height}x{slice_width}, "
                    f"overlap: {overlap_height_ratio}x{overlap_width_ratio}")

        # Use SAHI's slice_coco function
        sliced_coco_dict, sliced_coco_path = slice_coco(
            coco_annotation_file_path=str(coco_json_path),
            image_dir=str(images_dir),
            output_dir=str(output_dir),
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            min_area_filter=min_area_filter if min_area_filter > 0 else None,
            max_area_filter=max_area_filter,
        )

        logger.info(f"Sliced dataset saved to {output_dir}")
        logger.info(f"Sliced COCO JSON: {sliced_coco_path}")

        # Calculate statistics
        sliced_coco = Coco.from_coco_dict_or_path(sliced_coco_dict)
        total_annotations = len(sliced_coco.annotations)
        total_slices = len(sliced_coco.image_list)
        
        images_with_objects = sum(
            1 for img in sliced_coco.image_list if len(img.annotations) > 0
        )
        empty_slices = total_slices - images_with_objects

        stats = SlicedDatasetStats(
            total_slices=total_slices,
            images_with_objects=images_with_objects,
            empty_slices=empty_slices,
            total_annotations=total_annotations,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )

        logger.info(f"Slicing stats: {total_slices} slices "
                    f"({images_with_objects} with objects, {empty_slices} empty), "
                    f"{total_annotations} annotations")

        return stats

    @staticmethod
    def prepare_training_dataset(
        coco_json_path: Path,
        images_dir: Path,
        output_dir: Path,
        train_split: float = 0.85,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_height_ratio: float = 0.25,
        overlap_width_ratio: float = 0.25,
    ) -> dict:
        """
        Prepare complete training dataset: slice COCO data and split into train/val.

        Args:
            coco_json_path: Path to COCO annotations JSON
            images_dir: Directory containing original images
            output_dir: Directory to save training dataset
            train_split: Fraction of data for training (0-1)
            slice_height: Height of each slice
            slice_width: Width of each slice
            overlap_height_ratio: Vertical overlap ratio (0-1)
            overlap_width_ratio: Horizontal overlap ratio (0-1)

        Returns:
            Dict with paths to train/val COCO JSONs and image directories
        """
        coco_json_path = Path(coco_json_path)
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Slice the dataset
        sliced_dir = output_dir / "sliced"
        sliced_stats = DatasetSlicer.slice_coco_dataset(
            coco_json_path=coco_json_path,
            images_dir=images_dir,
            output_dir=sliced_dir,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )

        # Step 2: Load sliced dataset and split
        sliced_coco_path = sliced_dir / "coco_annotations.json"
        sliced_coco = Coco.from_coco_dict_or_path(sliced_coco_path, image_dir=sliced_dir / "images")

        split_result = sliced_coco.split_coco_as_train_val(train_split_rate=train_split)

        # Step 3: Save split datasets
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)

        train_coco_path = train_dir / "coco_annotations.json"
        val_coco_path = val_dir / "coco_annotations.json"

        save_json(split_result["train_coco"].json, train_coco_path)
        save_json(split_result["val_coco"].json, val_coco_path)

        logger.info(f"Training dataset prepared:")
        logger.info(f"  Train: {len(split_result['train_coco'].image_list)} images")
        logger.info(f"  Val: {len(split_result['val_coco'].image_list)} images")

        return {
            "train_coco_json": str(train_coco_path),
            "val_coco_json": str(val_coco_path),
            "train_images_dir": str(sliced_dir / "images"),
            "val_images_dir": str(sliced_dir / "images"),
            "sliced_stats": {
                "total_slices": sliced_stats.total_slices,
                "images_with_objects": sliced_stats.images_with_objects,
                "empty_slices": sliced_stats.empty_slices,
                "total_annotations": sliced_stats.total_annotations,
            }
        }
