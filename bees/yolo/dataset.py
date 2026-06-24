"""
Dataset preparation for YOLO training.

Handles train/val split, data.yaml generation, and augmentation configuration.
"""

import json
import logging
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .config import YOLOConfig
from .converter import CVATToYOLOConverter
from .data_discovery import (
    collect_image_files,
    discover_dataset_dirs,
    find_coco_annotation_candidates,
    find_xml_annotation_candidates,
    resolve_images_root,
)

logger = logging.getLogger(__name__)


@dataclass
class SourceSample:
    """Training sample with source image and precomputed YOLO label lines."""

    source_dir: Path
    image_path: Path
    annotation_name: str
    yolo_lines: List[str]


class DatasetPreparer:
    """Prepares dataset for YOLO training."""

    def __init__(self, config: YOLOConfig):
        self.config = config
        self.converter = CVATToYOLOConverter(config.class_names)
        self._image_index_cache: Dict[Path, Dict[str, Path]] = {}

    def prepare_dataset(self, val_split: float = 0.4, seed: int = 42) -> Path:
        """Prepare complete YOLO dataset from XML or COCO annotations."""
        random.seed(seed)

        if self.config.output_dir.exists():
            logger.info(f"Cleaning old dataset at {self.config.output_dir}")
            # shutil.rmtree(self.config.output_dir)  # Keep disabled by default.

        self.config.ensure_dirs()

        source_samples = self._load_source_samples()
        if not source_samples:
            raise ValueError("No training samples found for the selected annotation source.")

        random.shuffle(source_samples)
        val_count = max(1, int(len(source_samples) * val_split))
        train_count = len(source_samples) - val_count

        train_samples = source_samples[:train_count]
        val_samples = source_samples[train_count:]

        logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} val")

        train_stats = self._process_split(train_samples, self.config.output_dir / "train")
        val_stats = self._process_split(val_samples, self.config.output_dir / "val")

        data_yaml_path = self._generate_data_yaml()
        logger.info(f"Dataset prepared: train={train_stats}, val={val_stats}")
        logger.info(f"Data config: {data_yaml_path}")
        return data_yaml_path

    def _load_source_samples(self) -> List[SourceSample]:
        """Load source samples using auto annotation discovery mode."""
        dataset_format = (self.config.annotations_format or "auto").strip().lower()
        if dataset_format != "auto":
            raise ValueError(
                f"Unsupported annotation format '{dataset_format}'. Only 'auto' mode is supported."
            )
        return self._load_source_samples_auto()

    def _process_split(self, samples: List[SourceSample], output_dir: Path) -> dict:
        """Process a dataset split (train or val)."""
        images_dir = output_dir / "images"
        labels_dir = output_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        stats = {"images": 0, "annotations": 0}
        used_output_stems: Dict[str, int] = {}

        for sample in samples:
            output_stem = self._build_unique_output_stem(
                annotation_name=sample.annotation_name,
                source_dir=sample.source_dir,
                used_stems=used_output_stems,
            )

            dst_image = images_dir / f"{output_stem}{sample.image_path.suffix}"
            shutil.copy2(sample.image_path, dst_image)

            label_file = labels_dir / f"{output_stem}.txt"
            with open(label_file, "w", encoding="utf-8") as file:
                file.write("\n".join(sample.yolo_lines))

            stats["images"] += 1
            stats["annotations"] += len(sample.yolo_lines)

        return stats

    def _get_dataset_dirs_or_raise(self) -> List[Path]:
        """Return dataset directories or raise a clear configuration error."""
        dataset_dirs = discover_dataset_dirs(self.config.datasets_root, self.config.dataset_folder_pattern)
        if dataset_dirs:
            return dataset_dirs
        raise FileNotFoundError(
            f"No dataset directories found in {self.config.datasets_root}. "
            "Set yolo_datasets_root to a folder that contains dataset portions."
        )

    def _load_source_samples_auto(self) -> List[SourceSample]:
        """
        Auto-detect annotation format per dataset portion.

        Preference order per portion: COCO first, then XML.
        """
        source_samples: List[SourceSample] = []
        dataset_dirs = self._get_dataset_dirs_or_raise()
        seen_coco_paths = set()
        skipped_missing_xml_images = 0

        for dataset_dir in dataset_dirs:
            coco_candidates = find_coco_annotation_candidates(
                dataset_dir=dataset_dir,
                annotations_relpath=self.config.annotations_relpath,
            )
            if coco_candidates:
                coco_path = coco_candidates[0]
                coco_key = str(coco_path.resolve())
                if coco_key not in seen_coco_paths:
                    seen_coco_paths.add(coco_key)
                    source_samples.extend(self._parse_coco_dataset(coco_path, dataset_dir))
                continue

            xml_candidates = find_xml_annotation_candidates(
                dataset_dir=dataset_dir,
                annotations_relpath=self.config.annotations_relpath,
            )
            if not xml_candidates:
                continue
            source_batch, missing_count = self._collect_xml_samples_from_dataset(
                dataset_dir=dataset_dir,
                xml_path=xml_candidates[0],
            )
            source_samples.extend(source_batch)
            skipped_missing_xml_images += missing_count

        if not source_samples:
            raise FileNotFoundError(
                f"No XML or COCO annotations found in dataset directories under {self.config.datasets_root}."
            )

        if skipped_missing_xml_images:
            logger.warning(
                "Skipped %d XML annotations due to missing source images",
                skipped_missing_xml_images,
            )

        logger.info("Using AUTO multi-dataset mode (%d samples)", len(source_samples))
        return source_samples

    def _collect_xml_samples_from_dataset(self, dataset_dir: Path, xml_path: Path) -> Tuple[List[SourceSample], int]:
        """Collect YOLO-ready samples from one XML file."""
        xml_images_root = resolve_images_root(
            dataset_dir=dataset_dir,
            images_subdir=self.config.images_subdir,
        )
        annotations = self.converter.parse_cvat_xml(xml_path)
        logger.info("Loaded %d annotated images from %s", len(annotations), xml_path)

        source_samples: List[SourceSample] = []
        missing_counter = 0
        for annotation in annotations:
            src_image = self._find_source_image(annotation.name, xml_images_root)
            if src_image is None and xml_images_root != dataset_dir.resolve():
                src_image = self._find_source_image(annotation.name, dataset_dir)
            if src_image is None:
                missing_counter += 1
                logger.warning("File %s is in XML but not found in %s", annotation.name, dataset_dir)
                continue

            source_samples.append(
                SourceSample(
                    source_dir=dataset_dir,
                    image_path=src_image,
                    annotation_name=annotation.name,
                    yolo_lines=self.converter.convert_annotation(annotation),
                )
            )
        return source_samples, missing_counter

    def _parse_coco_dataset(self, coco_path: Path, dataset_dir: Path) -> List[SourceSample]:
        """Parse one COCO JSON into source samples."""
        try:
            with open(coco_path, "r", encoding="utf-8") as file:
                coco_data = json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid COCO JSON at {coco_path}: {exc}") from exc

        images = coco_data.get("images")
        annotations = coco_data.get("annotations")
        categories = coco_data.get("categories")
        if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
            raise ValueError(
                f"COCO JSON {coco_path} must contain list keys: images, annotations, categories"
            )

        category_id_to_name: Dict[int, str] = {}
        for category in categories:
            if not isinstance(category, dict) or "id" not in category or "name" not in category:
                raise ValueError(f"Invalid category entry in {coco_path}: {category}")
            category_id_to_name[int(category["id"])] = str(category["name"])

        category_id_to_class_idx: Dict[int, int] = {}
        for category_id, category_name in category_id_to_name.items():
            if category_name not in self.converter.class_to_id:
                raise ValueError(
                    f"Unknown COCO category '{category_name}' in {coco_path}. "
                    f"Known classes: {self.config.class_names}"
                )
            category_id_to_class_idx[category_id] = self.converter.class_to_id[category_name]

        grouped_annotations: Dict[int, List[Tuple[int, List[float]]]] = defaultdict(list)
        for annotation in annotations:
            if not isinstance(annotation, dict):
                logger.warning("Skipping malformed COCO annotation entry: %s", annotation)
                continue

            image_id = annotation.get("image_id")
            category_id = annotation.get("category_id")
            bbox = annotation.get("bbox")
            if image_id is None or category_id is None or not isinstance(bbox, list) or len(bbox) < 4:
                logger.warning("Skipping malformed COCO annotation: %s", annotation)
                continue

            category_key = int(category_id)
            if category_key not in category_id_to_class_idx:
                raise ValueError(
                    f"Annotation references unknown category_id={category_id} in {coco_path}"
                )

            grouped_annotations[int(image_id)].append(
                (category_id_to_class_idx[category_key], [float(value) for value in bbox[:4]])
            )

        images_root = resolve_images_root(
            dataset_dir=dataset_dir,
            images_subdir=self.config.images_subdir,
        )
        source_samples: List[SourceSample] = []
        missing_images: List[str] = []

        for image in images:
            if not isinstance(image, dict):
                logger.warning("Skipping malformed COCO image entry: %s", image)
                continue

            image_id = image.get("id")
            file_name = str(image.get("file_name", "")).strip()
            width = int(image.get("width", 0) or 0)
            height = int(image.get("height", 0) or 0)
            if image_id is None or not file_name:
                logger.warning("Skipping COCO image with missing id/file_name: %s", image)
                continue
            if width <= 0 or height <= 0:
                logger.warning("Skipping COCO image with invalid dimensions: %s", file_name)
                continue

            src_image = self._find_source_image(file_name, images_root)
            if src_image is None and images_root != dataset_dir.resolve():
                src_image = self._find_source_image(file_name, dataset_dir)
            if src_image is None:
                missing_images.append(file_name)
                continue

            yolo_lines: List[str] = []
            for class_id, bbox in grouped_annotations.get(int(image_id), []):
                yolo_line = self._coco_bbox_to_yolo_line(bbox, width=width, height=height, class_id=class_id)
                if yolo_line is not None:
                    yolo_lines.append(yolo_line)

            source_samples.append(
                SourceSample(
                    source_dir=dataset_dir,
                    image_path=src_image,
                    annotation_name=file_name,
                    yolo_lines=yolo_lines,
                )
            )

        if missing_images:
            preview = ", ".join(missing_images[:5])
            raise FileNotFoundError(
                f"Missing {len(missing_images)} image files referenced by {coco_path}. "
                f"First missing: {preview}. Checked under {images_root}."
            )

        logger.info(
            "Loaded %d COCO images from %s (%d annotations)",
            len(source_samples),
            coco_path,
            len(annotations),
        )
        return source_samples

    @staticmethod
    def _coco_bbox_to_yolo_line(bbox: List[float], width: int, height: int, class_id: int) -> Optional[str]:
        """Convert one COCO bbox [x, y, w, h] to YOLO normalized format."""
        x, y, bbox_w, bbox_h = bbox
        if bbox_w <= 0 or bbox_h <= 0:
            logger.warning("Skipping invalid COCO bbox with non-positive size: %s", bbox)
            return None

        x_center_norm = (x + bbox_w / 2.0) / width
        y_center_norm = (y + bbox_h / 2.0) / height
        w_norm = bbox_w / width
        h_norm = bbox_h / height

        x_center_norm = max(0.0, min(1.0, x_center_norm))
        y_center_norm = max(0.0, min(1.0, y_center_norm))
        w_norm = max(0.001, min(1.0, w_norm))
        h_norm = max(0.001, min(1.0, h_norm))
        return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

    def _build_image_index(self, source_dir: Path) -> Dict[str, Path]:
        """Build lookup index for source images."""
        source_dir = source_dir.resolve()
        if source_dir in self._image_index_cache:
            return self._image_index_cache[source_dir]

        index: Dict[str, Path] = {}
        for file_path in collect_image_files(
            source_dir=source_dir,
            image_extensions=self.config.image_extensions,
            recursive=True,
        ):
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
        for key in (normalized, annotation_name):
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
        train_path = (self.config.output_dir / "train" / "images").resolve()
        val_path = (self.config.output_dir / "val" / "images").resolve()

        data_config = {
            "path": str(self.config.output_dir.resolve()),
            "train": str(train_path),
            "val": str(val_path),
            "names": {i: name for i, name in enumerate(self.config.class_names)},
            "nc": len(self.config.class_names),
        }

        with open(data_yaml_path, "w", encoding="utf-8") as file:
            yaml.dump(data_config, file, default_flow_style=False, allow_unicode=True)
        return data_yaml_path

    def get_augmentation_config(self) -> dict:
        """Get augmentation configuration for YOLO training."""
        return {
            "mosaic": self.config.mosaic,
            "mixup": self.config.mixup,
            "hsv_h": self.config.hsv_h,
            "hsv_s": self.config.hsv_s,
            "hsv_v": self.config.hsv_v,
            "degrees": self.config.degrees,
            "scale": self.config.scale,
            "fliplr": self.config.fliplr,
            "flipud": self.config.flipud,
        }


def prepare_yolo_dataset(config: YOLOConfig, val_split: float = 0.4) -> Path:
    """Convenience function to prepare YOLO dataset."""
    preparer = DatasetPreparer(config)
    return preparer.prepare_dataset(val_split=val_split)
