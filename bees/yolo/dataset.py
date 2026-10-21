"""
Dataset preparation for YOLO training.

Handles train/val split, data.yaml generation, and augmentation configuration.
"""

import json
import logging
import random
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from .config import YOLOConfig
from .cvat import xyxy_to_yolo_line
from .cvat_to_yolo import CVATToYOLOConverter
from .data_discovery import (
    collect_image_files,
    discover_dataset_dirs,
    find_coco_annotation_candidates,
    find_xml_annotation_candidates,
    resolve_images_root,
)

logger = logging.getLogger(__name__)

_PREPARED_NAMES = ("train", "val", "data.yaml")


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

    def prepare_dataset(self, val_split: Optional[float] = None, seed: Optional[int] = None) -> Path:
        """Prepare complete YOLO dataset from XML or COCO annotations."""
        if val_split is None:
            val_split = self.config.validation_split
        if seed is None:
            seed = self.config.split_seed
        random.seed(seed)

        source_samples = self._load_source_samples()
        if not source_samples:
            raise ValueError("No training samples found for the selected annotation source.")

        random.shuffle(source_samples)
        val_count = max(1, int(len(source_samples) * val_split))
        train_count = len(source_samples) - val_count

        train_samples = source_samples[:train_count]
        val_samples = source_samples[train_count:]

        logger.info(f"Split: {len(train_samples)} train, {len(val_samples)} val")

        parent = self.config.output_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".yolo-prepare-", dir=parent))
        try:
            train_stats = self._process_split(train_samples, staging / "train")
            val_stats = self._process_split(val_samples, staging / "val")
            self._generate_data_yaml(staging)
            data_yaml_path = self._publish_prepared_dataset(staging)
        finally:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)

        self.config.models_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Dataset prepared: train={train_stats}, val={val_stats}")
        logger.info(f"Data config: {data_yaml_path}")
        return data_yaml_path

    def _load_source_samples(self) -> List[SourceSample]:
        """Load samples using the annotation format selected in YAML."""
        dataset_format = str(self.config.annotations_format or "").strip().lower()
        if dataset_format not in {"auto", "xml", "coco"}:
            raise ValueError(
                f"Unsupported annotation format '{dataset_format}'. Expected auto, xml, or coco."
            )
        return self._load_source_samples_for_format(dataset_format)

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

    def _load_source_samples_for_format(self, dataset_format: str) -> List[SourceSample]:
        """
        Load annotations from each dataset portion.

        Auto mode prefers COCO, then XML. Explicit modes use only their
        requested format.
        """
        source_samples: List[SourceSample] = []
        dataset_dirs = self._get_dataset_dirs_or_raise()
        seen_annotation_paths = set()
        skipped_missing_xml_images = 0

        for dataset_dir in dataset_dirs:
            if dataset_format in {"auto", "coco"}:
                coco_candidates = find_coco_annotation_candidates(
                    dataset_dir=dataset_dir,
                    annotations_relpath=self.config.annotations_relpath,
                )
                if len(coco_candidates) > 1:
                    raise _multiple_annotations_error(dataset_dir, "COCO", coco_candidates)
                if coco_candidates:
                    coco_path = coco_candidates[0]
                    coco_key = str(coco_path.resolve())
                    if coco_key not in seen_annotation_paths:
                        seen_annotation_paths.add(coco_key)
                        source_samples.extend(self._parse_coco_dataset(coco_path, dataset_dir))
                    continue

            if dataset_format not in {"auto", "xml"}:
                continue

            xml_candidates = find_xml_annotation_candidates(
                dataset_dir=dataset_dir,
                annotations_relpath=self.config.annotations_relpath,
            )
            if not xml_candidates:
                continue
            if len(xml_candidates) > 1:
                raise _multiple_annotations_error(dataset_dir, "XML", xml_candidates)
            xml_path = xml_candidates[0]
            xml_key = str(xml_path.resolve())
            if xml_key in seen_annotation_paths:
                continue
            seen_annotation_paths.add(xml_key)
            source_batch, missing_count = self._collect_xml_samples_from_dataset(
                dataset_dir=dataset_dir,
                xml_path=xml_path,
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

        logger.info("Using %s multi-dataset mode (%d samples)", dataset_format.upper(), len(source_samples))
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

        line = xyxy_to_yolo_line(x, y, x + bbox_w, y + bbox_h, width, height, class_id)
        if line is None:
            logger.warning("Skipping COCO bbox outside the image: %s", bbox)
        return line

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

    def _publish_prepared_dataset(self, staging: Path) -> Path:
        """Replace train, val, and data.yaml only after the new build exists."""
        output = self.config.output_dir
        output.mkdir(parents=True, exist_ok=True)
        backup = Path(tempfile.mkdtemp(prefix=".yolo-prepare-backup-", dir=output.parent))
        published: List[str] = []
        try:
            for name in _PREPARED_NAMES:
                current = output / name
                if current.exists():
                    shutil.move(str(current), str(backup / name))
                shutil.move(str(staging / name), str(output / name))
                published.append(name)
        except Exception:
            _restore_prepared_dataset(output, backup, published)
            raise
        shutil.rmtree(backup, ignore_errors=True)
        return output / "data.yaml"

    def _generate_data_yaml(self, directory: Optional[Path] = None) -> Path:
        """Generate data.yaml for YOLO training."""
        directory = self.config.output_dir if directory is None else directory
        data_yaml_path = directory / "data.yaml"
        output_dir = self.config.output_dir.resolve()
        train_path = output_dir / "train" / "images"
        val_path = output_dir / "val" / "images"

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


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _restore_prepared_dataset(output: Path, backup: Path, published: List[str]) -> None:
    """Drop a partial publish and put the previous train, val, and data.yaml back."""
    for name in reversed(published):
        _remove_path(output / name)
        saved = backup / name
        if saved.exists():
            shutil.move(str(saved), str(output / name))
    for name in _PREPARED_NAMES:
        saved = backup / name
        current = output / name
        if saved.exists() and not current.exists():
            shutil.move(str(saved), str(current))
    shutil.rmtree(backup, ignore_errors=True)


def _multiple_annotations_error(dataset_dir: Path, kind: str, candidates: List[Path]) -> ValueError:
    """Name every candidate and the config key that selects one."""
    shown = []
    for path in candidates:
        try:
            shown.append(path.relative_to(dataset_dir).as_posix())
        except ValueError:
            shown.append(path.name)
    joined = ", ".join(shown)
    return ValueError(
        f"Multiple {kind} annotations found in {dataset_dir}: {joined}. "
        "Set yolo_annotations_relpath to select one."
    )


def prepare_yolo_dataset(config: YOLOConfig, val_split: Optional[float] = None) -> Path:
    """Convenience function to prepare YOLO dataset."""
    preparer = DatasetPreparer(config)
    return preparer.prepare_dataset(val_split=val_split)
