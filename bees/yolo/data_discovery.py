"""
Shared data discovery helpers for YOLO pipelines.
"""

from pathlib import Path
from typing import List, Optional

from bees.io_utils import collect_image_files


def discover_dataset_dirs(datasets_root: Path, dataset_folder_pattern: str) -> List[Path]:
    """Discover unique dataset directories under root + pattern."""
    if not datasets_root.exists():
        return []

    portions = sorted(
        path
        for path in datasets_root.glob(dataset_folder_pattern)
        if path.is_dir()
    )
    # A root with portions is a container, not another portion. This prevents
    # recursive discovery from loading each child's annotations twice.
    return portions or [datasets_root]


def find_xml_annotation_candidates(
    dataset_dir: Path,
    annotations_relpath: Optional[str] = None,
) -> List[Path]:
    """Return XML annotation candidates for one dataset portion."""
    if annotations_relpath:
        candidate = dataset_dir / annotations_relpath
        return [candidate] if candidate.is_file() and candidate.suffix.lower() == ".xml" else []

    return sorted(path for path in dataset_dir.rglob("*.xml") if path.is_file())


def find_coco_annotation_candidates(
    dataset_dir: Path,
    annotations_relpath: Optional[str] = None,
) -> List[Path]:
    """Return COCO annotation candidates for one dataset portion."""
    if annotations_relpath:
        candidate = dataset_dir / annotations_relpath
        return [candidate] if candidate.is_file() and candidate.suffix.lower() == ".json" else []

    # A conventional COCO file gets priority. Only fall back to another JSON
    # when it contains the required top-level COCO collections.
    for pattern in ("**/*coco*.json", "**/coco*.json"):
        matches = sorted(path for path in dataset_dir.glob(pattern) if path.is_file())
        if matches:
            return matches

    matches: List[Path] = []
    for path in sorted(dataset_dir.rglob("*.json")):
        try:
            import json
            with open(path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and all(isinstance(payload.get(key), list) for key in ("images", "annotations", "categories")):
            matches.append(path)
    if matches:
        return matches
    return []


def resolve_images_root(
    dataset_dir: Path,
    images_subdir: str,
) -> Path:
    """Resolve image root folder for a dataset portion."""
    return (dataset_dir / images_subdir).resolve()

