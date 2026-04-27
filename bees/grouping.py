"""
Image grouping module for bee spore analysis.

Provides unified file discovery for both legacy (_1/_2/_3 suffix)
and hierarchical (Type/Probe/Sample folder) structures.
"""

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from PIL import Image

logger = logging.getLogger(__name__)


# Default extensions (was hardcoded .jpg only)
DEFAULT_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


@dataclass
class ImageInfo:
    """Metadata for a single image file."""
    path: Path
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass
class HierarchicalStructure:
    """
    Nested structure: Type / Probe / Sample / [ImageInfo].

    Example:
        types = {
            "Вид_А": {
                "Control": {
                    "Сэмпл_1": [ImageInfo(path1, 1920, 1080), ...],
                    "Сэмпл_2": [...]
                },
                "Проба_1": {...}
            }
        }
    """
    types: Dict[str, Dict[str, Dict[str, List[ImageInfo]]]] = field(default_factory=dict)

    def add_sample(
        self,
        type_name: str,
        probe_name: str,
        sample_name: str,
        images: List[ImageInfo]
    ) -> None:
        """Add a sample folder's images to the structure."""
        if type_name not in self.types:
            self.types[type_name] = {}
        if probe_name not in self.types[type_name]:
            self.types[type_name][probe_name] = {}
        self.types[type_name][probe_name][sample_name] = images

    @property
    def n_types(self) -> int:
        return len(self.types)

    def iter_samples(self):
        """Yield (type_name, probe_name, sample_name, images) tuples."""
        for type_name, probes in sorted(self.types.items()):
            for probe_name, samples in sorted(probes.items()):
                for sample_name, images in sorted(samples.items()):
                    yield type_name, probe_name, sample_name, images

    def get_probe_names(self, type_name: str) -> List[str]:
        """Get all probe names for a given type."""
        return list(self.types.get(type_name, {}).keys())

    def get_sample_names(self, type_name: str, probe_name: str) -> List[str]:
        """Get all sample names for a given type+probe."""
        return list(self.types.get(type_name, {}).get(probe_name, {}).keys())


# ─────────────────────────────────────────────────────────────────────
# UNIFIED FILE DISCOVERY
# ─────────────────────────────────────────────────────────────────────

def find_images(
    folder: Union[str, Path],
    extensions: Optional[set] = None
) -> List[Path]:
    """
    Find all image files in a folder.

    Args:
        folder: Directory to scan
        extensions: Set of extensions to include (default: jpg, jpeg, png, bmp, tiff, webp)

    Returns:
        Sorted list of Path objects
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    exts = extensions or DEFAULT_EXTENSIONS
    images = []

    for ext in exts:
        images.extend(folder.glob(f"*{ext.lower()}"))
        images.extend(folder.glob(f"*{ext.upper()}"))

    return sorted(set(images))  # deduplicate


def scan_folder_with_metadata(
    folder: Union[str, Path],
    extensions: Optional[set] = None
) -> List[ImageInfo]:
    """
    Scan folder and read image dimensions for all found images.

    Args:
        folder: Directory to scan
        extensions: Image extensions to include

    Returns:
        List of ImageInfo with path and dimensions
    """
    images = find_images(folder, extensions)
    result = []

    for img_path in images:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
            result.append(ImageInfo(path=img_path, width=w, height=h))
            logger.debug(f"Scanned {img_path.name}: {w}x{h}")
        except Exception as e:
            logger.warning(f"Failed to read {img_path}: {e}")
            continue

    return result



# ─────────────────────────────────────────────────────────────────────
# HIERARCHICAL MODE: Type / Probe / Sample folders
# ─────────────────────────────────────────────────────────────────────

def scan_hierarchy(
    root_dir: Union[str, Path],
    extensions: Optional[set] = None
) -> HierarchicalStructure:
    """
    Scan nested folder structure: Type / Probe / Sample / images.

    Args:
        root_dir: Root directory containing type folders
        extensions: Image extensions to include

    Returns:
        HierarchicalStructure with all metadata pre-loaded
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    structure = HierarchicalStructure()

    for type_path in sorted(root_dir.iterdir()):
        if not type_path.is_dir():
            continue
        type_name = type_path.name

        for probe_path in sorted(type_path.iterdir()):
            if not probe_path.is_dir():
                continue
            probe_name = probe_path.name

            for sample_path in sorted(probe_path.iterdir()):
                if not sample_path.is_dir():
                    continue
                sample_name = sample_path.name

                images = scan_folder_with_metadata(sample_path, extensions)
                if images:
                    structure.add_sample(type_name, probe_name, sample_name, images)
                    logger.debug(
                        f"Added sample {type_name}/{probe_name}/{sample_name}: "
                        f"{len(images)} images"
                    )

    logger.info(
        f"Scanned hierarchy: {structure.n_types} types, "
        f"{sum(len(p) for t in structure.types.values() for p in t.values())} probes, "
        f"{sum(len(s) for t in structure.types.values() for p in t.values() for s in p.values())} samples"
    )
    return structure


