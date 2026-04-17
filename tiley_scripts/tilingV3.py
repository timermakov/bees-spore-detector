"""
Slice a CVAT annotation file into fixed-size training tiles (Pascal VOC per tile).

Prefer the integrated CLI (from repository root):

  python -m bees.main --export-tiled-cvat

Defaults: ``tiley.export`` in ``config.yaml``. This script can read the same section when
``--xml`` / ``--img`` are omitted.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bees.config_loader import create_config_manager  # noqa: E402
from bees.yolo.cvat_tiled_export import CvatTiledPascalExporter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Export CVAT annotations to tiled Pascal VOC dataset")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="YAML config (tiley.export)")
    parser.add_argument("--xml", type=str, default=None, help="CVAT annotations.xml (default: tiley.export.cvat_xml)")
    parser.add_argument("--img", type=str, default=None, help="Source images folder (default: tiley.export.images_dir)")
    parser.add_argument("--out", type=str, default=None, help="Output folder (default: tiley.export.out)")
    parser.add_argument("--tile", type=int, default=None, help="Tile size (default: tiley.export.tile_size)")
    parser.add_argument("--overlap", type=float, default=None, help="Overlap (default: tiley.export.overlap)")
    parser.add_argument("--negative_ratio", type=float, default=None, help="Default: tiley.export.negative_ratio")
    parser.add_argument("--seed", type=int, default=None, help="Seed (default: tiley.export.seed; null = random)")
    args = parser.parse_args()

    cm = create_config_manager(args.config)
    e = cm.get_tiley()["export"]
    xml = args.xml or e.get("cvat_xml")
    img = args.img or e.get("images_dir")
    if not xml or not img:
        parser.error("Provide --xml and --img or set tiley.export.cvat_xml and images_dir in config.yaml")
    out = args.out if args.out is not None else str(e["out"])
    tile = args.tile if args.tile is not None else int(e["tile_size"])
    overlap = args.overlap if args.overlap is not None else float(e["overlap"])
    neg = args.negative_ratio if args.negative_ratio is not None else float(e["negative_ratio"])
    seed = args.seed if args.seed is not None else e.get("seed")

    exporter = CvatTiledPascalExporter()
    exporter.export(
        Path(xml),
        Path(img),
        Path(out),
        tile_size=tile,
        overlap=overlap,
        negative_ratio=neg,
        seed=seed,
    )


if __name__ == "__main__":
    main()
