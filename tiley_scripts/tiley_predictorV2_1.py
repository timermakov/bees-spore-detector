"""
Tiled YOLO inference on a folder (Pascal VOC XML + optional preview images).

Prefer the integrated CLI (from repository root):

  python -m bees.main --predict-tiled

Defaults for this script come from ``tiley.predict`` in ``config.yaml`` when flags are omitted.
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bees.config_loader import create_config_manager  # noqa: E402
from bees.yolo import YOLOConfig, SporeDetector  # noqa: E402
from bees.yolo.tiled_predict import run_tiled_prediction_folder  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiled spore detection (YOLO)")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="YAML config path")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pt (or set tiley.predict.weights)")
    parser.add_argument("--source", type=str, default=None, help="Image folder (default: tiley.predict.source)")
    parser.add_argument("--out", type=str, default=None, help="Output folder (default: tiley.predict.out)")
    parser.add_argument("--imgsz", type=int, default=None, help="YOLO size (default: tiley.predict.imgsz or yolo)")
    parser.add_argument("--tile_size", type=int, default=None, help="Tile size (default: tiley.predict.tile_size or imgsz)")
    parser.add_argument("--conf", type=float, default=None, help="Confidence (default: tiley.predict.conf or yolo)")
    parser.add_argument("--overlap", type=float, default=None, help="Default: tiley.predict.overlap")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU (default: tiley.predict.merge_iou)")
    parser.add_argument("--no-clahe", action="store_true", help="Disable CLAHE (overrides tiley.predict.use_clahe)")
    parser.add_argument("--no-preview", action="store_true", help="No pred_*.jpg (overrides write_previews)")
    args = parser.parse_args()

    cm = create_config_manager(args.config)
    p = cm.get_tiley()["predict"]
    yolo_config = YOLOConfig.from_config_manager(cm)
    detector = SporeDetector(yolo_config)

    weights = args.weights or p.get("weights")
    if not weights:
        parser.error("Provide --weights or set tiley.predict.weights in config.yaml")
    detector.load_weights(weights)

    source = args.source if args.source is not None else str(p["source"])
    out = args.out if args.out is not None else str(p["out"])
    imgsz = args.imgsz if args.imgsz is not None else (p.get("imgsz") if p.get("imgsz") is not None else yolo_config.imgsz)
    tile_size = args.tile_size if args.tile_size is not None else (p.get("tile_size") if p.get("tile_size") is not None else imgsz)
    conf = args.conf if args.conf is not None else (p.get("conf") if p.get("conf") is not None else yolo_config.confidence_threshold)
    overlap = args.overlap if args.overlap is not None else float(p["overlap"])
    merge_iou = args.iou if args.iou is not None else float(p["merge_iou"])
    use_clahe = bool(p["use_clahe"]) and not args.no_clahe
    write_previews = bool(p["write_previews"]) and not args.no_preview

    run_tiled_prediction_folder(
        detector,
        Path(source),
        Path(out),
        tile_size=tile_size,
        overlap=overlap,
        merge_iou=merge_iou,
        confidence=conf,
        imgsz=imgsz,
        use_clahe=use_clahe,
        write_previews=write_previews,
    )


if __name__ == "__main__":
    main()
