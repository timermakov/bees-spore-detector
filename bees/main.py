"""
Main module for bee spore analysis.

This module provides the command-line interface and main processing pipeline
for analyzing bee spore images using computer vision techniques.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

from bees.config_loader import create_config_manager, ConfigurationError
from bees.reporting import ReportManager
from bees.cvat_exporter import CVATExporter
from bees.spore_analysis_pipeline import SporeAnalysisPipeline

# YOLO / inference stack (optional: ultralytics, torch, ...)
try:
    from bees.yolo import (
        YOLOConfig,
        SporeDetector,
        SporeCounter,
        SporeTrainer,
        DatasetPreparer,
        PseudoLabeler,
    )

    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnalysisRunner:
    """Main runner class for the analysis pipeline."""

    def __init__(self, config_path: str, data_dir: Optional[str] = None,
                 results_dir: Optional[str] = None, use_yolo: bool = False):
        """
        Initialize the analysis runner.

        Args:
            config_path: Path to configuration file
            data_dir: Optional override for data directory
            results_dir: Optional override for results directory
            use_yolo: Whether to use YOLO-based detection
        """
        self.config_path = Path(config_path)
        self.config_manager = create_config_manager(config_path)
        self.use_yolo = use_yolo

        # Get directories
        self.data_dir = data_dir or self.config_manager.get_param('data_dir')
        self.results_dir = results_dir or self.config_manager.get_param('results_dir')

        # Create pipeline
        self.pipeline = SporeAnalysisPipeline(
            self.config_manager, self.data_dir, self.results_dir, use_yolo=use_yolo
        )

        # Create exporters
        self.cvat_exporter = CVATExporter(str(self.results_dir))
        self.report_manager = ReportManager(str(self.results_dir))

    def run(self, export_cvat: bool = False) -> Dict[str, any]:
        """
        Run the complete analysis.

        Args:
            export_cvat: Whether to export CVAT format

        Returns:
            Dictionary containing analysis results and report paths
        """
        logger.info("Starting bee spore analysis")

        try:
            # Run analysis
            groups_results = self.pipeline.run_analysis()

            # Generate reports
            reports = self.report_manager.generate_all_reports(groups_results)

            # Export CVAT if requested
            if export_cvat:
                # Получаем список всех обработанных изображений из pipeline
                all_results = self.pipeline.group_results.get('all_images', [])
                if all_results:
                    image_files = [res['image_path'] for res in all_results]
                    spore_objects_list = [res['spore_objects'] for res in all_results]
                    cvat_path = self.cvat_exporter.export_task(
                        'bees_task', image_files, spore_objects_list
                    )
                    reports['cvat'] = cvat_path

            self._save_run_parameters()
            logger.info("Analysis completed successfully")
            return {'groups_results': groups_results, 'reports': reports}
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    def _save_run_parameters(self) -> None:
        """Save run parameters for reproducibility."""
        params_txt = Path(self.results_dir) / 'params_used.txt'
        log_lines = [
            "--------------------------------",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Config file: {self.config_path.absolute()}",
            f"Data dir: {Path(self.data_dir).absolute()}",
            f"Results dir: {Path(self.results_dir).absolute()}",
            "Parameters:"
        ]
        all_params = self.config_manager.get_all_params()
        for key, value in all_params.items():
            log_lines.append(f"  {key}: {value}")
        log_lines.append("--------------------------------")
        try:
            with open(params_txt, 'a', encoding='utf-8') as f:
                f.write('\n'.join(log_lines) + '\n')
        except Exception as e:
             logger.warning(f"Could not write parameters log: {e}")


def train_yolo_model(config_path: str, quick_test: bool = False) -> int:
    """
    Train YOLO model for spore detection.

    Args:
        config_path: Path to config file
        quick_test: If True, use reduced settings for fast testing (~15 min)
    """
    if not YOLO_AVAILABLE:
        logger.error("YOLO module not available. Install ultralytics: pip install ultralytics")
        return 1

    try:
        config_manager = create_config_manager(config_path)
        yolo_config = YOLOConfig.from_config_manager(config_manager)

        if quick_test:
            logger.info("Starting YOLO QUICK TEST training (30 epochs, 640px)...")
        else:
            logger.info("Starting YOLO training pipeline...")

        preparer = DatasetPreparer(yolo_config)
        data_yaml = preparer.prepare_dataset(val_split=0.4)

        trainer = SporeTrainer(yolo_config)
        metrics = trainer.train(data_yaml, quick_test=quick_test)

        logger.info(f"Training complete. Metrics: {metrics}")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


def pseudo_label(config_path: str, source_dir: str, output_dir: str = "pseudo_labels", confidence: float = 0.5, max_det: int = 1000) -> int:
    """Generate pseudo-labels for unlabeled images using trained model."""
    if not YOLO_AVAILABLE:
        logger.error("YOLO module not available")
        return 1

    try:
        config_manager = create_config_manager(config_path)
        yolo_config = YOLOConfig.from_config_manager(config_manager)

        logger.info(f"Generating pseudo-labels from {source_dir}")
        logger.info(f"Confidence threshold: {confidence}, max_det: {max_det}")

        labeler = PseudoLabeler(yolo_config)
        stats = labeler.generate_labels(
            Path(source_dir),
            Path(output_dir),
            confidence=confidence,
            max_det=max_det
        )

        logger.info(f"Pseudo-labeling complete: {stats['images']} images, {stats['detections']} detections")
        logger.info(f"Output saved to: {output_dir}")
        logger.info("Review the labels, then merge with: --merge-pseudo")
        return 0

    except Exception as e:
        logger.error(f"Pseudo-labeling failed: {e}")
        return 1


def _resolve_tiley_export_config(config_manager, input_override: Optional[str], out_override: Optional[str]) -> Dict[str, Any]:
    e = config_manager.get_tiley()["export"]
    images_dir = input_override or e.get("input") or e.get("images_dir")
    cvat_xml = e.get("cvat_xml")
    if not cvat_xml and images_dir:
        candidate = Path(images_dir) / "annotations.xml"
        if candidate.exists():
            cvat_xml = str(candidate)
    return {
        "images_dir": images_dir,
        "cvat_xml": cvat_xml,
        "out": out_override or e["out"],
        "tile_size": int(e["tile_size"]),
        "overlap": float(e["overlap"]),
    }


def _resolve_tiley_predict_config(config_manager, input_override: Optional[str], out_override: Optional[str]) -> Dict[str, Any]:
    p = config_manager.get_tiley()["predict"]
    default_model = config_manager.get_param("yolo_model", "yolo11s.pt")
    default_conf = config_manager.get_param("yolo_confidence", 0.25)
    default_tile = config_manager.get_param("yolo_imgsz", 1024)
    return {
        "source": input_override or p["input"],
        "out": out_override or p["out"],
        "tile_size": int(p.get("tile_size") or default_tile),
        "overlap": float(p["overlap"]),
        "conf": float(p.get("conf") if p.get("conf") is not None else default_conf),
        "weights": p.get("weights") or default_model,
        "device": p.get("device", "cuda:0"),
        "model_type": p.get("model_type", "ultralytics"),
        "write_previews": bool(p.get("write_previews", True)),
    }


def _import_sahi_export_modules():
    try:
        from bees.yolo.converter_coco import CVATToCocoConverter
        from bees.yolo.dataset_slicer import DatasetSlicer
        return CVATToCocoConverter, DatasetSlicer
    except ImportError as exc:
        logger.error(
            "SAHI tiled export is unavailable: %s. Install dependencies: pip install sahi ultralytics",
            exc,
        )
        return None, None


def _import_sahi_predict_modules():
    try:
        from bees.yolo.sahi_inference import SAHIDetector, run_sliced_inference_folder
        return SAHIDetector, run_sliced_inference_folder
    except ImportError as exc:
        logger.error(
            "SAHI tiled prediction is unavailable: %s. Install dependencies: pip install sahi ultralytics",
            exc,
        )
        return None, None


def run_tile_export(config_path: str, input_dir: Optional[str], out_dir: Optional[str]) -> int:
    config_manager = create_config_manager(config_path)
    resolved = _resolve_tiley_export_config(config_manager, input_dir, out_dir)

    CVATToCocoConverter, DatasetSlicer = _import_sahi_export_modules()
    if CVATToCocoConverter is None or DatasetSlicer is None:
        return 1

    if not resolved["images_dir"] or not resolved["cvat_xml"]:
        logger.error(
            "tile-export needs CVAT XML and image directory from config.tiley.export (cvat_xml/input) or annotations.xml in input"
        )
        return 1

    images_dir = Path(str(resolved["images_dir"]))
    cvat_xml = Path(str(resolved["cvat_xml"]))
    output_dir = Path(str(resolved["out"]))

    if not images_dir.is_dir():
        logger.error("Input images directory not found: %s", images_dir)
        return 1
    if not cvat_xml.is_file():
        logger.error("CVAT XML not found: %s", cvat_xml)
        return 1

    coco_dir = output_dir / "coco"
    sliced_dir = output_dir / "sliced"
    coco_dir.mkdir(parents=True, exist_ok=True)
    coco_json = coco_dir / "dataset.json"

    converter = CVATToCocoConverter(class_names=["spore"])
    converter.parse_cvat_to_coco(cvat_xml, images_dir=images_dir)
    converter.export_to_coco_json(coco_json)

    stats = DatasetSlicer.slice_coco_dataset(
        coco_json_path=coco_json,
        images_dir=images_dir,
        output_dir=sliced_dir,
        slice_height=int(resolved["tile_size"]),
        slice_width=int(resolved["tile_size"]),
        overlap_height_ratio=float(resolved["overlap"]),
        overlap_width_ratio=float(resolved["overlap"]),
    )
    stats_path = sliced_dir / "slice_stats.json"

    logger.info(
        "Tile export complete: %s slices (%s with objects, %s empty)",
        stats.total_slices,
        stats.images_with_objects,
        stats.empty_slices,
    )
    logger.info("Tile export stats saved to %s", stats_path)
    print(f"Tile export written to: {output_dir.resolve()}")
    return 0


def run_tile_predict(config_path: str, input_dir: Optional[str], out_dir: Optional[str]) -> int:
    config_manager = create_config_manager(config_path)
    resolved = _resolve_tiley_predict_config(config_manager, input_dir, out_dir)

    SAHIDetector, run_sliced_inference_folder = _import_sahi_predict_modules()
    if SAHIDetector is None or run_sliced_inference_folder is None:
        return 1

    source_dir = Path(str(resolved["source"]))
    output_dir = Path(str(resolved["out"]))

    if not source_dir.is_dir():
        logger.error("Input images directory not found: %s", source_dir)
        return 1

    try:
        detector = SAHIDetector(
            model_path=str(resolved["weights"]),
            model_type=str(resolved["model_type"]),
            confidence_threshold=float(resolved["conf"]),
            device=str(resolved["device"]),
        )
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    stats = run_sliced_inference_folder(
        detector=detector,
        source_dir=source_dir,
        output_dir=output_dir,
        slice_height=int(resolved["tile_size"]),
        slice_width=int(resolved["tile_size"]),
        overlap_height_ratio=float(resolved["overlap"]),
        overlap_width_ratio=float(resolved["overlap"]),
        confidence=float(resolved["conf"]),
        write_previews=bool(resolved["write_previews"]),
    )

    logger.info("Tile prediction complete: %s images, %s detections", stats["images"], stats["detections"])
    print(f"Tile predictions written to: {output_dir.resolve()}")
    return 0


def merge_pseudo_labels(config_path: str, pseudo_dir: str = "pseudo_labels") -> int:
    """Merge verified pseudo-labels into training set."""
    if not YOLO_AVAILABLE:
        logger.error("YOLO module not available")
        return 1

    try:
        config_manager = create_config_manager(config_path)
        yolo_config = YOLOConfig.from_config_manager(config_manager)

        logger.info(f"Merging pseudo-labels from {pseudo_dir}")

        labeler = PseudoLabeler(yolo_config)
        stats = labeler.merge_with_training(Path(pseudo_dir))

        logger.info(f"Merge complete: {stats['added']} images added to training set")
        return 0

    except Exception as e:
        logger.error(f"Merge failed: {e}")
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build command-line parser."""
    parser = argparse.ArgumentParser(description='Bees Spore Counter CLI')
    parser.add_argument('-c', '--config', required=False, default='config.yaml',
                        help='Path to YAML config')
    parser.add_argument('-d', '--data', required=False,
                        help='Override data directory (images)')
    parser.add_argument('-o', '--output', required=False,
                        help='Override results directory')
    parser.add_argument('--export-cvat-zip', action='store_true',
                        help='Export CVAT 1.1 ZIP with ellipses')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')

    # YOLO arguments
    parser.add_argument('--use-yolo', action='store_true',
                        help='Use YOLO-based detection instead of OpenCV')
    parser.add_argument('--train-yolo', action='store_true',
                        help='Train YOLO model on annotated data')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick training mode: 30 epochs, 640px images (~15 min on CPU)')

    # Pseudo-labeling arguments
    parser.add_argument('--pseudo-label', action='store_true',
                        help='Generate pseudo-labels for unlabeled images')
    parser.add_argument('--pseudo-source', type=str, default='dataset_test',
                        help='Source directory for pseudo-labeling (default: dataset_test)')
    parser.add_argument('--pseudo-output', type=str, default='pseudo_labels',
                        help='Output directory for pseudo-labels (default: pseudo_labels)')
    parser.add_argument('--pseudo-conf', type=float, default=0.5,
                        help='Confidence threshold for pseudo-labeling (default: 0.5)')
    parser.add_argument('--pseudo-max-det', type=int, default=1000,
                        help='Max detections per image for pseudo-labeling (default: 1000)')
    parser.add_argument('--merge-pseudo', action='store_true',
                        help='Merge pseudo-labels into training set')
    parser.add_argument('--export-yolo-cvat', action='store_true',
                        help='Run YOLO on a folder and export predictions as CVAT ZIP (box format)')
    parser.add_argument('--yolo-source', type=str, default='dataset_test',
                        help='Source directory for YOLO->CVAT export (default: dataset_test)')
    parser.add_argument('--yolo-cvat-output', type=str, default='results',
                        help='Output directory for YOLO->CVAT export (default: results)')
    parser.add_argument('--yolo-cvat-task', type=str, default='yolo_auto_annotations',
                        help='Task name for YOLO->CVAT export ZIP/XML (default: yolo_auto_annotations)')
    parser.add_argument('--yolo-cvat-conf', type=float, default=None,
                        help='Confidence threshold override for YOLO->CVAT export (default: from config)')
    parser.add_argument('--yolo-cvat-max-det', type=int, default=1000,
                        help='Max detections per image for YOLO->CVAT export (default: 1000)')
    parser.add_argument('--yolo-cvat-shape', choices=['box', 'ellipse'], default='box',
                        help='Shape for YOLO->CVAT export (box or ellipse, default: box)')

    subparsers = parser.add_subparsers(dest='command')

    tile_export_parser = subparsers.add_parser(
        'tile-export',
        help='Export tiled SAHI dataset from CVAT XML (config-driven)',
    )
    tile_export_parser.add_argument('--input', type=str, default=None,
                                    help='Override input images folder (otherwise config tiley.export.input)')
    tile_export_parser.add_argument('--out', type=str, default=None,
                                    help='Override output folder (otherwise config tiley.export.out)')

    tile_predict_parser = subparsers.add_parser(
        'tile-predict',
        help='Run SAHI tiled prediction for a folder (config-driven)',
    )
    tile_predict_parser.add_argument('--input', type=str, default=None,
                                     help='Override input folder (otherwise config tiley.predict.input)')
    tile_predict_parser.add_argument('--out', type=str, default=None,
                                     help='Override output folder (otherwise config tiley.predict.out)')

    # === HIERARCHICAL ANALYSIS SUBPARSER ===
    hierarchy_parser = subparsers.add_parser(
        'hierarchy-analyze',
        help='Analyze nested folder structure: Type/Probe/Sample(photos)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Structure:
      input_dir/
        Type_A/
          Control/
            Sample_1/          ← Repetition 1
              photo1.jpg
            Sample_2/          ← Repetition 2
              photo1.jpg
          Probe_1/
            Sample_1/
            Sample_2/

    Examples:
      python -m bees.main hierarchy-analyze --input-dir ./data --output-dir ./results
      python -m bees.main hierarchy-analyze --input-dir ./data --no-yolo
            """
    )
    hierarchy_parser.add_argument('--input-dir', type=str, required=True,
                                  help='Root directory with Type/Probe/Sample(photos) structure')
    hierarchy_parser.add_argument('--output-dir', type=str, default='hierarchical_output',
                                  help='Output directory (default: hierarchical_output)')
    hierarchy_parser.add_argument('--no-yolo', action='store_true',
                                  help='Use OpenCV instead of YOLO')

    return parser


def main() -> int:
    """Main entry point for the command-line interface."""
    parser = build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.command == 'tile-export':
        try:
            return run_tile_export(args.config, args.input, args.out)
        except ConfigurationError as exc:
            logger.error("%s", exc)
            return 1

    if args.command == 'tile-predict':
        try:
            return run_tile_predict(args.config, args.input, args.out)
        except ConfigurationError as exc:
            logger.error("%s", exc)
            return 1
    # === HIERARCHICAL ANALYSIS HANDLER ===
    if args.command == 'hierarchy-analyze':
        try:
            from .yolo.hierarchical_analysis import HierarchicalAnalyzer

            logging.getLogger().setLevel(logging.INFO)
            logger.info("Starting hierarchical analysis")
            logger.info(f"Input: {args.input_dir}, Output: {args.output_dir}")

            analyzer = HierarchicalAnalyzer(
                config_path=args.config,
                use_yolo=not args.no_yolo
            )

            reports = analyzer.run_complete_analysis(
                root_dir=Path(args.input_dir),
                output_dir=Path(args.output_dir)
            )

            if not reports:
                logger.error("No reports generated")
                return 1

            logger.info("✓ Hierarchical analysis completed!")
            for report_type, report_path in reports.items():
                logger.info(f"  - {report_type}: {report_path}")
            return 0

        except ConfigurationError as exc:
            logger.error(f"Configuration error: {exc}")
            return 1
        except Exception as e:
            logger.error(f"Hierarchical analysis failed: {e}")
            return 1

    if args.pseudo_label:
        return pseudo_label(args.config, args.pseudo_source, args.pseudo_output, args.pseudo_conf, args.pseudo_max_det)

    if args.merge_pseudo:
        return merge_pseudo_labels(args.config, args.pseudo_output)

    if args.export_yolo_cvat:
        if not YOLO_AVAILABLE:
            logger.error("YOLO module not available. Install ultralytics: pip install ultralytics")
            return 1
        try:
            config_manager = create_config_manager(args.config)
            yolo_config = YOLOConfig.from_config_manager(config_manager)
            detector = SporeDetector(yolo_config)
            exporter = CVATExporter(args.yolo_cvat_output)
            zip_path = exporter.export_yolo_predictions(
                detector=detector,
                source_dir=Path(args.yolo_source),
                task_name=args.yolo_cvat_task,
                confidence=args.yolo_cvat_conf,
                max_det=args.yolo_cvat_max_det,
                shape_type=args.yolo_cvat_shape
            )
            print(f"YOLO CVAT export: {zip_path}")
            return 0
        except Exception as e:
            logger.error(f"YOLO -> CVAT export failed: {e}")
            return 1

    if args.train_yolo:
        return train_yolo_model(args.config, quick_test=args.quick_test)

    if args.use_yolo and not YOLO_AVAILABLE:
        logger.warning("YOLO not available, falling back to OpenCV. Install: pip install ultralytics")
        args.use_yolo = False

    try:
        runner = AnalysisRunner(args.config, args.data, args.output, use_yolo=args.use_yolo)
        results = runner.run(export_cvat=args.export_cvat_zip)

        method = "YOLO" if args.use_yolo else "OpenCV"
        print(f"\n===== Analysis Summary ({method}) =====")
        print(f"Processed {len(results['groups_results'])} groups")
        print(f"Reports generated: {len(results['reports'])}")
        if 'cvat' in results['reports']:
            print(f"CVAT export: {results['reports']['cvat']}")
        print("============================\n")

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
