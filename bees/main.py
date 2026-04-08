"""
Main module for bee spore analysis.

This module provides the command-line interface and main processing pipeline
for analyzing bee spore images using computer vision techniques.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from bees.config_loader import create_config_manager, ConfigurationError
from bees.grouping import create_group_manager
from bees.reporting import ReportManager
from bees.cvat_exporter import CVATExporter
from bees.spore_analysis_pipeline import SporeAnalysisPipeline

# YOLO imports (optional)
try:
    from bees.yolo import YOLOConfig, SporeDetector, SporeCounter, SporeTrainer, DatasetPreparer, PseudoLabeler
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
                # Collect all image files and spore objects
                image_files = []
                spore_objects_list = []
                
                for prefix, rows in groups_results.items():
                    group_paths = self._get_group_image_paths(prefix)
                    group_objects = self._get_group_spore_objects(prefix, rows)
                    
                    image_files.extend(group_paths)
                    spore_objects_list.extend(group_objects)
                
                if image_files:
                    cvat_path = self.cvat_exporter.export_task(
                        'bees_task', image_files, spore_objects_list
                    )
                    reports['cvat'] = cvat_path
            
            # Save run parameters
            self._save_run_parameters()
            
            logger.info("Analysis completed successfully")
            return {
                'groups_results': groups_results,
                'reports': reports
            }
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def _get_group_image_paths(self, prefix: str) -> List[str]:
        """Get image paths for a group."""
        group_manager = create_group_manager(str(self.data_dir))
        if group_manager and group_manager.has_group(prefix):
            return group_manager.get_group(prefix)
        return []
    
    def _get_group_spore_objects(self, prefix: str, rows: List[Tuple[int, float]]) -> List[List]:
        """Get spore objects for a group from stored results."""
        if hasattr(self.pipeline, 'group_results') and prefix in self.pipeline.group_results:
            return [result['spore_objects'] for result in self.pipeline.group_results[prefix]]
        # Fallback to empty lists if no stored results
        return [[] for _ in rows]
    
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
        
        # Add all parameters
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
        
        # Prepare dataset with 75/25 split
        preparer = DatasetPreparer(yolo_config)
        data_yaml = preparer.prepare_dataset(val_split=0.4)  # 60/40 split for better validation
        
        # Train model
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


def main():
    """Main entry point for the command-line interface."""
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
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle pseudo-labeling
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
    
    # Handle YOLO training mode
    if args.train_yolo:
        return train_yolo_model(args.config, quick_test=args.quick_test)
    
    # Check YOLO availability
    if args.use_yolo and not YOLO_AVAILABLE:
        logger.warning("YOLO not available, falling back to OpenCV. Install: pip install ultralytics")
        args.use_yolo = False
    
    try:
        # Create and run analysis
        runner = AnalysisRunner(args.config, args.data, args.output, use_yolo=args.use_yolo)
        results = runner.run(export_cvat=args.export_cvat_zip)
        
        # Print summary
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