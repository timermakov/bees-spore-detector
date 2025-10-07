"""
Main module for bee spore analysis.

This module provides the command-line interface and main processing pipeline
for analyzing bee spore images using computer vision techniques.
"""

import os
import argparse
import logging
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from bees import io_utils, image_proc, spores, titer
from bees.config_loader import create_config_manager, ConfigurationError
from bees.grouping import create_group_manager, GroupedImageManager
from bees.reporting import ReportManager
from bees.image_proc import SporeDetectionPipeline
from bees.titer import TiterCalculator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SporeAnalysisPipeline:
    """Main pipeline for spore analysis."""
    
    def __init__(self, config_manager, data_dir: str, results_dir: str):
        """
        Initialize the analysis pipeline.
        
        Args:
            config_manager: Configuration manager instance
            data_dir: Directory containing input images
            results_dir: Directory for output results
        """
        self.config_manager = config_manager
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.detection_pipeline = SporeDetectionPipeline()
        self.titer_calculator = TiterCalculator()
        
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_parameters(self) -> Dict[str, any]:
        """Get detection parameters from configuration."""
        return {
            'min_contour_area': self.config_manager.get_int_param('min_contour_area'),
            'max_contour_area': self.config_manager.get_int_param('max_contour_area'),
            'min_ellipse_area': self.config_manager.get_int_param('min_ellipse_area'),
            'max_ellipse_area': self.config_manager.get_int_param('max_ellipse_area'),
            'canny_threshold1': self.config_manager.get_int_param('canny_threshold1'),
            'canny_threshold2': self.config_manager.get_int_param('canny_threshold2'),
            'min_spore_contour_length': self.config_manager.get_int_param('min_spore_contour_length'),
            'intensity_threshold': self.config_manager.get_int_param('intensity_threshold'),
        }
    
    def process_image(self, 
                     image_path: str, 
                     xml_path: str, 
                     debug_prefix: Optional[str] = None) -> Dict[str, any]:
        """
        Process a single image for spore detection.
        
        Args:
            image_path: Path to the image file
            xml_path: Path to the metadata XML file
            debug_prefix: Optional prefix for debug output
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Load image and metadata
            image = io_utils.ImageLoader.load_image(image_path)
            metadata = io_utils.MetadataLoader.load_metadata(xml_path)
            
            # Process image (instantiate pipeline with debug path to emit all debug artifacts)
            params = self.get_parameters()
            debug_base = Path(self.results_dir) / Path(image_path).stem
            detection_pipeline = SporeDetectionPipeline(debug_path=str(debug_base))
            spore_objects = detection_pipeline.detect_spores(image, **params)
            
            # Count spores and calculate titer
            count = spores.count_spores(spore_objects)
            titer_value = self.titer_calculator.calculate_titer(count)
            
            # Save final overlay debug image
            if debug_prefix:
                debug_path = str(debug_prefix) + '_debug'
                image_proc.save_debug_image(image, spore_objects, debug_path)

            # Collect available debug images for convenience
            debug_candidates = [
                f"{debug_base}_blur.jpg",
                f"{debug_base}_clahe.jpg",
                f"{debug_base}_edges.jpg",
                f"{debug_base}_edges_morph.jpg",
                f"{debug_base}_edges_nolines.jpg",
                f"{debug_base}_edges_close.jpg",
                f"{debug_base}_ellipses.jpg",
                f"{debug_base}_debug.jpg",
            ]
            debug_images = [str(p) for p in map(Path, debug_candidates) if Path(p).exists()]
            
            return {
                'image': image,
                'spore_objects': spore_objects,
                'count': count,
                'titer': titer_value,
                'metadata': metadata,
                'image_path': image_path,
                'debug_images': debug_images
            }
            
        except Exception as e:
            logger.error(f"Failed to process image {image_path}: {e}")
            raise
    
    def process_group(self, 
                     group_prefix: str, 
                     image_paths: List[str]) -> Tuple[List[int], float, List[Dict]]:
        """
        Process a group of three images.
        
        Args:
            group_prefix: Group prefix name
            image_paths: List of three image paths
            
        Returns:
            Tuple of (counts, group_titer, results)
        """
        counts = []
        results = []
        
        for idx, image_path in enumerate(image_paths, 1):
            xml_path = image_path + '_meta.xml'
            
            if not os.path.exists(xml_path):
                logger.warning(f"Metadata file not found: {xml_path}")
                continue
            
            debug_prefix = Path(self.results_dir) / f"{Path(image_path).stem}"
            result = self.process_image(image_path, xml_path, debug_prefix)
            
            counts.append(result['count'])
            results.append(result)
            
            logger.info(f"Processed {group_prefix} sample {idx}: {result['count']} spores")
        
        if not counts:
            raise ValueError(f"No valid images processed for group {group_prefix}")
        
        group_titer = self.titer_calculator.calculate_titer(counts)
        logger.info(f"Group {group_prefix} titer: {group_titer:.2f} million spores/ml")
        
        return counts, group_titer, results
    
    def run_analysis(self) -> Dict[str, List[Tuple[int, float]]]:
        """
        Run the complete analysis pipeline.
        
        Returns:
            Dictionary mapping group prefixes to lists of (count, titer) tuples
        """
        # Create group manager
        group_manager = create_group_manager(str(self.data_dir))
        if not group_manager:
            raise RuntimeError("Failed to create group manager")
        
        logger.info(f"Found {group_manager.get_group_count()} image groups")
        
        # Process each group
        groups_results = {}
        self.group_results = {}  # Store detailed results for CVAT export
        
        for prefix in group_manager.list_group_prefixes():
            image_paths = group_manager.get_group(prefix)
            if image_paths:
                counts, group_titer, results = self.process_group(prefix, image_paths)
                # Store as (count, group_titer) tuples for each sample
                groups_results[prefix] = [(count, group_titer) for count in counts]
                # Store detailed results for CVAT export
                self.group_results[prefix] = results
        
        return groups_results


class CVATExporter:
    """Handles export of results to CVAT format."""
    
    def __init__(self, output_dir: str):
        """
        Initialize the CVAT exporter.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.exporter = io_utils.CVATExporter()
    
    def export_task(self, 
                   task_name: str, 
                   image_files: List[str], 
                   spore_objects_list: List[List]) -> str:
        """
        Export analysis results to CVAT format.
        
        Args:
            task_name: Name for the CVAT task
            image_files: List of image file paths
            spore_objects_list: List of spore object lists for each image
            
        Returns:
            Path to the generated ZIP file
        """
        export_dir = self.output_dir / task_name
        images_dir = export_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images
        for image_path in image_files:
            shutil.copy(image_path, images_dir)
        
        # Build XML annotations
        root = self._build_annotations_xml(image_files, spore_objects_list)
        
        # Save XML
        xml_path = export_dir / 'annotations.xml'
        
        # Pretty-print the XML
        io_utils.XMLFormatter.indent_xml(root)
        
        # Convert to string and save
        import xml.etree.ElementTree as ET
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        
        # Create ZIP file
        zip_path = self.output_dir / f'{task_name}.zip'
        self._create_zip(export_dir, zip_path)
        
        logger.info(f"CVAT export completed: {zip_path}")
        return str(zip_path)
    
    def _build_annotations_xml(self, 
                              image_files: List[str], 
                              spore_objects_list: List[List]) -> any:
        """Build the annotations XML structure."""
        import xml.etree.ElementTree as ET
        
        root = ET.Element('annotations')
        ET.SubElement(root, 'version').text = '1.1'
        
        # Add meta and image elements
        for i, (image_path, spore_objects) in enumerate(zip(image_files, spore_objects_list)):
            meta, image_elem = self.exporter.export_image_elements(
                image_path, spore_objects, i
            )
            if i == 0:  # Use first meta
                root.insert(1, meta)
            root.append(image_elem)
        
        return root
    
    def _create_zip(self, source_dir: Path, zip_path: Path) -> None:
        """Create a ZIP file from the source directory."""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in source_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(source_dir)
                    zipf.write(file_path, arcname)


class AnalysisRunner:
    """Main runner class for the analysis pipeline."""
    
    def __init__(self, config_path: str, data_dir: Optional[str] = None, 
                 results_dir: Optional[str] = None):
        """
        Initialize the analysis runner.
        
        Args:
            config_path: Path to configuration file
            data_dir: Optional override for data directory
            results_dir: Optional override for results directory
        """
        self.config_path = Path(config_path)
        self.config_manager = create_config_manager(config_path)
        
        # Get directories
        self.data_dir = data_dir or self.config_manager.get_param('data_dir')
        self.results_dir = results_dir or self.config_manager.get_param('results_dir')
        
        # Create pipeline
        self.pipeline = SporeAnalysisPipeline(
            self.config_manager, self.data_dir, self.results_dir
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
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Create and run analysis
        runner = AnalysisRunner(args.config, args.data, args.output)
        results = runner.run(export_cvat=args.export_cvat_zip)
        
        # Print summary
        print("\n===== Analysis Summary =====")
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