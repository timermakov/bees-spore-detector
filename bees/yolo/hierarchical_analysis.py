"""
Hierarchical Analysis Pipeline for Nested Folder Structures

Analyzes spore images in nested folder structures without requiring XML annotations.
Generates statistical reports (mean titer, std dev, p-value) for each sample group.

Structure:
  Вид_А/
    Проба_001/
      Сэмпл_1/
        image1.jpg, image2.jpg, ...
      Сэмпл_2/
        image1.jpg, image2.jpg, ...
    Проба_002/
      ...
  Вид_Б/
    ...
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy import stats

from bees.yolo import SporeDetector, YOLOConfig
from bees.config_loader import create_config_manager
from bees.reporting import ExcelReporter, MarkdownReporter
from bees.titer import TiterCalculator

logger = logging.getLogger(__name__)


@dataclass
class SampleStatistics:
    """Statistics for a single sample group."""
    sample_name: str
    mean_titer: float
    std_titer: float
    count_measurements: int
    all_titer_values: List[float]
    
    @property
    def p_value(self) -> Optional[float]:
        """Calculate p-value using one-sample t-test against population mean."""
        if self.count_measurements < 2:
            return None
        # One-sample t-test: H0 = mean is same as expected value
        # Using 0 as null hypothesis (typical for normalized data)
        t_stat, p = stats.ttest_1samp(self.all_titer_values, 0)
        return p


class HierarchicalAnalyzer:
    """Analyzes spore images in nested folder structures."""

    def __init__(self, config_path: str = "config.yaml", use_yolo: bool = True):
        """
        Initialize the hierarchical analyzer.

        Args:
            config_path: Path to configuration file
            use_yolo: Use YOLO detection (True) or OpenCV (False)
        """
        self.config_manager = create_config_manager(config_path)
        self.use_yolo = use_yolo
        self.detector = None
        self.titer_calculator = TiterCalculator()
        self._initialize_detector()

    def _initialize_detector(self):
        """Initialize the spore detector."""
        if self.use_yolo:
            from bees.yolo import YOLOConfig, SporeDetector
            yolo_config = YOLOConfig.from_config_manager(self.config_manager)
            self.detector = SporeDetector(yolo_config)
            logger.info("✓ YOLO detector initialized")
        else:
            # Use OpenCV-based detection from pipeline
            from bees.spore_analysis_pipeline import SporeAnalysisPipeline
            self.pipeline = SporeAnalysisPipeline(
                self.config_manager, 
                None, 
                None, 
                use_yolo=False
            )
            logger.info("✓ OpenCV detector initialized")

    def find_sample_folders(self, root_dir: Path) -> Dict[str, Path]:
        """
        Find all sample folders in nested structure.
        
        Returns dict of {sample_name: folder_path}
        """
        samples = {}
        
        # Walk through type/probe/sample structure
        for type_path in sorted(root_dir.iterdir()):
            if not type_path.is_dir():
                continue
            
            for probe_path in sorted(type_path.iterdir()):
                if not probe_path.is_dir():
                    continue
                
                for sample_path in sorted(probe_path.iterdir()):
                    if not sample_path.is_dir():
                        continue
                    
                    # Create hierarchical sample name
                    sample_name = f"{type_path.name}/{probe_path.name}/{sample_path.name}"
                    samples[sample_name] = sample_path
                    logger.debug(f"Found sample: {sample_name}")
        
        return samples

    def find_images(self, folder: Path) -> List[Path]:
        """Find all image files in a folder."""
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        images = []
        
        for ext in extensions:
            images.extend(folder.glob(f"*{ext}"))
            images.extend(folder.glob(f"*{ext.upper()}"))
        
        return sorted(images)

    def analyze_sample_folder(self, sample_path: Path) -> Tuple[List[int], List[float]]:
        """
        Analyze all images in a sample folder.
        
        Returns:
            Tuple of (spore_counts, titer_values)
        """
        images = self.find_images(sample_path)
        
        if not images:
            logger.warning(f"No images found in {sample_path}")
            return [], []
        
        spore_counts = []
        titer_values = []
        
        for image_path in images:
            try:
                import cv2
                image = cv2.imread(str(image_path))
                
                if image is None:
                    logger.warning(f"Could not read image: {image_path}")
                    continue
                
                # Detect spores
                if self.use_yolo:
                    detections = self.detector.detect(image)
                    count = len(detections)
                else:
                    # Use OpenCV pipeline
                    spores = self.pipeline.pipeline.detect_spores(
                        image, 
                        **self.config_manager.get_all_params()
                    )
                    count = len(spores)
                
                # Calculate titer
                titer = self.titer_calculator.calculate_titer(count)
                
                spore_counts.append(count)
                titer_values.append(titer)
                
                logger.debug(f"  {image_path.name}: {count} spores, titer={titer:.2f}")
                
            except Exception as e:
                logger.error(f"Failed to analyze {image_path}: {e}")
                continue
        
        return spore_counts, titer_values

    def analyze_hierarchy(self, root_dir: Path) -> Dict[str, SampleStatistics]:
        """
        Analyze complete folder hierarchy.
        
        Returns:
            Dictionary mapping sample names to statistics
        """
        logger.info(f"Starting hierarchical analysis of {root_dir}")
        
        samples = self.find_sample_folders(root_dir)
        logger.info(f"Found {len(samples)} samples to analyze")
        
        results = {}
        
        for sample_name, sample_path in samples.items():
            logger.info(f"Analyzing {sample_name}...")
            
            counts, titers = self.analyze_sample_folder(sample_path)
            
            if not titers:
                logger.warning(f"No titer values for {sample_name}")
                continue
            
            stats_obj = SampleStatistics(
                sample_name=sample_name,
                mean_titer=float(np.mean(titers)),
                std_titer=float(np.std(titers)),
                count_measurements=len(titers),
                all_titer_values=titers
            )
            
            results[sample_name] = stats_obj
            
            logger.info(
                f"  ✓ {sample_name}: mean={stats_obj.mean_titer:.2f}, "
                f"std={stats_obj.std_titer:.4f}, n={stats_obj.count_measurements}"
            )
        
        logger.info(f"✓ Analysis complete: {len(results)} samples")
        return results

    def generate_excel_report(
        self, 
        analysis_results: Dict[str, SampleStatistics],
        output_dir: Path,
        filename: str = "hierarchical_analysis.xlsx"
    ) -> Path:
        """Generate Excel report with statistics."""
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill, Border, Side
        from openpyxl.utils import get_column_letter
        
        wb = Workbook()
        ws = wb.active
        ws.title = 'Hierarchical Analysis'
        
        # Headers
        headers = [
            'Вид (Type)',
            'Проба (Probe)',
            'Сэмпл (Sample)',
            'Mean Titer',
            'Std Dev',
            'N (measurements)',
            'P-value'
        ]
        ws.append(headers)
        
        # Styling
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_font = Font(bold=True, color="FFFFFF")
        
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
        
        # Add data rows
        for sample_name, stats_obj in sorted(analysis_results.items()):
            parts = sample_name.split('/')
            type = parts[0] if len(parts) > 0 else ""
            probe = parts[1] if len(parts) > 1 else ""
            sample = parts[2] if len(parts) > 2 else ""
            
            p_value = stats_obj.p_value if stats_obj.p_value is not None else "N/A"
            
            ws.append([
                type,
                probe,
                sample,
                f"{stats_obj.mean_titer:.4f}",
                f"{stats_obj.std_titer:.6f}",
                stats_obj.count_measurements,
                f"{p_value:.6f}" if isinstance(p_value, float) else p_value
            ])
        
        # Format columns
        column_widths = [20, 20, 20, 15, 15, 15, 15]
        for col_idx, width in enumerate(column_widths, 1):
            ws.column_dimensions[get_column_letter(col_idx)].width = width
        
        # Apply borders
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row):
            for cell in row:
                cell.border = thin_border
                if cell.row > 1:  # Data rows
                    cell.alignment = Alignment(horizontal="center", vertical="center")
        
        # Save
        output_path = output_dir / filename
        wb.save(output_path)
        logger.info(f"✓ Excel report saved: {output_path}")
        
        return output_path

    def generate_markdown_summary(
        self,
        analysis_results: Dict[str, SampleStatistics],
        output_dir: Path,
        filename: str = "analysis_summary.md"
    ) -> Path:
        """Generate Markdown summary report."""
        output_path = output_dir / filename
        
        content = """# Hierarchical Spore Analysis Report

## Summary Statistics

"""
        
        for sample_name, stats_obj in sorted(analysis_results.items()):
            content += f"""
### {sample_name}
- **Mean Titer:** {stats_obj.mean_titer:.4f} million spores/ml
- **Std Dev:** {stats_obj.std_titer:.6f}
- **N (measurements):** {stats_obj.count_measurements}
- **P-value:** {stats_obj.p_value:.6f if stats_obj.p_value else 'N/A'}

"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"✓ Markdown summary saved: {output_path}")
        return output_path

    def run_complete_analysis(
        self,
        root_dir: Path,
        output_dir: Path,
        excel_filename: str = "hierarchical_analysis.xlsx",
        markdown_filename: str = "analysis_summary.md"
    ) -> Dict[str, Path]:
        """
        Run complete analysis pipeline.
        
        Returns:
            Dictionary with paths to generated reports
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze
        results = self.analyze_hierarchy(root_dir)
        
        if not results:
            logger.error("No analysis results generated")
            return {}
        
        # Generate reports
        reports = {}
        
        excel_path = self.generate_excel_report(
            results, 
            output_dir, 
            excel_filename
        )
        reports['excel'] = excel_path
        
        md_path = self.generate_markdown_summary(
            results,
            output_dir,
            markdown_filename
        )
        reports['markdown'] = md_path
        
        logger.info("✓ All reports generated successfully")
        return reports


def main():
    """Command-line interface for hierarchical analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Hierarchical Analysis Pipeline for Nested Folder Structures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze nested structure with YOLO detection
  python -m bees.yolo.hierarchical_analysis --input-dir dataset_test --output-dir analysis_output

  # Use OpenCV detection instead
  python -m bees.yolo.hierarchical_analysis --input-dir dataset_test --output-dir analysis_output --no-yolo

  # Custom config file
  python -m bees.yolo.hierarchical_analysis --input-dir dataset_test --config my_config.yaml
        """
    )
    
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Root directory with Type/Probe/Sample structure"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("hierarchical_output"),
        help="Output directory for reports (default: hierarchical_output)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--no-yolo",
        action="store_true",
        help="Use OpenCV detection instead of YOLO"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    try:
        # Validate input
        if not args.input_dir.exists():
            logger.error(f"Input directory not found: {args.input_dir}")
            return 1
        
        # Run analysis
        analyzer = HierarchicalAnalyzer(
            config_path=args.config,
            use_yolo=not args.no_yolo
        )
        
        reports = analyzer.run_complete_analysis(
            root_dir=args.input_dir,
            output_dir=args.output_dir
        )
        
        logger.info("✓ Analysis completed successfully!")
        logger.info(f"Reports:")
        for report_type, report_path in reports.items():
            logger.info(f"  - {report_type}: {report_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
