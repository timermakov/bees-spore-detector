"""
SAHI Pipeline for Bee Spore Detection

Complete end-to-end pipeline for bee spore detection using SAHI (Sliced Aided Hyper Inference).
This pipeline handles CVAT annotation conversion, dataset preparation, and inference.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from bees.yolo.converter_coco import CVATToCocoConverter
from bees.yolo.sahi_inference import SAHIDetector, run_sliced_inference_folder
from bees.yolo.dataset_slicer import DatasetSlicer

logger = logging.getLogger(__name__)


class SAHIPipeline:
    """Complete SAHI pipeline for bee spore detection."""

    def __init__(self, model_path: str = "yolo11s.pt", device: str = "cuda:0"):
        """
        Initialize the SAHI pipeline.

        Args:
            model_path: Path to YOLO model weights
            device: Device for inference (cuda:0, cpu, etc.)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.detector = None

    def initialize_detector(self):
        """Initialize the SAHI detector."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        logger.info(f"Initializing SAHI detector with model: {self.model_path}")
        self.detector = SAHIDetector(
            model_path=str(self.model_path),
            model_type="ultralytics",
            confidence_threshold=0.25,
            device=self.device,
        )
        logger.info("✓ SAHI detector initialized")

    def convert_cvat_to_coco(
        self,
        cvat_xml: Path,
        images_dir: Path,
        output_coco: Path,
        class_names: list = None
    ) -> Path:
        """
        Convert CVAT XML annotations to COCO format.

        Args:
            cvat_xml: Path to CVAT annotations.xml
            images_dir: Directory containing images
            output_coco: Output path for COCO JSON
            class_names: List of class names (default: ["spore"])

        Returns:
            Path to created COCO JSON file
        """
        if class_names is None:
            class_names = ["spore"]

        logger.info(f"Converting CVAT XML to COCO format...")
        logger.info(f"  CVAT XML: {cvat_xml}")
        logger.info(f"  Images dir: {images_dir}")
        logger.info(f"  Output: {output_coco}")

        converter = CVATToCocoConverter(class_names=class_names)
        coco = converter.parse_cvat_to_coco(cvat_xml, images_dir=images_dir)
        converter.export_to_coco_json(output_coco)

        logger.info("✓ CVAT conversion complete")
        logger.info(f"  Images: {len(coco.image_list)}")
        logger.info(f"  Annotations: {sum(len(img.annotations) for img in coco.image_list)}")

        return output_coco

    def slice_dataset_for_training(
        self,
        coco_json: Path,
        images_dir: Path,
        output_dir: Path,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        train_split: float = 0.8
    ) -> Dict[str, Path]:
        """
        Slice COCO dataset for YOLO training with train/val split.

        Args:
            coco_json: Path to COCO JSON file
            images_dir: Directory containing source images
            output_dir: Output directory for sliced dataset
            slice_height: Height of each slice
            slice_width: Width of each slice
            overlap_height_ratio: Overlap ratio for height
            overlap_width_ratio: Overlap ratio for width
            train_split: Fraction of data for training (rest for validation)

        Returns:
            Dictionary with paths to train/val COCO JSON files
        """
        logger.info(f"Slicing dataset for training...")
        logger.info(f"  COCO JSON: {coco_json}")
        logger.info(f"  Images dir: {images_dir}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Slice size: {slice_height}x{slice_width}")
        logger.info(f"  Overlap: {overlap_height_ratio:.1%} x {overlap_width_ratio:.1%}")
        logger.info(f"  Train split: {train_split:.1%}")

        dataset_paths = DatasetSlicer.prepare_training_dataset(
            coco_json_path=coco_json,
            images_dir=images_dir,
            output_dir=output_dir,
            train_split=train_split,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )

        logger.info("✓ Dataset slicing complete")
        logger.info(f"  Train COCO: {dataset_paths['train_coco_json']}")
        logger.info(f"  Val COCO: {dataset_paths['val_coco_json']}")

        return dataset_paths

    def run_sliced_inference(
        self,
        source_dir: Path,
        output_dir: Path,
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_height_ratio: float = 0.2,
        overlap_width_ratio: float = 0.2,
        confidence: float = 0.25,
        write_previews: bool = True
    ) -> Dict[str, Any]:
        """
        Run sliced inference on a directory of images.

        Args:
            source_dir: Directory containing images to process
            output_dir: Output directory for results
            slice_height: Height of each slice
            slice_width: Width of each slice
            overlap_height_ratio: Overlap ratio for height
            overlap_width_ratio: Overlap ratio for width
            confidence: Confidence threshold for detections
            write_previews: Whether to write preview images

        Returns:
            Dictionary with inference statistics
        """
        if self.detector is None:
            self.initialize_detector()

        logger.info(f"Running sliced inference...")
        logger.info(f"  Source dir: {source_dir}")
        logger.info(f"  Output dir: {output_dir}")
        logger.info(f"  Slice size: {slice_height}x{slice_width}")
        logger.info(f"  Overlap: {overlap_height_ratio:.1%} x {overlap_width_ratio:.1%}")
        logger.info(f"  Confidence: {confidence}")
        logger.info(f"  Write previews: {write_previews}")

        results = run_sliced_inference_folder(
            detector=self.detector,
            source_dir=source_dir,
            output_dir=output_dir,
            slice_height=slice_height,
            slice_width=slice_width,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
            confidence=confidence,
            write_previews=write_previews,
        )

        logger.info("✓ Sliced inference complete")
        logger.info(f"  Images processed: {results['images']}")
        logger.info(f"  Total detections: {results['detections']}")

        return results

    def run_complete_pipeline(
        self,
        cvat_xml: Path,
        images_dir: Path,
        test_images_dir: Optional[Path] = None,
        output_base_dir: Path = Path("sahi_output"),
        slice_height: int = 512,
        slice_width: int = 512,
        overlap_ratio: float = 0.2,
        train_split: float = 0.8,
        confidence: float = 0.25
    ) -> Dict[str, Any]:
        """
        Run the complete SAHI pipeline from CVAT annotations to inference results.

        Args:
            cvat_xml: Path to CVAT annotations.xml
            images_dir: Directory with training images
            test_images_dir: Optional directory with test images for inference
            output_base_dir: Base directory for all outputs
            slice_height: Height of each slice
            slice_width: Width of each slice
            overlap_ratio: Overlap ratio for slicing
            train_split: Fraction of data for training
            confidence: Confidence threshold for inference

        Returns:
            Dictionary with all pipeline results
        """
        logger.info("🚀 Starting complete SAHI pipeline")
        logger.info("=" * 50)

        # Create output directories
        output_base_dir.mkdir(parents=True, exist_ok=True)
        coco_dir = output_base_dir / "coco"
        training_dir = output_base_dir / "training_dataset"
        inference_dir = output_base_dir / "inference_results"
        coco_dir.mkdir(exist_ok=True)
        training_dir.mkdir(exist_ok=True)
        inference_dir.mkdir(exist_ok=True)

        results = {}

        try:
            # Step 1: Convert CVAT to COCO
            logger.info("\n📝 Step 1: Converting CVAT to COCO format")
            coco_json = coco_dir / "dataset.json"
            results["coco_json"] = self.convert_cvat_to_coco(
                cvat_xml=cvat_xml,
                images_dir=images_dir,
                output_coco=coco_json
            )

            # Step 2: Slice dataset for training
            logger.info("\n✂️  Step 2: Slicing dataset for training")
            dataset_paths = self.slice_dataset_for_training(
                coco_json=coco_json,
                images_dir=images_dir,
                output_dir=training_dir,
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_height_ratio=overlap_ratio,
                overlap_width_ratio=overlap_ratio,
                train_split=train_split
            )
            results["training_dataset"] = dataset_paths

            # Step 3: Run inference on test images (if provided)
            if test_images_dir and test_images_dir.exists():
                logger.info("\n🔍 Step 3: Running sliced inference on test images")
                inference_results = self.run_sliced_inference(
                    source_dir=test_images_dir,
                    output_dir=inference_dir,
                    slice_height=slice_height,
                    slice_width=slice_width,
                    overlap_height_ratio=overlap_ratio,
                    overlap_width_ratio=overlap_ratio,
                    confidence=confidence,
                    write_previews=True
                )
                results["inference_results"] = inference_results
            else:
                logger.info("\n⚠️  Step 3: Skipping inference (no test images provided)")

            logger.info("\n✅ Pipeline completed successfully!")
            logger.info("=" * 50)
            logger.info("📁 Output locations:")
            logger.info(f"  COCO dataset: {coco_dir}")
            logger.info(f"  Training data: {training_dir}")
            if test_images_dir and test_images_dir.exists():
                logger.info(f"  Inference results: {inference_dir}")

            results["output_dirs"] = {
                "coco": coco_dir,
                "training": training_dir,
                "inference": inference_dir if test_images_dir and test_images_dir.exists() else None
            }

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

        return results


def main():
    """Command-line interface for SAHI pipeline."""
    parser = argparse.ArgumentParser(
        description="SAHI Pipeline for Bee Spore Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert CVAT to COCO only
  python -m bees.yolo.sahi_pipeline --cvat-xml dataset_test/annotations.xml --images-dir dataset_test --output-dir sahi_output --step convert

  # Slice dataset for training
  python -m bees.yolo.sahi_pipeline --coco-json sahi_output/coco/dataset.json --images-dir dataset_test --output-dir sahi_output --step slice

  # Run inference on test images
  python -m bees.yolo.sahi_pipeline --test-images dataset_test2 --output-dir sahi_output --step inference

  # Run complete pipeline
  python -m bees.yolo.sahi_pipeline --cvat-xml dataset_test/annotations.xml --images-dir dataset_test --test-images dataset_test2 --output-dir sahi_output --step complete
        """
    )

    parser.add_argument("--cvat-xml", type=Path,
                       help="Path to CVAT annotations.xml file")
    parser.add_argument("--images-dir", type=Path,
                       help="Directory containing training images")
    parser.add_argument("--coco-json", type=Path,
                       help="Path to existing COCO JSON file (for slice/inference steps)")
    parser.add_argument("--test-images", type=Path,
                       help="Directory containing test images for inference")
    parser.add_argument("--output-dir", type=Path, default=Path("sahi_output"),
                       help="Base output directory (default: sahi_output)")
    parser.add_argument("--model", type=str, default="yolo11s.pt",
                       help="Path to YOLO model weights (default: yolo11s.pt)")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for inference (default: cuda:0)")
    parser.add_argument("--step", choices=["convert", "slice", "inference", "complete"],
                       default="complete", help="Pipeline step to run (default: complete)")

    # Slicing parameters
    parser.add_argument("--slice-height", type=int, default=512,
                       help="Slice height in pixels (default: 512)")
    parser.add_argument("--slice-width", type=int, default=512,
                       help="Slice width in pixels (default: 512)")
    parser.add_argument("--overlap", type=float, default=0.2,
                       help="Overlap ratio for slicing (default: 0.2)")
    parser.add_argument("--train-split", type=float, default=0.8,
                       help="Training data fraction (default: 0.8)")

    # Inference parameters
    parser.add_argument("--confidence", type=float, default=0.25,
                       help="Confidence threshold for detections (default: 0.25)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    try:
        # Initialize pipeline
        pipeline = SAHIPipeline(model_path=args.model, device=args.device)

        if args.step == "convert":
            # CVAT to COCO conversion only
            if not args.cvat_xml or not args.images_dir:
                parser.error("--cvat-xml and --images-dir required for convert step")
            coco_dir = args.output_dir / "coco"
            coco_dir.mkdir(parents=True, exist_ok=True)
            pipeline.convert_cvat_to_coco(
                cvat_xml=args.cvat_xml,
                images_dir=args.images_dir,
                output_coco=coco_dir / "dataset.json"
            )

        elif args.step == "slice":
            # Dataset slicing only
            if not args.coco_json or not args.images_dir:
                parser.error("--coco-json and --images-dir required for slice step")
            training_dir = args.output_dir / "training_dataset"
            pipeline.slice_dataset_for_training(
                coco_json=args.coco_json,
                images_dir=args.images_dir,
                output_dir=training_dir,
                slice_height=args.slice_height,
                slice_width=args.slice_width,
                overlap_height_ratio=args.overlap,
                overlap_width_ratio=args.overlap,
                train_split=args.train_split
            )

        elif args.step == "inference":
            # Inference only
            if not args.test_images:
                parser.error("--test-images required for inference step")
            inference_dir = args.output_dir / "inference_results"
            pipeline.run_sliced_inference(
                source_dir=args.test_images,
                output_dir=inference_dir,
                slice_height=args.slice_height,
                slice_width=args.slice_width,
                overlap_height_ratio=args.overlap,
                overlap_width_ratio=args.overlap,
                confidence=args.confidence
            )

        elif args.step == "complete":
            # Complete pipeline
            if not args.cvat_xml or not args.images_dir:
                parser.error("--cvat-xml and --images-dir required for complete pipeline")
            pipeline.run_complete_pipeline(
                cvat_xml=args.cvat_xml,
                images_dir=args.images_dir,
                test_images_dir=args.test_images,
                output_base_dir=args.output_dir,
                slice_height=args.slice_height,
                slice_width=args.slice_width,
                overlap_ratio=args.overlap,
                train_split=args.train_split,
                confidence=args.confidence
            )

        logger.info("🎉 Pipeline execution completed successfully!")

    except Exception as e:
        logger.error(f"❌ Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
