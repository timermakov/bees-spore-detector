#!/usr/bin/env python
"""
SAHI Installation & Integration Test

Run this script to verify SAHI is installed and working correctly.

Usage:
    python test_sahi_integration.py
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)

# Color codes for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def test_import(module_name: str, package_name: str = None) -> bool:
    """Test if a module can be imported."""
    try:
        __import__(module_name)
        logger.info(f"{GREEN}✓{RESET} {module_name} available")
        return True
    except ImportError as e:
        if package_name:
            logger.error(f"{RED}✗{RESET} {module_name} not found. Install: pip install {package_name}")
        else:
            logger.error(f"{RED}✗{RESET} {module_name} not found: {e}")
        return False


def test_sahi_features():
    """Test SAHI-specific features."""
    logger.info("\n" + BOLD + "Testing SAHI Features..." + RESET)
    
    try:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        from sahi.slicing import slice_image
        from sahi.utils.coco import Coco, CocoImage, CocoAnnotation
        from sahi.utils.file import save_json
        
        logger.info(f"{GREEN}✓{RESET} AutoDetectionModel")
        logger.info(f"{GREEN}✓{RESET} get_sliced_prediction")
        logger.info(f"{GREEN}✓{RESET} slice_image")
        logger.info(f"{GREEN}✓{RESET} COCO utilities")
        logger.info(f"{GREEN}✓{RESET} File utilities")
        
        return True
    except Exception as e:
        logger.error(f"{RED}✗{RESET} SAHI features test failed: {e}")
        return False


def test_bee_modules():
    """Test custom bee modules."""
    logger.info("\n" + BOLD + "Testing Bee Modules..." + RESET)
    
    all_ok = True
    
    try:
        from bees.yolo.converter_coco import CVATToCocoConverter
        logger.info(f"{GREEN}✓{RESET} CVATToCocoConverter")
    except Exception as e:
        logger.error(f"{RED}✗{RESET} CVATToCocoConverter: {e}")
        all_ok = False
    
    try:
        from bees.yolo.sahi_inference import SAHIDetector, run_sliced_inference_folder
        logger.info(f"{GREEN}✓{RESET} SAHIDetector")
        logger.info(f"{GREEN}✓{RESET} run_sliced_inference_folder")
    except Exception as e:
        logger.error(f"{RED}✗{RESET} SAHIDetector: {e}")
        all_ok = False
    
    try:
        from bees.yolo.dataset_slicer import DatasetSlicer, SlicedDatasetStats
        logger.info(f"{GREEN}✓{RESET} DatasetSlicer")
    except Exception as e:
        logger.error(f"{RED}✗{RESET} DatasetSlicer: {e}")
        all_ok = False
    
    try:
        from bees.yolo import (
            CVATToCocoConverter,
            SAHIDetector,
            DatasetSlicer,
            run_sliced_inference_folder,
        )
        logger.info(f"{GREEN}✓{RESET} Package __init__ exports")
    except Exception as e:
        logger.error(f"{RED}✗{RESET} Package exports: {e}")
        all_ok = False
    
    return all_ok


def test_dependencies():
    """Test all required dependencies."""
    logger.info("\n" + BOLD + "Testing Dependencies..." + RESET)
    
    dependencies = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
        ("yaml", "PyYAML"),
        ("ultralytics", "ultralytics"),
        ("sahi", "sahi"),
    ]
    
    all_ok = True
    for module, package in dependencies:
        if not test_import(module, package):
            all_ok = False
    
    return all_ok


def test_file_structure():
    """Test if all new files exist."""
    logger.info("\n" + BOLD + "Testing File Structure..." + RESET)
    
    expected_files = [
        Path("bees/yolo/converter_coco.py"),
        Path("bees/yolo/sahi_inference.py"),
        Path("bees/yolo/dataset_slicer.py"),
        Path("bees/yolo/migration_guide.py"),
        Path("SAHI_MIGRATION.md"),
        Path("SAHI_QUICKREF.md"),
        Path("SAHI_IMPLEMENTATION_SUMMARY.md"),
    ]
    
    all_ok = True
    for filepath in expected_files:
        if filepath.exists():
            logger.info(f"{GREEN}✓{RESET} {filepath}")
        else:
            logger.error(f"{RED}✗{RESET} {filepath} not found")
            all_ok = False
    
    return all_ok


def print_summary(all_tests_passed: bool):
    """Print summary of tests."""
    logger.info("\n" + "="*80)
    
    if all_tests_passed:
        logger.info(f"{GREEN}{BOLD}✓ ALL TESTS PASSED!{RESET}")
        logger.info("\nYou can now use SAHI modules:")
        logger.info("  from bees.yolo import SAHIDetector, DatasetSlicer, CVATToCocoConverter")
        logger.info("\nNext steps:")
        logger.info("  1. Read SAHI_MIGRATION.md")
        logger.info("  2. Try examples in migration_guide.py")
        logger.info("  3. Check SAHI_QUICKREF.md for copy-paste patterns")
    else:
        logger.info(f"{RED}{BOLD}✗ SOME TESTS FAILED{RESET}")
        logger.info("\nPlease install missing dependencies:")
        logger.info("  pip install -r requirements.txt")
        logger.info("  pip install sahi>=0.11.36")
    
    logger.info("="*80 + "\n")


def main():
    """Run all tests."""
    logger.info(BOLD + "SAHI Integration Test" + RESET)
    logger.info("="*80 + "\n")
    
    results = {
        "Dependencies": test_dependencies(),
        "SAHI Features": test_sahi_features(),
        "File Structure": test_file_structure(),
        "Bee Modules": test_bee_modules(),
    }
    
    all_passed = all(results.values())
    
    logger.info("\n" + BOLD + "Test Summary:" + RESET)
    for test_name, passed in results.items():
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        logger.info(f"  {test_name}: {status}")
    
    print_summary(all_passed)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
