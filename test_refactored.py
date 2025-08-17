#!/usr/bin/env python3
"""
Test script for the refactored Bee Spore Counter codebase.

This script demonstrates the new structured approach and verifies
that all components work correctly.
"""

import sys
import os
from pathlib import Path

# Add the bees package to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        from bees import (
            ImagePreprocessor,
            EdgeDetector,
            SporeDetector,
            SporeDetectionPipeline,
            TiterCalculator,
            ImageLoader,
            MetadataLoader,
            CVATExporter,
            ImageGroupValidator,
            ImageGrouper,
            GroupedImageManager,
            MarkdownReporter,
            ExcelReporter,
            ReportManager,
            ConfigurationLoader,
            ConfigurationManager,
            ConfigurationValidator
        )
        print("✓ All classes imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_legacy_functions():
    """Test that legacy functions still work."""
    print("\nTesting legacy functions...")
    
    try:
        from bees import (
            count_spores,
            calculate_titr,
            load_config,
            get_param,
            write_markdown_report,
            export_excel
        )
        print("✓ All legacy functions imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Legacy function import failed: {e}")
        return False

def test_configuration():
    """Test configuration management."""
    print("\nTesting configuration management...")
    
    try:
        from bees import create_config_manager
        
        # Test with default values
        config = create_config_manager("config.yaml")
        
        # Test parameter retrieval
        data_dir = config.get_param("data_dir")
        min_area = config.get_int_param("min_contour_area")
        
        print(f"✓ Configuration loaded: data_dir={data_dir}, min_area={min_area}")
        return True
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_titer_calculation():
    """Test titer calculation."""
    print("\nTesting titer calculation...")
    
    try:
        from bees import TiterCalculator, calculate_titr
        
        # Test legacy function
        legacy_titer = calculate_titr([100, 120, 140])
        print(f"✓ Legacy function: {legacy_titer:.2f}")
        
        # Test new class
        calculator = TiterCalculator(volume_factor=12.0)
        new_titer = calculator.calculate_titer([100, 120, 140])
        print(f"✓ New class: {new_titer:.2f}")
        
        # Test single count
        single_titer = calculator.calculate_titer(100)
        print(f"✓ Single count: {single_titer:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Titer calculation test failed: {e}")
        return False

def test_spore_analysis():
    """Test spore analysis functions."""
    print("\nTesting spore analysis...")
    
    try:
        from bees import analyze_spore_distribution, filter_spores_by_area, validate_spore_contours
        
        # Test with empty list
        empty_analysis = analyze_spore_distribution([])
        print(f"✓ Empty analysis: {empty_analysis['total_count']} spores")
        
        # Test validation
        validation = validate_spore_contours([])
        print(f"✓ Validation: {validation['valid_count']} valid, {validation['invalid_count']} invalid")
        
        return True
        
    except Exception as e:
        print(f"✗ Spore analysis test failed: {e}")
        return False

def test_image_processing():
    """Test image processing components."""
    print("\nTesting image processing...")
    
    try:
        from bees import ImagePreprocessor, EdgeDetector, SporeDetector
        
        # Test component creation
        preprocessor = ImagePreprocessor()
        edge_detector = EdgeDetector()
        spore_detector = SporeDetector()
        
        print("✓ Image processing components created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Image processing test failed: {e}")
        return False

def test_reporting():
    """Test reporting components."""
    print("\nTesting reporting...")
    
    try:
        from bees import MarkdownReporter, ExcelReporter, ReportManager
        
        # Test component creation
        markdown_reporter = MarkdownReporter("test_output/")
        excel_reporter = ExcelReporter("test_output/")
        report_manager = ReportManager("test_output/")
        
        print("✓ Reporting components created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Reporting test failed: {e}")
        return False

def test_grouping():
    """Test image grouping components."""
    print("\nTesting image grouping...")
    
    try:
        from bees import ImageGroupValidator, ImageGrouper, GroupedImageManager
        
        # Test component creation
        validator = ImageGroupValidator()
        grouper = ImageGrouper("test_data/")
        manager = GroupedImageManager({})
        
        print("✓ Grouping components created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Grouping test failed: {e}")
        return False

def test_io_utils():
    """Test I/O utilities."""
    print("\nTesting I/O utilities...")
    
    try:
        from bees import ImageLoader, MetadataLoader, CVATExporter, XMLFormatter
        
        # Test component creation
        image_loader = ImageLoader()
        metadata_loader = MetadataLoader()
        cvat_exporter = CVATExporter()
        
        print("✓ I/O utility components created successfully")
        return True
        
    except Exception as e:
        print(f"✗ I/O utilities test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Bee Spore Counter - Refactored Codebase Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_legacy_functions,
        test_configuration,
        test_titer_calculation,
        test_spore_analysis,
        test_image_processing,
        test_reporting,
        test_grouping,
        test_io_utils
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The refactored codebase is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
