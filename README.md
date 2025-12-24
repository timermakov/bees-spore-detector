# Bee Spore Counter

A computer vision-based system for detecting and counting bee spores in microscopic images using the Goryaev chamber method.

## Overview

The Bee Spore Counter is a Python-based tool that automates the detection and counting of bee spores in microscopic images. It uses advanced computer vision techniques including edge detection, morphological operations, and ellipse fitting to identify and count spores accurately.

## Features

- **Automated Spore Detection**: Uses computer vision algorithms to detect spores in microscopic images
- **Batch Processing**: Processes groups of three images (triplicate analysis) according to scientific standards
- **Multiple Output Formats**: Generates reports in Markdown, Excel, and CVAT XML formats
- **Configurable Parameters**: Adjustable detection parameters for different image conditions
- **Debug Output**: Comprehensive debug images for algorithm validation
- **CVAT Export**: Export results for annotation tools like CVAT

## Architecture

The codebase has been refactored to follow modern Python best practices with a clear separation of concerns:

### Core Modules

- **`image_proc.py`**: Image preprocessing and spore detection pipeline
- **`spores.py`**: Spore counting and analysis utilities
- **`titer.py`**: Titer calculation using the Goryaev chamber method
- **`io_utils.py`**: Image loading, metadata handling, and CVAT export
- **`grouping.py`**: Image grouping and validation
- **`reporting.py`**: Report generation in various formats
- **`config_loader.py`**: Configuration management and validation
- **`main.py`**: Main CLI interface and pipeline orchestration

### Key Classes

#### Image Processing
- `ImagePreprocessor`: Handles image preprocessing operations
- `EdgeDetector`: Manages edge detection and morphological operations
- `SporeDetector`: Handles spore detection and filtering
- `SporeDetectionPipeline`: Main pipeline for spore detection

#### Analysis
- `TiterCalculator`: Calculates titer values with configurable volume factors
- `SporeAnalyzer`: Provides spore analysis and validation functions

#### I/O and Export
- `ImageLoader`: Loads and validates image files
- `MetadataLoader`: Handles XML metadata files
- `CVATExporter`: Exports results to CVAT format
- `FilePairFinder`: Finds image-metadata file pairs

#### Reporting
- `MarkdownReporter`: Generates Markdown reports
- `ExcelReporter`: Creates Excel spreadsheets
- `ReportManager`: Orchestrates multiple report types

#### Configuration
- `ConfigurationLoader`: Loads and validates YAML configuration
- `ConfigurationManager`: Manages configuration with defaults and type conversion
- `ConfigurationValidator`: Validates configuration structure and values

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV (cv2)
- PIL/Pillow
- NumPy
- PyYAML
- openpyxl

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Configuration

Create a `config.yaml` file with your analysis parameters:

```yaml
# Directory configuration
data_dir: "dataset2"
results_dir: "results2"

# Detection parameters
min_contour_area: 25
max_contour_area: 500
min_ellipse_area: 25
max_ellipse_area: 500
canny_threshold1: 40
canny_threshold2: 125
min_spore_contour_length: 5
intensity_threshold: 50

# Параметры зоны анализа (зелёный квадрат)
# Размер квадрата в пикселях
analysis_square_size: 780
# Толщина линии квадрата в пикселях
analysis_square_line_width: 2
```

## Usage

### Command Line Interface

```bash
# Basic usage with default configuration
python -m bees.main

# Custom configuration file
python -m bees.main -c my_config.yaml

# Override data and output directories
python -m bees.main -d /path/to/images -o /path/to/results

# Enable CVAT export
python -m bees.main --export-cvat-zip

# Verbose logging
python -m bees.main -v
```

### Programmatic Usage

```python
from bees import SporeDetectionPipeline, TiterCalculator, ReportManager

# Create detection pipeline
pipeline = SporeDetectionPipeline()

# Process an image
spores = pipeline.detect_spores(image, **params)

# Calculate titer
titer_calculator = TiterCalculator(volume_factor=12.0)
titer = titer_calculator.calculate_titer(spore_count)

# Generate reports
report_manager = ReportManager("results/")
reports = report_manager.generate_all_reports(groups_results)
```

### Image Requirements

Images should be named according to the pattern:
- `prefix_1.jpg`, `prefix_2.jpg`, `prefix_3.jpg`

Each image should have a corresponding metadata file:
- `prefix_1.jpg_meta.xml`, `prefix_2.jpg_meta.xml`, `prefix_3.jpg_meta.xml`

## Algorithm Details

### Preprocessing Pipeline

1. **Grayscale Conversion**: Convert to grayscale for processing
2. **Noise Reduction**: Apply Gaussian blur to reduce noise
3. **Contrast Enhancement**: Use CLAHE for adaptive histogram equalization

### Edge Detection

1. **Bilateral Filtering**: Reduce noise while preserving edges
2. **Canny Edge Detection**: Dual-threshold edge detection for better coverage
3. **Morphological Operations**: Clean edges and remove noise
4. **Line Removal**: Remove long straight lines (hairs, boundaries)

### Spore Detection

1. **Contour Finding**: Extract contours from edge images
2. **Geometric Filtering**: Filter by area, contour length, and ellipse properties
3. **Ellipse Fitting**: Fit ellipses to valid contours
4. **Validation**: Check eccentricity, axis ratio, and intensity characteristics
5. **Deduplication**: Remove overlapping detections

## Output

### Reports

- **Individual Image Reports**: Markdown files with per-image results
- **Group Reports**: Summary reports for triplicate groups
- **Excel Reports**: Comprehensive spreadsheets with merged cells

### Debug Images

- `_blur.jpg`: Blurred image
- `_clahe.jpg`: Contrast-enhanced image
- `_edges.jpg`: Edge detection result
- `_edges_morph.jpg`: Morphologically cleaned edges
- `_ellipses.jpg`: Final detection results

### CVAT Export

- **ZIP Archive**: Contains images and annotations.xml
- **Ellipse Annotations**: Spores marked as ellipse objects
- **CVAT 1.1 Format**: Compatible with CVAT annotation tool

## API Reference

### Core Functions

```python
from bees import (
    count_spores,
    SporeDetectionPipeline,
    TiterCalculator,
    ReportManager,
    ConfigurationManager,
)
```

### Configuration Management

```python
from bees import create_config_manager

# Load configuration manager
config = create_config_manager("config.yaml")

# Get typed parameters
min_area = config.get_int_param("min_contour_area")
threshold = config.get_float_param("intensity_threshold")

# Validate configuration
missing = config.validate_required_params(["data_dir", "results_dir"])
```

### Image Processing

```python
# Create detection pipeline
pipeline = SporeDetectionPipeline(debug_path="debug/")

# Process image
spores = pipeline.detect_spores(
    image,
    min_contour_area=25,
    max_contour_area=500,
    canny_threshold1=40,
    canny_threshold2=125
)
```

## Development

### Code Structure

The codebase follows these principles:
- **Single Responsibility**: Each class has a single, well-defined purpose
- **Dependency Injection**: Dependencies are injected rather than hardcoded
- **Type Hints**: Comprehensive type annotations for better IDE support
- **Error Handling**: Proper exception handling with meaningful error messages
- **Logging**: Structured logging for debugging and monitoring
- **Testing**: Unit tests for all major components

### Adding New Features

1. **Create New Class**: Follow the established naming conventions
2. **Add Type Hints**: Include comprehensive type annotations
3. **Document**: Add docstrings with examples
4. **Test**: Create unit tests for new functionality
5. **Update Exports**: Add new classes to `__init__.py`

### Testing

```bash
# Run tests (example)
python test_refactored.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the coding standards
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenCV community for computer vision algorithms
- Scientific community for the Goryaev chamber method
- Contributors and maintainers

## Changelog

### Version 2.0.0
- Complete codebase refactoring
- Object-oriented architecture
- Enhanced error handling and validation
- Improved logging and debugging
- Better type hints and documentation
- Maintained backward compatibility

### Version 1.x
- Initial implementation
- Basic spore detection
- Simple reporting


## Commands to train
### default
python -m bees.main --train-yolo
### quick test
python -m bees.main --train-yolo --quick-test

## Commands to predict
yolo predict model="models\yolo11s_spores\weights\best.pt" source="dataset_test" imgsz=1280 conf=0.25 save=True project="results" name="test_predictions"

### do not show text and confidence on image - too many overlays
yolo predict model="models\yolo11s_spores\weights\best.pt" source="dataset_test" imgsz=1280 conf=0.25 save=True project="results" name="clean_predictions" show_labels=False show_conf=False



## Commands to Run
1. Test on dataset_test/ (clean, no labels):
```yolo predict model="models\yolo11s_spores\weights\best.pt" source="dataset_test" imgsz=1280 conf=0.25 save=True project="results" name="test_v2" show_labels=False show_conf=False```

2. Test using OpenVINO (faster on Intel Iris Xe):
```yolo predict model="models\yolo11s_spores\weights\best_openvino_model" source="dataset_test" imgsz=1280 conf=0.25 save=True project="results" name="test_openvino" show_labels=False show_conf=False```

3. Run full pipeline with YOLO:
```python -m bees.main --use-yolo -d dataset_test```

4. Validate model metrics:
```yolo val model="models\yolo11s_spores\weights\best.pt" data="yolo_dataset\data.yaml" imgsz=1280```

Note: Still at 45.8% mAP (target 90%). To improve further:
Need more labeled training data
Or run full training with 1280px images (currently 640px quick test)




Good ideas! Let me implement both:

1. **60/40 split** = 6-7 train, 4-5 val images → more representative validation
2. **Pseudo-labeling** = use current model to label `dataset_test/` images → expand training data

[2 tools called]

Now let me create the pseudo-labeling script:

[10 tools called]

Done! Here's what I implemented:

---

## Changes Made

### 1. Train/Val Split Changed to 60/40
- With 11 images: **6-7 train**, **4-5 val**
- More representative validation metrics

### 2. Pseudo-Labeling System
New commands added to expand dataset:

---

## Commands

### Step 1: Retrain with 60/40 split
```powershell
# Clean old dataset first
Remove-Item -Recurse -Force yolo_dataset

# Train with new split
python -m bees.main --train-yolo --quick-test
```

### Step 2: Generate pseudo-labels from test images
```powershell
# Uses trained model to label dataset_test/ images
python -m bees.main --pseudo-label --pseudo-source dataset_test --pseudo-conf 0.5
```

This creates `pseudo_labels/` with:
- `images/` - copied test images  
- `labels/` - YOLO format labels (model predictions)

### Step 3: Review pseudo-labels
Open `pseudo_labels/labels/` and review the `.txt` files. Delete or edit incorrect ones.

### Step 4: Merge verified pseudo-labels into training
```powershell
python -m bees.main --merge-pseudo
```

### Step 5: Retrain with expanded dataset
```powershell
python -m bees.main --train-yolo --quick-test
```

---

## Workflow Summary

```
11 labeled images (60/40 split)
        ↓
   Train model
        ↓
   Pseudo-label 25 test images
        ↓
   Review & verify predictions
        ↓
   Merge verified labels
        ↓
   Retrain with 11 + N images
        ↓
   Better accuracy!
```
