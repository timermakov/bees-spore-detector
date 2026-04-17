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
- **SAHI Tiling Support**: Advanced sliced inference for large images using SAHI framework

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

## SAHI Tiling Support

The system includes SAHI (Sliced Aided Hyper Inference) for advanced tiled inference on large images. SAHI provides production-ready sliced inference with automatic detection merging.

### SAHI Installation

```bash
pip install sahi>=0.11.36
```

### SAHI Pipeline Commands

#### Complete Pipeline (CVAT → Training Dataset → Inference)

```bash
# Run the complete SAHI pipeline: CVAT conversion, dataset slicing, and inference
python -m bees.yolo.sahi_pipeline \
  --cvat-xml dataset_test/annotations.xml \
  --images-dir dataset_test \
  --test-images dataset_test2 \
  --output-dir sahi_output \
  --model yolo11s.pt \
  --device cuda:0
```

#### Individual Pipeline Steps

**Step 1: Convert CVAT to COCO**
```bash
python -m bees.yolo.sahi_pipeline \
  --step convert \
  --cvat-xml dataset_test/annotations.xml \
  --images-dir dataset_test \
  --output-dir sahi_output
```

**Step 2: Slice Dataset for Training**
```bash
python -m bees.yolo.sahi_pipeline \
  --step slice \
  --coco-json sahi_output/coco/dataset.json \
  --images-dir dataset_test \
  --output-dir sahi_output \
  --slice-height 512 \
  --slice-width 512 \
  --overlap 0.2 \
  --train-split 0.8
```

**Step 3: Run Sliced Inference**
```bash
python -m bees.yolo.sahi_pipeline \
  --step inference \
  --test-images dataset_test2 \
  --output-dir sahi_output \
  --model yolo11s.pt \
  --confidence 0.25
```

#### Custom Parameters

```bash
# Large slice size for very large images
python -m bees.yolo.sahi_pipeline \
  --cvat-xml dataset_test/annotations.xml \
  --images-dir dataset_test \
  --slice-height 1024 \
  --slice-width 1024 \
  --overlap 0.3 \
  --confidence 0.5

# CPU inference (slower but works without GPU)
python -m bees.yolo.sahi_pipeline \
  --cvat-xml dataset_test/annotations.xml \
  --images-dir dataset_test \
  --device cpu
```

### SAHI Pipeline Features

- **Automatic Detection Merging**: No manual NMS logic required
- **Framework Agnostic**: Works with YOLOv5/v8/v11, MMDet, HuggingFace, TorchVision
- **Multiple Export Formats**: COCO JSON, Pascal VOC XML, CSV, FiftyOne
- **Progress Tracking**: Real-time progress bars and statistics
- **Memory Efficient**: Processes large images without loading everything into memory
- **Production Ready**: Based on 600+ academic citations and community testing

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


## YOLO Multi-Dataset Training

### Default (auto-detects Nvidia GPU)
The training process automatically detects and uses the first available Nvidia GPU. If no Nvidia GPU is found, it falls back to CPU.

YOLO training now uses a strict multi-dataset layout. Legacy single-folder mode is removed.

### 1) Required `config.yaml` keys

```yaml
# YOLO dataset root (contains dataset portions)
yolo_datasets_root: dataset_train

# Which subfolders of yolo_datasets_root to include
yolo_dataset_folder_pattern: "*"

# Optional: set when every portion uses the same XML filename
# yolo_annotations_filename: annotations_orig_2025-08-22.xml
```

### 2) Required folder structure

```text
dataset_train/
  portion_001/
    annotations.xml
    image_1.jpg
    image_2.jpg
    ...
  portion_002/
    annotations.xml
    image_1.jpg
    subfolder/
      image_2.jpg
```

**Note**: The training system automatically detects Nvidia GPUs and skips Intel 
integrated graphics. You only need to manually specify the device if you want to 
use a specific GPU or force CPU usage.

- Every dataset portion must contain at least one XML file with CVAT annotations.
- If `yolo_annotations_filename` is set, that exact filename must exist in each portion.
- Add new portions into `dataset_train` and retrain; no code changes needed.

### 3) Train commands

```bash
# Full training
python -m bees.main --train-yolo

# Quick smoke training
python -m bees.main --train-yolo --quick-test
```

Optional device override in `config.yaml`:

```yaml
yolo_device: "cuda:0"  # or "cuda:1", or "cpu"
```

### Optional: Generate pseudo-labels from images
```powershell
# Uses trained model to label dataset_test/ images, use conf level you need
python -m bees.main --pseudo-label --pseudo-source dataset_test --pseudo-conf 0.5
```

This creates `pseudo_labels/` with:
- `images/` - copied test images  
- `labels/` - YOLO format labels (model predictions)

Review pseudo-labels: open `pseudo_labels/labels/` and review `.txt` files.

Merge verified pseudo-labels into training:
```powershell
python -m bees.main --merge-pseudo
```

Retrain with expanded dataset:
```powershell
python -m bees.main --train-yolo --quick-test
```


### 4) Inference / validation commands

Make annotations for humans/CVAT -> --export-yolo-cvat
Make labels for retraining -> --pseudo-label
Run production analysis -> --use-yolo

```bash
# Predict on test folder (clean output without labels/conf text)
yolo predict model="models\yolo11s_spores\weights\best.pt" source="dataset_test" imgsz=1280 conf=0.25 save=True project="results" name="clean_predictions" show_labels=False show_conf=False

# Validate trained model on generated YOLO dataset split
yolo val model="models\yolo11s_spores\weights\best.pt" data="yolo_dataset\data.yaml" imgsz=1280
```

Export YOLO predictions directly to CVAT 1.1 ZIP (box format):
```bash
python -m bees.main --export-yolo-cvat --yolo-source dataset_test --yolo-cvat-output results --yolo-cvat-task yolo_auto_annotations
```

Optional export tuning:
```bash
python -m bees.main --export-yolo-cvat --yolo-source dataset_test --yolo-cvat-conf 0.35 --yolo-cvat-max-det 1500
```

Export as ellipses (derived from YOLO boxes):
```bash
python -m bees.main --export-yolo-cvat --yolo-source dataset_test --yolo-cvat-shape ellipse
```

Run full triplicate analysis pipeline with YOLO:
```bash
python -m bees.main --use-yolo -d dataset_test
```

Export CVAT ZIP from pipeline detections (OpenCV or YOLO mode):
```bash
python -m bees.main --use-yolo -d dataset_test --export-cvat-zip
```

### 5) Incremental retraining workflow

1. Add a new portion folder under `dataset_train` with images + XML.
2. Run training again:

```bash
python -m bees.main --train-yolo
```

The dataset builder will automatically include all matching portions under `yolo_datasets_root`.

### 6) Annotation merge scripts (when to use)

- Use `python -m bees.main --export-yolo-cvat ...` for model-driven CVAT export.
- Use `annotation_merger.py` only when you need to merge existing XML annotations from multiple sources into one CVAT XML.
- `annotation_mergerV1.py` contains extra legacy modes (split Pascal VOC / ellipse/box switching / IoU merge). Keep it only for old workflows; for current YOLO->CVAT export it is not required.

#