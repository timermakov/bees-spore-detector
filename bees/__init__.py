"""
Bee Spore Counter Package

A computer vision-based system for detecting and counting bee spores in microscopic images.
"""

# Version information
__version__ = "2.0.0"
__author__ = "Timofei Ermakov, ITMO University. Email: ts.ermakov@yandex.ru"

# Import main modules
from . import opencv
from . import yolo
from . import titer
from . import io_utils
from . import grouping
from . import reporting
from . import config_loader

# Import from opencv module (OpenCV-based image processing)
from .opencv import (
    ImagePreprocessor,
    EdgeDetector,
    SporeDetector,
    SporeDetectionPipeline,
    count_spores,
    analyze_spore_distribution,
    filter_spores_by_area,
    validate_spore_contours
)

# For backward compatibility, also export as submodules
image_proc = opencv
spores = opencv

from .titer import (
    TiterCalculator
)

from .io_utils import (
    ImageLoader,
    MetadataLoader,
    FilePairFinder,
    CVATExporter,
    XMLFormatter
)

from .grouping import (

    ImageInfo,  # ДОБАВЛЕНО: есть в grouping.py
    HierarchicalStructure
)

from .reporting import (
    MarkdownReporter,
    ExcelReporter,
    ReportManager
)

from .config_loader import (
    ConfigurationLoader,
    ConfigurationManager,
    ConfigurationValidator,
    create_config_manager,
    load_config_context
)

__all__ = [
    # OpenCV-based image processing classes
    'ImagePreprocessor',
    'EdgeDetector',
    'SporeDetector',
    'SporeDetectionPipeline',
    # OpenCV-based spore analysis functions
    'count_spores',
    'analyze_spore_distribution',
    'filter_spores_by_area',
    'validate_spore_contours',
    # Other modules
    'TiterCalculator',
    'ImageLoader',
    'MetadataLoader',
    'FilePairFinder',
    'ImageInfo',
    'HierarchicalStructure',
    'CVATExporter',
    'XMLFormatter',
    'MarkdownReporter',
    'ExcelReporter',
    'ReportManager',
    'ConfigurationLoader',
    'ConfigurationManager',
    'ConfigurationValidator',
    # Functions
    'create_config_manager',
    'load_config_context',
    # Submodules
    'opencv',
    'yolo',
    'titer',
    'io_utils',
    'grouping',
    'reporting',
    'config_loader',
    # Backward compatibility aliases
    'image_proc',
    'spores',
]



