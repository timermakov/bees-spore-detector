"""
Bee Spore Counter Package

A computer vision-based system for detecting and counting bee spores in microscopic images.
"""

# Version information
__version__ = "2.0.0"
__author__ = "Timofei Ermakov, ITMO University. Email: ts.ermakov@yandex.ru"

# Import main modules
from . import image_proc
from . import spores
from . import titer
from . import io_utils
from . import grouping
from . import reporting
from . import config_loader

# Import main classes for easy access
from .image_proc import (
    ImagePreprocessor,
    EdgeDetector,
    SporeDetector,
    SporeDetectionPipeline
)

from .spores import (
    count_spores,
    analyze_spore_distribution,
    filter_spores_by_area,
    validate_spore_contours
)

from .titer import (
    TiterCalculator,
    create_standard_calculator,
    create_custom_calculator
)

from .io_utils import (
    ImageLoader,
    MetadataLoader,
    FilePairFinder,
    CVATExporter,
    XMLFormatter
)

from .grouping import (
    ImageGroupValidator,
    ImageGrouper,
    GroupedImageManager,
    list_grouped_images,
    create_group_manager
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
    # Main classes
    'ImagePreprocessor',
    'EdgeDetector', 
    'SporeDetector',
    'SporeDetectionPipeline',
    'TiterCalculator',
    'ImageLoader',
    'MetadataLoader',
    'FilePairFinder',
    'CVATExporter',
    'XMLFormatter',
    'ImageGroupValidator',
    'ImageGrouper',
    'GroupedImageManager',
    'MarkdownReporter',
    'ExcelReporter',
    'ReportManager',
    'ConfigurationLoader',
    'ConfigurationManager',
    'ConfigurationValidator',
    
    # Functions
    'count_spores',
    'analyze_spore_distribution',
    'filter_spores_by_area',
    'validate_spore_contours',
    'create_standard_calculator',
    'create_custom_calculator',
    'list_grouped_images',
    'create_group_manager',
    'create_config_manager',
    'load_config_context',
    'image_proc',
    'spores',
    'titer',
    'io_utils',
    'grouping',
    'reporting',
    'config_loader'
] 