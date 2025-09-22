"""
Bee Spore Counter Package

A computer vision-based system for detecting and counting bee spores in microscopic images.
"""

# Version information
__version__ = "2.0.0"
__author__ = "Bee Spore Counter Team"

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
    calculate_titer,
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
    ReportManager,
    write_markdown_report,
    export_excel
)

from .config_loader import (
    ConfigurationLoader,
    ConfigurationManager,
    ConfigurationValidator,
    load_config,
    get_param,
    create_config_manager,
    load_config_context
)

# Legacy imports for backward compatibility
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
    'calculate_titer',
    'create_standard_calculator',
    'create_custom_calculator',
    'list_grouped_images',
    'create_group_manager',
    'write_markdown_report',
    'export_excel',
    'load_config',
    'get_param',
    'create_config_manager',
    'load_config_context',
    
    # Legacy functions (maintained for compatibility)
    'image_proc',
    'spores',
    'titer',
    'io_utils',
    'grouping',
    'reporting',
    'config_loader'
] 