"""
Input/Output utilities for bee spore analysis.

This module provides functionality for loading images and metadata, as well as
exporting results in various formats including CVAT XML.
"""

import os
import logging
from typing import List, Tuple, Optional, Union
from pathlib import Path
from PIL import Image
import xml.etree.ElementTree as ET
import cv2
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


class ImageLoader:
    """Handles loading and validation of image files."""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    @classmethod
    def load_image(cls, image_path: Union[str, Path]) -> Image.Image:
        """
        Load an image from file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Loaded PIL Image object
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If file format is not supported
            OSError: If image cannot be loaded
            
        Example:
            >>> image = ImageLoader.load_image("path/to/image.jpg")
            >>> print(f"Image size: {image.size}")
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if image_path.suffix.lower() not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {image_path.suffix}")
        
        try:
            image = Image.open(image_path)
            logger.debug(f"Successfully loaded image: {image_path}")
            return image
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            raise OSError(f"Cannot load image {image_path}: {e}")
    
    @classmethod
    def validate_image(cls, image_path: Union[str, Path]) -> bool:
        """
        Validate if a file is a valid image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            True if file is a valid image, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception:
            return False


class MetadataLoader:
    """Handles loading and parsing of XML metadata files."""
    
    @classmethod
    def load_metadata(cls, xml_path: Union[str, Path]) -> ET.Element:
        """
        Load XML metadata from file.
        
        Args:
            xml_path: Path to the XML metadata file
            
        Returns:
            Root element of the XML tree
            
        Raises:
            FileNotFoundError: If XML file doesn't exist
            ET.ParseError: If XML is malformed
            
        Example:
            >>> root = MetadataLoader.load_metadata("path/to/metadata.xml")
            >>> print(f"Root tag: {root.tag}")
        """
        xml_path = Path(xml_path)
        
        if not xml_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {xml_path}")
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            logger.debug(f"Successfully loaded metadata: {xml_path}")
            return root
        except ET.ParseError as e:
            logger.error(f"Failed to parse XML {xml_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load metadata {xml_path}: {e}")
            raise


class FilePairFinder:
    """Finds pairs of image files and their corresponding metadata files."""
    
    @classmethod
    def list_image_pairs(cls, data_dir: Union[str, Path]) -> List[Tuple[Path, Path]]:
        """
        Find all image files with corresponding metadata files.
        
        Args:
            data_dir: Directory to search for image files
            
        Returns:
            List of (image_path, metadata_path) tuples
            
        Example:
            >>> pairs = FilePairFinder.list_image_pairs("dataset/")
            >>> for img_path, meta_path in pairs:
            >>>     print(f"Image: {img_path.name}, Metadata: {meta_path.name}")
        """
        data_dir = Path(data_dir)
        
        if not data_dir.is_dir():
            logger.warning(f"Directory does not exist: {data_dir}")
            return []
        
        pairs = []
        for img_file in data_dir.glob("*.jpg"):
            meta_file = img_file.with_suffix(img_file.suffix + "_meta.xml")
            if meta_file.exists():
                pairs.append((img_file, meta_file))
                logger.debug(f"Found pair: {img_file.name} + {meta_file.name}")
        
        logger.info(f"Found {len(pairs)} image-metadata pairs in {data_dir}")
        return pairs


class CVATExporter:
    """Handles export of detection results to CVAT format."""
    
    def __init__(self, label_name: str = 'spore'):
        """
        Initialize the CVAT exporter.
        
        Args:
            label_name: Label name for detected objects
        """
        self.label_name = label_name
    
    def export_image_elements(self, 
                            image_path: Union[str, Path], 
                            spore_objects: List[np.ndarray], 
                            image_id: int = 0) -> Tuple[ET.Element, ET.Element]:
        """
        Create CVAT XML elements for an image.
        
        Args:
            image_path: Path to the image file
            spore_objects: List of spore contours
            image_id: Unique identifier for the image
            
        Returns:
            Tuple of (meta_element, image_element) for CVAT XML
            
        Example:
            >>> exporter = CVATExporter()
            >>> meta, image_elem = exporter.export_image_elements("img.jpg", contours, 0)
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Get image dimensions
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Create image element
        image_elem = ET.Element('image', {
            'id': str(image_id),
            'name': image_path.name,
            'width': str(width),
            'height': str(height),
        })
        
        # Add spore annotations as ellipses
        for idx, contour in enumerate(spore_objects):
            if len(contour) < 5:
                logger.warning(f"Skipping contour {idx}: insufficient points ({len(contour)})")
                continue
            
            try:
                ellipse = cv2.fitEllipse(contour)
                ellipse_elem = self._create_ellipse_element(ellipse, idx)
                image_elem.append(ellipse_elem)
            except Exception as e:
                logger.warning(f"Failed to fit ellipse for contour {idx}: {e}")
                continue
        
        # Create meta element
        meta = self._create_meta_element(image_path, width, height)
        
        return meta, image_elem
    
    def _create_ellipse_element(self, ellipse: Tuple, group_id: int) -> ET.Element:
        """Create an ellipse XML element for CVAT."""
        (cx, cy), (major, minor), rotation = ellipse
        
        return ET.Element('ellipse', {
            'label': self.label_name,
            'occluded': '0',
            'source': 'auto',
            'cx': f"{float(cx):.2f}",
            'cy': f"{float(cy):.2f}",
            'rx': f"{float(major/2):.2f}",
            'ry': f"{float(minor/2):.2f}",
            'rotation': f"{float(rotation):.2f}",
            'z_order': '0',
            'group_id': str(group_id),
        })
    
    def _create_meta_element(self, image_path: Path, width: int, height: int) -> ET.Element:
        """Create a meta XML element for CVAT."""
        meta = ET.Element('meta')
        task = ET.SubElement(meta, 'task')
        
        ET.SubElement(task, 'name').text = image_path.name
        ET.SubElement(task, 'size').text = '1'
        ET.SubElement(task, 'mode').text = 'annotation'
        ET.SubElement(task, 'overlap').text = '0'
        
        labels = ET.SubElement(task, 'labels')
        label = ET.SubElement(labels, 'label')
        ET.SubElement(label, 'name').text = self.label_name
        ET.SubElement(label, 'type').text = 'ellipse'
        ET.SubElement(label, 'attributes')
        
        original_size = ET.SubElement(task, 'original_size')
        ET.SubElement(original_size, 'width').text = str(width)
        ET.SubElement(original_size, 'height').text = str(height)
        
        return meta


class XMLFormatter:
    """Handles XML formatting and pretty-printing."""
    
    @staticmethod
    def indent_xml(elem: ET.Element, level: int = 0) -> None:
        """
        Recursively pretty-print XML with indentation.
        
        Args:
            elem: XML element to format
            level: Current indentation level
        """
        indent = "\n" + level * "  "
        
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = indent + "  "
            
            for child in elem:
                XMLFormatter.indent_xml(child, level + 1)
            
            if not elem.tail or not elem.tail.strip():
                elem.tail = indent
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = indent


# Legacy functions for backward compatibility
def load_image(image_path: Union[str, Path]) -> Image.Image:
    """Legacy function for loading images."""
    return ImageLoader.load_image(image_path)


def load_metadata(xml_path: Union[str, Path]) -> ET.Element:
    """Legacy function for loading metadata."""
    return MetadataLoader.load_metadata(xml_path)


def list_image_pairs(data_dir: Union[str, Path]) -> List[Tuple[Path, Path]]:
    """Legacy function for finding image pairs."""
    return FilePairFinder.list_image_pairs(data_dir)


def export_cvat_xml_elements(image_path: Union[str, Path], 
                           spore_objects: List[np.ndarray], 
                           image_id: int = 0, 
                           label_name: str = 'spore') -> Tuple[ET.Element, ET.Element]:
    """Legacy function for CVAT export."""
    exporter = CVATExporter(label_name)
    return exporter.export_image_elements(image_path, spore_objects, image_id)


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Legacy function for XML indentation."""
    XMLFormatter.indent_xml(elem, level) 