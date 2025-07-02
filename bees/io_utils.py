import os
from PIL import Image
import xml.etree.ElementTree as ET

def load_image(image_path):
    return Image.open(image_path)

def load_metadata(xml_path):
    tree = ET.parse(xml_path)
    return tree.getroot()

def save_markdown(result_path, content):
    with open(result_path, 'w', encoding='utf-8') as f:
        f.write(content) 