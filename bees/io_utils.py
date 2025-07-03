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

def list_image_pairs(data_dir):
    """
    Возвращает список пар (image_path, xml_path) для всех .jpg файлов с соответствующими .jpg_meta.xml
    """
    pairs = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith('.jpg'):
            image_path = os.path.join(data_dir, fname)
            xml_path = image_path + '_meta.xml'
            if os.path.exists(xml_path):
                pairs.append((image_path, xml_path))
    return pairs 