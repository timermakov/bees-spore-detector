import os
from PIL import Image
import xml.etree.ElementTree as ET
import datetime

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

def export_cvat_xml_elements(image_path, spore_objs, image_id=0, label_name='spore'):
    """
    Return (meta, image_element) for CVAT for images 1.1 XML export.
    image_path: path to the image file
    spore_objs: list of contours (each contour is a numpy array of shape (N, 1, 2))
    image_id: integer id for the image
    label_name: label for all spores
    """
    import xml.etree.ElementTree as ET
    from PIL import Image
    import os
    if not os.path.exists(image_path):
        raise FileNotFoundError(image_path)
    with Image.open(image_path) as img:
        width, height = img.size
    # Create <image> element
    image_elem = ET.Element('image', {
        'id': str(image_id),
        'name': os.path.basename(image_path),
        'width': str(width),
        'height': str(height),
    })
    for idx, cnt in enumerate(spore_objs):
        points = cnt.reshape(-1, 2)
        points_str = ';'.join([f"{float(x):.2f},{float(y):.2f}" for x, y in points])
        polygon = ET.SubElement(image_elem, 'polygon', {
            'label': label_name,
            'occluded': '0',
            'source': 'auto',
            'points': points_str,
            'z_order': '0',
            'group_id': str(idx),
            'outside': '0',
        })
    # Create <meta> (minimal, for merging)
    meta = ET.Element('meta')
    task = ET.SubElement(meta, 'task')
    ET.SubElement(task, 'name').text = os.path.basename(image_path)
    ET.SubElement(task, 'size').text = '1'
    ET.SubElement(task, 'mode').text = 'annotation'
    ET.SubElement(task, 'overlap').text = '0'
    labels = ET.SubElement(task, 'labels')
    label = ET.SubElement(labels, 'label')
    ET.SubElement(label, 'name').text = label_name
    ET.SubElement(label, 'attributes')
    original_size = ET.SubElement(task, 'original_size')
    ET.SubElement(original_size, 'width').text = str(width)
    ET.SubElement(original_size, 'height').text = str(height)
    return meta, image_elem 