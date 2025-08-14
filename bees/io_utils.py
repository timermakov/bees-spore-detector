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
    Return (meta, image_element) for CVAT for images 1.1 XML export, using <ellipse> elements.
    image_path: path to the image file
    spore_objs: list of contours (each contour is a numpy array of shape (N, 1, 2))
    image_id: integer id for the image
    label_name: label for all spores
    """
    import xml.etree.ElementTree as ET
    from PIL import Image
    import os
    import cv2
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
        if len(cnt) < 5:
            # fitEllipse requires at least 5 points, skip or fallback to polygon if needed
            continue
        ellipse = cv2.fitEllipse(cnt)
        (cx, cy), (major, minor), rotation = ellipse
        ellipse_elem = ET.SubElement(image_elem, 'ellipse', {
            'label': label_name,
            'occluded': '0',
            'source': 'auto',
            'cx': f"{float(cx):.2f}",
            'cy': f"{float(cy):.2f}",
            'rx': f"{float(major/2):.2f}",
            'ry': f"{float(minor/2):.2f}",
            'rotation': f"{float(rotation):.2f}",
            'z_order': '0',
            'group_id': str(idx),
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
    ET.SubElement(label, 'type').text = 'ellipse'
    ET.SubElement(label, 'attributes')
    original_size = ET.SubElement(task, 'original_size')
    ET.SubElement(original_size, 'width').text = str(width)
    ET.SubElement(original_size, 'height').text = str(height)
    return meta, image_elem 

# Add pretty-print utility for XML

def indent_xml(elem, level=0):
    """Recursively pretty-print XML with indentation."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i 