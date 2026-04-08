import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import argparse
from datetime import datetime


def process_xml_file(file_path):
    img_data = {}  # { filename: { 'width': w, 'height': h, 'boxes': set((xtl, ytl, xbr, ybr)) } }
    try:
        tree = ET.parse(file_path)
        src_root = tree.getroot()

        # CVAT формат
        images = src_root.findall('image')
        if images:
            for img in images:
                name = img.get('name')
                if name not in img_data:
                    img_data[name] = {'width': img.get('width'), 'height': img.get('height'), 'boxes': set()}

                for box in img.findall('box'):
                    coords = (box.get('xtl'), box.get('ytl'), box.get('xbr'), box.get('ybr'))
                    img_data[name]['boxes'].add(coords)

        # Pascal VOC формат
        elif src_root.tag == 'annotation' or src_root.find('object') is not None:
            name = src_root.find('filename').text if src_root.find('filename') is not None else file_path.stem
            size = src_root.find('size')
            w = size.find('width').text if size is not None else "0"
            h = size.find('height').text if size is not None else "0"

            if name not in img_data:
                img_data[name] = {'width': w, 'height': h, 'boxes': set()}

            for obj in src_root.findall('object'):
                bbox = obj.find('bndbox')
                if bbox is not None:
                    coords = (bbox.find('xmin').text, bbox.find('ymin').text,
                              bbox.find('xmax').text, bbox.find('ymax').text)
                    img_data[name]['boxes'].add(coords)

    except Exception:
        pass
    return img_data


def universal_merge(input_sources, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_xml_path = output_path / "FINAL_CLEAN_ANNOTATION.xml"

    # Глобальное хранилище для объединения: { name: {width, height, boxes} }
    global_storage = {}

    print(f"\n{'Источник':<40} | {'Фото':<8} | {'Спор (новые)':<12}")
    print("-" * 70)

    for source in input_sources:
        src_path = Path(source)
        files = []
        if src_path.is_file() and src_path.suffix.lower() == '.xml':
            files = [src_path]
            display_name = src_path.name
        elif src_path.is_dir():
            files = sorted(list(src_path.glob('*.xml')))
            display_name = f"Folder: {src_path.name}"
        else:
            continue

        src_new_boxes = 0
        src_new_images = 0

        for f in files:
            file_data = process_xml_file(f)
            for img_name, data in file_data.items():
                if img_name not in global_storage:
                    global_storage[img_name] = {'width': data['width'], 'height': data['height'], 'boxes': set()}
                    src_new_images += 1

                before_count = len(global_storage[img_name]['boxes'])
                global_storage[img_name]['boxes'].update(data['boxes'])
                src_new_boxes += (len(global_storage[img_name]['boxes']) - before_count)

        print(f"{display_name[:40]:<40} | {src_new_images:<8} | {src_new_boxes:<12}")

    # Сборка финального XML
    root = ET.Element('annotations')
    ET.SubElement(root, 'version').text = "1.1"

    total_objs = 0
    for idx, (name, data) in enumerate(global_storage.items()):
        img_node = ET.SubElement(root, 'image', {
            'id': str(idx),
            'name': name,
            'width': data['width'],
            'height': data['height']
        })
        for box in data['boxes']:
            ET.SubElement(img_node, 'box', {
                'label': 'spore', 'xtl': box[0], 'ytl': box[1],
                'xbr': box[2], 'ybr': box[3], 'occluded': '0'
            })
            total_objs += 1

    # Красивое сохранение
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    clean_xml = "\n".join([line for line in xml_str.split('\n') if line.strip()])
    with open(final_xml_path, "w", encoding="utf-8") as f:
        f.write(clean_xml)

    print("-" * 70)
    print(f"{'ИТОГО УНИКАЛЬНЫХ:':<40} | {len(global_storage):<8} | {total_objs:<12}")
    print(f"\n🚀 Очищенная аннотация готова: {final_xml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--out', type=str, default="merged_output")
    args = parser.parse_args()
    universal_merge(args.input, args.out)