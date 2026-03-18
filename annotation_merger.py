import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import argparse
from datetime import datetime


def process_xml_file(file_path):
    """Вспомогательная функция для подсчета данных в одном XML"""
    img_nodes = []
    obj_count = 0
    try:
        tree = ET.parse(file_path)
        src_root = tree.getroot()

        # CVAT формат
        images = src_root.findall('image')
        if images:
            for img in images:
                # Создаем копию узла
                new_img = ET.Element('image', img.attrib)
                objs = img.findall('box') + img.findall('ellipse')
                for obj in objs:
                    new_img.append(obj)
                    obj_count += 1
                img_nodes.append(new_img)

        # Pascal VOC формат
        elif src_root.tag == 'annotation' or src_root.find('object') is not None:
            fname = src_root.find('filename').text if src_root.find('filename') is not None else file_path.stem
            size = src_root.find('size')
            w = size.find('width').text if size is not None else "0"
            h = size.find('height').text if size is not None else "0"

            img_node = ET.Element('image', {'name': fname, 'width': w, 'height': h})
            for obj in src_root.findall('object'):
                bbox = obj.find('bndbox')
                if bbox is not None:
                    ET.SubElement(img_node, 'box', {
                        'label': 'spore',
                        'xtl': bbox.find('xmin').text,
                        'ytl': bbox.find('ymin').text,
                        'xbr': bbox.find('xmax').text,
                        'ybr': bbox.find('ymax').text,
                        'occluded': '0'
                    })
                    obj_count += 1
            img_nodes.append(img_node)

    except Exception:
        pass
    return img_nodes, obj_count


def universal_merge(input_sources, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_xml_path = output_path / "FINAL_AGGREGATED_ANNOTATION.xml"

    root = ET.Element('annotations')
    ET.SubElement(root, 'version').text = "1.1"

    # Метаданные
    meta = ET.SubElement(root, 'meta')
    task = ET.SubElement(meta, 'task')
    ET.SubElement(task, 'name').text = f"Merged_Project_{datetime.now().strftime('%Y%m%d')}"
    labels = ET.SubElement(task, 'labels')
    label = ET.SubElement(labels, 'label')
    ET.SubElement(label, 'name').text = "spore"

    total_images = 0
    total_objects_found = 0

    print(f"\n{'Источник (Путь/Файл)':<45} | {'Фото':<8} | {'Спор':<10}")
    print("-" * 70)

    for source in input_sources:
        src_path = Path(source)
        src_img_nodes = []
        src_obj_total = 0

        # Собираем файлы из источника
        files_in_source = []
        if src_path.is_file() and src_path.suffix.lower() == '.xml':
            files_in_source = [src_path]
            display_name = src_path.name
        elif src_path.is_dir():
            files_in_source = sorted(list(src_path.glob('*.xml')))
            display_name = f"Folder: {src_path.name}"
        else:
            continue

        # Обрабатываем все файлы источника
        for f in files_in_source:
            nodes, count = process_xml_file(f)
            for n in nodes:
                n.set('id', str(total_images + len(src_img_nodes)))
                src_img_nodes.append(n)
            src_obj_total += count

        # Добавляем в общий корень
        for n in src_img_nodes:
            root.append(n)

        # Обновляем глобальные счетчики
        count_imgs = len(src_img_nodes)
        total_images += count_imgs
        total_objects_found += src_obj_total

        # Вывод статистики по источнику (одна строка на папку или файл)
        print(f"{display_name[:45]:<45} | {count_imgs:<8} | {src_obj_total:<10}")

    # Сохранение
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    clean_xml = "\n".join([line for line in xml_str.split('\n') if line.strip()])
    with open(final_xml_path, "w", encoding="utf-8") as f:
        f.write(clean_xml)

    print("-" * 70)
    print(f"{'ИТОГО СОБРАНО:':<45} | {total_images:<8} | {total_objects_found:<10}")
    print(f"\n Результат: {final_xml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--out', type=str, default="merged_output")
    args = parser.parse_args()
    universal_merge(args.input, args.out)