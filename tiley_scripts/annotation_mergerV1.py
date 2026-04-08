import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
import argparse

# Цвета для терминала
G, B, RESET = "\033[32m", "\033[34;1m", "\033[0m"


def calculate_iou(boxA, boxB):
    xA, yA, xB, yB = map(float, boxA)
    xAe, yAe, xBe, yBe = map(float, boxB)
    inter_xA, inter_yA = max(xA, xAe), max(yA, yAe)
    inter_xB, inter_yB = min(xB, xBe), min(yB, yBe)
    interArea = max(0, inter_xB - inter_xA) * max(0, inter_yB - inter_yA)
    areaA = (xB - xA) * (yB - yA)
    areaB = (xBe - xAe) * (yBe - yAe)
    union = areaA + areaB - interArea
    return interArea / union if union > 0 else 0


def process_xml_file(file_path, min_size=5):
    img_data = {}
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        items = root.findall('image') if root.findall('image') else [root]
        for item in items:
            if item.tag == 'image':
                name, w, h = item.get('name'), item.get('width'), item.get('height')
                boxes_raw = [(b.get('xtl'), b.get('ytl'), b.get('xbr'), b.get('ybr')) for b in item.findall('box')]
            else:
                name = root.find('filename').text if root.find('filename') is not None else file_path.name
                size = root.find('size')
                w, h = size.find('width').text, size.find('height').text
                boxes_raw = []
                for obj in root.findall('object'):
                    b = obj.find('bndbox')
                    boxes_raw.append(
                        (b.find('xmin').text, b.find('ymin').text, b.find('xmax').text, b.find('ymax').text))

            # Фильтр меньше 5 пикселей
            clean_boxes = [b for b in boxes_raw if
                           (float(b[2]) - float(b[0])) >= min_size and (float(b[3]) - float(b[1])) >= min_size]
            img_data[name] = {'w': w, 'h': h, 'boxes': clean_boxes}
    except:
        pass
    return img_data


def save_output(storage, output_dir, mode, shape_type):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if mode == 'single':
        root = ET.Element('annotations')
        ET.SubElement(root, 'version').text = "1.1"
        for i, (name, data) in enumerate(storage.items()):
            img_node = ET.SubElement(root, 'image', id=str(i), name=name, width=str(data['w']), height=str(data['h']))
            for b in data['boxes']:
                xtl, ytl, xbr, ybr = map(float, b)
                if shape_type == 'ellipse':
                    el = ET.SubElement(img_node, 'ellipse', label='spore', source='manual', z_order='0')
                    el.set('cx', f"{(xtl + xbr) / 2:.2f}");
                    el.set('cy', f"{(ytl + ybr) / 2:.2f}")
                    el.set('rx', f"{(xbr - xtl) / 2:.2f}");
                    el.set('ry', f"{(ybr - ytl) / 2:.2f}")
                else:
                    bx = ET.SubElement(img_node, 'box', label='spore', source='manual', z_order='0')
                    bx.set('xtl', f"{xtl:.2f}");
                    bx.set('ytl', f"{ytl:.2f}")
                    bx.set('xbr', f"{xbr:.2f}");
                    bx.set('ybr', f"{ybr:.2f}")

        with open(out_path / "annotations.xml", "w", encoding="utf-8") as f:
            f.write(minidom.parseString(ET.tostring(root)).toprettyxml(indent="  "))

    else:  # mode == 'split'
        for name, data in storage.items():
            root = ET.Element('annotation')
            ET.SubElement(root, 'filename').text = name
            sz = ET.SubElement(root, 'size')
            ET.SubElement(sz, 'width').text = str(data['w'])
            ET.SubElement(sz, 'height').text = str(data['h'])
            for b in data['boxes']:
                obj = ET.SubElement(root, 'object')
                ET.SubElement(obj, 'name').text = 'spore'
                bx = ET.SubElement(obj, 'bndbox')
                for i, tag in enumerate(['xmin', 'ymin', 'xmax', 'ymax']):
                    ET.SubElement(bx, tag).text = str(b[i])
            with open(out_path / f"{Path(name).stem}.xml", "w", encoding="utf-8") as f:
                f.write(minidom.parseString(ET.tostring(root)).toprettyxml(indent="  "))


def universal_merge(input_sources, output_dir, mode, shape_type, iou_thresh=0.5, verbose=False):
    global_storage = {}
    print(f"\n{'Источник':<45} | {'Фото':<6} | {'Прирост'}")
    print("-" * 80)

    for idx, source in enumerate(input_sources):
        src_path = Path(source)
        files = list(src_path.glob('**/*.xml')) if src_path.is_dir() else [src_path]
        folder_added, folder_images = 0, 0

        if verbose and idx > 0:
            print(f"\n{B}[АНАЛИЗ ДОПОЛНЕНИЯ: {src_path.name}]{RESET}")
            print(f"{'Файл':<35} | {'Было':<6} | {'Стало':<6} | {'Прибавка (%)'}")
            print("-" * 75)

        for f in sorted(files):
            file_data = process_xml_file(f)
            for img_name, data in file_data.items():
                if img_name not in global_storage:
                    global_storage[img_name] = {'w': data['w'], 'h': data['h'], 'boxes': []}
                    folder_images += 1

                before = len(global_storage[img_name]['boxes'])
                added = 0
                for n_box in data['boxes']:
                    if not any(calculate_iou(n_box, e_box) > iou_thresh for e_box in global_storage[img_name]['boxes']):
                        global_storage[img_name]['boxes'].append(n_box)
                        added += 1
                folder_added += added

                if verbose and idx > 0 and added > 0:
                    perc = (added / before * 100) if before > 0 else 0
                    color = B if perc > 50 else (G if perc > 20 else "")
                    print(
                        f"  {img_name:<32} | {before:<6} | {len(global_storage[img_name]['boxes']):<6} | {color}+{added} ({perc:.1f}%){RESET}")

        print(f"{src_path.name[:45]:<45} | {folder_images:<6} | +{folder_added:<7}")

    save_output(global_storage, output_dir, mode, shape_type)
    print(f"\nГОТОВО! Итого: {len(global_storage)} фото. Режим: {mode}, Фигуры: {shape_type.upper()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--out', type=str, default="merged_output")
    parser.add_argument('--mode', choices=['single', 'split'], default='single')
    parser.add_argument('--type', choices=['box', 'ellipse'], default='box')  # Вот он, аргумент --type
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    universal_merge(args.input, args.out, args.mode, args.type, 0.5, args.verbose)