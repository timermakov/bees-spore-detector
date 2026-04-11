# нарезает датасет для обучения 

import os
import xml.etree.ElementTree as ET
import cv2
import shutil
import random
import argparse
import time
from pathlib import Path
from xml.dom import minidom


def save_pascal_xml(filename, bboxes, size):
    width, height, depth = size
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'filename').text = filename
    s_el = ET.SubElement(annotation, 'size')
    ET.SubElement(s_el, 'width').text = str(width)
    ET.SubElement(s_el, 'height').text = str(height)
    ET.SubElement(s_el, 'depth').text = str(depth)
    for box in bboxes:
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = 'spore'
        bnd = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bnd, 'xmin').text = str(int(box[0]))
        ET.SubElement(bnd, 'ymin').text = str(int(box[1]))
        ET.SubElement(bnd, 'xmax').text = str(int(box[2]))
        ET.SubElement(bnd, 'ymax').text = str(int(box[3]))
    return annotation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', type=str, required=True, help="Общий XML файл")
    parser.add_argument('--img', type=str, required=True, help="Папка с оригиналами")
    parser.add_argument('--out', type=str, default='datasetV3_final')
    parser.add_argument('--tile', type=int, default=512)
    parser.add_argument('--overlap', type=float, default=0.25)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--negative_ratio', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=None, help="Сид для рандома (по дефолту None - рандом)")
    args = parser.parse_args()

    # Настройка рандома
    current_seed = args.seed if args.seed is not None else int(time.time())
    random.seed(current_seed)

    out_path = Path(args.out)
    if out_path.exists(): shutil.rmtree(out_path)

    for s in ['train', 'val']:
        (out_path / 'images' / s).mkdir(parents=True, exist_ok=True)
        (out_path / 'labels_xml' / s).mkdir(parents=True, exist_ok=True)

    tree = ET.parse(args.xml)
    root = tree.getroot()

    # Готовим общий XML
    common_xml_root = ET.Element("annotations")
    meta_node = root.find('meta')
    if meta_node is not None: common_xml_root.append(meta_node)

    images_data = []
    for img_tag in root.findall('image'):
        boxes = []
        for ell in img_tag.findall('ellipse'):
            cx, cy, rx, ry = float(ell.get('cx')), float(ell.get('cy')), float(ell.get('rx')), float(ell.get('ry'))
            boxes.append([cx - rx, cy - ry, cx + rx, cy + ry])
        images_data.append({'name': img_tag.get('name'), 'boxes': boxes})

    random.shuffle(images_data)
    split_idx = int(len(images_data) * (1 - args.val_split))

    stats = {
        'train': {'total': 0, 'objs': 0, 'empty': 0, 'box_count': 0, 'tile': args.tile},
        'val': {'total': 0, 'objs': 0, 'empty': 0, 'box_count': 0, 'tile': args.tile}
    }

    print(f"Нарезка тайлов...")
    print(f"  Размер (Train/Val): {args.tile} px")
    print(f"  Negative ratio: {args.negative_ratio * 100:.0f}%")
    print(f"  Используемый Seed: {current_seed} {'(fixed)' if args.seed else '(random)'}\n")

    img_id_counter = 0
    for i, data in enumerate(images_data):
        mode = 'train' if i < split_idx else 'val'
        t_size = stats[mode]['tile']
        stride = int(t_size * (1 - args.overlap))

        img = cv2.imread(str(Path(args.img) / data['name']))
        if img is None: continue
        h, w, c = img.shape

       # y_pos = [min(j * stride, h - t_size) for j in range((h - t_size) // stride + 2) if j * stride < h - t_size / 2]
        if h <= t_size:
            y_pos = [0]
        else:
            y_pos = list(range(0, h - t_size + 1, stride))
            if y_pos[-1] != h - t_size:
                y_pos.append(h - t_size)
       # x_pos = [min(j * stride, w - t_size) for j in range((w - t_size) // stride + 2) if j * stride < w - t_size / 2]
        if w <= t_size:
            x_pos = [0]
        else:
            x_pos = list(range(0, w - t_size + 1, stride))
            if x_pos[-1] != w - t_size:
                x_pos.append(w - t_size)

        for y1 in y_pos:
            for x1 in x_pos:
                tile_bboxes = []
                for box in data['boxes']:
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    if x1 <= cx < x1 + t_size and y1 <= cy < y1 + t_size:
                        tile_bboxes.append([max(0, box[0] - x1), max(0, box[1] - y1), min(t_size, box[2] - x1),
                                            min(t_size, box[3] - y1)])

                if len(tile_bboxes) == 0:
                    if random.random() > args.negative_ratio: continue
                    stats[mode]['empty'] += 1
                else:
                    stats[mode]['objs'] += 1
                    stats[mode]['box_count'] += len(tile_bboxes)

                stats[mode]['total'] += 1
                t_name = f"{Path(data['name']).stem}_tile_{y1}_{x1}.jpg"
                cv2.imwrite(str(out_path / 'images' / mode / t_name), img[y1:y1 + t_size, x1:x1 + t_size])

                tile_xml = save_pascal_xml(t_name, tile_bboxes, (t_size, t_size, c))
                with open(out_path / 'labels_xml' / mode / f"{t_name.replace('.jpg', '.xml')}", "w",
                          encoding="utf-8") as f:
                    f.write(minidom.parseString(ET.tostring(tile_xml)).toprettyxml(indent="   "))

                img_el = ET.SubElement(common_xml_root, "image",
                                       {"id": str(img_id_counter), "name": t_name, "width": str(t_size),
                                        "height": str(t_size)})
                for b in tile_bboxes:
                    ET.SubElement(img_el, "box",
                                  {"label": "spore", "xtl": str(b[0]), "ytl": str(b[1]), "xbr": str(b[2]),
                                   "ybr": str(b[3])})
                img_id_counter += 1

    with open(out_path / "annotations_all.xml", "w", encoding="utf-8") as f:
        f.write(minidom.parseString(ET.tostring(common_xml_root)).toprettyxml(indent="  "))

    t_all = stats['train']['total'] + stats['val']['total']
    e_all = stats['train']['empty'] + stats['val']['empty']
    o_all = stats['train']['objs'] + stats['val']['objs']
    b_all = stats['train']['box_count'] + stats['val']['box_count']

    print(f"--- СТАТИСТИКА НАРЕЗКИ ---")
    print(f"Всего тайлов сохранено: {t_all}")
    print(f"Всего спор (аннотаций): {b_all}")
    print(f"Из них со спорами:      {o_all}")
    print(f"Из них пустых (фон):    {e_all} ({(e_all / t_all * 100 if t_all > 0 else 0):.1f}% от выборки)")
    print(f"--------------------------")
    print(f"TRAIN: {stats['train']['total']} тайлов | {stats['train']['box_count']} спор")
    print(f"VAL:   {stats['val']['total']} тайлов | {stats['val']['box_count']} спор")
    print(f"Seed для повтора: {current_seed}")


if __name__ == "__main__":
    main()