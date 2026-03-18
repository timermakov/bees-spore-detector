import os
import xml.etree.ElementTree as ET
import cv2
import shutil
import random
import argparse
from pathlib import Path
from xml.dom import minidom


def create_tiled_dataset_v5(xml_path, img_src_dir, output_dir=None, tile_size=640, overlap=0.25, val_split=0.2,
                            negative_ratio=0.1):
    # Если выходная папка не указана, создаем имя на основе исходной папки
    if output_dir is None:
        src_folder_name = Path(img_src_dir).stem
        output_dir = f"{src_folder_name}_tiles"

    base = Path(output_dir)
    if base.exists():
        shutil.rmtree(base)

    # Временные папки для сбора всех данных
    all_imgs_dir = base / 'all_images'
    all_labels_dir = base / 'all_labels'
    all_imgs_dir.mkdir(parents=True)
    all_labels_dir.mkdir(parents=True)

    # Читаем исходный XML
    tree = ET.parse(xml_path)
    root = tree.getroot()

    meta_node = root.find('meta')
    if meta_node is not None:
        meta_copy = ET.Element('meta')
        for child in meta_node:
            meta_copy.append(child)
    else:
        meta_copy = None

    version_node = root.find('version')
    version_text = version_node.text if version_node is not None else '1.1'

    image_data = []
    for img_tag in root.findall('image'):
        name = img_tag.get('name')
        w, h = int(img_tag.get('width')), int(img_tag.get('height'))
        ellipses = []
        for ell in img_tag.findall('ellipse'):
            cx, cy = float(ell.get('cx')), float(ell.get('cy'))
            rx, ry = float(ell.get('rx')), float(ell.get('ry'))
            ell_data = {
                'bbox': (cx - rx, cy - ry, cx + rx, cy + ry),
                'center': (cx, cy),
                'group_id': ell.get('group_id'),
                'source': ell.get('source', 'manual'),
                'occluded': ell.get('occluded', '0'),
                'label': ell.get('label', 'spore')
            }
            ellipses.append(ell_data)
        image_data.append({'name': name, 'w': w, 'h': h, 'annotations': ellipses})

    stride = int(tile_size * (1 - overlap))
    common_xml_root = ET.Element("annotations")
    ver_elem = ET.SubElement(common_xml_root, "version")
    ver_elem.text = version_text
    if meta_copy is not None:
        common_xml_root.append(meta_copy)

    img_id = 0
    generated_tiles = []

    # СЧЕТЧИКИ ДЛЯ ТЕБЯ
    tiles_with_objects = 0
    tiles_empty_saved = 0
    total_tiles_attempted = 0

    print(f"Нарезка тайлов...")
    print(f"  Размер тайла: {tile_size}px")
    print(f"  Перекрытие: {overlap * 100:.0f}%")
    print(f"  Negative ratio: {negative_ratio * 100:.0f}%")
    print(f"  Выходная папка: {output_dir}")

    for data in image_data:
        img_path = Path(img_src_dir) / data['name']
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        h, w = data['h'], data['w']

        y_positions = [min(j * stride, h - tile_size) for j in range(max(1, (h - tile_size) // stride + 1))]
        x_positions = [min(j * stride, w - tile_size) for j in range(max(1, (w - tile_size) // stride + 1))]

        for y1 in set(y_positions):
            for x1 in set(x_positions):
                total_tiles_attempted += 1
                y2, x2 = y1 + tile_size, x1 + tile_size
                tile_labels_yolo = []
                tile_objects_xml = []

                for obj in data['annotations']:
                    acx, acy = obj['center']
                    if x1 <= acx < x2 and y1 <= acy < y2:
                        ax1, ay1, ax2, ay2 = obj['bbox']
                        lx1, ly1 = max(ax1, x1) - x1, max(ay1, y1) - y1
                        lx2, ly2 = min(ax2, x2) - x1, min(ay2, y2) - y1

                        tcx = ((lx1 + lx2) / 2) / tile_size
                        tcy = ((ly1 + ly2) / 2) / tile_size
                        tw = (lx2 - lx1) / tile_size
                        th = (ly2 - ly1) / tile_size
                        tile_labels_yolo.append(f"0 {tcx:.6f} {tcy:.6f} {tw:.6f} {th:.6f}")

                        box_attrs = {
                            'label': obj['label'], 'source': obj['source'], 'occluded': obj['occluded'],
                            'xtl': f"{lx1:.2f}", 'ytl': f"{ly1:.2f}", 'xbr': f"{lx2:.2f}", 'ybr': f"{ly2:.2f}",
                            'z_order': '0'
                        }
                        if obj.get('group_id'): box_attrs['group_id'] = obj['group_id']
                        tile_objects_xml.append(box_attrs)

                tile_name = f"{Path(data['name']).stem}_tile_{y1}_{x1}.jpg"

                # Решаем, сохранять ли тайл
                is_empty = len(tile_objects_xml) == 0
                if not is_empty:
                    save_this_tile = True
                    tiles_with_objects += 1
                else:
                    if random.random() < negative_ratio:
                        save_this_tile = True
                        tiles_empty_saved += 1
                    else:
                        save_this_tile = False

                if save_this_tile:
                    cv2.imwrite(str(all_imgs_dir / tile_name), img[y1:y2, x1:x2])
                    with open(all_labels_dir / tile_name.replace('.jpg', '.txt'), 'w') as f:
                        f.write('\n'.join(tile_labels_yolo))

                    img_el = ET.SubElement(common_xml_root, "image", {
                        "id": str(img_id), "name": tile_name, "width": str(tile_size), "height": str(tile_size)
                    })
                    for box_attrs in tile_objects_xml:
                        ET.SubElement(img_el, "box", box_attrs)

                    generated_tiles.append(tile_name)
                    img_id += 1

    # Сохраняем общий XML
    xml_str = minidom.parseString(ET.tostring(common_xml_root)).toprettyxml(indent="  ")
    with open(base / "annotations_all.xml", "w", encoding="utf-8") as f:
        f.write(xml_str)

    # Деление на выборки
    random.shuffle(generated_tiles)
    split_idx = int(len(generated_tiles) * (1 - val_split))
    train_files = generated_tiles[:split_idx]
    val_files = generated_tiles[split_idx:]

    for split, files in [('train', train_files), ('val', val_files)]:
        (base / 'images' / split).mkdir(parents=True)
        (base / 'labels' / split).mkdir(parents=True)
        for f_name in files:
            shutil.move(str(all_imgs_dir / f_name), str(base / 'images' / split / f_name))
            label_name = f_name.replace('.jpg', '.txt')
            shutil.move(str(all_labels_dir / label_name), str(base / 'labels' / split / label_name))

    shutil.rmtree(all_imgs_dir)
    shutil.rmtree(all_labels_dir)

    # ВЫВОД СТАТИСТИКИ
    total_saved = len(generated_tiles)
    empty_percent = (tiles_empty_saved / total_saved * 100) if total_saved > 0 else 0
    print(f"\n--- СТАТИСТИКА НАРЕЗКИ ---")
    print(f"Всего тайлов сохранено: {total_saved}")
    print(f"Из них со спорами:      {tiles_with_objects}")
    print(f"Из них пустых (фон):    {tiles_empty_saved} ({empty_percent:.1f}% от выборки)")
    print(f"--------------------------")
    print(f"Train: {len(train_files)} | Val: {len(val_files)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml', type=str, required=True)
    parser.add_argument('--img_dir', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--tile_size', type=int, default=640)
    parser.add_argument('--overlap', type=float, default=0.25)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--negative_ratio', type=float, default=0.1)
    args = parser.parse_args()

    create_tiled_dataset_v5(
        xml_path=args.xml, img_src_dir=args.img_dir, output_dir=args.output,
        tile_size=args.tile_size, overlap=args.overlap,
        val_split=args.val_split, negative_ratio=args.negative_ratio
    )