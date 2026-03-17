import os
import xml.etree.ElementTree as ET
import cv2
import shutil
import random
import argparse
from pathlib import Path

def create_tiled_dataset_v2(xml_path, img_src_dir, output_dir, tile_size=640, overlap=0.25, val_split=0.2,
                            negative_ratio=0.1):
    base = Path(output_dir)
    if base.exists():
        shutil.rmtree(base)

    for split in ['train', 'val']:
        (base / 'images' / split).mkdir(parents=True)
        (base / 'labels' / split).mkdir(parents=True)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    image_data = []
    for img_tag in root.findall('image'):
        name = img_tag.get('name')
        w, h = int(img_tag.get('width')), int(img_tag.get('height'))
        ellipses = []
        for ell in img_tag.findall('ellipse'):
            cx, cy = float(ell.get('cx')), float(ell.get('cy'))
            rx, ry = float(ell.get('rx')), float(ell.get('ry'))
            # сохраняем bbox и центр
            ellipses.append((cx - rx, cy - ry, cx + rx, cy + ry, cx, cy))
        image_data.append({'name': name, 'w': w, 'h': h, 'annotations': ellipses})

    random.shuffle(image_data)
    split_idx = int(len(image_data) * (1 - val_split))

    stride = int(tile_size * (1 - overlap))
    print(f"Параметры нарезки: tile={tile_size}, overlap={overlap}, stride={stride}")

    pos_tiles = 0
    neg_tiles = 0
    total_tiles = 0

    for i, data in enumerate(image_data):
        split_name = 'train' if i < split_idx else 'val'
        img_path = Path(img_src_dir) / data['name']
        if not img_path.exists():
            print(f"Предупреждение: не найден {img_path}")
            continue

        img = cv2.imread(str(img_path))
        h, w = data['h'], data['w']

        # вычисляем позиции тайлов
        y_steps = max(1, (h - tile_size) // stride + 1)
        x_steps = max(1, (w - tile_size) // stride + 1)

        y_positions = [min(i * stride, h - tile_size) for i in range(y_steps)]
        if y_positions[-1] < h - tile_size:
            y_positions.append(h - tile_size)

        x_positions = [min(i * stride, w - tile_size) for i in range(x_steps)]
        if x_positions[-1] < w - tile_size:
            x_positions.append(w - tile_size)

        for y1 in y_positions:
            for x1 in x_positions:
                y2 = y1 + tile_size
                x2 = x1 + tile_size
                if y2 > h:
                    y1 = h - tile_size
                    y2 = h
                if x2 > w:
                    x1 = w - tile_size
                    x2 = w

                tile_labels = []
                for (ax1, ay1, ax2, ay2, acx, acy) in data['annotations']:
                    if x1 <= acx < x2 and y1 <= acy < y2:
                        lx1 = max(ax1, x1)
                        ly1 = max(ay1, y1)
                        lx2 = min(ax2, x2)
                        ly2 = min(ay2, y2)
                        tcx = ((lx1 + lx2) / 2 - x1) / tile_size
                        tcy = ((ly1 + ly2) / 2 - y1) / tile_size
                        tw = (lx2 - lx1) / tile_size
                        th = (ly2 - ly1) / tile_size
                        if 0 <= tcx <= 1 and 0 <= tcy <= 1 and tw > 0 and th > 0:
                            tile_labels.append(f"0 {tcx:.6f} {tcy:.6f} {tw:.6f} {th:.6f}")

                tile_name = f"{Path(data['name']).stem}_tile_{y1}_{x1}.jpg"
                tile_img = img[y1:y2, x1:x2]

                if tile_labels:
                    cv2.imwrite(str(base / 'images' / split_name / tile_name), tile_img)
                    with open(base / 'labels' / split_name / tile_name.replace('.jpg', '.txt'), 'w') as f:
                        f.write('\n'.join(tile_labels))
                    pos_tiles += 1
                else:
                    if random.random() < negative_ratio:
                        cv2.imwrite(str(base / 'images' / split_name / tile_name), tile_img)
                        open(base / 'labels' / split_name / tile_name.replace('.jpg', '.txt'), 'w').close()
                        neg_tiles += 1
                total_tiles += 1

    print(f"Создано тайлов: всего {total_tiles}, положительных {pos_tiles}, пустых {neg_tiles}")
    print(f"Положительные тайлы составляют {pos_tiles/total_tiles*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Нарезка изображений на тайлы для YOLO")
    parser.add_argument('--xml', type=str, required=True, help='Путь к XML-файлу с аннотациями')
    parser.add_argument('--img_dir', type=str, required=True, help='Папка с исходными изображениями')
    parser.add_argument('--output', type=str, help='Выходная папка для тайлов (если не указана, генерируется автоматически)')
    parser.add_argument('--tile_size', type=int, default=640, help='Размер тайла (по умолчанию 640)')
    parser.add_argument('--overlap', type=float, default=0.25, help='Перекрытие тайлов (0.0-1.0, по умолчанию 0.25)')
    parser.add_argument('--negative_ratio', type=float, default=0.1, help='Доля пустых тайлов (0.0-1.0, по умолчанию 0.1)')
    parser.add_argument('--val_split', type=float, default=0.2, help='Доля валидационной выборки (по умолчанию 0.2)')
    parser.add_argument('--auto_output', action='store_true', help='Автоматически генерировать имя выходной папки (по умолчанию включено, если output не указан)')

    args = parser.parse_args()

    # Определяем выходную папку
    if args.output:
        output_dir = args.output
    else:
        # Базовое имя исходной папки (без пути)
        src_path = Path(args.img_dir)
        base_name = src_path.name
        # Добавляем суффикс с параметрами для информативности
        output_dir = f"{base_name}_tiled_{args.tile_size}_{args.overlap}"
        print(f"Выходная папка не указана, генерируем: {output_dir}")

    create_tiled_dataset_v2(
        xml_path=args.xml,
        img_src_dir=args.img_dir,
        output_dir=output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        val_split=args.val_split,
        negative_ratio=args.negative_ratio
    )