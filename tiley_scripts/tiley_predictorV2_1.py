# оценивать тестовые изображения
# веса

import cv2
import os
import torch
import argparse
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from ultralytics import YOLO
from torchvision.ops import nms


def save_pascal_voc(filename, img_shape, bboxes, output_path):
    height, width, depth = img_shape
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'filename').text = filename

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)

    for box in bboxes:
        x1, y1, x2, y2 = map(int, box[:4])
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = 'spore'
        bbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bbox, 'xmin').text = str(max(0, x1))
        ET.SubElement(bbox, 'ymin').text = str(max(0, y1))
        ET.SubElement(bbox, 'xmax').text = str(min(width, x2))
        ET.SubElement(bbox, 'ymax').text = str(min(height, y2))

    xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(xml_str)


def apply_clahe(img):
    """Улучшает контраст, чтобы проявить бледные споры в светлых зонах."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # clipLimit 3.0 — золотая середина
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_img = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    return enhanced_img


def main():
    parser = argparse.ArgumentParser(description="Spore Tiley Predictor v2.0")
    parser.add_argument('--weights', type=str, required=True, help="Путь к best.pt")
    parser.add_argument('--source', type=str, required=True, help="Папка с оригиналами")
    parser.add_argument('--imgsz', type=int, default=1024, help="Размер для модели")
    parser.add_argument('--tile_size', type=int, default=None, help="Размер нарезки (по дефолту = imgsz)")
    parser.add_argument('--conf', type=float, default=0.3, help="Порог уверенности")
    # ИСПРАВЛЕНО: убран знак процента, вызывавший ValueError
    parser.add_argument('--overlap', type=float, default=0.2, help="Нахлест тайлов (0.2 это 20 процентов)")
    parser.add_argument('--iou', type=float, default=0.15, help="Порог склейки рамок NMS")

    args = parser.parse_args()

    if args.tile_size is None:
        args.tile_size = args.imgsz

    output_dir = Path("runs/detect/results") / f"clean_prediction_{args.conf}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    stride = int(args.tile_size * (1 - args.overlap))

    files = [f for f in os.listdir(args.source) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\n[START] Найдено файлов: {len(files)}")

    for filename in files:
        img_path = os.path.join(args.source, filename)
        orig_img = cv2.imread(img_path)
        if orig_img is None: continue

        # Улучшаем контраст для детекции
        enhanced_img = apply_clahe(orig_img)

        h, w, c = orig_img.shape

        if h <= args.tile_size:
            y_positions = [0]
        else:
            y_positions = list(range(0, h - args.tile_size + 1, stride))
            if y_positions[-1] != h - args.tile_size:
                y_positions.append(h - args.tile_size)

        if w <= args.tile_size:
            x_positions = [0]
        else:
            x_positions = list(range(0, w - args.tile_size + 1, stride))
            if x_positions[-1] != w - args.tile_size:
                x_positions.append(w - args.tile_size)

        boxes_list = []
        scores_list = []

        for y1 in y_positions:
            for x1 in x_positions:
                tile = enhanced_img[y1:y1 + args.tile_size, x1:x1 + args.tile_size]
                results = model.predict(tile, imgsz=args.imgsz, conf=args.conf, verbose=False)

                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    boxes_list.append([xyxy[0] + x1, xyxy[1] + y1, xyxy[2] + x1, xyxy[3] + y1])
                    scores_list.append(box.conf.item())

        if boxes_list:
            boxes_tensor = torch.tensor(boxes_list)
            scores_tensor = torch.tensor(scores_list)

            # Склеиваем дубликаты
            keep = nms(boxes_tensor, scores_tensor, args.iou)
            final_boxes = boxes_tensor[keep]

            # Сохраняем XML
            xml_path = output_dir / (Path(filename).stem + ".xml")
            save_pascal_voc(filename, (h, w, c), final_boxes, xml_path)

            # Рисуем на оригинале
            for box in final_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.imwrite(str(output_dir / f"pred_{filename}"), orig_img)
            print(f"[OK] {filename}: {len(final_boxes)} спор")
        else:
            print(f"[!] {filename}: ничего не найдено")

    print(f"\n[FINISH] Результаты в: {output_dir}")


if __name__ == "__main__":
    main()