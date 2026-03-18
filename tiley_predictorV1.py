import cv2
import os
import torch
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from ultralytics import YOLO
from torchvision.ops import nms


def save_pascal_voc(filename, img_shape, bboxes, output_path):
    # ИСПРАВЛЕНО: распаковываем высоту и ширину из img_shape
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--source', type=str, required=True)
    parser.add_argument('--imgsz', type=int, default=1024)
    # По умолчанию None, чтобы потом приравнять к imgsz
    parser.add_argument('--tile_size', type=int, default=None)
    parser.add_argument('--conf', type=float, default=0.45)
    parser.add_argument('--overlap', type=float, default=0.15)
    parser.add_argument('--iou', type=float, default=0.3)
    args = parser.parse_args()

    # ЛОГИКА: если тайл не задан, берем imgsz
    if args.tile_size is None:
        args.tile_size = args.imgsz

    output_dir = Path("runs/detect/results") / f"clean_prediction_{args.conf}"
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    stride = int(args.tile_size * (1 - args.overlap))

    files = [f for f in os.listdir(args.source) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Обработка {len(files)} файлов из {args.source}...")

    for filename in files:
        img_path = os.path.join(args.source, filename)
        img = cv2.imread(img_path)
        if img is None: continue
        h, w, c = img.shape

        # Твоя логика расчета позиций тайлов
        y_positions = [min(j * stride, h - args.tile_size) for j in range(max(1, (h - args.tile_size) // stride + 1))]
        x_positions = [min(j * stride, w - args.tile_size) for j in range(max(1, (w - args.tile_size) // stride + 1))]

        boxes_list = []
        scores_list = []

        for y1 in set(y_positions):
            for x1 in set(x_positions):
                tile = img[y1:y1 + args.tile_size, x1:x1 + args.tile_size]
                results = model.predict(tile, imgsz=args.imgsz, conf=args.conf, verbose=False)

                for box in results[0].boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    boxes_list.append([xyxy[0] + x1, xyxy[1] + y1, xyxy[2] + x1, xyxy[3] + y1])
                    scores_list.append(box.conf.item())

        if boxes_list:
            boxes_tensor = torch.tensor(boxes_list)
            scores_tensor = torch.tensor(scores_list)

            keep = nms(boxes_tensor, scores_tensor, args.iou)
            final_boxes = boxes_tensor[keep]

            xml_path = output_dir / (Path(filename).stem + ".xml")
            save_pascal_voc(filename, (h, w, c), final_boxes, xml_path)

            for box in final_boxes:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imwrite(str(output_dir / f"pred_{filename}"), img)
            print(f"Файл {filename}: Найдено {len(final_boxes)} спор")
        else:
            print(f"Файл {filename}: Споры не найдены")


if __name__ == "__main__":
    main()