#!/usr/bin/env python3
"""
COCO 사전학습 yolov8n.pt(범용 car 클래스) vs 파인튜닝된 target_vehicle(v2.0.0)을
같은 프레임 4장에 나란히 돌려서 위/아래 2행(before/after) 몽타주를 만든다.
"""
import cv2
import numpy as np
from ultralytics import YOLO

BASE_MODEL = '/home/foscar/umk_yolo_vehicle/yolov8n.pt'
FT_MODEL = '/home/foscar/umk_yolo_vehicle/runs/target_vehicle_v2/weights/best.pt'
COCO_CAR_CLASS = 2

SAMPLES = [
    '/home/foscar/yolo-V8-KMU-xycar/target_vehicle/data/montage_samples/004947_1787224054.912.jpg',
    '/home/foscar/yolo-V8-KMU-xycar/target_vehicle/data/montage_samples/001721_1787223946.787.jpg',
    '/home/foscar/yolo-V8-KMU-xycar/target_vehicle/data/montage_samples/002081_1787224250.457.jpg',
    '/home/foscar/yolo-V8-KMU-xycar/target_vehicle/data/montage_samples/000150_1787224185.737.jpg',
]
OUT_PATH = '/home/foscar/yolo-V8-KMU-xycar/target_vehicle/docs/before_after_montage.jpg'

base_model = YOLO(BASE_MODEL)
ft_model = YOLO(FT_MODEL)

THUMB_W = 300


def draw(img, boxes_data, color, label_prefix):
    img = img.copy()
    for cls_name, conf, (x1, y1, x2, y2) in boxes_data:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f'{cls_name} {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty = max(th + 6, y1)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), color, -1)
        cv2.putText(img, text, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def add_banner(img, text, color):
    banner_h = 26
    banner = np.full((banner_h, img.shape[1], 3), color, dtype=np.uint8)
    cv2.putText(banner, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, img])


top_row, bottom_row = [], []
for path in SAMPLES:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    scale = THUMB_W / w
    img_small = cv2.resize(img, (THUMB_W, int(h * scale)))

    # Before: COCO yolov8n, car class만
    base_res = base_model.predict(img, conf=0.25, classes=[COCO_CAR_CLASS], verbose=False)[0]
    base_boxes = []
    for b in base_res.boxes:
        x1, y1, x2, y2 = (b.xyxy[0].cpu().numpy() * scale).astype(int)
        base_boxes.append(('car(COCO)', float(b.conf[0]), (x1, y1, x2, y2)))
    before_img = draw(img_small, base_boxes, (0, 0, 255), 'before')
    before_img = add_banner(before_img, f'BEFORE: yolov8n COCO ({len(base_boxes)} det)', (60, 60, 180))

    # After: 파인튜닝 target_vehicle
    ft_res = ft_model.predict(img, conf=0.25, verbose=False)[0]
    ft_boxes = []
    for b in ft_res.boxes:
        x1, y1, x2, y2 = (b.xyxy[0].cpu().numpy() * scale).astype(int)
        ft_boxes.append(('target_vehicle', float(b.conf[0]), (x1, y1, x2, y2)))
    after_img = draw(img_small, ft_boxes, (0, 200, 0), 'after')
    after_img = add_banner(after_img, f'AFTER: target_vehicle v2.0.0 ({len(ft_boxes)} det)', (30, 140, 30))

    top_row.append(before_img)
    bottom_row.append(after_img)

top = np.hstack(top_row)
bottom = np.hstack(bottom_row)
montage = np.vstack([top, bottom])

import os
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
cv2.imwrite(OUT_PATH, montage)
print('saved:', OUT_PATH, montage.shape)
