#!/usr/bin/env python3
"""
signal_state v1.0.0(기존 739장 사람 라벨링 모델) vs v1.1.0(raw데이터 세션 매핑으로
보강한 7,377장 재학습 모델)을 같은 프레임에 나란히 돌려서 위/아래 2행(before/after)
몽타주를 만든다. v1.0.0이 특히 green_straight/green_left를 헷갈리던 프레임 위주로 선정.
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO

V1_MODEL = '/home/foscar/yolo_train_env/runs/detect/train-4/weights/2차학습/best.pt'
V11_MODEL = '/home/foscar/yolo-V8-traffic_light-finetune/runs/signal_state_v2/weights/best.pt'

# (파일, 실제 정답) — val split에서 선정, 세션 매핑이 곧 정답
SAMPLES = [
    ('/home/foscar/yolo-V8-traffic_light-finetune/data/merged_dataset/images/val/20260820_225835_001979_1787234381.863.jpg', 'green_straight'),
    ('/home/foscar/yolo-V8-traffic_light-finetune/data/merged_dataset/images/val/20260820_230011_001808_1787234472.090.jpg', 'red'),
    ('/home/foscar/yolo-V8-traffic_light-finetune/data/merged_dataset/images/val/20260820_230136_002107_1787234567.038.jpg', 'green_left'),
    ('/home/foscar/yolo-V8-traffic_light-finetune/data/merged_dataset/images/val/20260820_225835_002200_1787234389.267.jpg', 'green_straight'),
]
OUT_PATH = '/home/foscar/yolo-V8-KMU-xycar/signal_state/docs/before_after_montage.jpg'
THUMB_W = 300

v1_model = YOLO(V1_MODEL)
v11_model = YOLO(V11_MODEL)


def draw(img, boxes_data, color, gt):
    img = img.copy()
    for cls_name, conf, (x1, y1, x2, y2) in boxes_data:
        ok = (cls_name == gt)
        box_color = (0, 200, 0) if ok else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 2)
        text = f'{cls_name} {conf:.2f}' + ('' if ok else ' (X)')
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        ty = max(th + 6, y1)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 6, ty), box_color, -1)
        cv2.putText(img, text, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def add_banner(img, text, color):
    banner_h = 26
    banner = np.full((banner_h, img.shape[1], 3), color, dtype=np.uint8)
    cv2.putText(banner, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([banner, img])


top_row, bottom_row = [], []
for path, gt in SAMPLES:
    img = cv2.imread(path)
    h, w = img.shape[:2]
    scale = THUMB_W / w
    img_small = cv2.resize(img, (THUMB_W, int(h * scale)))

    v1_res = v1_model.predict(img, conf=0.25, verbose=False)[0]
    v1_boxes = []
    for b in v1_res.boxes:
        x1, y1, x2, y2 = (b.xyxy[0].cpu().numpy() * scale).astype(int)
        cls_name = v1_res.names[int(b.cls[0])]
        v1_boxes.append((cls_name, float(b.conf[0]), (x1, y1, x2, y2)))
    before_img = draw(img_small, v1_boxes, None, gt)
    before_img = add_banner(before_img, f'BEFORE: v1.0.0 (GT={gt})', (60, 60, 180))

    v11_res = v11_model.predict(img, conf=0.25, verbose=False)[0]
    v11_boxes = []
    for b in v11_res.boxes:
        x1, y1, x2, y2 = (b.xyxy[0].cpu().numpy() * scale).astype(int)
        cls_name = v11_res.names[int(b.cls[0])]
        v11_boxes.append((cls_name, float(b.conf[0]), (x1, y1, x2, y2)))
    after_img = draw(img_small, v11_boxes, None, gt)
    after_img = add_banner(after_img, f'AFTER: v1.1.0 (GT={gt})', (30, 140, 30))

    top_row.append(before_img)
    bottom_row.append(after_img)

top = np.hstack(top_row)
bottom = np.hstack(bottom_row)
montage = np.vstack([top, bottom])

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
cv2.imwrite(OUT_PATH, montage)
print('saved:', OUT_PATH, montage.shape)
