#!/usr/bin/env python3
"""
기존 1027_40epoch.pt 신호등 검출기로 raw_pool 전체를 스캔해서
박스+클래스+신뢰도를 CSV로 저장한다. (yolo-V8-KMU-xycar/scripts/scan_dedicated_capture.py 패턴)
"""
import os, glob, csv
from ultralytics import YOLO

MODEL_PATH = '/home/foscar/yolo_train_env/runs/detect/train-4/weights/2차학습/best.pt'
POOL_DIR = '/home/foscar/yolo-V8-traffic_light-finetune/data/raw_pool'
OUT_CSV = '/home/foscar/yolo-V8-traffic_light-finetune/data/scan_result.csv'
CONF_MIN = 0.1
BATCH_SIZE = 64

model = YOLO(MODEL_PATH)
names = model.names
print('classes:', names)

files = sorted(glob.glob(os.path.join(POOL_DIR, '*', '*.jpg')))
print(f'total frames: {len(files)}')

rows = []
for i in range(0, len(files), BATCH_SIZE):
    batch = files[i:i + BATCH_SIZE]
    results = model.predict(batch, verbose=False, conf=CONF_MIN, device=0)
    for f, r in zip(batch, results):
        boxes = r.boxes
        if len(boxes) == 0:
            rows.append({'file': f, 'ndet': 0, 'cls': -1, 'cls_name': '', 'conf': 0,
                         'x1': 0, 'y1': 0, 'x2': 0, 'y2': 0, 'img_w': r.orig_shape[1], 'img_h': r.orig_shape[0]})
            continue
        # 신뢰도 가장 높은 박스 하나를 대표로(신호등은 프레임당 보통 1개가 유의미)
        best_idx = int(boxes.conf.argmax())
        conf = float(boxes.conf[best_idx])
        cls_id = int(boxes.cls[best_idx])
        x1, y1, x2, y2 = boxes.xyxy[best_idx].tolist()
        rows.append({'file': f, 'ndet': len(boxes), 'cls': cls_id, 'cls_name': names[cls_id],
                     'conf': conf, 'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                     'img_w': r.orig_shape[1], 'img_h': r.orig_shape[0]})
    if (i // BATCH_SIZE) % 10 == 0:
        n_hit = sum(1 for x in rows if x['ndet'] > 0)
        print(f'  {i}/{len(files)}  검출성공 {n_hit}/{len(rows)}')

with open(OUT_CSV, 'w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=['file', 'ndet', 'cls', 'cls_name', 'conf', 'x1', 'y1', 'x2', 'y2', 'img_w', 'img_h'])
    w.writeheader()
    w.writerows(rows)

n_hit = sum(1 for x in rows if x['ndet'] > 0)
print(f'\n총 {len(rows)}장 중 검출 {n_hit}장({n_hit/len(rows)*100:.1f}%) -> {OUT_CSV}')

from collections import Counter
c = Counter(x['cls_name'] for x in rows if x['ndet'] > 0)
for k, v in c.most_common():
    print(f'  {k}: {v}')
