#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
signal_state YOLOv8 증분 파인튜닝 (기존 best.pt에서 이어학습).
기존 739장 -> 병합 7,377장(seed + raw데이터 session 매핑 라벨).
"""
import os
os.environ['WANDB_MODE'] = 'disabled'

from ultralytics import YOLO

BASE_WEIGHTS = '/home/foscar/yolo_train_env/runs/detect/train-4/weights/2차학습/best.pt'
DATA_YAML = '/home/foscar/yolo-V8-traffic_light-finetune/data/merged_dataset/data.yaml'
RUNS_DIR = '/home/foscar/yolo-V8-traffic_light-finetune/runs'

model = YOLO(BASE_WEIGHTS)
model.train(
    data=DATA_YAML,
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    seed=42,
    patience=20,
    project=RUNS_DIR,
    name='signal_state_v2',
)

metrics = model.val()
print('mAP50:', metrics.box.map50)
print('mAP50-95:', metrics.box.map)

best_pt = model.trainer.best
best_model = YOLO(best_pt)
onnx_path = best_model.export(format='onnx', imgsz=640, opset=12, simplify=True, nms=True)
print('ONNX 저장 위치:', onnx_path)
