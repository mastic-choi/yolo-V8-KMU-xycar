#!/usr/bin/env python3
"""
ultralytics 8.3.0의 `model.export(format='onnx', nms=True)`는 일반 DetectionModel
(YOLOv8n 등, end2end 아닌 모델)에는 사실상 적용되지 않는다(CoreML 전용 옵션이라 ONNX엔
안 먹힘 — export_onnx()에 self.args.nms 참조 자체가 없음, exporter.py 확인 결과).

그래서 NMS를 직접 그래프에 넣는 wrapper를 만들어 export한다 — torchvision.ops.batched_nms
(ONNX opset>=11에서 심볼릭 지원)를 forward()에서 호출해, 최종 output0을
track_drive가 기대하는 [x1,y1,x2,y2,conf,cls] 6열 포맷으로 직접 뱉게 만든다.
"""
import argparse
import torch
import torch.nn as nn
import torchvision
from ultralytics import YOLO
from ultralytics.utils.ops import xywh2xyxy


class NMSWrapper(nn.Module):
    def __init__(self, model, conf_thres=0.25, iou_thres=0.45, max_det=100):
        super().__init__()
        self.model = model
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det

    def forward(self, x):
        pred = self.model(x)[0]  # [1, 4+nc, N] — box(cxcywh) + per-class conf(sigmoid 적용됨)
        pred = pred.transpose(1, 2)[0]  # [N, 4+nc]

        boxes = xywh2xyxy(pred[:, :4])
        scores_all = pred[:, 4:]
        conf, cls = scores_all.max(-1)

        mask = conf > self.conf_thres
        boxes, conf, cls = boxes[mask], conf[mask], cls[mask].float()

        keep = torchvision.ops.batched_nms(boxes, conf, cls, self.iou_thres)
        keep = keep[: self.max_det]

        out = torch.cat([boxes[keep], conf[keep].unsqueeze(-1), cls[keep].unsqueeze(-1)], dim=-1)  # [M, 6]
        return out.unsqueeze(0)  # [1, M, 6]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--opset', type=int, default=12)
    args = ap.parse_args()

    yolo = YOLO(args.weights)
    base_model = yolo.model.eval().cpu()
    wrapped = NMSWrapper(base_model, conf_thres=args.conf, iou_thres=args.iou).eval()

    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(
        wrapped,
        dummy,
        args.out,
        input_names=['images'],
        output_names=['output0'],
        opset_version=args.opset,
        dynamic_axes={
            'images': {0: 'batch', 2: 'height', 3: 'width'},
            'output0': {0: 'batch', 1: 'num_dets'},
        },
        do_constant_folding=True,
    )
    print('saved:', args.out)


if __name__ == '__main__':
    main()
