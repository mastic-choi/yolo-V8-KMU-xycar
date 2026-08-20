# Bootstrap 2차 라운드 — 학습 PC(Ubuntu + RTX 3070 Ti)용 가이드

v1.0.0(`data/seed_labeled` 2,127장으로 학습)로 `data/temp/no_detection`(기존 COCO
모델이 못 잡던 4,214장)을 재검출한 결과, 93.3%(3,930장)를 새로 잡아냈다(신뢰도
중앙값 0.92). 이 중 사람 다리/발을 오검출한 프레임(16장)을 사람이 육안 검수로
제거하고 남은 **3,914장**을 2차 시드로 추가한다.

## 0. 왜 이 순서인가

- `conf≥0.5`로 자동 필터링하면 될 것 같았지만 실제로는 **conf 0.5~0.85 구간에도
  사람 오검출이 광범위하게 섞여 있고, 반대로 conf 0.86 근처엔 진짜 차량(사람이
  손으로 차를 조작하는 장면)도 많아서 신뢰도만으로는 못 걸렀다** — 그래서 이번에도
  전수 육안 검수(박스 위치만 확인, 사람이 배경에 있는 건 무관)를 거쳤다.
- 검수 기준은 "박스가 실제로 vehA(#46)를 감싸는가" 하나뿐. 차 옆에 사람이 같이
  나온 프레임은 문제없음(오히려 실전 시나리오에 가까워서 좋음).

## 1. 데이터 전송

맥에서 만든 `data/round2_new_detections.zip`(352MB, `images/`+`labels/` 각 3,914개)을
이 PC로 옮긴다(scp/USB/공유 드라이브 등 편한 방법으로). 예시:

```bash
scp mac:~/code/yolo-V8-KMU-xycar/target_vehicle/data/round2_new_detections.zip ~/yolo-V8-KMU-xycar/target_vehicle/data/
```

## 2. 기존 `seed_labeled`에 병합

```bash
cd ~/yolo-V8-KMU-xycar/target_vehicle
git pull                       # 맥에서 커밋한 labels/*.txt(라벨 텍스트만) 반영
unzip -q data/round2_new_detections.zip -d data/

cp -n data/round2_new_detections/images/* data/seed_labeled/images/
cp -n data/round2_new_detections/labels/* data/seed_labeled/labels/

# 파일명이 car1_/car2_ 접두어라 기존 세트와 겹칠 일은 없음. 개수만 확인.
ls data/seed_labeled/images | wc -l   # 2127 + 3914 = 6041 나와야 정상
ls data/seed_labeled/labels | wc -l
```

## 3. 노트북 수정 후 재학습

`notebooks/finetune_yolov8_local_rtx.ipynb`를 열고 학습 셀(§3)에서 두 가지만 바꾼다:

```python
BASE_WEIGHTS = os.path.join(RUNS_DIR, 'target_vehicle_v1', 'weights', 'best.pt')  # yolov8n.pt 대신 v1 체크포인트에서 이어학습
...
model.train(
    ...
    name='target_vehicle_v2',   # 'target_vehicle_v1' -> v2 (기존 결과 덮어쓰지 않도록)
    ...
)
```

나머지 셀(데이터 준비, 검증, export)은 그대로 실행. 데이터가 3배 가까이 늘었으니
`patience`는 그대로 두되 학습 시간은 더 걸릴 수 있다.

## 4. 검증 시 반드시 확인할 것

- v1.0.0 릴리즈 노트에 적어뒀듯 **train/val을 프레임 단위로 랜덤 분할**하고
  있어서(§1 코드 그대로) val mAP 숫자를 과신하지 말 것. 데이터가 늘었어도 이
  분할 방식 자체는 안 고쳤으니 이번에도 마찬가지. 정말 일반화 성능을 보려면
  `data/temp/no_detection`에서 이번에도 여전히 안 잡히는 케이스(전체 4,214장 중
  나머지 ~284장)로 다시 확인하는 게 더 정확하다.
- **`model.export(..., nms=True)`가 실제로 raw(NMS 미적용) 출력을 냈던 문제**를
  이번엔 export 직후 바로 확인할 것:
  ```python
  import onnxruntime as ort
  sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
  print(sess.get_outputs()[0].shape)   # [1, N, 6]이어야 함 (nms 적용됨)
                                        # [1, 5, 8400]이면 아직도 안 먹힌 것 - track_drive에 못 씀
  ```
  안 먹혔다면 ultralytics 버전을 확인하거나, `nms=True` 대신 export 후 별도로
  onnx-graphsurgeon 등으로 NMS 노드를 붙이는 방법을 찾아야 한다.

## 5. 결과 공유

학습 끝나면 맥에서 했던 것과 동일하게 GitHub Release로 올린다:

```bash
gh release create v2.0.0 \
  ~/umk_yolo_vehicle/runs/target_vehicle_v2/weights/best.onnx \
  --repo mastic-choi/yolo-V8-KMU-xycar \
  --title "v2.0.0 — YOLOv8n target_vehicle (seed_labeled 6041장, round2 bootstrap)" \
  --notes "여기에 최종 mAP, epoch, 학습시간, nms export 확인결과 적을 것"
```

## 6. 그 다음 (3차 라운드가 필요하면)

`data/temp/no_detection`에서 이번에도 안 잡힌 나머지(~284장 근방)는 진짜 어려운
케이스다 — 자동 검출로는 더 못 늘리니, 이제부터는 CVAT에 올려서 사람이 직접
박스를 그려야 한다(원래 README 방법론 5번 단계).
