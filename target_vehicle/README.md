# target_vehicle — 방해차량(vehA #46) YOLOv8 파인튜닝

국민대 자율주행 경진대회용 `track_drive` 저장소(`~/orca/workspaces/UMK/...`)의
`yolo_vehicle.py`(방해차량 검출)가 신뢰도 0.45대로 낮게 나오는 문제를 고치기 위한
**전용 차량(방해차량) YOLOv8 파인튜닝** 프로젝트.

COCO 사전학습 `yolov8n.pt`의 범용 `car` 클래스를 그대로 쓰는 대신, 대회에서 실제로
회피해야 하는 **그 차량 한 대**(#46, TRAXXAS 검정/연두)의 뒷모습을 잘 잡도록
파인튜닝하는 것이 목표. 회피 시나리오상 우리 차가 뒤에서 접근하므로 뒷모습 검출이
1차 목표.

## Results

### Before / After — 같은 프레임 4장 비교

왼쪽부터 4장, 위=파인튜닝 전(COCO `yolov8n`의 범용 `car` 클래스), 아래=파인튜닝 후
(`target_vehicle` v1.1.0). COCO 모델은 4장 중 2장에서 놓치거나(0 det) 신뢰도가
낮은데(0.4~0.5대), 파인튜닝 모델은 4장 전부 0.9대 신뢰도로 검출.

![Before/After 비교](docs/before_after_montage.jpg)

### 정량 비교 (val split 기준)

| Model | 학습 데이터 | epoch | 학습 시간 | mAP50 | mAP50-95 | Precision | Recall |
|:-----:|:-----------:|:-----:|:---------:|:-----:|:--------:|:---------:|:------:|
| [v1.0.0](https://github.com/mastic-choi/yolo-V8-KMU-xycar/releases/tag/v1.0.0) | seed_labeled 2,127장 (train 1808/val 319) | 80 (best@50) | 10.5분 | 0.995 | 0.974 | 1.0 | 1.0 |
| **[v1.1.0](https://github.com/mastic-choi/yolo-V8-KMU-xycar/releases/tag/v1.1.0)** (round2 bootstrap) | seed+round2 6,041장 (train 5135/val 906) | 139 (best@76) | 49분 | 0.995 | **0.985** :arrow_up: | 1.0 | 1.0 |

RTX 3070 Ti 기준. v1.1.0이 v1.0.0 대비 mAP50-95 +0.011, 데이터는 3배(2,127→6,041장)
늘었지만 Precision/Recall은 이미 1.0이라 큰 차이는 mAP50-95(더 엄격한 IoU 기준)에서
드러남 — 데이터 다양성이 늘면서 박스 타이트니스가 개선된 것으로 보임.

**[v1.2.0](https://github.com/mastic-choi/yolo-V8-KMU-xycar/releases/tag/v1.2.0)**:
가중치는 v1.1.0과 동일(재학습 없음) — ONNX export만 NMS 내장 방식으로 수정
(아래 "알려진 함정" 참고). 표의 지표는 v1.1.0과 동일.

## 데이터 소스 (전부 로컬 경로, git에 이미지 자체는 안 올림)

| 풀 | 경로 | 장수 | 성격 |
|---|---|---|---|
| dataset | `~/code/fine-tune/dataset/` | 2123 | 원래 차선 파인튜닝용 원본 주행 프레임(차량은 우연히 등장) |
| lap_005 | `~/Downloads/lap_005/` | 2734 | 지그재그 보강 주행 프레임(차량은 우연히 등장) |
| ~~lap_001 3~~ | `~/Downloads/lap_001 3/` | 2262 | **`dataset`과 파일 내용이 완전히 동일함(MD5 일치, 2026-08-20 확인)** — 중복이라 스캔 제외 |
| 자동차1 | `~/Downloads/20260820/자동차1/` | 7223 | **차량 전용 촬영**(사람이 손으로 각도·거리 바꿔가며 촬영) |
| 자동차2 | `~/Downloads/20260820/자동차2/` | 3061 | 위와 동일 성격, 두 번째 세션 — car1/car2는 촬영 세션 구분일 뿐 **같은 차량 한 대**, 라벨 클래스는 통합 |

`dataset`/`lap_005`는 차량이 "우연히" 찍힌 프레임을 찾아야 해서 스캔 후 사람이
골라내는 과정이 필요했다. `자동차1`/`자동차2`(총 10,284장)는 거의 모든 프레임에
차량이 있어 이번 1차 학습의 실제 데이터 소스가 됐다 — 아래 `data/seed_labeled`가
그 결과물.

**`data/seed_labeled/`(6,041장, git엔 `labels/*.txt`만 커밋)** — 지금까지 실제로
쓴 최종 학습 데이터(v1.0.0 시드 2,127 + round2 3,914). `images/`는 로컬에만 있음.

## 방법론 — bootstrap(반복) 라벨링, CVAT 없이 진행

`fine-tune` 저장소가 TwinLiteNet 차선 모델을 파인튜닝할 때 썼던 것과 같은 큰 틀
(자동 라벨→사람 보정→재학습 반복)을 재사용하되, 원래 계획했던 CVAT 수작업 없이
**"자동검출 → 사람이 틀린 것만 삭제"** 방식으로 진행했다 — 처음부터 박스를 그리는
것보다 훨씬 빨랐다.

### 1차 라운드 (완료, [v1.0.0](https://github.com/mastic-choi/yolo-V8-KMU-xycar/releases/tag/v1.0.0))

1. **자동 검출**: COCO `yolov8n.pt`로 자동차1/2 전체(10,284장) 스캔
   (`scripts/scan_dedicated_capture.py`) → 검출 6,070장(59%), `conf≥0.5` 2,181장 /
   `conf≥0.7` 903장.
2. **사람이 "틀린 것만" 삭제**: 박스를 그려서 `data/temp/{conf_ge_0.7,conf_ge_0.5,
   wrong_vehicle_suspect,no_detection}`로 분류 → 사람이 폴더를 훑으며 오검출만
   지움(그리기 없음, 삭제만). `wrong_vehicle_suspect`는 색상 휴리스틱(연두=vehA/
   빨강=vehB)으로 걸러낸 의심 후보였는데, 실제로는 대부분 정상 검출로 확인됨(색상
   판별기 자체가 부정확했음 — 아래 "알려진 함정" 참고).
3. **라벨 자동 생성**: 검수 후 남은 파일들의 CSV 박스 좌표를 그대로 YOLO 포맷으로
   변환(정규화 `cx,cy,w,h`, 클래스 0 하나) → `data/seed_labeled/` 2,127장 완성.
   사람이 박스를 직접 그린 적은 한 번도 없음 — "그리기"는 COCO 모델이, "판단"만
   사람이 함.
4. **파인튜닝**: `notebooks/finetune_yolov8_local_rtx.ipynb`(Ubuntu + RTX 3070 Ti),
   `yolov8n.pt`에서 시작, train 1808/val 319. Best epoch 50, patience 30으로
   epoch 80 조기종료(총 10.5분). val mAP50 0.995, mAP50-95 0.974.
   - **주의**: train/val을 프레임 단위로 랜덤 분할해서, 연속 촬영 영상이라 이웃
     프레임끼리 train/val에 나뉘어 들어갔을 수 있음(사실상 같은 장면) — 이 val
     숫자를 곧이곧대로 믿지 말 것.

### 2차 라운드 (완료, [v1.1.0](https://github.com/mastic-choi/yolo-V8-KMU-xycar/releases/tag/v1.1.0))

5. **의사라벨(pseudo-label) 확장**: v1.0.0 모델로 `data/temp/no_detection`
   (4,214장, 기존 COCO가 못 잡던 것들)을 재검출 → 93.3%(3,930장)를 새로 잡아냄
   (신뢰도 중앙값 0.92). 사람 오검출(다리/발 등, 16장)만 육안 검수로 제거하고
   남은 3,914장을 2차 시드로 추가.
6. **재학습**: v1.0.0 체크포인트에서 이어학습(scratch 아님), 총 6,041장(train
   5135/val 906). Best epoch 76, patience 30으로 epoch 139 조기종료(49분).
   val mAP50 0.995, mAP50-95 **0.985**(v1.0.0 대비 +0.011).

### 3차 라운드 (필요 시)

`data/temp/no_detection`에서 v1.1.0으로도 안 잡힌 나머지(~284장 근방)는 진짜 어려운
케이스 — 자동 검출로는 더 못 늘리므로, 다음 라운드가 필요하면 CVAT에 올려서 사람이
직접 박스를 그려야 함. (옵션) SAM2 정밀화: `~/code/fine-tune/sam2.1_b.pt`로 YOLO
박스를 프롬프트 삼아 마스크를 뽑고, 그 마스크의 최소외접사각형을 최종 박스로 쓰면
더 타이트해진다.

## 스크립트

- `scripts/scan_coco_car.py` — COCO `yolov8n.pt`로 정적 프레임 풀에서 `car` 클래스
  검출, 결과를 CSV로 저장(신뢰도/박스좌표).
- `scripts/scan_color_hsv.py` — 차체 고유 연두색(HSV) 기반 스캔. YOLO가 놓치는
  "부분적으로 가려진" 프레임을 잡아내는 보조 수단(단, 주황 콘/초록 비상구 표지판
  등과 색이 겹쳐 오탐 많음 — 결과를 contact sheet로 반드시 육안 검토할 것).
- `scripts/scan_dedicated_capture.py` — 자동차1/2(전용 촬영) 전체를 스캔해 검출
  성공률/분포 파악용.
- `scripts/curate_seed_candidates.py` — 1차 세션에서 사람이 육안 확인한 뒷모습
  구간(dataset/lap_005 프레임 범위)을 `data/candidates/`로 복사해 재현.
- `scripts/make_before_after_montage.py` — 위 Results 몽타주를 만드는 스크립트
  (COCO yolov8n vs 파인튜닝 v1.1.0, 같은 프레임 4장 비교).
- `scripts/export_onnx_with_nms.py` — [v1.2.0](https://github.com/mastic-choi/yolo-V8-KMU-xycar/releases/tag/v1.2.0)
  에서 추가. `model.export(nms=True)`가 안 먹히는 문제(아래 "알려진 함정") 해결용 —
  `torchvision.ops.batched_nms`를 forward()에 심은 wrapper로 직접 재학습 없이
  같은 `best.pt`를 NMS 내장 ONNX로 재export.

## 알려진 함정 (실제로 겪은 것들)

- **작은 썸네일(contact sheet)만 보고 판단하면 놓친다** — 의자 카트를 차로
  오인하거나, 반대로 카트에 가려진 차량 뒷부분을 놓치는 사례가 실제로 있었다.
  애매하면 반드시 원본 해상도로 재확인할 것.
- **색상 기반 스캔은 오탐이 많다** — 주황 라바콘, 초록 "EXIT" 표지판, 목재 패널이
  차체의 연두색 HSV 범위와 겹친다. 후보 목록을 그대로 믿지 말고 contact sheet로
  걸러낼 것.
- **주차 스팟(정지 상태) 프레임은 매 랩마다 중복 등장** — 같은 차량이 트랙 옆
  같은 자리에 세워져 있어 여러 랩 영상에 반복 출현한다. 실제 "통과 중" 장면과
  구분해서 과대표집되지 않게 주의.
- **색상 기반 오탐 판별기도 틀릴 수 있다** — `wrong_vehicle_suspect`로 격리했던
  116장 중 62장이 사람 검수 결과 실제로는 정상 검출이었다(반사광/그림자로 HSV
  판정이 흔들림). 자동 필터는 후보를 줄이는 용도로만 쓰고, 최종 판단은 항상
  사람이 원본을 보고 할 것.
- **`model.export(format='onnx', nms=True)`가 항상 먹히는 게 아니다 — (v1.2.0에서
  해결)** v1.0.0/v1.1.0 `best.onnx` 둘 다 output shape이 `[1,5,8400]`(NMS 미적용
  raw 출력)이었다. 원인은 ultralytics 8.3.0의 `export_onnx()`가 `self.args.nms`를
  아예 참조하지 않는다는 것 — `nms=True`는 CoreML export 전용 옵션이고, 일반
  `DetectionModel`(yolov8n 등)의 ONNX export에는 적용되지 않는다(`end2end` 모델만
  예외). **해결**: `scripts/export_onnx_with_nms.py`로 `torchvision.ops.batched_nms`를
  forward()에 심은 wrapper를 만들어 재export — 재학습 없이 같은 `best.pt`로
  [v1.2.0](https://github.com/mastic-choi/yolo-V8-KMU-xycar/releases/tag/v1.2.0)
  `best_nms.onnx`(output0 `[1,N,6]` = `[x1,y1,x2,y2,conf,cls]`) 생성 완료.
- **val 숫자는 세션 랜덤 분할이라 과대평가 가능** — 두 라운드 다 프레임 단위 랜덤
  분할이라 인접 프레임이 train/val에 나뉘어 들어갈 수 있음. `signal_state` 프로젝트
  에서는 이 교훈을 반영해 세션(시간 구간) 단위로 분할함.
