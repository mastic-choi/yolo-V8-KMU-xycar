# signal_state — 신호등(signal_state) YOLOv8 파인튜닝

이 저장소의 두 번째 하위 프로젝트. `target_vehicle`(저장소 루트)이 방해차량 검출용이라면,
이건 실내 트랙에 설치된 소형 신호등 장치의 **점등 상태 분류**(red / green_straight /
green_left)가 목표. 기존 모델(`yolo_train_env`, 로컬 macOS 학습, 739장)이 특히
`green_straight` vs `green_left`를 자주 혼동해서, 데이터를 늘려 재학습하기로 함.

## Results

### Before / After — 같은 프레임 4장 비교

위=v1.0.0(기존 739장, 사람이 직접 라벨링), 아래=v1.1.0(raw데이터 세션 매핑으로
보강한 7,377장 재학습). 빨간 박스(X)=오분류, 초록 박스=정분류. v1.0.0이 실제로
`green_straight`를 `green_left`로 오분류하던 프레임 2장(1번/4번 컬럼)이 v1.1.0에서
전부 정확히 고쳐졌고, 원래 맞았던 `red`/`green_left`(2번/3번)도 신뢰도가 올라감.

![signal_state before/after](docs/before_after_montage.jpg)

### 정량 비교 (val split 기준)

| Model | 학습 데이터 | epoch | 학습 시간 | mAP50 | mAP50-95 | Precision | Recall |
|:-----:|:-----------:|:-----:|:---------:|:-----:|:--------:|:---------:|:------:|
| v1.0.0 (기존, 사람이 직접 라벨링) | 739장 (train 629/val 110) | 50 (best@27) | 39분 | 0.995 | 0.807 | 0.995 | 0.996 |
| **v1.1.0** (raw데이터 세션 매핑 보강) | 7,377장 (train 6273/val 1104) | 87 (best@67) | 37.7분 | 0.995 | **0.956** :arrow_up: | 1.0 | 1.0 |

RTX 3070 Ti 기준. 데이터를 10배(739→7,377장) 늘리면서도 학습 시간은 오히려 비슷
(patience 조기종료 덕분) — mAP50-95가 **0.807 → 0.956(+0.149)**로 크게 개선됐는데,
이는 몽타주에서 보이듯 실제로 헷갈리던 클래스(green_straight/green_left) 오분류가
줄어든 결과.

### 알려진 함정 — `nms=True` export가 여기서도 안 먹힘

`target_vehicle`이 겪었던 것과 똑같은 문제 재현: v1.1.0 `best.onnx`를
`onnxruntime`으로 열어보면 output shape이 `[1, 7, 8400]`(NMS 미적용 raw, 4 박스
좌표 + 클래스 3개)이다. 실차/ROS 노드에 바로 넣기 전 export를 다시 확인/수정할 것.

## 데이터 소스

| 풀 | 장수 | 성격 |
|---|---|---|
| 기존 시드 (`yolo_train_env/final_dataset`) | 739 | 최초 학습에 쓴 원본 라벨 세트 |
| `카메라 raw데이터.zip` (2026-08-20 촬영, 3세션) | 6,888 | 신호를 한 상태로 고정해두고 찍은 신규 raw 프레임 |

## 방법론 — 이번엔 detector 예측 클래스를 안 믿고 세션=정답으로 라벨링

`target_vehicle` 1차 라운드는 "COCO 모델이 박스+존재여부를 검출 → 사람이 오검출만 삭제"
방식이었음. 이번엔 다름 — **기존 모델 자체가 클래스 분류를 못 해서** 데이터를 보강하는
것이므로, 기존 모델의 예측 클래스를 pseudo-label로 쓰면 같은 오류를 그대로 학습해버림.

대신:
1. **박스 위치(localization)는 신뢰**: 기존 모델의 검출 성공률이 6,888장 중 6,886장
   (100%)이었고, 애초에 문제는 "박스를 못 찾는 것"이 아니라 "찾은 박스의 색상 상태를
   잘못 분류하는 것"이었음 → 박스 좌표만 그대로 재사용.
2. **클래스는 촬영 세션으로 강제 라벨링**: raw데이터 zip의 세션 폴더 3개가 신호 상태
   3개와 정확히 대응 — 촬영할 때 사람이 신호를 하나의 상태로 고정해두고 찍은 것.
   - `20260820_225835` → `green_straight` (앞 250프레임은 사람이 카메라 앞을 가려서
     장치가 안 보임 — 제외)
   - `20260820_230011` → `red`
   - `20260820_230136` → `green_left`
3. **검증**: 각 클래스에서 무작위 샘플을 뽑아 박스+라벨을 원본에 그려 육안 확인 —
   램프 위치가 항상 일관됨(1번=red, 3번=green_left, 4번=green_straight)을 확인함.

이렇게 만든 `data/seed_labeled/labels/`(train 6,273 / val 1,104, 기존 739장 포함
병합)로 재학습. 이미지 자체는 로컬에만 있음(리포 컨벤션대로 라벨 txt만 커밋).

## train/val 분할 시 주의

`target_vehicle` 1차 라운드가 프레임 단위 랜덤 분할로 val 수치를 과신하지 말라고 했던
것과 같은 이유로, 이번엔 **세션(시간 구간) 단위**로 분할함 — 각 세션을 프레임 번호순
정렬 후 앞 85% train / 뒤 15% val. 완전한 랜덤 셔플보다는 인접 프레임 누수가 훨씬 적음.

## 스크립트

- `scripts/scan_traffic_light.py` — 기존 `signal_state` best.pt로 raw_pool 전체를
  스캔해 박스 좌표+예측 신뢰도를 CSV로 저장 (`target_vehicle/scripts/
  scan_dedicated_capture.py`와 같은 패턴). **주의**: 여기서 나온 예측 클래스는
  신뢰하지 않음 — 박스 좌표만 다음 단계에서 재사용.
- `scripts/build_dataset.py` — CSV의 박스 좌표 + 세션→클래스 매핑으로 YOLO 라벨을
  만들고, 기존 시드와 병합해 `data.yaml`까지 생성.
- `train.py` — 기존 `signal_state` best.pt에서 이어학습(scratch 아님), epochs=100,
  patience=20, device=0(CUDA), 학습 후 검증+ONNX export(`nms=True`)까지 자동 실행.
  **`target_vehicle`가 실제로 겪었던 함정 참고**: export 직후 ONNX output shape이
  `[1,N,6]`(NMS 적용됨)인지 반드시 확인할 것 — 실제로 `[1,7,8400]`으로 안 먹혔음
  (위 "알려진 함정" 참고).
- `scripts/make_before_after_montage.py` — 위 Results 몽타주를 만드는 스크립트
  (v1.0.0 vs v1.1.0, 같은 프레임 4장에 GT 대비 정오답 색상 표시).
