#!/usr/bin/env python3
"""
raw_pool 세션 폴더 = 촬영 시 고정해둔 신호 상태(사람이 확인한 사실)이므로,
기존(분류를 잘 못하는) 모델의 예측 클래스는 버리고 세션->클래스 매핑으로 라벨을 만든다.
바운딩 박스 위치(x1,y1,x2,y2)는 detector의 localization 결과를 그대로 신뢰해서 재사용한다
(문제는 분류였지, 박스 위치 검출 성공률은 100%였음).

기존 final_dataset(739장, 시드)과 합쳐 새 데이터셋을 만든다.
세션(시간 구간) 단위로 train/val을 나눠서 인접 프레임 누수를 최소화한다.
"""
import os, csv, shutil

SCAN_CSV = '/home/foscar/yolo-V8-traffic_light-finetune/data/scan_result.csv'
EXISTING_DATASET = '/home/foscar/yolo_train_env/final_dataset'
OUT_DATASET = '/home/foscar/yolo-V8-traffic_light-finetune/data/merged_dataset'
VAL_RATIO = 0.15

CLASS_NAMES = ['red', 'green_straight', 'green_left']
# 세션 -> (class_id, 사용할 최소 프레임 번호[포함])
SESSION_LABEL = {
    '20260820_225835': (1, 250),   # green_straight, 초반 ~220프레임은 사람이 가려서 제외
    '20260820_230011': (0, 0),     # red
    '20260820_230136': (2, 0),     # green_left
}

for split in ('train', 'val'):
    os.makedirs(os.path.join(OUT_DATASET, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(OUT_DATASET, 'labels', split), exist_ok=True)

# 1) 기존 시드 데이터(739장) 그대로 복사
n_seed = {'train': 0, 'val': 0}
for split in ('train', 'val'):
    src_img_dir = os.path.join(EXISTING_DATASET, 'images', split)
    src_lbl_dir = os.path.join(EXISTING_DATASET, 'labels', split)
    for fname in os.listdir(src_img_dir):
        shutil.copy2(os.path.join(src_img_dir, fname), os.path.join(OUT_DATASET, 'images', split, fname))
        stem = os.path.splitext(fname)[0]
        lbl_src = os.path.join(src_lbl_dir, stem + '.txt')
        if os.path.isfile(lbl_src):
            shutil.copy2(lbl_src, os.path.join(OUT_DATASET, 'labels', split, stem + '.txt'))
        n_seed[split] += 1
print('기존 시드 복사:', n_seed)

# 2) 신규 데이터: 박스는 detector 결과, 클래스는 세션 매핑으로 강제
rows = list(csv.DictReader(open(SCAN_CSV)))
kept = []
skipped_no_det = 0
skipped_offphase = 0
for r in rows:
    session = r['file'].split('/')[-2]
    if session not in SESSION_LABEL:
        continue
    if int(r['ndet']) == 0:
        skipped_no_det += 1
        continue
    frame_idx = int(os.path.basename(r['file']).split('_')[0])
    cls_id, min_frame = SESSION_LABEL[session]
    if frame_idx < min_frame:
        skipped_offphase += 1
        continue
    kept.append((r, session, cls_id))

print(f'검출 실패로 제외: {skipped_no_det}, 초반 미점등 구간 제외: {skipped_offphase}, 사용: {len(kept)}')

by_session = {}
for r, session, cls_id in kept:
    by_session.setdefault(session, []).append((r, cls_id))

n_new = {'train': 0, 'val': 0}
for session, items in by_session.items():
    items.sort(key=lambda t: t[0]['file'])
    n_val = max(1, int(len(items) * VAL_RATIO))
    split_map = [('train', items[:-n_val]), ('val', items[-n_val:])]
    for split, group in split_map:
        for r, cls_id in group:
            src = r['file']
            fname = f'{session}_{os.path.basename(src)}'
            dst_img = os.path.join(OUT_DATASET, 'images', split, fname)
            shutil.copy2(src, dst_img)

            img_w, img_h = float(r['img_w']), float(r['img_h'])
            x1, y1, x2, y2 = float(r['x1']), float(r['y1']), float(r['x2']), float(r['y2'])
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h

            stem = os.path.splitext(fname)[0]
            with open(os.path.join(OUT_DATASET, 'labels', split, stem + '.txt'), 'w') as f:
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            n_new[split] += 1

print('신규 라벨 추가:', n_new)

data_yaml = f"""path: {OUT_DATASET}
train: images/train
val: images/val
nc: {len(CLASS_NAMES)}
names: {CLASS_NAMES}
"""
with open(os.path.join(OUT_DATASET, 'data.yaml'), 'w') as f:
    f.write(data_yaml)

total_train = n_seed['train'] + n_new['train']
total_val = n_seed['val'] + n_new['val']
print(f'\n최종 데이터셋: train {total_train}장, val {total_val}장 -> {OUT_DATASET}')
