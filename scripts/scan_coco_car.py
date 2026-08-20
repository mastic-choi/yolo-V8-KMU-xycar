import os, glob, csv
from ultralytics import YOLO

MODEL_PATH = '/Users/mastic-choi/orca/workspaces/UMK/feature-lavacon-box-pairing-s1-start/yolo_ros/yolov8n.pt'
POOLS = [
    ('dataset', '/Users/mastic-choi/code/fine-tune/dataset'),
    ('lap005', '/Users/mastic-choi/code/fine-tune/lap_005_raw_2734'),
]
CAR_CLASS = 2  # COCO 'car'
CONF_MIN = 0.15  # 낮게 잡고 나중에 사람이 골라냄

model = YOLO(MODEL_PATH)

out_csv = os.path.join(os.path.dirname(__file__), 'car_scan_result.csv')
rows = []
for pool_name, pool_dir in POOLS:
    files = sorted(glob.glob(os.path.join(pool_dir, '*.png')))
    print(f'{pool_name}: {len(files)} frames')
    # 배치 추론
    batch_size = 32
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        results = model.predict(batch, verbose=False, conf=CONF_MIN, classes=[CAR_CLASS])
        for f, r in zip(batch, results):
            if len(r.boxes) == 0:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                rows.append({
                    'pool': pool_name, 'file': f, 'conf': conf,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'w': w, 'h': h,
                })
        if (i // batch_size) % 10 == 0:
            print(f'  {i}/{len(files)} done, candidates so far: {len(rows)}')

with open(out_csv, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['pool','file','conf','x1','y1','x2','y2','w','h'])
    writer.writeheader()
    writer.writerows(rows)

print(f'총 후보: {len(rows)}장 -> {out_csv}')
