import os, glob, csv
from ultralytics import YOLO

MODEL_PATH = '/Users/mastic-choi/orca/workspaces/UMK/feature-lavacon-box-pairing-s1-start/yolo_ros/yolov8n.pt'
POOLS = [
    ('car1', '/Users/mastic-choi/Downloads/20260820/자동차1'),
    ('car2', '/Users/mastic-choi/Downloads/20260820/자동차2'),
]
CAR_CLASS = 2
CONF_MIN = 0.10

model = YOLO(MODEL_PATH)
out_csv = os.path.join(os.path.dirname(__file__), 'new_car_scan_result.csv')
rows = []
for pool_name, pool_dir in POOLS:
    files = sorted(glob.glob(os.path.join(pool_dir, '*.jpg')))
    print(f'{pool_name}: {len(files)} frames')
    batch_size = 32
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        results = model.predict(batch, verbose=False, conf=CONF_MIN, classes=[CAR_CLASS])
        for f, r in zip(batch, results):
            if len(r.boxes) == 0:
                rows.append({'pool': pool_name, 'file': f, 'conf': 0, 'x1':0,'y1':0,'x2':0,'y2':0,'w':0,'h':0,'ndet':0})
                continue
            # 가장 큰 박스(면적)만 대표로 기록
            best = max(r.boxes, key=lambda b: float((b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1])))
            conf = float(best.conf[0])
            x1,y1,x2,y2 = best.xyxy[0].tolist()
            rows.append({'pool': pool_name, 'file': f, 'conf': conf, 'x1':x1,'y1':y1,'x2':x2,'y2':y2,
                         'w': x2-x1, 'h': y2-y1, 'ndet': len(r.boxes)})
        if (i // batch_size) % 20 == 0:
            n_hit = sum(1 for x in rows if x['ndet']>0)
            print(f'  {i}/{len(files)}  검출성공 {n_hit}/{len(rows)}')

with open(out_csv, 'w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=['pool','file','conf','x1','y1','x2','y2','w','h','ndet'])
    w.writeheader()
    w.writerows(rows)
n_hit = sum(1 for x in rows if x['ndet']>0)
print(f'총 {len(rows)}장 중 검출 {n_hit}장 -> {out_csv}')
