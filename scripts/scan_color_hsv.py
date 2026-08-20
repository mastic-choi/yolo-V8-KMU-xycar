import os, glob, csv
import cv2
import numpy as np

POOLS = [
    ('dataset', '/Users/mastic-choi/code/fine-tune/dataset'),
    ('lap005', '/Users/mastic-choi/Downloads/lap_005'),
    ('lap001_3', '/Users/mastic-choi/Downloads/lap_001 3'),
]

LOWER = np.array([32, 100, 100])
UPPER = np.array([48, 255, 255])
MIN_PIXELS = 60  # 이 이상이면 후보

rows = []
for pool_name, pool_dir in POOLS:
    files = sorted(glob.glob(os.path.join(pool_dir, '*.png')))
    print(f'{pool_name}: {len(files)} frames')
    for i, f in enumerate(files):
        img = cv2.imread(f)
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER, UPPER)
        cnt = int(np.count_nonzero(mask))
        if cnt >= MIN_PIXELS:
            ys, xs = np.nonzero(mask)
            rows.append({
                'pool': pool_name, 'file': f, 'pixels': cnt,
                'cx': int(xs.mean()), 'cy': int(ys.mean()),
                'x0': int(xs.min()), 'x1': int(xs.max()),
                'y0': int(ys.min()), 'y1': int(ys.max()),
            })
        if i % 500 == 0:
            print(f'  {i}/{len(files)}, candidates so far: {len(rows)}')

out_csv = os.path.join(os.path.dirname(__file__), 'green_scan_result.csv')
with open(out_csv, 'w', newline='') as fp:
    w = csv.DictWriter(fp, fieldnames=['pool','file','pixels','cx','cy','x0','x1','y0','y1'])
    w.writeheader()
    w.writerows(rows)
print(f'총 후보: {len(rows)} -> {out_csv}')
