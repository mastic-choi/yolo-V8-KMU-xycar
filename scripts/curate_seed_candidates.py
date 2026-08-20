#!/usr/bin/env python3
"""
1차 세션에서 눈으로 확인한 vehA(#46 TRAXXAS 검정/연두) 뒷모습 구간을
원본 raw 프레임 풀에서 복사해 시드 후보 폴더로 모은다.

이 스크립트가 하는 일은 "발견"이 아니라 "재현"이다 — 실제 프레임 범위는
사람이 contact sheet를 눈으로 보고 확정한 것(대화 세션 기록)이라, 새 데이터를
스캔하려면 scan_coco_car.py / scan_color_hsv.py / scan_dedicated_capture.py를
먼저 돌려서 후보를 뽑고, contact sheet로 검토한 뒤 이 스크립트의 RANGES를
갱신할 것.

출력: data/candidates/<label>/<tag>_frame_NNNNNN.<ext>
"""
import os
import shutil

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, 'data', 'candidates')

POOLS = {
    'dataset': '/Users/mastic-choi/code/fine-tune/dataset',
    'lap005': '/Users/mastic-choi/Downloads/lap_005',
    # NOTE: /Users/mastic-choi/Downloads/lap_001 3 는 dataset과 완전 동일 파일(MD5 일치,
    # 2026-08-20 확인) — 중복 소스이므로 스캔 대상에서 제외.
}

# (pool, lo, hi, tag, label) — label: 'clear_rear' | 'ambiguous_parked'
RANGES = [
    ('dataset', 24, 24, 'ds', 'clear_rear'),
    ('dataset', 196, 205, 'ds', 'clear_rear'),
    ('dataset', 280, 289, 'ds', 'clear_rear'),
    ('dataset', 686, 698, 'ds', 'clear_rear'),
    ('dataset', 1484, 1493, 'ds', 'clear_rear'),
    ('lap005', 345, 350, 'lap5', 'clear_rear'),
    ('lap005', 760, 792, 'lap5', 'clear_rear'),

    ('lap005', 818, 818, 'lap5', 'ambiguous_parked'),
    ('lap005', 3188, 3195, 'lap5', 'ambiguous_parked'),
    ('lap005', 1794, 1811, 'lap5', 'ambiguous_parked'),
    ('lap005', 2203, 2216, 'lap5', 'ambiguous_parked'),
    ('lap005', 2859, 2869, 'lap5', 'ambiguous_parked'),
    ('lap005', 1270, 1284, 'lap5', 'ambiguous_parked'),
]


def main():
    for pool, lo, hi, tag, label in RANGES:
        pool_dir = POOLS[pool]
        dest_dir = os.path.join(OUT_DIR, label)
        os.makedirs(dest_dir, exist_ok=True)
        for n in range(lo, hi + 1):
            fname = f'frame_{n:06d}.png'
            src = os.path.join(pool_dir, fname)
            if not os.path.isfile(src):
                continue
            dst = os.path.join(dest_dir, f'{tag}_{fname}')
            shutil.copy2(src, dst)
    for label in ('clear_rear', 'ambiguous_parked'):
        d = os.path.join(OUT_DIR, label)
        n = len(os.listdir(d)) if os.path.isdir(d) else 0
        print(f'{label}: {n}장 -> {d}')


if __name__ == '__main__':
    main()
