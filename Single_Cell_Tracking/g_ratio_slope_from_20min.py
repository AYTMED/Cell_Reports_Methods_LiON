#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute average Ratio change rate (Ratio/frame) from the 20-min timepoint (t=1)
until the end of continuous tracking from t=1, for tracks that start at t=0 or t=1.

Inputs
  - track_output/tracks_long.csv  (must contain: track_id, t, red_sum, green_sum)

Outputs (default outdir=track_output/analysis)
  - ratio_slope_from_20min.csv

Definitions
  * We consider tracks whose first observed frame is 0 or 1.
  * For each eligible track, we find the longest **continuous** run starting at frame t=1
    (i.e., frames 1,2,3,...,L are all present). If L<=1 (no continuation),
    the slope is undefined and the track is skipped unless --keep-single is set.
  * Average change rate per frame = (ratio[L] - ratio[1]) / (L-1).
    We also report a per-hour value by multiplying by (60/min_per_frame).

Usage
  python ratio_slope_from_20min.py \
    --tracks track_output/tracks_long.csv \
    --outdir track_output/analysis \
    --frames 19 --interval-min 20
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser(description='Average Ratio slope from 20-min (t=1) to end of continuous tracking from t=1')
    ap.add_argument('--tracks', default='track_output/tracks_long.csv')
    ap.add_argument('--outdir', default='track_output/analysis')
    ap.add_argument('--frames', type=int, default=19, help='Total frames (default 19 for 0..18)')
    ap.add_argument('--interval-min', type=float, default=20.0, help='Minutes per frame (default 20)')
    ap.add_argument('--keep-single', dest='keep_single', action='store_true',
                    help='Keep tracks with only t=1 (no continuation); slope set to NaN')
    return ap.parse_args()


def longest_run_from_one(present: set[int], max_frame: int) -> int:
    """Return last frame L >= 1 of the longest consecutive run 1..L contained in `present`.
    If frame 1 is missing, return 0.
    """
    if 1 not in present:
        return 0
    L = 1
    while (L + 1) in present and (L + 1) <= max_frame - 1:
        L += 1
    return L


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_outdir(outdir)

    df = pd.read_csv(args.tracks)
    required = {'track_id', 't', 'red_sum', 'green_sum'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing columns in {args.tracks}: {required - set(df.columns)}')

    df['track_id'] = df['track_id'].astype(int)
    df['t'] = df['t'].astype(int)
    df['ratio'] = df['red_sum'] / (df['green_sum'] + 1e-12)

    rows = []
    for tid, g in df.groupby('track_id'):
        ts = sorted(g['t'].tolist())
        if not ts:
            continue
        start_frame = min(ts)
        if start_frame not in (0, 1):
            continue  # only tracks starting at 0 or 1

        present = set(ts)
        L = longest_run_from_one(present, args.frames)
        if L == 0:
            continue  # missing t=1 entirely

        if 1 not in present:
            continue
        r1 = float(g.loc[g['t'] == 1, 'ratio'].iloc[0])
        if L in present:
            rL = float(g.loc[g['t'] == L, 'ratio'].iloc[0])
        else:
            continue

        frames_cont = L - 1 + 1  # number of frames in [1..L]
        if L == 1 and not args.keep_single:
            # no continuation beyond t=1 → slope undefined; skip
            continue

        slope_per_frame = (rL - r1) / (L - 1) if L > 1 else float('nan')
        slope_per_hour = slope_per_frame * (60.0 / args.interval_min)
        duration_min = (L - 1) * args.interval_min

        rows.append({
            'track_id': tid,
            'start_frame': start_frame,
            'has_t1': True,
            'end_frame': L,
            'frames_cont_from_t1': frames_cont,
            'duration_min_from_t1': duration_min,
            'ratio_t1': r1,
            'ratio_t_end': rL,
            'avg_slope_ratio_per_frame': slope_per_frame,
            'avg_slope_ratio_per_hour': slope_per_hour,
        })

    out_csv = outdir / 'ratio_slope_from_20min.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print('[OK] wrote', out_csv)
    print('rows:', len(rows))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
