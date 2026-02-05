#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Find start time and continuous duration where both duration length and the
number of tracks achieving it are maximized, under several reasonable criteria.

Inputs:
  - track_output/coverage/coverage_matrix.csv  (preferred, from coverage_by_start_time.py)
    OR
  - track_output/tracks_long.csv  (will compute coverage on the fly)

Outputs:
  - Prints a textual summary to stdout
  - Saves a compact CSV with the top picks at track_output/coverage/argmax_summary.csv

Selection criteria reported:
  1) Max COUNT first, tie-break by longest DURATION    → (s*, d*) maximizing count(s,d), then max d
  2) Max DURATION with at least one track (count>0)    → (s†, d†) maximizing d with count>0
  3) Max PRODUCT (duration * count)                    → (s‡, d‡) maximizing d*count (trade-off summary)

All times are reported in frames and converted to hours/minutes using interval.

Usage:
  python argmax_coverage_summary.py \
    --coverage track_output/coverage/coverage_matrix.csv \
    --tracks track_output/tracks_long.csv \
    --frames 19 --interval-min 20
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser(description='Summarize argmax of continuous coverage over start times and durations')
    ap.add_argument('--coverage', default='track_output/coverage/coverage_matrix.csv', help='CSV from coverage_by_start_time.py')
    ap.add_argument('--tracks', default='track_output/tracks_long.csv', help='tracks_long.csv (fallback if coverage CSV missing)')
    ap.add_argument('--frames', type=int, default=19)
    ap.add_argument('--interval-min', type=float, default=20.0)
    return ap.parse_args()


def build_from_tracks(tracks_csv: Path, frames: int, interval_min: float) -> pd.DataFrame:
    df = pd.read_csv(tracks_csv)
    if not {'track_id','t'}.issubset(df.columns):
        raise ValueError('tracks_long.csv must contain columns: track_id, t')
    df['t'] = df['t'].astype(int)

    # presence per track
    presence = {}
    for tid, g in df.groupby('track_id'):
        arr = np.zeros(frames, dtype=bool)
        ts = g['t'].to_numpy(int)
        ts = ts[(ts>=0)&(ts<frames)]
        arr[ts] = True
        presence[int(tid)] = arr

    hours = np.arange(frames) * (interval_min/60.0)

    # durations: all possible run lengths 1..frames
    durations_frames = np.arange(1, frames+1, dtype=int)
    durations_min = durations_frames * interval_min

    # counts matrix
    counts = np.zeros((len(durations_frames), frames), dtype=int)
    for s in range(frames):
        # run length for each track from s
        runlens = []
        for arr in presence.values():
            if s < len(arr) and arr[s]:
                # count consecutive True
                i = s
                while i < len(arr) and arr[i]:
                    i += 1
                runlens.append(i-s)
            else:
                runlens.append(0)
        runlens = np.asarray(runlens, int)
        for j, req in enumerate(durations_frames):
            counts[j, s] = int((runlens >= req).sum())

    # to long-form like coverage_matrix.csv
    recs=[]
    for j, req in enumerate(durations_frames):
        for s in range(frames):
            recs.append({
                'start_frame': s,
                'start_h': hours[s],
                'duration_min': float(durations_min[j]),
                'required_frames': int(req),
                'count': int(counts[j, s]),
            })
    return pd.DataFrame(recs)


def choose_argmax(df: pd.DataFrame, interval_min: float):
    # 1) Max count, tie-break by longest duration
    idx_max_c = df['count'].idxmax()
    max_count = int(df.loc[idx_max_c, 'count'])
    df_maxc = df[df['count'] == max_count]
    idx1 = df_maxc['required_frames'].idxmax()
    pick1 = df.loc[idx1]

    # 2) Max duration with count>0
    df_pos = df[df['count'] > 0]
    if len(df_pos):
        idx2 = df_pos['required_frames'].idxmax()
        pick2 = df_pos.loc[idx2]
    else:
        pick2 = None

    # 3) Max product (duration * count)
    df = df.copy()
    df['product'] = df['required_frames'] * df['count']
    idx3 = df['product'].idxmax()
    pick3 = df.loc[idx3]

    return pick1, pick2, pick3


def main():
    args = parse_args()
    cov_path = Path(args.coverage)
    if cov_path.exists():
        cov = pd.read_csv(cov_path)
    else:
        cov = build_from_tracks(Path(args.tracks), args.frames, args.interval_min)

    # Normalize columns
    for col in ['start_frame','required_frames','count']:
        cov[col] = cov[col].astype(int)

    # Pick
    p1, p2, p3 = choose_argmax(cov, args.interval_min)

    # Assemble summary
    rows = []
    def row_from_pick(tag, p):
        if p is None: return None
        return {
            'criterion': tag,
            'start_frame': int(p['start_frame']),
            'start_h': float(p['start_h']),
            'required_frames': int(p['required_frames']),
            'duration_min': float(p['required_frames']*args.interval_min),
            'count': int(p['count'])
        }
    rows.append(row_from_pick('max_count_then_longest_duration', p1))
    rows.append(row_from_pick('longest_duration_with_count>0', p2))
    rows.append(row_from_pick('max_product_durationxcount', p3))
    rows = [r for r in rows if r is not None]
    summary = pd.DataFrame(rows)

    outdir = cov_path.parent if cov_path.exists() else Path(args.tracks).parent / 'coverage'
    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / 'argmax_summary.csv'
    summary.to_csv(out_csv, index=False)

    # Pretty print
    def fmt_row(r):
        return (f"start: frame {r['start_frame']} (t={r['start_h']:.2f} h), "
                f"duration: {r['required_frames']} frames ({r['duration_min']:.0f} min), "
                f"count: {r['count']}")

    print('[Result]')
    for _, r in summary.iterrows():
        print(f" - {r['criterion']}: " + fmt_row(r))

    print('\n[Saved] ', out_csv)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
