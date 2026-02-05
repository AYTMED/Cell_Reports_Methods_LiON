#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute, for each start timepoint s, how many tracks are continuously
observable for at least a given duration starting at s.

Input:
  - track_output/tracks_long.csv  (requires columns: track_id, t)

Output (default outdir=track_output/coverage):
  - coverage_matrix.csv  : long-form table with (start_frame, start_h, duration_min, required_frames, count)
  - coverage_summary.pdf : PDF with a heatmap and a few line plots

Assumptions / Defaults:
  - Total frames = 19 (0..18)
  - Frame interval = 20 min → 0..6 h

Examples:
  python coverage_by_start_time.py \
    --tracks track_output/tracks_long.csv \
    --outdir track_output/coverage \
    --frames 19 --interval-min 20
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def parse_args():
    ap = argparse.ArgumentParser(description='Count continuous tracking coverage from each start timepoint')
    ap.add_argument('--tracks', default='track_output/tracks_long.csv')
    ap.add_argument('--outdir', default='track_output/coverage')
    ap.add_argument('--frames', type=int, default=19, help='Total number of frames (default 19 for 0..18)')
    ap.add_argument('--interval-min', type=float, default=20.0, help='Minutes per frame (default 20)')
    # durations in minutes; default 20,40,...,360
    ap.add_argument('--durations-min', type=str, default='20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360',
                    help='Comma-separated durations (minutes) to test (>= interval)')
    return ap.parse_args()


def build_presence_matrix(df: pd.DataFrame, frames: int) -> dict:
    """Return dict track_id -> boolean array len=frames indicating presence per frame."""
    presence = {}
    for tid, g in df.groupby('track_id'):
        arr = np.zeros(frames, dtype=bool)
        ts = g['t'].to_numpy(dtype=int)
        ts = ts[(ts >= 0) & (ts < frames)]
        arr[ts] = True
        presence[int(tid)] = arr
    return presence


def consecutive_run_length_from(arr: np.ndarray, start: int) -> int:
    """Length (in frames) of consecutive True run starting at index start."""
    if start < 0 or start >= arr.size or not arr[start]:
        return 0
    i = start
    while i < arr.size and arr[i]:
        i += 1
    return i - start


def compute_counts(presence: dict, frames: int, durations_frames: np.ndarray) -> np.ndarray:
    """Return counts[d_idx, start] for each duration and start frame."""
    D = len(durations_frames)
    counts = np.zeros((D, frames), dtype=int)
    for s in range(frames):
        # for each track, compute its consecutive run length from s
        runlens = []
        for arr in presence.values():
            runlens.append(consecutive_run_length_from(arr, s))
        runlens = np.asarray(runlens, dtype=int)
        for d_idx, req in enumerate(durations_frames):
            counts[d_idx, s] = int((runlens >= req).sum())
    return counts


def main():
    args = parse_args()
    outdir = Path(args.outdir); ensure_outdir(outdir)

    # Read tracks and basic checks
    df = pd.read_csv(args.tracks)
    if not {'track_id','t'}.issubset(df.columns):
        raise ValueError('tracks_long.csv must contain columns: track_id, t')
    df['t'] = df['t'].astype(int)

    frames = int(args.frames)
    interval_min = float(args.interval_min)
    hours = np.arange(frames) * (interval_min / 60.0)

    # Parse durations (min) → frames, clipped to frames
    durations_min = [float(x) for x in args.durations_min.split(',') if x.strip()]
    durations_frames = np.array([max(1, int(round(m / interval_min))) for m in durations_min], dtype=int)
    # Remove duplicates and ensure ascending order, and limit to possible length
    durations_frames, unique_idx = np.unique(durations_frames, return_index=True)
    durations_min = [durations_min[i] for i in sorted(unique_idx, key=lambda i: durations_frames[list(unique_idx).index(i)])]
    durations_frames = np.clip(durations_frames, 1, frames)

    # Build presence arrays per track
    presence = build_presence_matrix(df, frames)

    # Compute counts matrix: shape (len(durations), frames)
    counts = compute_counts(presence, frames, durations_frames)

    # Save long-form CSV
    rows = []
    for d_idx, req in enumerate(durations_frames):
        for s in range(frames):
            rows.append({
                'start_frame': s,
                'start_h': hours[s],
                'duration_min': durations_min[d_idx] if d_idx < len(durations_min) else float(req)*interval_min,
                'required_frames': int(req),
                'count': int(counts[d_idx, s])
            })
    out_csv = outdir / 'coverage_matrix.csv'
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # Plot to a single PDF (heatmap + selected line plots)
    pdf_path = outdir / 'coverage_summary.pdf'
    with PdfPages(pdf_path) as pdf:
        # Heatmap
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(counts, aspect='auto', origin='lower',
                       extent=[hours[0], hours[-1] if len(hours)>1 else 0, durations_min[0], durations_min[-1]],
                       interpolation='nearest')
        ax.set_xlabel('Start time (h)')
        ax.set_ylabel('Required continuous duration (min)')
        ax.set_title('Number of tracks with continuous coverage by start time')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Count of tracks')
        pdf.savefig(fig); plt.close(fig)

        # Line plots for a subset of durations (choose up to 6 evenly)
        K = min(6, len(durations_frames))
        idxs = np.linspace(0, len(durations_frames)-1, K, dtype=int)
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1,1,1)
        for i in idxs:
            ax.plot(hours, counts[i], label=f'>= {int(durations_min[i])} min', linewidth=1.6)
        ax.set_xlabel('Start time (h)')
        ax.set_ylabel('Count of tracks')
        ax.set_title('Continuous coverage vs start time (selected durations)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        pdf.savefig(fig); plt.close(fig)

    print('[OK] wrote:')
    print(' -', out_csv)
    print(' -', pdf_path)

    return 0

if __name__ == '__main__':
    raise SystemExit(main())
