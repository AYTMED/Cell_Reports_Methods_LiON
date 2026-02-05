#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot ratio-only (Red/Green) per ROI over time, with optional XY trajectory
--------------------------------------------------------------------------
Reads:  track_output/tracks_long.csv
Writes: track_output/plots_ratio/
  - ratio_per_track.pdf         (multi-page, one track per page)
  - ratio_summary_grid.png      (small multiples overview)
  - per_track/track_XXXX.png    (optional individual PNGs with --save-per-png)

Options:
  --rolling N         : moving average window (frames); 1 disables smoothing
  --min-frames N      : skip tracks with <N frames (default 3)
  --traj              : add a second panel showing the XY trajectory (x vs y)
  --grid-cols C       : number of columns in summary grid (default 5)

Usage:
  python plot_tracks_ratio_only_with_traj.py \
    --tracks track_output/tracks_long.csv \
    --outdir track_output/plots_ratio \
    --rolling 3 --traj

Notes:
  - X-axis is fixed to frames 0..18 by default (19 frames) and converted to hours
    assuming 20 min/frame (0 h .. 6 h). You can change these via --frames and
    --interval-min.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def moving_average(a: np.ndarray, w: int) -> np.ndarray:
    """Simple centered moving average on observed sequence only (no interpolation)."""
    if w is None or w <= 1:
        return a
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return a
    k = max(1, int(w))
    kernel = np.ones(k, dtype=float) / k
    # centered smoothing; for boundaries, fall back to 'same' which tapers automatically
    return np.convolve(a, kernel, mode='same')


def parse_args():
    ap = argparse.ArgumentParser(description='Plot ratio-only (R/G) timecourses per track, optionally with XY trajectory')
    ap.add_argument('--tracks', default='track_output/tracks_long.csv')
    ap.add_argument('--outdir', default='track_output/plots_ratio')
    ap.add_argument('--rolling', type=int, default=1)
    ap.add_argument('--min-frames', type=int, default=3)
    ap.add_argument('--traj', action='store_true', help='Add trajectory subplot (x vs y) with time color')
    ap.add_argument('--grid-cols', type=int, default=5)
        # New: frame/time control
    ap.add_argument('--frames', type=int, default=19, help='Total frame count to plot (default 19 for 0..18)')
    ap.add_argument('--interval-min', type=float, default=20.0, help='Minutes per frame (default 20)')
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = Path(args.outdir); ensure_outdir(outdir)

    df = pd.read_csv(args.tracks)
    need = {'track_id','t','red_sum','green_sum','x','y'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {args.tracks}: {missing}')

    df['t'] = df['t'].astype(int)
    df.sort_values(['track_id','t'], inplace=True)

    # keep tracks with sufficient frames
    keep_ids = df.groupby('track_id')['t'].count()
    keep_ids = keep_ids[keep_ids >= args.min_frames].index.tolist()
    df = df[df['track_id'].isin(keep_ids)].copy()

    if df.empty:
        print('[WARN] No tracks after filtering; nothing to plot.')
        return 0

    # Multipage PDF
    from matplotlib.backends.backend_pdf import PdfPages
    pdf_path = outdir / 'ratio_per_track.pdf'

    # Fixed frame axis 0..frames-1 and hours conversion
    t_full = np.arange(0, args.frames, dtype=int)
    hours_full = t_full * (args.interval_min / 60.0)
    h_max = hours_full[-1] if len(hours_full) else 0.0

    with PdfPages(pdf_path) as pdf:
        for tid, g in df.groupby('track_id'):
            # Observed frames only
            t_obs = g['t'].to_numpy()
            h_obs = t_obs * (args.interval_min / 60.0)
            ratio_obs = (g['red_sum'].to_numpy()) / (g['green_sum'].to_numpy() + 1e-12)
            # Optional smoothing BUT keep values only at observed samples
            ratio_line = moving_average(ratio_obs, args.rolling)

            if args.traj:
                fig = plt.figure(figsize=(8.0, 5.5))
                ax1 = fig.add_subplot(2,1,1)
                ax2 = fig.add_subplot(2,1,2)
            else:
                fig = plt.figure(figsize=(7.5, 3.0))
                ax1 = fig.add_subplot(1,1,1)

            # --- Ratio vs time (hours) ---
            # Connect only observed points as a polyline (no interpolation/smoothing)
            if len(h_obs) >= 1:
                ax1.plot(h_obs, ratio_obs, '-o', linewidth=1.4, markersize=3)
            ax1.set_title(f'track {tid}  (frames={len(g)})')
            ax1.set_xlabel('Time (h)')
            ax1.set_ylabel('Ratio (R/G)')
            ax1.set_xlim(0, h_max)
            ax1.grid(True, alpha=0.3)

            if args.traj:
                # XY trajectory colored by time (observed frames only)
                x = g['x'].to_numpy(); y = g['y'].to_numpy()
                sc = ax2.scatter(x, y, c=h_obs, s=14)
                ax2.plot(x, y, linewidth=0.8, alpha=0.6)
                ax2.set_aspect('equal', adjustable='box')
                ax2.set_xlabel('x (px)')
                ax2.set_ylabel('y (px)')
                ax2.set_title('trajectory (color = time, h)')
                cb = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
                cb.set_label('h')

            fig.tight_layout()
            pdf.savefig(fig)

            # PNG export disabled per user request
            # plt.close(fig)

    print(f'[OK] wrote {pdf_path}')

    # Small multiples grid output has been removed per user request (no PNG outputs).
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
