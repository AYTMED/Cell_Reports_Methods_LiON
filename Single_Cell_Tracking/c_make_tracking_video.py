#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracking video generator
 - Uses track_output/per_frame/XXXX/overlay.png as base frames
 - Uses track_output/tracks_long.csv to draw track paths and current positions
"""

import cv2
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# Input paths
outdir = Path("track_output")
tracks_csv = outdir / "tracks_long.csv"
per_frame_dir = outdir / "per_frame"
video_out = outdir / "tracking_video.mp4"

# Load tracking data
df = pd.read_csv(tracks_csv)
T = df["t"].max() + 1

# Assign a fixed color to each track_id
track_ids = df["track_id"].unique()
np.random.seed(0)
colors = {tid: tuple(np.random.randint(0, 255, size=3).tolist()) for tid in track_ids}

# Determine video frame size
sample_frame = cv2.imread(str(per_frame_dir / f"{0:04d}" / "overlay.png"))
H, W, _ = sample_frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(str(video_out), fourcc, 5, (W, H))  # Adjust FPS as needed (default: 5 fps)

# Dictionary to store and accumulate track paths
track_paths = {tid: [] for tid in track_ids}

for t in range(T):
    frame_path = per_frame_dir / f"{t:04d}" / "overlay.png"
    if not frame_path.exists():
        continue
    frame = cv2.imread(str(frame_path))
    if frame is None:
        continue

    # Add points for the current frame
    for _, row in df[df["t"] == t].iterrows():
        tid = int(row["track_id"])
        x, y = int(row["x"]), int(row["y"])
        track_paths[tid].append((x, y))

    # Draw accumulated trajectories and current positions
    for tid, pts in track_paths.items():
        if len(pts) > 1:
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i - 1], pts[i], colors[tid], 1)
        if len(pts) > 0:
            cv2.circle(frame, pts[-1], 4, colors[tid], -1)

    writer.write(frame)

writer.release()
print(f"[OK] tracking_video.mp4 written to {video_out}")
