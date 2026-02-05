# stack_frames_to_tif.py
import sys, glob
import numpy as np
from pathlib import Path
from skimage import io
import tifffile

# Usage: python stack_frames_to_tif.py /path/to/frames/*.tif movie.tif
frames_glob = sys.argv[1]
out_path = sys.argv[2] if len(sys.argv) > 2 else "movie.tif"

files = sorted(glob.glob(frames_glob))
assert files, "No files matched."

frames = []
for f in files:
    arr = io.imread(f)          # (H, W, 3) or (H, W, 4)
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        arr = arr[..., :3]      # Ensure RGB format
    frames.append(arr)

stack = np.stack(frames, axis=0)  # (T, H, W, 3)
tifffile.imwrite(out_path, stack)
print("Wrote", out_path, "with shape", stack.shape)
