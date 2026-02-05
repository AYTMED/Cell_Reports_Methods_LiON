#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ratiometric pseudo-color for TIFFs (full-resolution safe, arbitrary colors):

- Input:
  * Multi-page grayscale (page0/page1)
  * Single-page RGB composite (R,G,B)
- Outputs (per input):
  * <name>_mask_otsu.tif
  * <name>_pseudocolor.tif
- Shared colorbar (vector PDF):
  * colorbar.pdf
- Features:
  * Full-resolution loading (tifffile → automatic fallback to Pillow on failure)
  * Supports multi-sample gray pages (choose sample with --page-sample)
  * Inherit resolution tags (X/YResolution, ResolutionUnit) when possible
  * Ratio→color mapping is **linear only** (no nonlinearity)
  * Arbitrary colors: --colors "low[,mid[,...],high]" (color names or #RRGGBB)
    - Examples: "green,red", "#00c8ff,white,#ffa000", "cyan,white,orange,purple"
    - Any number of colors are interpolated evenly
    - Positions can be specified with --stops "0,0.4,1" (0–1, same count as --colors)

Examples:
  # Default (green↔red)
  python ratiomap.py FAC.tif DFO.tif --mode autorounded --ticks 3 --outdir out_rg

  # Custom colors: cyan → white → orange
  python ratiomap.py FAC.tif DFO.tif --mode autorounded --ticks 3 --outdir out_cyo \
      --colors "#00c8ff,white,#ffa000"

  # Custom colors: purple → blue → green → yellow → red (5 colors)
  python ratiomap.py FAC.tif DFO.tif --mode autorounded --ticks 4 --outdir out_rain \
      --colors "purple,blue,green,yellow,red"

  # Non-uniform with stops (emphasize low values): blue → white → yellow, place white at 0.3
  python ratiomap.py FAC.tif DFO.tif --mode autorounded --ticks 3 --outdir out_skew \
      --colors "blue,white,yellow" --stops "0,0.3,1"
"""

import argparse, os, sys
import numpy as np
import tifffile as tiff
from PIL import Image, ImageSequence

# colorbar (vector)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# --------------------- TIFF loading (full resolution & fallback) ---------------------
def _page_info_list(tf: tiff.TiffFile):
    infos = []
    for i, p in enumerate(tf.pages):
        try: h = int(getattr(p, "imagelength")); w = int(getattr(p, "imagewidth"))
        except Exception: h = w = 0
        try: spp = int(getattr(p, "samplesperpixel", 1))
        except Exception: spp = 1
        try: phot = str(getattr(p, "photometric", ""))
        except Exception: phot = ""
        infos.append({"index": i, "h": h, "w": w, "area": h*w, "spp": spp, "photometric": phot})
    return infos

def _as_fullres(tf: tiff.TiffFile, page_index: int, tif_path: str):
    try:
        p = tf.pages[page_index]
        try:
            for s in tf.series:
                if p in s.pages:
                    return s.levels[0].asarray()
        except Exception:
            pass
        return p.asarray()
    except Exception as e:
        try:
            im = Image.open(tif_path)
            try:
                im.seek(page_index)
                return np.array(im)
            except EOFError:
                best = None; best_area = -1
                for fr in ImageSequence.Iterator(im):
                    a = np.array(fr)
                    area = a.shape[0] * a.shape[1]
                    if area > best_area:
                        best_area, best = area, a
                if best is not None:
                    return best
                raise e
        except Exception:
            raise e

def _split_rg_from_array(arr: np.ndarray):
    if arr.ndim != 3:
        raise RuntimeError("RGB composite expected a 3D array")
    if arr.shape[-1] >= 3:     # (H,W,C)
        R = arr[..., 0]; G = arr[..., 1]
    elif arr.shape[0] >= 3:    # (C,H,W)
        R = arr[0, ...]; G = arr[1, ...]
    else:
        raise RuntimeError("Unable to locate RGB channels")
    return R.astype(np.float32), G.astype(np.float32)

def _ensure_2d_gray(arr: np.ndarray, sample_index: int = 0) -> np.ndarray:
    a = arr
    if a.ndim == 2:
        return a.astype(np.float32)
    if a.ndim == 3:
        if a.shape[0] in (1,2,3,4):
            axis = 0
        elif a.shape[-1] in (1,2,3,4):
            axis = -1
        else:
            axis = 0
        if a.shape[axis] == 1:
            a = np.take(a, 0, axis=axis)
            return a.astype(np.float32)
        else:
            idx = max(0, min(sample_index, a.shape[axis]-1))
            a = np.take(a, idx, axis=axis)
            if a.ndim != 2 and 1 in a.shape:
                a = np.squeeze(a)
            return a.astype(np.float32)
    return a.astype(np.float32)

def load_two_channels(path, force="auto", order="TRITC,EGFP", rgb_order="R,G", page_sample=0):
    with tiff.TiffFile(path) as tf:
        infos = _page_info_list(tf)
        gray_pages = sorted([d for d in infos if d["spp"] == 1], key=lambda x: x["area"], reverse=True)
        rgb_pages  = sorted([d for d in infos if d["spp"] >= 3 or "RGB" in str(d["photometric"]).upper()],
                            key=lambda x: x["area"], reverse=True)

        if force == "pages":
            mode = "pages"
        elif force == "rgb":
            mode = "rgb"
        else:
            mode = "pages" if len(gray_pages) >= 2 and gray_pages[1]["area"] > 0 else ("rgb" if rgb_pages else "pages")

        if mode == "pages":
            if len(gray_pages) < 2:
                raise RuntimeError(f"{os.path.basename(path)}: need ≥2 grayscale pages for 'pages' mode.")
            p0_idx, p1_idx = gray_pages[0]["index"], gray_pages[1]["index"]
            A0 = _ensure_2d_gray(_as_fullres(tf, p0_idx, path), sample_index=page_sample)
            A1 = _ensure_2d_gray(_as_fullres(tf, p1_idx, path), sample_index=page_sample)
            key = order.upper().replace(" ", "")
            if key == "TRITC,EGFP": TRITC, EGFP = A0, A1
            elif key == "EGFP,TRITC": TRITC, EGFP = A1, A0
            else: raise ValueError(f"Invalid --order: {order}")
            return TRITC, EGFP

        if not rgb_pages:
            raise RuntimeError(f"{os.path.basename(path)}: RGB page not found.")
        p_idx = rgb_pages[0]["index"]
        A = _as_fullres(tf, p_idx, path)
        R, G = _split_rg_from_array(A)
        key = rgb_order.upper().replace(" ", "")
        if key == "R,G": TRITC, EGFP = R, G
        elif key == "G,R": TRITC, EGFP = G, R
        else: raise ValueError(f"Invalid --rgb-order: {rgb_order} (use 'R,G' or 'G,R')")
        return TRITC, EGFP

# --------------------- Otsu & morphology ---------------------
def otsu_threshold(img):
    im = img.astype(np.float64)
    im -= im.min()
    vmax = im.max()
    if vmax == 0:
        return float(img.min())
    im /= vmax
    hist, bin_edges = np.histogram(im.ravel(), bins=256, range=(0.0, 1.0))
    prob = hist.astype(np.float64) / max(1.0, hist.sum())
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]
    k = int(np.argmax((mu_t * omega - mu) ** 2 / (omega * (1.0 - omega) + 1e-12)))
    thr = bin_edges[0] + (k / 255.0) * (bin_edges[-1] - bin_edges[0])
    return float(img.min() + thr * (img.max() - img.min()))

def box_blur(img, k=3):
    if k is None or k < 3 or (k % 2) == 0:
        return img
    pad = k // 2
    im = np.pad(img, ((pad, pad), (pad, pad)), mode='edge')
    S = im.cumsum(0).cumsum(1)
    out = (S[k:, k:] - S[:-k, k:] - S[k:, :-k] + S[:-k, :-k]) / (k * k)
    return out.astype(img.dtype)

def binary_dilate(mask, iters=1):
    m = mask
    for _ in range(max(0, iters)):
        H, W = m.shape
        pm = np.pad(m, ((1, 1), (1, 1)), mode='constant')
        nb = [pm[1+dy:1+dy+H, 1+dx:1+dx+W] for dy in (-1,0,1) for dx in (-1,0,1)]
        m = np.logical_or.reduce(nb)
    return m

def binary_erode(mask, iters=1):
    m = mask
    for _ in range(max(0, iters)):
        H, W = m.shape
        pm = np.pad(m, ((1, 1), (1, 1)), mode='constant')
        nb = [pm[1+dy:1+dy+H, 1+dx:1+dx+W] for dy in (-1,0,1) for dx in (-1,0,1)]
        m = np.logical_and.reduce(nb)
    return m

# --------------------- Arbitrary color mapping ---------------------
def _parse_colors(colors_str: str | None, palette: str):
    """Parse --colors. If None, return palette defaults."""
    if colors_str:
        parts = [s.strip() for s in colors_str.split(",") if s.strip()]
        if len(parts) < 2:
            raise ValueError("--colors must specify at least 2 colors, comma-separated")
        rgb = [mcolors.to_rgb(p) for p in parts]  # 0..1 floats
        return np.array(rgb, dtype=np.float32)
    # fallback to palette presets
    if palette == "rg":   # green -> red
        return np.array([mcolors.to_rgb("green"), mcolors.to_rgb("red")], dtype=np.float32)
    if palette == "by":   # blue -> white -> yellow
        return np.array([mcolors.to_rgb("#005cff"), (1,1,1), mcolors.to_rgb("#ffdc00")], dtype=np.float32)
    if palette == "oc":   # cyan -> white -> orange
        return np.array([mcolors.to_rgb("#00c8ff"), (1,1,1), mcolors.to_rgb("#ffa000")], dtype=np.float32)
    # default
    return np.array([mcolors.to_rgb("green"), mcolors.to_rgb("red")], dtype=np.float32)

def _parse_stops(stops_str: str | None, n: int):
    """Parse --stops. If None, use even spacing. If provided: 0..1, ascending, count == n."""
    if not stops_str:
        return np.linspace(0.0, 1.0, n, dtype=np.float32)
    parts = [float(s.strip()) for s in stops_str.split(",") if s.strip()]
    if len(parts) != n:
        raise ValueError("--stops count must match --colors")
    arr = np.array(parts, dtype=np.float32)
    if np.any(arr < 0) or np.any(arr > 1):
        raise ValueError("--stops values must be within 0..1")
    if not np.all(np.diff(arr) >= 0):
        raise ValueError("--stops must be in ascending order")
    if arr[0] != 0.0 or arr[-1] != 1.0:
        raise ValueError("--stops must include both 0 and 1 (first=0, last=1)")
    return arr

def build_cmap(colors_arr: np.ndarray, stops_arr: np.ndarray) -> LinearSegmentedColormap:
    """Build a continuous Matplotlib colormap (linear interpolation)."""
    tuples = list(zip(stops_arr.tolist(), [tuple(c) for c in colors_arr.tolist()]))
    return LinearSegmentedColormap.from_list("custom_linear", tuples)

def apply_colormap(norm: np.ndarray, valid: np.ndarray, cmap: LinearSegmentedColormap) -> np.ndarray:
    """norm[0..1] → RGB8. Invalid pixels are black."""
    rgba = cmap(norm)  # (..., 4) in 0..1
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    rgb[~valid] = 0
    return rgb

def to_rgb_map_arbitrary(ratio_masked, lo, hi, colors_arr, stops_arr):
    valid = np.isfinite(ratio_masked)
    norm = np.zeros_like(ratio_masked, dtype=np.float32)
    rng = hi - lo if hi > lo else 1.0
    norm[valid] = np.clip((ratio_masked[valid] - lo) / rng, 0, 1)
    cmap = build_cmap(colors_arr, stops_arr)
    return apply_colormap(norm, valid, cmap)

# --------------------- range helpers ---------------------
def compute_auto_range(arr_list, p_lo=2, p_hi=98):
    vals = []
    for a in arr_list:
        a = a.astype(np.float32)
        vals.append(a[np.isfinite(a)])
    vals = np.concatenate(vals) if vals else np.array([], dtype=np.float32)
    if vals.size == 0:
        return 0.5, 2.0
    lo, hi = np.percentile(vals, [p_lo, p_hi])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(vals)); mx = float(np.nanmax(vals))
        hi = mx if mx > lo else lo + 1.0
    return float(lo), float(hi)

def nice_number(x, mode='ceil'):
    if x == 0 or not np.isfinite(x): return 0.0
    k = np.floor(np.log10(abs(x))); m = abs(x) / (10 ** k)
    if mode == 'ceil': m_n = 1 if m <= 1 else 2 if m <= 2 else 5 if m <= 5 else 10
    elif mode == 'floor': m_n = 5 if m >= 5 else 2 if m >= 2 else 1 if m >= 1 else 0.5
    else:
        candidates = np.array([1, 2, 5, 10], dtype=float)
        m_n = candidates[np.argmin(np.abs(candidates - m))]
    return np.sign(x) * m_n * (10 ** k)

def nice_bounds(lo, hi, n_ticks_target=3):
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return 0.5, 2.0, [0.5, 1.0, 2.0]
    rng = hi - lo
    step_raw = rng / max(1, (n_ticks_target - 1))
    step = nice_number(step_raw, mode='ceil')
    lo_nice = np.floor(lo / step) * step
    hi_nice = np.ceil(hi / step) * step
    n_steps = int(round((hi_nice - lo_nice) / step)) if step > 0 else 0
    if n_steps < 2:
        step = max(0.1, np.round(rng / max(1, (n_ticks_target - 1)), 1))
        lo_nice = np.floor(lo / step) * step
        hi_nice = np.ceil(hi / step) * step
        n_steps = int(round((hi_nice - lo_nice) / step))
    ticks = [lo_nice + i * step for i in range(n_steps + 1)]
    if lo_nice <= 1.0 <= hi_nice and all(abs(t - 1.0) > 1e-6 for t in ticks):
        ticks.append(1.0); ticks = sorted(ticks)
    if len(ticks) > 6:
        stride = int(np.ceil(len(ticks) / 6))
        ticks = ticks[::stride]
        if lo_nice <= 1.0 <= hi_nice and all(abs(t - 1.0) > 1e-6 for t in ticks):
            ticks.append(1.0); ticks = sorted(ticks)
    return float(lo_nice), float(hi_nice), [float(np.round(t, 3)) for t in ticks]

# --------------------- colorbar (vector) ---------------------
def save_colorbar_vector(path_pdf, lo, hi, ticks, steps, colors_arr, stops_arr,
                         width_in=6.0, height_in=1.2, dpi=300):
    cmap = build_cmap(colors_arr, stops_arr)
    fig = plt.figure(figsize=(width_in, height_in))
    ax = fig.add_axes([0.08, 0.45, 0.84, 0.3])
    ax.set_xlim(lo, hi); ax.set_ylim(0, 1); ax.axis("off")

    if steps and steps > 0:
        xs = np.linspace(lo, hi, steps + 1)
        for i in range(steps):
            x0, x1 = xs[i], xs[i + 1]
            v = (i + 0.5) / steps
            rect = patches.Rectangle((x0, 0), x1 - x0, 1,
                                     facecolor=cmap(v), edgecolor=cmap(v), linewidth=0)
            ax.add_patch(rect)
    else:
        X = np.linspace(lo, hi, 1024)
        Z = np.tile(X[None, :], (2, 1))
        ax.imshow(Z, extent=[lo, hi, 0, 1], origin="lower", aspect="auto", cmap=cmap)

    for t in ticks:
        if t < lo or t > hi: continue
        ax.plot([t, t], [0, -0.1], color="black", linewidth=1.0, clip_on=False)
        if abs(t - round(t)) < 1e-6: label = f"{t:.0f}"
        elif abs(t*10 - round(t*10)) < 1e-6: label = f"{t:.1f}"
        else: label = f"{t:.2f}"
        ax.text(t, -0.25, label, ha="center", va="top", fontsize=10, color="black")

    fig.savefig(path_pdf, format="pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# --------------------- Mask ---------------------
def build_mask_relaxed(TRITC, EGFP, mode="and", scale=1.0, blur=0, dilate=0, erode=0):
    Rb = box_blur(TRITC, blur) if blur and blur >= 3 else TRITC
    Gb = box_blur(EGFP,  blur) if blur and blur >= 3 else EGFP

    if mode == "and":
        thr_r, thr_g = otsu_threshold(Rb) * scale, otsu_threshold(Gb) * scale
        mask = (TRITC >= thr_r) & (EGFP >= thr_g)
    elif mode == "or":
        thr_r, thr_g = otsu_threshold(Rb) * scale, otsu_threshold(Gb) * scale
        mask = (TRITC >= thr_r) | (EGFP >= thr_g)
    elif mode == "sum":
        S = Rb + Gb; thr_s = otsu_threshold(S) * scale
        mask = (TRITC + EGFP) >= thr_s
    elif mode == "max":
        M = np.maximum(Rb, Gb); thr_m = otsu_threshold(M) * scale
        mask = np.maximum(TRITC, EGFP) >= thr_m
    else:
        raise ValueError(f"Unknown --mask-mode: {mode}")

    if erode and erode > 0:  mask = binary_erode(mask, erode)
    if dilate and dilate > 0: mask = binary_dilate(mask, dilate)
    return mask

# --------------------- Inherit resolution tags ---------------------
def _read_resolution_tags(tif_path):
    try:
        with tiff.TiffFile(tif_path) as tf:
            page = tf.pages[0]
            x = page.tags.get('XResolution'); y = page.tags.get('YResolution'); u = page.tags.get('ResolutionUnit')
            def _to_float(v):
                try:
                    if isinstance(v, tuple) and len(v) == 2:
                        num, den = v; return float(num) / float(den) if den else float(num)
                    return float(v)
                except Exception: return None
            xres = _to_float(x.value) if x else None
            yres = _to_float(y.value) if y else None
            unit = None
            if u:
                val = u.value[0] if isinstance(u.value, (list, tuple)) else u.value
                unit = {2: 'INCH', 3: 'CENTIMETER'}.get(int(val), None)
            if xres and yres and unit: return xres, yres, unit
    except Exception:
        pass
    return None, None, None

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser(description="Ratiometric pseudo-color (TRITC/EGFP or RGB R/G) for TIFFs.")
    ap.add_argument("inputs", nargs="+", help="Input TIFF(s): multi-page or RGB composites.")
    ap.add_argument("--force", choices=["auto", "pages", "rgb"], default="auto", help="How to read channels.")
    ap.add_argument("--order", default="TRITC,EGFP", help="For pages mode: TRITC,EGFP or EGFP,TRITC.")
    ap.add_argument("--rgb-order", default="R,G", help="For RGB mode: 'R,G' or 'G,R'.")
    ap.add_argument("--outdir", default="outputs")

    # range
    ap.add_argument("--mode", choices=["fixed", "auto", "autorounded"], default="auto")
    ap.add_argument("--lo", type=float); ap.add_argument("--hi", type=float)
    ap.add_argument("--plo", type=float, default=2.0); ap.add_argument("--phi", type=float, default=98.0)
    ap.add_argument("--ticks", type=int, default=3)

    # colorbar (vector)
    ap.add_argument("--colorbar_name", default="colorbar.pdf")
    ap.add_argument("--colorbar_steps", type=int, default=0, help=">0 for fully vector bar (e.g., 64).")

    # mask
    ap.add_argument("--mask-mode", choices=["and","or","sum","max"], default="and")
    ap.add_argument("--mask-scale", type=float, default=1.0, help="<1.0 to relax Otsu thresholds.")
    ap.add_argument("--mask-blur",  type=int,   default=0,   help="Odd kernel for pre-Otsu blur (0=off).")
    ap.add_argument("--mask-dilate",type=int,   default=0,   help="Binary dilation iterations.")
    ap.add_argument("--mask-erode", type=int,   default=0,   help="Binary erosion iterations.")

    # palette (backward compatible). Ignored if --colors is provided.
    ap.add_argument("--palette", choices=["rg","by","oc"], default="rg",
                    help="Preset palette (ignored if --colors is set)")

    # Custom colors
    ap.add_argument("--colors", type=str, default=None,
                    help="Comma-separated colors (>=2). e.g., 'green,red' or '#00c8ff,white,#ffa000'")
    ap.add_argument("--stops", type=str, default=None,
                    help="Comma-separated stops (0..1, same count as --colors). e.g., '0,0.4,1'")

    # multi-sample gray page handling
    ap.add_argument("--page-sample", type=int, default=0,
                    help="Sample index to use when a gray page is multi-sample like (S,H,W)/(H,W,S) (default 0).")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # --- parse colors / stops ---
    colors_arr = _parse_colors(args.colors, args.palette)      # Nx3, 0..1
    stops_arr  = _parse_stops(args.stops, colors_arr.shape[0]) # N

    # 1st pass: compute shared range
    masked_ratios, loaded = [], []
    for p in args.inputs:
        TRITC, EGFP = load_two_channels(
            p, force=args.force, order=args.order, rgb_order=args.rgb_order, page_sample=args.page_sample
        )
        if TRITC.shape != EGFP.shape:
            raise RuntimeError(f"Size mismatch in {os.path.basename(p)}: {TRITC.shape} vs {EGFP.shape}")
        mask = build_mask_relaxed(TRITC, EGFP,
                                  mode=args.mask_mode, scale=args.mask_scale,
                                  blur=args.mask_blur, dilate=args.mask_dilate, erode=args.mask_erode)
        ratio = (TRITC / (EGFP + 1e-6)).astype(np.float32)
        ratio[~mask] = np.nan
        masked_ratios.append(ratio)
        loaded.append((p, mask, ratio))

    # Shared range
    if args.mode == "fixed":
        if args.lo is None or args.hi is None or args.hi <= args.lo:
            sys.exit("For --mode fixed, provide valid --lo < --hi.")
        lo, hi = args.lo, args.hi
        mid = (lo + hi) / 2
        ticks = [lo, 1.0, hi] if (lo < 1.0 < hi) else [lo, mid, hi]
    elif args.mode == "auto":
        lo, hi = compute_auto_range(masked_ratios, args.plo, args.phi)
        mid = (lo + hi) / 2
        ticks = [lo, 1.0, hi] if (lo < 1.0 < hi) else [lo, mid, hi]
    else:
        lo_auto, hi_auto = compute_auto_range(masked_ratios, args.plo, args.phi)
        lo, hi, ticks = nice_bounds(lo_auto, hi_auto, n_ticks_target=args.ticks)

    # 2nd pass: write outputs (inherit resolution tags)
    for p, mask, ratio in loaded:
        name = os.path.splitext(os.path.basename(p))[0]
        xres, yres, unit = _read_resolution_tags(p)
        kwres = dict(resolution=(xres, yres), resolutionunit=unit) if (xres and yres and unit) else {}

        # mask
        tiff.imwrite(os.path.join(args.outdir, f"{name}_mask_otsu.tif"),
                     (mask.astype(np.uint8) * 255), **kwres)

        # pseudocolor
        rgb = to_rgb_map_arbitrary(ratio, lo, hi, colors_arr, stops_arr)
        tiff.imwrite(os.path.join(args.outdir, f"{name}_pseudocolor.tif"),
                     rgb, photometric='rgb', **kwres)

    # shared colorbar
    save_colorbar_vector(os.path.join(args.outdir, args.colorbar_name),
                         lo, hi, ticks, steps=args.colorbar_steps,
                         colors_arr=colors_arr, stops_arr=stops_arr)

    print(f"Shared range: {lo:.3f} – {hi:.3f} | ticks: {ticks}")
    print(f"Mask mode={args.mask_mode}, scale={args.mask_scale}, blur={args.mask_blur}, dilate={args.mask_dilate}, erode={args.mask_erode}")
    print(f"Read mode={args.force}, rgb-order={args.rgb_order}, pages-order={args.order}, page-sample={args.page_sample}")
    if args.colors:
        print(f"Colors(custom)={args.colors}  Stops={args.stops or 'even'}")
    else:
        print(f"Palette(preset)={args.palette}")
    print(f"Outputs in: {os.path.abspath(args.outdir)}")

if __name__ == "__main__":
    sys.exit(main())
