#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timelapse tracking & R/G quantification
Cellpose (cyto3) for segmentation  →  btrack if available, else TrackPy fallback
---------------------------------------------------------------------------
- If your btrack build lacks `Detection` (has Detection? False), we fall back to TrackPy.
- Inputs: single TIF (THWC/TCHW/...) or two globs (--red-glob/--green-glob)
- Outputs: per-frame artifacts + tracks_long.csv + tracks_summary.csv

Usage (RGB timelapse in THWC):
  python timelapse_tracking_pipeline_v3_fallback_trackpy.py movie.tif --axes THWC \
    --save-labels --save-overlay --save-roi-csv

Install deps:
  pip install cellpose btrack trackpy tifffile scikit-image pandas numpy matplotlib
"""

import argparse, os, sys, gc, glob
from pathlib import Path
import numpy as np, pandas as pd
from skimage import io, segmentation
from skimage.util import img_as_ubyte
from skimage.measure import regionprops

# --- optional libs ---
try:
    import tifffile
except Exception:
    tifffile = None

try:
    from cellpose import models
except Exception as e:
    models = None
    _cellpose_err = str(e)

# btrack (optional)
try:
    import btrack
    from btrack.datasets import cell_config
    _has_btrack = True
    _has_detection = hasattr(btrack, 'Detection')
except Exception as e:
    btrack = None
    cell_config = None
    _has_btrack = False
    _has_detection = False
    _btrack_err = str(e)

# TrackPy fallback (optional)
try:
    import trackpy as tp
    _has_trackpy = True
except Exception as e:
    tp = None
    _has_trackpy = False
    _tp_err = str(e)

# ---------------- utils ----------------

def ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo)

def labels_to_overlay(base_gray: np.ndarray, labels: np.ndarray) -> np.ndarray:
    g = normalize01(base_gray)
    rgb = np.dstack([g, g, g])
    boundaries = segmentation.find_boundaries(labels, mode='outer')
    rgb[boundaries, 0] = 1.0
    rgb[boundaries, 1] = 0.0
    rgb[boundaries, 2] = 0.0
    return img_as_ubyte(np.clip(rgb, 0, 1))

# -------------- load as (T,C,H,W) --------------

def parse_axes(arr: np.ndarray, axes: str):
    axes = axes.upper()
    if axes == 'THW':
        T,H,W = arr.shape; arr = arr.reshape(T,1,H,W)
    elif axes == 'THWC':
        T,H,W,C = arr.shape; arr = np.moveaxis(arr, -1, 1)
    elif axes == 'TCHW':
        pass
    elif axes == 'HWC':
        H,W,C = arr.shape; arr = np.moveaxis(arr, -1, 0); arr = arr.reshape(1,*arr.shape)
    elif axes == 'CHW':
        C,H,W = arr.shape; arr = arr.reshape(1,C,H,W)
    else:
        raise ValueError(f'Unsupported axes: {axes}')
    return arr

ess = (3,4)

def infer_axes(arr: np.ndarray) -> str:
    if arr.ndim == 2: return 'CHW'
    if arr.ndim == 3:
        if arr.shape[-1] in ess: return 'HWC'
        if arr.shape[0] <= 6 and (arr.shape[1]>32 and arr.shape[2]>32): return 'CHW'
        return 'THW'
    if arr.ndim == 4:
        if arr.shape[-1] in ess: return 'THWC'
        return 'TCHW'
    raise ValueError('Unsupported ndim')

def imread_any(path: Path):
    if tifffile is not None and path.suffix.lower() in ('.tif','.tiff'):
        return tifffile.imread(str(path))
    return io.imread(str(path))

def load_single(path: Path, axes: str|None):
    arr = imread_any(path)
    if axes is None: axes = infer_axes(arr)
    return parse_axes(arr, axes)

def load_from_globs(red_glob: str, green_glob: str):
    rfs = sorted(glob.glob(red_glob)); gfs = sorted(glob.glob(green_glob))
    assert len(rfs)==len(gfs) and len(rfs)>0, 'Red/Green counts must match and >0'
    frames=[]
    for rf,gf in zip(rfs,gfs):
        r = normalize01(imread_any(Path(rf)).astype(np.float32))
        g = normalize01(imread_any(Path(gf)).astype(np.float32))
        assert r.shape==g.shape, f'Shape mismatch: {rf} vs {gf}'
        frames.append(np.stack([r,g],axis=0))
    return np.stack(frames,axis=0)

def select_rg(arr_tchw, red_idx, green_idx):
    T,C,H,W = arr_tchw.shape
    assert 0<=red_idx<C and 0<=green_idx<C
    R = arr_tchw[:,red_idx].astype(np.float32,copy=True)
    G = arr_tchw[:,green_idx].astype(np.float32,copy=True)
    for t in range(T):
        R[t]=normalize01(R[t]); G[t]=normalize01(G[t])
    return R,G

# -------------- segmentation --------------

def run_cellpose(img, model, diameter, flow_th, prob_th):
    masks, flows, styles, diams = model.eval(img, channels=[0,0], diameter=diameter or None,
                                             flow_threshold=flow_th, cellprob_threshold=prob_th)
    if masks is None: return np.zeros_like(img, dtype=np.int32)
    return masks.astype(np.int32, copy=False)

def segment_series(R, G, diameter, flow_th, prob_th, gpu):
    if models is None:
        raise RuntimeError(f'cellpose not available: {_cellpose_err}')
    model = models.Cellpose(model_type='cyto3', gpu=gpu)
    labels=[]
    for t in range(R.shape[0]):
        seg = np.maximum(R[t], G[t])
        labels.append(run_cellpose(seg, model, diameter, flow_th, prob_th))
    return labels

# -------------- measure per frame --------------

def measure_frame(Rt, Gt, L, eps=1e-12):
    rows=[]
    for p in regionprops(L):
        rr=p.coords[:,0]; cc=p.coords[:,1]
        rsum=float(Rt[rr,cc].sum()); gsum=float(Gt[rr,cc].sum())
        rows.append({
            'x':float(p.centroid[1]), 'y':float(p.centroid[0]), 'area_px':int(p.area),
            'red_sum':rsum, 'green_sum':gsum, 'ratio_rg': rsum/(gsum+eps)
        })
    return pd.DataFrame(rows)

# -------------- tracking backends --------------

def track_with_btrack(measures_by_t, frame_shape=None):
    dets=[]
    for t,df in enumerate(measures_by_t):
        for _,row in df.iterrows():
            d=btrack.Detection(x=row['x'], y=row['y'], t=t)
            d.features={'area_px':float(row['area_px']), 'red_sum':float(row['red_sum']),
                        'green_sum':float(row['green_sum']), 'ratio_rg':float(row['ratio_rg'])}
            dets.append(d)
    tr=btrack.BayesianTracker()
    tr.configure_from_file(cell_config())
    if frame_shape is not None:
        H,W=frame_shape; tr.volume=((0,W),(0,H),(0,1))
    tr.append(dets); tr.track(); tr.optimize()
    # Convert to long table via nearest matching per frame
    from scipy.spatial import cKDTree
    rows=[]
    for t,df in enumerate(measures_by_t):
        if len(df)==0: continue
        coords=np.column_stack([df['x'].values, df['y'].values])
        pts=[]; tids=[]
        for trk in tr.tracks:
            for p in trk.points:
                if p.t==t: pts.append([p.x,p.y]); tids.append(trk.ID)
        if not pts: continue
        tree=cKDTree(np.array(pts,float))
        dists,idx=tree.query(coords,k=1)
        for r,(dist,i) in zip(df.to_dict('records'), zip(dists,idx)):
            rows.append({**r,'t':t,'track_id':int(tids[i]),'nn_dist_px':float(dist)})
    return pd.DataFrame(rows)


def track_with_trackpy(measures_by_t, search_range=20, memory=2):
    # Build one table with frame column
    recs=[]
    for t,df in enumerate(measures_by_t):
        if len(df)==0: continue
        tmp=df.copy(); tmp['frame']=t; recs.append(tmp)
    if not recs: return pd.DataFrame([])
    spots=pd.concat(recs, ignore_index=True)
    # TrackPy linking
    linked=tp.link_df(spots, search_range=search_range, memory=memory)
    # Rename particle -> track_id, keep long format
    linked.rename(columns={'particle':'track_id'}, inplace=True)
    linked['track_id']=linked['track_id'].astype(int)
    linked.rename(columns={'frame':'t'}, inplace=True)
    return linked[['track_id','t','x','y','area_px','red_sum','green_sum','ratio_rg']]

# -------------- CLI --------------

def parse_args():
    ap=argparse.ArgumentParser(description='Timelapse tracking with Cellpose; btrack or TrackPy fallback')
    ap.add_argument('input', nargs='?', default=None, help='Single TIF path')
    ap.add_argument('--axes', default=None, help='THW, THWC, TCHW, HWC, CHW')
    ap.add_argument('--red-glob', default=None); ap.add_argument('--green-glob', default=None)
    ap.add_argument('--use-single-as-both', action='store_true')
    ap.add_argument('--red-idx', type=int, default=0); ap.add_argument('--green-idx', type=int, default=1)
    ap.add_argument('--diameter', type=float, default=0.0); ap.add_argument('--flow-th', type=float, default=0.4)
    ap.add_argument('--prob-th', type=float, default=0.0); ap.add_argument('--no-gpu', action='store_true')
    ap.add_argument('--outdir', default='track_output')
    ap.add_argument('--save-labels', action='store_true'); ap.add_argument('--save-overlay', action='store_true')
    ap.add_argument('--save-roi-csv', action='store_true'); ap.add_argument('--save-btrack-json', action='store_true')
    ap.add_argument('--tp-search', type=float, default=20.0, help='TrackPy search_range (px) fallback')
    ap.add_argument('--tp-memory', type=int, default=2, help='TrackPy memory frames fallback')
    return ap.parse_args()


def main():
    args=parse_args()
    outroot=Path(args.outdir); ensure_outdir(outroot)

    # Load to (T,C,H,W)
    if args.red_glob and args.green_glob:
        arr=load_from_globs(args.red_glob,args.green_glob)
    else:
        if not args.input:
            print('[ERR] Provide input TIF or both --red-glob/--green-glob'); return 2
        arr=load_single(Path(args.input), axes=args.axes)
    T,C,H,W=arr.shape

    if C==1:
        if args.use_single_as_both:
            arr=np.concatenate([arr,arr],axis=1); C=2
        else:
            print('[ERR] Single-channel input. Use --use-single-as-both or provide two channels.'); return 2

    R,G = select_rg(arr, args.red_idx, args.green_idx)

    # Segment
    labels_list = segment_series(R,G,args.diameter,args.flow_th,args.prob_th,gpu=(not args.no_gpu))

    # Measure per frame
    measures_by_t=[]; per_frame_dir=outroot/'per_frame'; ensure_outdir(per_frame_dir)
    for t in range(T):
        L=labels_list[t]; df=measure_frame(R[t],G[t],L); measures_by_t.append(df)
        fdir=per_frame_dir/f'{t:04d}'; ensure_outdir(fdir)
        if args.save_labels: io.imsave(str(fdir/'labels.tif'), L.astype(np.int32), check_contrast=False)
        if args.save_overlay: io.imsave(str(fdir/'overlay.png'), labels_to_overlay(np.maximum(R[t],G[t]), L), check_contrast=False)
        if args.save_roi_csv: df.to_csv(fdir/'perROI.csv', index=False)

    # Track with btrack if possible and Detection exists; else TrackPy fallback
    if _has_btrack and _has_detection:
        try:
            H0,W0 = labels_list[0].shape if len(labels_list) else (None,None)
            tracks_long = track_with_btrack(measures_by_t, frame_shape=(H0,W0))
        except Exception as e:
            print('[WARN] btrack failed, falling back to TrackPy:', e)
            if not _has_trackpy:
                print('[ERR] TrackPy not available:', globals().get('_tp_err','')); return 2
            tracks_long = track_with_trackpy(measures_by_t, args.tp_search, args.tp_memory)
    else:
        if not _has_trackpy:
            print('[ERR] btrack unavailable and TrackPy not installed.');
            if not _has_btrack: print(' - btrack import error:', globals().get('_btrack_err',''))
            else: print(' - btrack present but missing Detection attribute.')
            print('Try: pip install -U btrack trackpy'); return 2
        tracks_long = track_with_trackpy(measures_by_t, args.tp_search, args.tp_memory)

    # Save outputs
    tracks_long.to_csv(outroot/'tracks_long.csv', index=False)
    g=tracks_long.groupby('track_id', as_index=False)
    summary=g.agg(frames=('t','count'), t_min=('t','min'), t_max=('t','max'),
                  mean_ratio=('ratio_rg','mean'), median_ratio=('ratio_rg','median'),
                  mean_red=('red_sum','mean'), mean_green=('green_sum','mean'),
                  mean_area_px=('area_px','mean'))
    summary['duration_frames']=summary['t_max']-summary['t_min']+1
    summary.to_csv(outroot/'tracks_summary.csv', index=False)
    print('[OK] wrote:'); print(' -', outroot/'tracks_long.csv'); print(' -', outroot/'tracks_summary.csv')
    return 0

if __name__=='__main__':
    sys.exit(main())
