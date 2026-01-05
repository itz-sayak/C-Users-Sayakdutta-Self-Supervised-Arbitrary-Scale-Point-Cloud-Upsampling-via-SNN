#!/usr/bin/env python3
"""
Evaluate `fn` predicted normals against ground-truth normals using ShapeNet test splits.
Saves: out/metrics/normals_shapenet.json and per-sample histogram PNGs in out/metrics/histograms_shapenet/

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_normals_shapenet.py --data_root data/ShapeNet --out_dir out/metrics --device cuda
"""
import os
import argparse
from pathlib import Path
import json
import numpy as np
import torch

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

import fn.config
import fn.checkpoints


def load_npz_arrays(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    keys = list(d.files)
    arrays = {k: d[k] for k in keys}
    return arrays


def find_arrays_for_model(model_dir):
    # Try fn.npz first
    p_fn = Path(model_dir) / 'fn.npz'
    if p_fn.exists():
        try:
            arrs = load_npz_arrays(str(p_fn))
            return arrs
        except Exception:
            pass
    # fallback: pointcloud.npz
    p_pc = Path(model_dir) / 'pointcloud.npz'
    if p_pc.exists():
        try:
            arrs = load_npz_arrays(str(p_pc))
            return arrs
        except Exception:
            pass
    # no usable file
    return None


def get_points_from_arrays(arrs):
    # prefer commonly used keys
    for k in ['points', 'pointcloud', 'cloud', 'pts', 'points_xyz', 'points_np', 'pcd']:
        if k in arrs:
            return np.asarray(arrs[k])
    # otherwise return first array if it's Nx3
    for k,v in arrs.items():
        a = np.asarray(v)
        if a.ndim==2 and a.shape[1]==3:
            return a
    return None


def get_normals_from_arrays(arrs):
    for k in ['normals','fn','gt_normals','fn_normals','normal','vertex_normals']:
        if k in arrs:
            return np.asarray(arrs[k])
    # try to infer: any Nx3 array not equal to points
    pts = get_points_from_arrays(arrs)
    for k,v in arrs.items():
        a = np.asarray(v)
        if a.ndim==2 and a.shape[1]==3:
            if pts is None or a.shape[0]==pts.shape[0]:
                # candidate
                if not np.allclose(a, pts[:a.shape[0]]):
                    return a
    return None


def angular_error_deg(a, b):
    # normalize
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    dot = np.clip(np.sum(an*bn, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(np.abs(dot)))
    return ang


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='data/ShapeNet')
    parser.add_argument('--out_dir', default='out/metrics')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_models', type=int, default=0)
    parser.add_argument('--model_list', default=None, help='comma-separated list of model ids or cat/mid to evaluate')
    args = parser.parse_args()

    model_list_set = None
    if args.model_list:
        model_list_set = set([s.strip() for s in args.model_list.split(',') if s.strip()])

    device = torch.device('cuda' if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu')
    print('Using device:', device)

    cfg = fn.config.load_config('config/fn.yaml')
    model = fn.config.get_model(cfg, device)
    ck = fn.checkpoints.CheckpointIO('out/fn', model=model)
    ck.load('model_best.pt')
    model.eval()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_dir = out_dir / 'histograms_shapenet'
    hist_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    processed = 0

    # collect iterator of (cat, mid) pairs to process
    cats = [p.name for p in data_root.iterdir() if p.is_dir()]
    iterator = []
    if model_list_set is not None:
        # search for matching pairs by mid, cat, cat/mid, or npz basename
        for cat in cats:
            lst_path = data_root / cat / 'test.lst'
            if not lst_path.exists():
                continue
            with open(lst_path, 'r') as f:
                ids = [l.strip() for l in f.readlines() if l.strip()]
            for mid in ids:
                model_dir = data_root / cat / mid
                if not model_dir.exists():
                    continue
                if mid in model_list_set or cat in model_list_set or f'{cat}/{mid}' in model_list_set:
                    iterator.append((cat, mid))
                    continue
                for fpath in model_dir.glob('*.npz'):
                    if fpath.stem in model_list_set:
                        iterator.append((cat, mid))
                        break
        # deduplicate while preserving order
        seen = set()
        dedup = []
        for pair in iterator:
            if pair not in seen:
                dedup.append(pair)
                seen.add(pair)
        iterator = dedup
    else:
        for cat in cats:
            lst_path = data_root / cat / 'test.lst'
            if not lst_path.exists():
                continue
            with open(lst_path, 'r') as f:
                ids = [l.strip() for l in f.readlines() if l.strip()]
            for mid in ids:
                iterator.append((cat, mid))

    # process each selected (cat, mid)
    for cat, mid in iterator:
        model_dir = data_root / cat / mid
        if not model_dir.exists():
            continue
        arrs = find_arrays_for_model(model_dir)
        if arrs is None:
            continue
        pts = get_points_from_arrays(arrs)
        norms = get_normals_from_arrays(arrs)
        if pts is None or norms is None:
            continue

        # normalize pts as generate.py
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        loc = (bbox_min + bbox_max) / 2.0
        scale = (bbox_max - bbox_min).max()
        normalized = (pts - loc) * (1.0/scale)
        input_arr = np.expand_dims(normalized, 0).astype(np.float32)

        try:
            with torch.no_grad():
                inp = torch.from_numpy(input_arr).float().to(device)
                outp = model(inp)
                if isinstance(outp, (list,tuple)):
                    pred = outp[0]
                else:
                    pred = outp
                pred = pred.detach().cpu().numpy().reshape(-1,3)
        except Exception as e:
            print('Model forward failed for', mid, e)
            continue

        # if sizes differ, match pred->gt by nearest neighbor
        if pred.shape[0] != norms.shape[0]:
            from sklearn.neighbors import KDTree
            tree = KDTree(norms)
            _, idx = tree.query(pred, k=1)
            matched_gt = norms[idx[:,0]]
            ang = angular_error_deg(pred, matched_gt)
        else:
            ang = angular_error_deg(pred, norms)

        mean_ang = float(np.mean(ang))
        median_ang = float(np.median(ang))
        results[f'{cat}/{mid}'] = {'mean_deg': mean_ang, 'median_deg': median_ang, 'count': int(ang.shape[0])}

        # save histogram
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(4,3), dpi=120)
            plt.hist(ang, bins=50)
            plt.xlabel('Angular error (deg)')
            plt.ylabel('Count')
            plt.title(f'{cat}/{mid}')
            plt.tight_layout()
            plt.savefig(hist_dir / (f'{cat}_{mid}.png'))
            plt.close()
        except Exception:
            pass

        processed += 1
        if args.max_models>0 and processed>=args.max_models:
            break

    out_json = out_dir / 'normals_shapenet.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print('Processed', processed, 'models. Saved', out_json)

if __name__=='__main__':
    main()
