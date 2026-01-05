#!/usr/bin/env python3
"""
Evaluate `fn` predicted normals against ground-truth normals and save per-sample angular error statistics.
Saves: out/metrics/normals.json and per-sample histogram PNGs in out/metrics/histograms/

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_normals.py --test_dir test --out_dir out/metrics --device cuda
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


def load_xyz(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :3].astype(np.float32)


def find_gt_normals(test_root, name):
    # Look for fn.npz or pointcloud.npz under category folders
    p = Path(test_root) / (name + '.npz')
    if p.exists():
        try:
            d = np.load(str(p))
            # common keys: 'normals', 'fn', 'points'
            for k in ['normals', 'fn', 'fn_points', 'points', 'xyz']:
                if k in d:
                    return d[k]
            # fallback: first array
            return d[list(d.files)[0]]
        except Exception:
            return None
    # Search inside category folders
    for cat in os.listdir(test_root):
        p2 = Path(test_root) / cat / (name + '.npz')
        if p2.exists():
            try:
                d = np.load(str(p2))
                for k in ['normals', 'fn', 'fn_points', 'points', 'xyz']:
                    if k in d:
                        return d[k]
                return d[list(d.files)[0]]
            except Exception:
                continue
    return None


def angular_error_deg(a, b):
    # a,b: (N,3)
    # ensure same shape by nearest neighbor mapping if lengths differ
    if a.shape[0] != b.shape[0]:
        # map a -> nearest in b
        from sklearn.neighbors import KDTree
        tree = KDTree(b)
        dists, idx = tree.query(a, k=1)
        b_matched = b[idx[:,0]]
        b = b_matched
    # normalize
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    dot = np.clip(np.sum(an*bn, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(np.abs(dot)))
    return ang


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', default='test')
    parser.add_argument('--out_dir', default='out/metrics')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_files', type=int, default=0)
    args = parser.parse_args()

    device = torch.device('cuda' if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu')
    print('Using device:', device)

    cfg = fn.config.load_config('config/fn.yaml')
    model = fn.config.get_model(cfg, device)
    ck = fn.checkpoints.CheckpointIO('out/fn', model=model)
    ck.load('model_best.pt')
    model.eval()

    pred_dir = Path('testout')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_dir = out_dir / 'histograms'
    hist_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in (Path(args.test_dir)).glob('*.xyz')])
    if args.max_files>0:
        files = files[:args.max_files]

    results = {}

    for f in tqdm(files):
        name = f.stem
        # Load input points
        pts = load_xyz(str(f))
        # try to find ground-truth normals
        gt_normals = find_gt_normals(args.test_dir, name)
        if gt_normals is None:
            print('No GT normals for', name, '- skipping')
            continue
        gt_normals = np.asarray(gt_normals)
        # Prepare input batch as generate.py: normalize and expand
        # Note: generate.py normalizes per-cloud; replicate that
        bbox_min = pts.min(axis=0)
        bbox_max = pts.max(axis=0)
        loc = (bbox_min + bbox_max) / 2.0
        scale = (bbox_max - bbox_min).max()
        normalized = (pts - loc) * (1.0/scale)
        input_arr = np.expand_dims(normalized, 0)

        # Run model to get normals for seed points - use generator internals if available
        # Here we'll call model directly with expected dataloader transform if possible.
        # The `fn` model expects patches; to keep it simple, call the model's forward with the normalized cloud if supported.
        try:
            with torch.no_grad():
                inp = torch.from_numpy(input_arr).float().to(device)
                # many model wrappers expect (B,N,3) and produce (B,N,3) normals
                out = model(inp)
                # try to extract normals from output
                if isinstance(out, tuple) or isinstance(out, list):
                    pred = out[0]
                else:
                    pred = out
                pred = pred.detach().cpu().numpy()
                # flatten to (N,3)
                pred = pred.reshape(-1, 3)
        except Exception as e:
            print('Model forward failed for', name, e)
            continue

        # compute angular error
        ang = angular_error_deg(pred, gt_normals)
        mean_ang = float(np.mean(ang))
        med_ang = float(np.median(ang))
        results[name] = {'mean_deg': mean_ang, 'median_deg': med_ang, 'count': int(ang.shape[0])}

        # save histogram
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(4,3), dpi=120)
            plt.hist(ang, bins=50)
            plt.xlabel('Angular error (deg)')
            plt.ylabel('Count')
            plt.title(name)
            plt.tight_layout()
            plt.savefig(hist_dir / (name + '.png'))
            plt.close()
        except Exception:
            pass

    out_json = out_dir / 'normals.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved', out_json)

if __name__=='__main__':
    main()
