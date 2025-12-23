#!/usr/bin/env python3
import os
import argparse
import glob
import json
import numpy as np
from tqdm import tqdm

import torch

try:
    from geomloss import SamplesLoss
except Exception:
    SamplesLoss = None


def load_xyz_or_np(file):
    if file.endswith('.ply'):
        import trimesh
        p = trimesh.load(file, process=False)
        return np.asarray(p.vertices)
    elif file.endswith('.xyz') or file.endswith('.txt'):
        return np.loadtxt(file)[:, :3]
    elif file.endswith('.npy'):
        return np.load(file)
    elif file.endswith('.npz'):
        d = np.load(file)
        if 'points' in d:
            return d['points']
        elif 'cloud' in d:
            return d['cloud']
        else:
            return d[list(d.files)[0]]
    else:
        raise ValueError('Unknown file format: '+file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--gt_root', required=True)
    parser.add_argument('--out_json', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--blur', type=float, default=0.05, help='Sinkhorn blur (regularization)')
    parser.add_argument('--p', type=int, default=2, help='Wasserstein p')
    parser.add_argument('--iters', type=int, default=200, help='Iterations for fallback Sinkhorn')
    parser.add_argument('--double', action='store_true', help='Use double precision for fallback computations')
    parser.add_argument('--downsample_gt', type=int, default=0, help='If >0, randomly downsample GT to this many points before computation')
    args = parser.parse_args()

    if SamplesLoss is None:
        raise RuntimeError('geomloss not available. Install with: pip install --user geomloss')

    device = torch.device('cuda' if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu')
    # Let geomloss select the best backend automatically.
    loss_fn = SamplesLoss(loss='sinkhorn', p=args.p, blur=args.blur, scaling=0.9, debias=False)
    results = {}

    files = sorted(glob.glob(os.path.join(args.pred_dir, '*')))
    for f in tqdm(files):
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            pred = load_xyz_or_np(f).astype(np.float32)
        except Exception as e:
            print('skip', f, e)
            continue
        # find gt
        cand = os.path.join(args.gt_root, name + '.npz')
        if os.path.exists(cand):
            gt = load_xyz_or_np(cand).astype(np.float32)
        else:
            cand2 = os.path.join(args.gt_root, name + '.xyz')
            if os.path.exists(cand2):
                gt = load_xyz_or_np(cand2).astype(np.float32)
            else:
                gt = None
                for cat in os.listdir(args.gt_root):
                    p = os.path.join(args.gt_root, cat, name + '.npz')
                    if os.path.exists(p):
                        gt = load_xyz_or_np(p).astype(np.float32)
                        break
                    p2 = os.path.join(args.gt_root, cat, name + '.xyz')
                    if os.path.exists(p2):
                        gt = load_xyz_or_np(p2).astype(np.float32)
                        break
                if gt is None:
                    print('GT not found for', name)
                    continue
        # optional downsampling of GT to reduce memory for tight Sinkhorn
        if args.downsample_gt and args.downsample_gt > 0 and gt.shape[0] > args.downsample_gt:
            idx = np.random.choice(gt.shape[0], args.downsample_gt, replace=False)
            gt = gt[idx]

        # build uniform weights (we'll rely on SamplesLoss's default uniform
        # weights to avoid shape-mismatch issues). Convert points to batched
        # tensors of shape (1, N, D) and (1, M, D).
        x = torch.from_numpy(pred).to(device)
        y = torch.from_numpy(gt).to(device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 2:
            y = y.unsqueeze(0)
        with torch.no_grad():
            try:
                # Try geomloss implementation first (fast, GPU-optimized when working)
                val = loss_fn(x, y).item()
            except Exception as e:
                print('geomloss failed for', name, 'falling back to stable log-domain Sinkhorn:', e)
                # fallback: numerically stable log-domain entropic Sinkhorn
                xx = x.squeeze(0)
                yy = y.squeeze(0)
                # choose dtype for fallback
                dtype = torch.float64 if args.double else torch.float32
                xx = xx.to(dtype=dtype)
                yy = yy.to(dtype=dtype)

                # squared cost matrix
                C = torch.cdist(xx, yy, p=2.0).pow(2)
                epsilon = float(args.blur)
                N = xx.shape[0]
                M = yy.shape[0]
                # uniform weights
                a = torch.full((N,), 1.0 / N, dtype=dtype, device=device)
                b = torch.full((M,), 1.0 / M, dtype=dtype, device=device)

                # log-domain variables
                loga = torch.log(a)
                logb = torch.log(b)
                # precompute kernel in log-space: logK = -C / epsilon
                logK = (-C / epsilon)

                # initialize logu, logv to zeros
                logu = torch.zeros(N, dtype=dtype, device=device)
                logv = torch.zeros(M, dtype=dtype, device=device)

                for _ in range(args.iters):
                    # log-sum-exp along rows: logsumexp(logK + logv)
                    # update logu
                    t = logK + logv.unsqueeze(0)  # (N, M)
                    logsum = torch.logsumexp(t, dim=1)
                    logu = loga - logsum
                    # update logv
                    t2 = logK.t() + logu.unsqueeze(0)  # (M, N)
                    logsum2 = torch.logsumexp(t2, dim=1)
                    logv = logb - logsum2

                # transport matrix in log-domain: logP = logu[:,None] + logK + logv[None,:]
                logP = logu.unsqueeze(1) + logK + logv.unsqueeze(0)
                P = torch.exp(logP)
                val = float(torch.sum(P * C).cpu().item())
        results[name] = {'sinkhorn': float(val), 'pred_points': int(pred.shape[0]), 'gt_points': int(gt.shape[0])}

    with open(args.out_json, 'w') as fh:
        json.dump(results, fh, indent=2)
    print('Saved', args.out_json)

if __name__=='__main__':
    main()
