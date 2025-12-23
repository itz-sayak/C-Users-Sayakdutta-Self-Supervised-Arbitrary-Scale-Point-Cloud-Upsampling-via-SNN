#!/usr/bin/env python3
"""
Visualize generated point clouds and color points by nearest-neighbor error to ground truth.
Saves PNGs into `out/visuals/`.

Usage:
    python scripts/visualize_results.py --pred_dir testout --gt_root test --out_dir out/visuals
"""
import os
import argparse
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from sklearn.neighbors import KDTree
except Exception:
    KDTree = None


def load_xyz(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :3].astype(np.float32)


def visualize(gen_pts, gt_pts, out_png, title=None, cmap='viridis'):
    if KDTree is None:
        raise RuntimeError('sklearn is required for KDTree; pip install scikit-learn')

    tree = KDTree(gt_pts)
    dists, _ = tree.query(gen_pts, k=1)
    dists = dists.ravel()

    fig = plt.figure(figsize=(10,5), dpi=150)
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2, projection='3d')

    # GT (simple gray)
    ax1.scatter(gt_pts[:,0], gt_pts[:,1], gt_pts[:,2], c='gray', s=1)
    ax1.set_title('Ground Truth')
    ax1.axis('off')

    # Generated colored by error
    p = ax2.scatter(gen_pts[:,0], gen_pts[:,1], gen_pts[:,2], c=dists, cmap=cmap, s=1)
    ax2.set_title('Generated (color = NN distance)')
    ax2.axis('off')

    fig.suptitle(title if title else '')

    # Place colorbar in its own axis on the right to avoid obstructing plots
    # create a narrow axis on the right side of the figure for the colorbar
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cb = fig.colorbar(p, cax=cax)
    cb.set_label('NN distance')

    # adjust layout to make room for the colorbar
    plt.subplots_adjust(left=0.05, right=0.9, top=0.88, bottom=0.05)
    fig.savefig(out_png, bbox_inches='tight', dpi=150)
    plt.close(fig)


def find_gt_for(pred_name, gt_root):
    # direct match
    cand = Path(gt_root) / (pred_name + '.xyz')
    if cand.exists():
        return str(cand)
    cand = Path(gt_root) / (pred_name + '.npz')
    if cand.exists():
        return str(cand)
    # search inside category folders
    for cat in os.listdir(gt_root):
        p = Path(gt_root) / cat / (pred_name + '.npz')
        if p.exists():
            return str(p)
        p2 = Path(gt_root) / cat / (pred_name + '.xyz')
        if p2.exists():
            return str(p2)
    return None


def load_any(path):
    p = Path(path)
    if p.suffix == '.npz' or p.suffix == '.npY' or p.suffix == '.npy':
        d = np.load(str(p))
        # choose first array if npz
        if isinstance(d, np.lib.npyio.NpzFile):
            key = list(d.files)[0]
            return d[key][:,:3]
        else:
            return d[:,:3]
    else:
        return load_xyz(str(p))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', default='testout')
    parser.add_argument('--gt_root', default='test')
    parser.add_argument('--out_dir', default='out/visuals')
    parser.add_argument('--max_files', type=int, default=0, help='0 = all')
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_root = args.gt_root
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(pred_dir.glob('*.xyz'))
    if args.max_files>0:
        files = files[:args.max_files]

    for f in files:
        name = f.stem
        gt_path = find_gt_for(name, gt_root)
        if gt_path is None:
            print(f"GT not found for {name}, skipping")
            continue
        try:
            gen = load_any(str(f))
            gt = load_any(gt_path)
        except Exception as e:
            print(f"Error loading {name}: {e}")
            continue

        out_png = out_dir / (name + '.png')
        title = name
        print(f"Visualizing {name} -> {out_png}")
        try:
            visualize(gen, gt, str(out_png), title=title)
        except Exception as e:
            print(f"Failed to visualize {name}: {e}")

if __name__=='__main__':
    main()
