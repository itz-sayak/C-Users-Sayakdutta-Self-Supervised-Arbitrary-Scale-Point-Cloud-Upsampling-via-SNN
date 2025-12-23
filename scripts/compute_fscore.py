#!/usr/bin/env python3
import os, sys, json, glob
import numpy as np
from tqdm import tqdm

try:
    from sklearn.neighbors import KDTree
except Exception:
    raise RuntimeError('scikit-learn is required: pip install scikit-learn')


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


def fscore_from_dists(d_pred_to_gt, d_gt_to_pred, thresh):
    # precision: fraction of pred points within thresh to GT
    prec = float((d_pred_to_gt <= thresh).sum()) / len(d_pred_to_gt)
    # recall: fraction of gt points within thresh to pred
    rec = float((d_gt_to_pred <= thresh).sum()) / len(d_gt_to_pred)
    f = 2*prec*rec/(prec+rec+1e-12)
    return prec, rec, f


def compute_for_pair(pred, gt, thresholds):
    tree_gt = KDTree(gt)
    d_pred_to_gt, _ = tree_gt.query(pred, k=1)
    d_pred_to_gt = d_pred_to_gt.ravel()
    tree_pred = KDTree(pred)
    d_gt_to_pred, _ = tree_pred.query(gt, k=1)
    d_gt_to_pred = d_gt_to_pred.ravel()
    out = {}
    for t in thresholds:
        p, r, f = fscore_from_dists(d_pred_to_gt, d_gt_to_pred, t)
        out[f'prec_{t}'] = p
        out[f'rec_{t}'] = r
        out[f'fscore_{t}'] = f
    return out


def main():
    if len(sys.argv)<4:
        print('Usage: compute_fscore.py <pred_dir> <gt_root> <out_json>')
        sys.exit(1)
    pred_dir = sys.argv[1]
    gt_root = sys.argv[2]
    out_json = sys.argv[3]
    thresholds = [0.01, 0.02]

    results = {}
    files = sorted(glob.glob(os.path.join(pred_dir, '*')))
    for f in tqdm(files):
        name = os.path.splitext(os.path.basename(f))[0]
        try:
            pred = load_xyz_or_np(f)
        except Exception as e:
            print('skip', f, e)
            continue
        # find gt
        cand = os.path.join(gt_root, name + '.npz')
        if os.path.exists(cand):
            gt = load_xyz_or_np(cand)
        else:
            cand2 = os.path.join(gt_root, name + '.xyz')
            if os.path.exists(cand2):
                gt = load_xyz_or_np(cand2)
            else:
                # search in subfolders
                gt = None
                for cat in os.listdir(gt_root):
                    p = os.path.join(gt_root, cat, name + '.npz')
                    if os.path.exists(p):
                        gt = load_xyz_or_np(p)
                        break
                    p2 = os.path.join(gt_root, cat, name + '.xyz')
                    if os.path.exists(p2):
                        gt = load_xyz_or_np(p2)
                        break
                if gt is None:
                    print('GT not found for', name, '-> skipping')
                    continue
        out = compute_for_pair(pred, gt, thresholds)
        results[name] = out

    with open(out_json, 'w') as fh:
        json.dump(results, fh, indent=2)
    print('Saved', out_json)

if __name__=='__main__':
    main()
