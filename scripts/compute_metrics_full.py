#!/usr/bin/env python3
import os
import argparse
import numpy as np
import glob
import json
from tqdm import tqdm
from sklearn.neighbors import KDTree


def chamfer_distance(a, b):
    tree_b = KDTree(b)
    dist_a, _ = tree_b.query(a, k=1)
    tree_a = KDTree(a)
    dist_b, _ = tree_a.query(b, k=1)
    return float(np.mean(dist_a**2) + np.mean(dist_b**2)), dist_a.ravel(), dist_b.ravel()


def hausdorff_distance(a, b):
    # symmetric Hausdorff: max(max(min_dists_a_to_b), max(min_dists_b_to_a))
    tree_b = KDTree(b)
    da, _ = tree_b.query(a, k=1)
    tree_a = KDTree(a)
    db, _ = tree_a.query(b, k=1)
    return float(max(np.max(da), np.max(db)))


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


def estimate_normals_pca(pts, k=16):
    # PCA normals for each point using k neighbors
    if pts.shape[0] < 3:
        return np.zeros_like(pts)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(k, len(pts))).fit(pts)
    _, idx = nbrs.kneighbors(pts)
    normals = np.zeros_like(pts)
    for i in range(len(pts)):
        neigh = pts[idx[i]]
        cov = np.cov((neigh - neigh.mean(axis=0)).T)
        w, v = np.linalg.eigh(cov)
        normal = v[:, 0]
        normals[i] = normal
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    normals = normals / norms
    return normals


def angular_error_deg(pred, gt):
    # pred, gt: (N,3) assumed aligned; if shapes differ, match pred->gt by NN
    if pred.shape[0] != gt.shape[0]:
        tree = KDTree(gt)
        _, idx = tree.query(pred, k=1)
        gt_matched = gt[idx[:,0]]
    else:
        gt_matched = gt
    an = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-9)
    bn = gt_matched / (np.linalg.norm(gt_matched, axis=1, keepdims=True) + 1e-9)
    dot = np.clip(np.sum(an*bn, axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(np.abs(dot)))
    return ang


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--gt_root', required=True)
    parser.add_argument('--out_json', required=True)
    parser.add_argument('--normal_k', type=int, default=16)
    args = parser.parse_args()

    pred_dir = args.pred_dir
    gt_root = args.gt_root
    out_json = args.out_json

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
        gt = None
        cand = os.path.join(gt_root, name + '.npz')
        if os.path.exists(cand):
            d = np.load(cand)
            if 'normals' in d:
                gt_normals = np.asarray(d['normals'])
            else:
                gt_normals = None
            if 'points' in d:
                gt = np.asarray(d['points'])
            else:
                gt = load_xyz_or_np(cand)
        else:
            # try gt_root/name.xyz
            cand2 = os.path.join(gt_root, name + '.xyz')
            if os.path.exists(cand2):
                gt = load_xyz_or_np(cand2)
                gt_normals = None
            else:
                # try subfolders
                for cat in os.listdir(gt_root):
                    pth = os.path.join(gt_root, cat, name + '.npz')
                    if os.path.exists(pth):
                        d = np.load(pth)
                        gt = np.asarray(d[list(d.files)[0]])
                        gt_normals = d.get('normals', None)
                        break
        if gt is None:
            print('GT not found for', name)
            continue
        cd, da, db = chamfer_distance(pred, gt)
        hd = hausdorff_distance(pred, gt)

        # normals: get GT normals if available, else try to compute from GT geometry
        if 'gt_normals' in locals() and gt_normals is not None:
            gtn = np.asarray(gt_normals)
        else:
            # try to load normals from same npz
            try:
                d = np.load(os.path.join(gt_root, name + '.npz'))
                if 'normals' in d:
                    gtn = np.asarray(d['normals'])
                else:
                    gtn = estimate_normals_pca(gt, k=args.normal_k)
            except Exception:
                gtn = estimate_normals_pca(gt, k=args.normal_k)

        # estimate normals on pred
        predn = estimate_normals_pca(pred, k=args.normal_k)
        angs = angular_error_deg(predn, gtn)
        mean_ang = float(np.mean(angs))
        median_ang = float(np.median(angs))

        results[name] = {
            'chamfer': float(cd),
            'hausdorff': float(hd),
            'normal_mean_deg': mean_ang,
            'normal_median_deg': median_ang,
            'count': int(len(angs))
        }

    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved', out_json)

if __name__=='__main__':
    main()
