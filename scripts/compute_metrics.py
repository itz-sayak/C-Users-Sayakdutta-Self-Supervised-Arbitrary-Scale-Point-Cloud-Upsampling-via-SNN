import os
import argparse
import numpy as np
import glob
import json
from tqdm import tqdm
from sklearn.neighbors import KDTree

# Chamfer distance (bidirectional) between two point sets
def chamfer_distance(a, b):
    # a,b: (N,3),(M,3)
    tree_b = KDTree(b)
    dist_a, _ = tree_b.query(a, k=1)
    tree_a = KDTree(a)
    dist_b, _ = tree_a.query(b, k=1)
    return float(np.mean(dist_a**2) + np.mean(dist_b**2))

# angular error between normals
def angular_error_deg(pred, gt):
    # pred,gt: (N,3)
    # align shapes
    assert pred.shape==gt.shape
    dot = np.sum(pred*gt, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    ang = np.arccos(np.abs(dot))
    return float(np.mean(np.degrees(ang)))


def load_xyz_or_np(file):
    if file.endswith('.ply'):
        try:
            import trimesh
            p = trimesh.load(file, process=False)
            return np.asarray(p.vertices)
        except Exception as e:
            raise
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
            # try first array
            return d[list(d.files)[0]]
    else:
        raise ValueError('Unknown file format: '+file)


def main(args):
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
        # find gt: search under gt_root for matching model id
        # try direct match: gt_root/name.npz
        gt_candidates = [
            os.path.join(gt_root, name + '.npz'),
            os.path.join(gt_root, name + '.xyz'),
        ]
        # fallback: if pred named like category_modelid, try to find model folder
        gt = None
        for cand in gt_candidates:
            if os.path.exists(cand):
                gt = load_xyz_or_np(cand)
                break
        if gt is None:
            # try search in ShapeNet dataset folders
            for cat in os.listdir(gt_root):
                model_path = os.path.join(gt_root, cat, name)
                if os.path.isdir(model_path):
                    p = os.path.join(model_path, 'pointcloud.npz')
                    if os.path.exists(p):
                        gt = load_xyz_or_np(p)
                        break
        if gt is None:
            print('GT not found for', name)
            continue
        # compute Chamfer
        cd = chamfer_distance(pred, gt)
        results[name] = {'chamfer': cd}
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print('Saved', out_json)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_dir', required=True)
    parser.add_argument('--gt_root', required=True)
    parser.add_argument('--out_json', required=True)
    args = parser.parse_args()
    main(args)
