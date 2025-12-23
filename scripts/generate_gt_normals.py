#!/usr/bin/env python3
"""
Generate normals for .off/.xyz files under data/ShapeNet_GT/gt and save as .npz files with keys 'points' and 'normals'.
Also copy generated .npz files to out/tmp_eval so eval scripts can find them alongside test xyz previews.
"""
import sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1]) if len(sys.argv)>1 else Path('data/ShapeNet_GT/gt')
out_tmp = Path('out/tmp_eval')
out_tmp.mkdir(parents=True, exist_ok=True)

# Try imports
try:
    import trimesh
except Exception:
    trimesh = None

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:
    NearestNeighbors = None


def load_xyz(path):
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, :3].astype(np.float32)


def estimate_normals_pca(pts, k=16):
    # simple PCA-based normal estimation using nearest neighbors
    if NearestNeighbors is None:
        raise RuntimeError('sklearn not available')
    nbrs = NearestNeighbors(n_neighbors=min(k, len(pts))).fit(pts)
    distances, indices = nbrs.kneighbors(pts)
    normals = np.zeros_like(pts)
    for i in range(len(pts)):
        neigh = pts[indices[i]]
        # subtract mean
        cov = np.cov((neigh - neigh.mean(axis=0)).T)
        # eigen decomposition
        w, v = np.linalg.eigh(cov)
        normal = v[:, 0]
        normals[i] = normal
    # normalize
    norms = np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    normals = normals / norms
    return normals.astype(np.float32)


def process_off(p):
    if trimesh is None:
        raise RuntimeError('trimesh not available')
    mesh = trimesh.load(str(p), process=False)
    # get vertices and vertex normals
    verts = np.asarray(mesh.vertices)
    if hasattr(mesh, 'vertex_normals') and mesh.vertex_normals is not None and len(mesh.vertex_normals)==len(verts):
        vnorms = np.asarray(mesh.vertex_normals)
    else:
        # compute per-vertex normals
        try:
            vnorms = mesh.vertex_normals
            vnorms = np.asarray(vnorms)
        except Exception:
            # fallback: estimate from faces
            vnorms = np.zeros_like(verts)
            if len(mesh.faces) > 0:
                face_normals = mesh.face_normals
                for fi, face in enumerate(mesh.faces):
                    for vid in face:
                        vnorms[vid] += face_normals[fi]
                # normalize
                vnorms = vnorms / (np.linalg.norm(vnorms, axis=1, keepdims=True) + 1e-9)
            else:
                vnorms = np.zeros_like(verts)
    return verts.astype(np.float32), vnorms.astype(np.float32)


def main():
    files = sorted(list(root.glob('*.off')) + list(root.glob('*.xyz')))
    if not files:
        print('No .off or .xyz files found under', root)
        return
    print('Found', len(files), 'files; processing...')
    for p in files:
        name = p.stem
        try:
            if p.suffix.lower() == '.off':
                pts, norms = process_off(p)
            else:
                pts = load_xyz(p)
                norms = estimate_normals_pca(pts, k=16)
            # save npz to gt folder
            out_npz = p.with_suffix('.npz')
            np.savez_compressed(str(out_npz), points=pts, normals=norms)
            # also copy to out/tmp_eval
            np.savez_compressed(str(out_tmp / (name + '.npz')), points=pts, normals=norms)
            print('Saved', out_npz)
        except Exception as e:
            print('Failed for', p, e)

if __name__=='__main__':
    main()
