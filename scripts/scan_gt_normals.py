#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np

root = Path(sys.argv[1]) if len(sys.argv)>1 else Path('data/ShapeNet_GT')
out = Path('out/metrics')
out.mkdir(parents=True, exist_ok=True)
index = {}

def check_npz(p):
    try:
        d = np.load(p, allow_pickle=True)
        keys = list(d.files)
        has_normals = any(k.lower() in ('normals','fn','gt_normals','fn_normals','normal','vertex_normals') for k in keys)
        shapes = {k: (np.asarray(d[k]).shape) for k in keys}
        return {'file':str(p), 'keys':keys, 'shapes':shapes, 'has_normals':has_normals}
    except Exception as e:
        return {'file':str(p), 'error':str(e)}

for p in root.rglob('*.npz'):
    info = check_npz(p)
    index[str(p)] = info

# also check .npy files
for p in root.rglob('*.npy'):
    try:
        arr = np.load(p, allow_pickle=True)
        info = {'file':str(p), 'shape':arr.shape, 'dtype':str(arr.dtype)}
        info['likely_normals'] = (arr.ndim==2 and arr.shape[1]==3)
    except Exception as e:
        info = {'file':str(p), 'error':str(e)}
    index[str(p)] = info

# check .xyz and .off for per-vertex normals present (not typical) — skip

with open(out / 'gt_normals_index.json','w') as f:
    json.dump(index, f, indent=2)
print('Wrote', out / 'gt_normals_index.json')
# print a short summary
tot = len(index)
have = sum(1 for v in index.values() if isinstance(v, dict) and ((v.get('has_normals')) or v.get('likely_normals')))
print(f'Found {tot} files; candidates with normals: {have}')
