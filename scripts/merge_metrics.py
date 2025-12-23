#!/usr/bin/env python3
import json
import glob
import os

metrics_dir = 'out/metrics'
out_file = os.path.join(metrics_dir, 'metrics_all_combined.json')

# list of metric files to include and special mapping for sinkhorn variants
files = glob.glob(os.path.join(metrics_dir, '*.json'))
# exclude the combined output if it exists
files = [f for f in files if not f.endswith('metrics_all_combined.json')]

# prefer these canonical files if present (load first to get core keys)
preferred_order = [
    'metrics_testout_full.json',
    'metrics_testout.json',
    'metrics_testout_fscore.json',
    'metrics_testout_sinkhorn.json',
    'metrics_testout_sinkhorn_tight.json',
    'metrics_testout_sinkhorn_down4096.json',
    'metrics_full.json',
    'metrics_subset.json',
    'normals.json',
    'normals_shapenet.json'
]

# sort files so preferred_order come first
def sort_key(p):
    base = os.path.basename(p)
    try:
        return (0, preferred_order.index(base))
    except ValueError:
        return (1, base)

files = sorted(files, key=sort_key)

combined = {}

for f in files:
    try:
        with open(f, 'r') as fh:
            data = json.load(fh)
    except Exception:
        continue
    name = os.path.basename(f)
    for sample, vals in data.items():
        if sample not in combined:
            combined[sample] = {}
        # map sinkhorn files to distinct keys
        if name == 'metrics_testout_sinkhorn.json':
            if 'sinkhorn' in vals:
                combined[sample]['sinkhorn_orig'] = vals.get('sinkhorn')
            # transfer preds/gt points if present
            if 'pred_points' in vals:
                combined[sample]['pred_points'] = vals.get('pred_points')
            if 'gt_points' in vals:
                combined[sample]['gt_points'] = vals.get('gt_points')
            continue
        if name == 'metrics_testout_sinkhorn_tight.json':
            if 'sinkhorn' in vals:
                combined[sample]['sinkhorn_tight'] = vals.get('sinkhorn')
            continue
        if name == 'metrics_testout_sinkhorn_down4096.json':
            if 'sinkhorn' in vals:
                combined[sample]['sinkhorn_down4096'] = vals.get('sinkhorn')
            # also record pred/gt counts
            if 'pred_points' in vals:
                combined[sample].setdefault('pred_points', vals.get('pred_points'))
            if 'gt_points' in vals:
                combined[sample]['gt_points_downsampled'] = vals.get('gt_points')
            continue
        # otherwise merge keys directly; if key exists and values differ, keep both with filename prefix
        for k, v in vals.items():
            if k in combined[sample]:
                if combined[sample][k] != v:
                    combined[sample][f'{k}__from__{name}'] = v
            else:
                combined[sample][k] = v

# write combined
os.makedirs(metrics_dir, exist_ok=True)
with open(out_file, 'w') as fh:
    json.dump(combined, fh, indent=2)

print('Wrote', out_file)
