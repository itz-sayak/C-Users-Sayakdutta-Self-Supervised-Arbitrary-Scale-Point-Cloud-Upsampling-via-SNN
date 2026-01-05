"""
Point Cloud 4× Upsampling Script (PU1K)
Faithful to original generate.py semantics
"""

import torch
import os
import numpy as np
from tqdm import tqdm
import time
import fd.config
import fn.config
import fn.checkpoints
import fd.checkpoints
from generation import Generator3D6


# ============================
# Fixed 4× configuration
# ============================

PU1K_INPUT_DIRS = [
    '/mnt/zone/B/Fimproved/data/PU1K/release/PU1K/test/input_256/input_256',
    '/mnt/zone/B/Fimproved/data/PU1K/release/PU1K/test/input_512/input_512',
    '/mnt/zone/B/Fimproved/data/PU1K/release/PU1K/test/input_1024/input_1024',
    '/mnt/zone/B/Fimproved/data/PU1K/release/PU1K/test/input_2048/input_2048',
]

OUTPUT_BASE = '/mnt/zone/B/Fimproved/testout/pu1k'

INPUT_TARGET = [
    (256, 1024),
    (512, 2048),
    (1024, 4096),
    (2048, 8192),
]


# ============================
# Utility functions
# ============================

def normalize_pointcloud(cloud):
    bbox = np.zeros((2, 3))
    bbox[0] = np.min(cloud, axis=0)
    bbox[1] = np.max(cloud, axis=0)

    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()
    scale_inv = 1.0 / scale if scale > 0 else 1.0

    cloud = (cloud - loc) * scale_inv
    return cloud, loc, scale


def farthest_point_sample(xyz, npoint):
    """Return indices (same as original script)."""
    device = 'cuda'
    xyz = torch.from_numpy(xyz).float().to(device)

    N, _ = xyz.shape
    centroids = torch.zeros(npoint, dtype=torch.long).to(device)
    distance = torch.ones(N).to(device) * 1e32
    farthest = torch.tensor([N // 2], dtype=torch.long).to(device)

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]

    return centroids.cpu().numpy()


# ============================
# Processing logic
# ============================

def process_file(input_path, output_path, generator, target_points):
    cloud = np.loadtxt(input_path)
    cloud = cloud[:, :3]

    cloud, loc, scale = normalize_pointcloud(cloud)
    cloud = np.expand_dims(cloud, 0)

    # Upsample
    upsampled = np.array(generator.upsample(cloud))

    # Denormalize
    upsampled = upsampled * scale + loc

    # FPS (exactly like original)
    assert upsampled.shape[0] >= target_points, \
        f"Generated {upsampled.shape[0]} points, expected ≥ {target_points}"

    indices = farthest_point_sample(upsampled, target_points)
    output_cloud = upsampled[indices]

    np.savetxt(output_path, output_cloud, fmt='%.6f')


# ============================
# Main
# ============================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("PU1K Fixed 4× Upsampling (Benchmark-Safe)")
    print("=" * 60)
    print(f"Device: {device}")
    print("Checkpoint: model_best.pt (fixed)")
    print()

    # Load configs
    cfg_fn = fn.config.load_config('config/fn.yaml')
    cfg_fd = fd.config.load_config('config/fd.yaml')

    # Models
    model_fn = fn.config.get_model(cfg_fn, device)
    model_fd = fd.config.get_model(cfg_fd, device)

    # Load checkpoints
    fn.checkpoints.CheckpointIO('out/fn', model=model_fn).load('model_best.pt')
    fd.checkpoints.CheckpointIO('out/fd', model=model_fd).load('model_best.pt')

    model_fn.eval()
    model_fd.eval()

    # Use smaller batch_size and k_neighbors to avoid OOM

    generator = Generator3D6(model_fn, model_fd, device, batch_size=256)

    total_files = 0
    total_time = 0.0

    for input_dir, (inp, tgt) in zip(PU1K_INPUT_DIRS, INPUT_TARGET):
        if not os.path.exists(input_dir):
            print(f"Missing: {input_dir}")
            continue

        out_dir = os.path.join(OUTPUT_BASE, f'output_{tgt}')
        os.makedirs(out_dir, exist_ok=True)

        files = sorted(f for f in os.listdir(input_dir) if f.endswith('.xyz'))

        print(f"\ninput_{inp} → output_{tgt} | files: {len(files)}")

        for fname in tqdm(files):
            t0 = time.time()
            process_file(
                os.path.join(input_dir, fname),
                os.path.join(out_dir, fname),
                generator,
                tgt
            )
            total_files += 1
            total_time += time.time() - t0

    print("\nDone.")
    print(f"Files processed: {total_files}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Avg/file: {total_time / max(total_files,1):.2f}s")
    print(f"Output root: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
