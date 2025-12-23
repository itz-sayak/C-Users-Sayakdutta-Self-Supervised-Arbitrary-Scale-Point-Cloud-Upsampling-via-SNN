"""
Point Cloud Upsampling Script for SNN-based Architecture

Usage:
    python generate.py                    # Process all test files
    python generate.py --input cow.xyz    # Process single file
    python generate.py --ratio 4          # Set upsampling ratio
    
Architecture:
    - fn model: SNN encoder + Standard decoder (normal estimation)
    - fd model: SNN encoder + Standard decoder (distance estimation)
"""

import torch
import os
import shutil
import argparse
from tqdm import tqdm
import time
import random
from collections import defaultdict
import fd.config
import fn.config
import fn.checkpoints
import fd.checkpoints
from generation import Generator3D6
import numpy as np


def farthest_point_sample(xyz, pointnumber):
    """
    Farthest point sampling for uniform point distribution.
    
    Args:
        xyz: Input points [N, 3]
        pointnumber: Target number of points
        
    Returns:
        Indices of selected points
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    N, C = xyz.shape
    torch.seed()
    xyz = torch.from_numpy(xyz).float().to(device)
    centroids = torch.zeros(pointnumber, dtype=torch.long).to(device)

    distance = torch.ones(N).to(device) * 1e32
    farthest = torch.randint(0, N, (1,), dtype=torch.long).to(device)
    farthest[0] = N // 2
    
    for i in tqdm(range(pointnumber), desc="FPS"):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        
    return centroids.detach().cpu().numpy().astype(int)


def normalize_pointcloud(cloud):
    """Normalize point cloud to unit bounding box centered at origin."""
    bbox = np.zeros((2, 3))
    bbox[0] = np.min(cloud, axis=0)
    bbox[1] = np.max(cloud, axis=0)
    
    loc = (bbox[0] + bbox[1]) / 2
    scale = (bbox[1] - bbox[0]).max()
    scale_inv = 1.0 / scale
    
    normalized = (cloud - loc) * scale_inv
    return normalized, loc, scale


def denormalize_pointcloud(cloud, loc, scale):
    """Denormalize point cloud back to original scale."""
    return cloud * scale + loc


def process_single_file(input_path, output_path, generator, target_points=8192):
    """
    Process a single point cloud file.
    
    Args:
        input_path: Path to input .xyz file
        output_path: Path to output .xyz file
        generator: Generator3D6 instance
        target_points: Target number of output points
    """
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")
    print(f"{'='*60}")
    
    # Load point cloud
    cloud = np.loadtxt(input_path)
    cloud = cloud[:, 0:3]
    print(f"Input points: {cloud.shape[0]}")
    
    # Normalize
    normalized, loc, scale = normalize_pointcloud(cloud)
    np.savetxt("test.xyz", normalized)
    cloud_input = np.expand_dims(normalized, 0)
    
    # Upsample using SNN-based models
    print("\nUpsampling with SNN encoder architecture...")
    start_time = time.time()
    upsampled = np.array(generator.upsample(cloud_input))
    upsample_time = time.time() - start_time
    print(f"Upsampling time: {upsample_time:.2f}s")
    print(f"Upsampled points: {upsampled.shape[0]}")
    
    # Denormalize
    upsampled = denormalize_pointcloud(upsampled, loc, scale)
    
    # Farthest point sampling to get target number of points
    if upsampled.shape[0] > target_points:
        print(f"\nApplying FPS to get {target_points} points...")
        centroids = farthest_point_sample(upsampled, target_points)
        output_cloud = upsampled[centroids]
    else:
        output_cloud = upsampled
    
    # Save output
    np.savetxt(output_path, output_cloud)
    print(f"Saved to: {output_path}")
    print(f"Output points: {output_cloud.shape[0]}")
    
    return output_cloud


def main():
    parser = argparse.ArgumentParser(description='SNN-based Point Cloud Upsampling')
    parser.add_argument('--input', type=str, default=None,
                        help='Path to single input file (processes all test files if not specified)')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output file (auto-generated if not specified)')
    parser.add_argument('--target_points', type=int, default=8192,
                        help='Target number of output points')
    parser.add_argument('--batch_size', type=int, default=400,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # Setup device
    is_cuda = torch.cuda.is_available() and args.device == 'cuda'
    device = torch.device("cuda" if is_cuda else "cpu")
    print(f"\n{'='*60}")
    print(f"SNN-based Point Cloud Upsampling")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Architecture: SNN encoder + Standard decoder")

    # Load configs - use correct path
    cfg1 = fn.config.load_config('config/fn.yaml')
    cfg2 = fd.config.load_config('config/fd.yaml')

    # Create models
    print("\nLoading models...")
    model = fn.config.get_model(cfg1, device)
    model2 = fd.config.get_model(cfg2, device)

    # Load checkpoints
    checkpoint_io1 = fn.checkpoints.CheckpointIO('out/fn', model=model)
    checkpoint_io2 = fd.checkpoints.CheckpointIO('out/fd', model=model2)
    
    try:
        load_dict1 = checkpoint_io1.load('model_best.pt')
        print(f"Loaded fn model from out/fn/model_best.pt")
    except FileNotFoundError:
        print("Warning: fn model checkpoint not found. Using untrained model.")
    
    try:
        load_dict2 = checkpoint_io2.load('model_best.pt')
        print(f"Loaded fd model from out/fd/model_best.pt")
    except FileNotFoundError:
        print("Warning: fd model checkpoint not found. Using untrained model.")

    # Set to eval mode
    model.eval()
    model2.eval()

    # Create generator
    generator = Generator3D6(model, model2, device, batch_size=args.batch_size)

    # Create output directory
    os.makedirs('testout', exist_ok=True)

    if args.input:
        # Process single file
        output_path = args.output or args.input.replace('test/', 'testout/')
        process_single_file(args.input, output_path, generator, args.target_points)
    else:
        # Process all test files
        datalist = [
            'test/cow.xyz', 'test/coverrear_Lp.xyz', 'test/chair.xyz',
            'test/camel.xyz', 'test/casting.xyz', 'test/duck.xyz',
            'test/eight.xyz', 'test/elephant.xyz', 'test/elk.xyz',
            'test/fandisk.xyz', 'test/genus3.xyz', 'test/horse.xyz',
            'test/Icosahedron.xyz', 'test/kitten.xyz', 'test/moai.xyz',
            'test/Octahedron.xyz', 'test/pig.xyz', 'test/quadric.xyz',
            'test/sculpt.xyz', 'test/star.xyz'
        ]
        
        outlist = [f.replace('test/', 'testout/') for f in datalist]
        
        print(f"\nProcessing {len(datalist)} test files...")
        
        for input_path, output_path in zip(datalist, outlist):
            if os.path.exists(input_path):
                process_single_file(input_path, output_path, generator, args.target_points)
            else:
                print(f"Warning: {input_path} not found, skipping.")
        
        print(f"\n{'='*60}")
        print("All processing complete!")
        print(f"Results saved to testout/")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
