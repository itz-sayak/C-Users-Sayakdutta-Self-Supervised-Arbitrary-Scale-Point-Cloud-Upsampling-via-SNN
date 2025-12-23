"""
Point Cloud Upsampling Generator for SNN-based Architecture

This module generates dense point clouds by:
1. Creating seed points using dense.cpp
2. Predicting normals using fn model (SNN encoder + Standard decoder)
3. Predicting distances using fd model (SNN encoder + Standard decoder)
4. Moving seed points along predicted normals by predicted distances

Architecture Note:
    - Both fn and fd models use SNN only in encoder for temporal feature extraction
    - Decoders are standard MLP for efficient inference
"""

import torch
import torch.optim as optim
from torch import autograd
import numpy as np
import math
from tqdm import trange
import trimesh
import copy
import time
from tqdm import tqdm
from sklearn.neighbors import KDTree
import torch.nn.functional as F
import os


def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2
    
    Args:
        vec1: A 3d "source" vector
        vec2: A 3d "destination" vector
    Returns:
        mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v):  # if not all zeros
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    else:
        return np.eye(3)  # cross of all zeros only occurs on identical directions


class Generator3D6(object):
    """
    Point Cloud Upsampling Generator using SNN-based models.
    
    This generator works with models that have SNN in encoder only:
        - fn model: ImprovedSNNNormalEstimation (SNN encoder + Standard decoder)
        - fd model: EnhancedSNNDistanceEstimation (SNN encoder + Standard decoder)
    
    Upsampling Process:
        1. Generate dense seed points on input point cloud surface
        2. For each seed point, find K nearest neighbors from input
        3. Predict surface normal at seed point using fn model
        4. Rotate patch to align normal with x-axis
        5. Predict distance to true surface using fd model
        6. Move seed point along normal by predicted distance
        7. Remove outliers from final point cloud
    """

    def __init__(self, model1, model2, device, k_neighbors=100, 
                 dense_spacing=0.004, outlier_threshold=1.5, batch_size=400):
        """
        Initialize the generator.
        
        Args:
            model1: Normal estimation model (fn - SNN encoder)
            model2: Distance estimation model (fd - SNN encoder)
            device: torch device
            k_neighbors: Number of neighbors for local patches
            dense_spacing: Spacing for seed point generation
            outlier_threshold: Threshold for outlier removal
            batch_size: Batch size for processing
        """
        self.model1 = model1  # fn model (normal estimation)
        self.model2 = model2  # fd model (distance estimation)
        self.device = device
        self.k_neighbors = k_neighbors
        self.dense_spacing = dense_spacing
        self.outlier_threshold = outlier_threshold
        self.batch_size = batch_size
        
        # Set models to eval mode
        self.model1.eval()
        self.model2.eval()

    def upsample(self, data):
        """Main upsampling entry point."""
        pc = self.generateiopoint(data)
        return pc

    def generateiopoint(self, data):
        """
        Generate upsampled point cloud.
        
        Args:
            data: Input point cloud [1, N, 3] or [N, 3]
            
        Returns:
            Upsampled point cloud [M, 3]
        """
        data = np.squeeze(data, 0)
        tree1 = KDTree(data)
        
        # Step 1: Generate seed points using dense.cpp
        print("Generating seed points...")
        wq = f"./dense {self.dense_spacing} {data.shape[0]}"
        print(wq)
        os.system(wq)
        data2 = np.loadtxt("target.xyz")
        xyz2 = data2[:, 0:3]
        
        # Step 2: Predict normals using fn model (SNN encoder + Standard decoder)
        print("Predicting normals (fn model - SNN encoder)...")
        pp = max(1, xyz2.shape[0] // self.batch_size)
        p_split = np.array_split(xyz2, pp, axis=0)
        normal = None
        
        for i in tqdm(range(len(p_split)), desc="Normal prediction"):
            dist, idx = tree1.query(p_split[i], self.k_neighbors)
            cloud = data[idx]
            cloud = cloud - np.tile(np.expand_dims(p_split[i], 1), (1, self.k_neighbors, 1))
            
            with torch.no_grad():
                # Reset SNN states for each batch
                if hasattr(self.model1, 'reset_states'):
                    self.model1.reset_states()
                
                # Direct forward pass (SNN encoder + Standard decoder)
                cloud_tensor = torch.from_numpy(cloud).float().to(self.device)
                n = self.model1(cloud_tensor)
                n = F.normalize(n, dim=-1)  # Ensure unit normals
            
            n = n.detach().cpu().numpy()
            if normal is None:
                normal = n
            else:
                normal = np.append(normal, n, axis=0)

        n_split = np.array_split(normal, pp, axis=0)
        xyzout = []
        
        # Step 3: Predict distances using fd model (SNN encoder + Standard decoder)
        print("Predicting distances (fd model - SNN encoder)...")
        for i in tqdm(range(len(p_split)), desc="Distance prediction"):
            dist, idx = tree1.query(p_split[i], self.k_neighbors)
            cloud = data[idx]
            cloud = cloud - np.tile(np.expand_dims(p_split[i], 1), (1, self.k_neighbors, 1))

            # Rotate patches to align normals with x-axis
            for j in range(cloud.shape[0]):
                M1 = rotation_matrix_from_vectors(n_split[i][j], [1, 0, 0])
                cloud[j] = (np.matmul(M1, cloud[j].T)).T

            with torch.no_grad():
                # Reset SNN states for each batch
                if hasattr(self.model2, 'reset_states'):
                    self.model2.reset_states()
                
                # Direct forward pass (SNN encoder + Standard decoder)
                cloud_tensor = torch.from_numpy(cloud).float().to(self.device)
                n = self.model2(cloud_tensor)

            length = np.tile(np.expand_dims(n.detach().cpu().numpy(), 1), (1, 3))
            xyzout.extend((p_split[i] + n_split[i] * length).tolist())

        xyzout = np.array(xyzout)
        
        # Step 4: Remove outliers
        print("Removing outliers...")
        tree3 = KDTree(xyzout)
        dist, idx = tree3.query(xyzout, 30)
        avg = np.mean(dist, axis=1)
        avgtotal = np.mean(dist)
        idx = np.where(avg < avgtotal * self.outlier_threshold)[0]
        xyzout = xyzout[idx, :]
        
        print(f"Output points: {xyzout.shape[0]} (removed {len(avg) - len(idx)} outliers)")

        return xyzout


# Extended generator with additional features
class SNNPointCloudGenerator(Generator3D6):
    """
    Extended generator with additional features for SNN-based upsampling.
    
    Additional features:
        - Configurable upsampling ratio
        - Multiple upsampling passes
        - Uncertainty estimation (optional)
    """
    
    def __init__(self, model1, model2, device, **kwargs):
        super().__init__(model1, model2, device, **kwargs)
        self.upsampling_ratio = kwargs.get('upsampling_ratio', 4)
    
    def multi_scale_upsample(self, data, num_passes=1):
        """
        Multi-pass upsampling for higher ratios.
        
        Args:
            data: Input point cloud
            num_passes: Number of upsampling passes
            
        Returns:
            Upsampled point cloud
        """
        result = data
        for i in range(num_passes):
            print(f"\n=== Upsampling pass {i+1}/{num_passes} ===")
            result = self.upsample(np.expand_dims(result, 0) if result.ndim == 2 else result)
        return result