import os
import logging
from torch.utils import data
import numpy as np
import yaml
import torch
import glob
from fn import transform as tsf

logger = logging.getLogger(__name__)


class PU1KMeshDataset(data.Dataset):
    """Mesh-based dataset for PU1K - computes normals from mesh faces.
    
    Samples point clouds from .off mesh files and computes ground truth normals.
    """
    
    def __init__(self, mesh_folder, split='train', num_points=512, 
                 num_patches=64, k_neighbors=12, transform=None):
        """
        Args:
            mesh_folder: Path to folder containing category subfolders with .off files
            split: 'train' or 'val'
            num_points: Number of points to sample from mesh
            num_patches: Number of patches to sample
            k_neighbors: Number of neighbors per patch
            transform: Optional transform
        """
        self.mesh_folder = mesh_folder
        self.split = split
        self.num_points = num_points
        self.num_patches = num_patches
        self.k_neighbors = k_neighbors
        self.transform = transform
        
        # Find all .off files
        self.mesh_files = []
        
        # Check if mesh_folder has category subdirectories
        categories = [d for d in os.listdir(mesh_folder) 
                     if os.path.isdir(os.path.join(mesh_folder, d))]
        
        if categories:
            for cat in categories:
                cat_path = os.path.join(mesh_folder, cat)
                off_files = glob.glob(os.path.join(cat_path, '*.off'))
                self.mesh_files.extend(off_files)
        else:
            # Direct .off files in folder
            self.mesh_files = glob.glob(os.path.join(mesh_folder, '*.off'))
        
        if not self.mesh_files:
            raise ValueError(f"No .off files found in {mesh_folder}")
        
        # Sort for reproducibility
        self.mesh_files.sort()
        
        # Split data (train: first 90%, val: last 10%)
        total = len(self.mesh_files)
        split_idx = int(total * 0.9)
        
        if split == 'train':
            self.mesh_files = self.mesh_files[:split_idx]
        elif split == 'val':
            self.mesh_files = self.mesh_files[split_idx:]
        
        logger.info(f"PU1KMeshDataset: {len(self.mesh_files)} meshes ({split})")
    
    def __len__(self):
        return len(self.mesh_files)
    
    def _load_off(self, filepath):
        """Load .off mesh file and return vertices and faces."""
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Skip header
        start = 0
        for i, line in enumerate(lines):
            if line.strip().upper() == 'OFF':
                start = i + 1
                break
            elif line.strip().upper().startswith('OFF'):
                # OFF header on same line as counts
                start = i
                break
        
        # Parse counts
        count_line = lines[start].strip()
        if count_line.upper().startswith('OFF'):
            count_line = count_line[3:].strip()
            if not count_line:
                start += 1
                count_line = lines[start].strip()
        
        parts = count_line.split()
        n_verts = int(parts[0])
        n_faces = int(parts[1])
        
        # Parse vertices
        vertices = []
        for i in range(start + 1, start + 1 + n_verts):
            parts = lines[i].strip().split()
            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
        vertices = np.array(vertices, dtype=np.float32)
        
        # Parse faces
        faces = []
        for i in range(start + 1 + n_verts, start + 1 + n_verts + n_faces):
            parts = lines[i].strip().split()
            n = int(parts[0])
            if n >= 3:
                face_indices = [int(parts[j+1]) for j in range(n)]
                # Triangulate if needed (fan triangulation)
                for j in range(1, n - 1):
                    faces.append([face_indices[0], face_indices[j], face_indices[j+1]])
        faces = np.array(faces, dtype=np.int32)
        
        return vertices, faces
    
    def _compute_face_normals(self, vertices, faces):
        """Compute face normals from vertices and faces."""
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        normals = np.cross(edge1, edge2)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        normals = normals / norms
        
        return normals
    
    def _compute_face_areas(self, vertices, faces):
        """Compute face areas for area-weighted sampling."""
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        cross = np.cross(edge1, edge2)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        return areas
    
    def _sample_points_from_mesh(self, vertices, faces, n_points):
        """Sample points uniformly from mesh surface with normals."""
        face_normals = self._compute_face_normals(vertices, faces)
        face_areas = self._compute_face_areas(vertices, faces)
        
        # Normalize areas to probabilities
        probs = face_areas / (face_areas.sum() + 1e-8)
        
        # Sample faces proportional to area
        face_indices = np.random.choice(len(faces), size=n_points, p=probs)
        
        # Sample random points within each triangle (barycentric coords)
        r1 = np.random.random(n_points).astype(np.float32)
        r2 = np.random.random(n_points).astype(np.float32)
        
        # Convert to barycentric
        sqrt_r1 = np.sqrt(r1)
        u = 1 - sqrt_r1
        v = sqrt_r1 * (1 - r2)
        w = sqrt_r1 * r2
        
        # Get triangle vertices
        v0 = vertices[faces[face_indices, 0]]
        v1 = vertices[faces[face_indices, 1]]
        v2 = vertices[faces[face_indices, 2]]
        
        # Compute sample points
        points = u[:, None] * v0 + v[:, None] * v1 + w[:, None] * v2
        
        # Get normals for sampled points
        normals = face_normals[face_indices]
        
        return points.astype(np.float32), normals.astype(np.float32)
    
    def __getitem__(self, idx):
        mesh_path = self.mesh_files[idx]
        
        try:
            vertices, faces = self._load_off(mesh_path)
        except Exception as e:
            logger.warning(f"Error loading {mesh_path}: {e}")
            # Return a different sample
            return self.__getitem__((idx + 1) % len(self))
        
        # Sample points from mesh
        points, normals = self._sample_points_from_mesh(
            vertices, faces, self.num_points
        )
        
        # Data augmentation for training
        if self.split == 'train':
            # Random rotation around Z-axis
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rotation = np.array([[cos_t, -sin_t, 0],
                                  [sin_t, cos_t, 0],
                                  [0, 0, 1]], dtype=np.float32)
            points = points @ rotation.T
            normals = normals @ rotation.T
            
            # Random scaling (don't scale normals)
            scale = np.random.uniform(0.8, 1.2)
            points *= scale
            
            # Random jitter
            points += np.random.normal(0, 0.002, points.shape).astype(np.float32)
        
        # Normalize to unit sphere
        centroid = points.mean(axis=0)
        points -= centroid
        
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points /= max_dist
        
        # Re-normalize normals after augmentation
        norm_lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / (norm_lengths + 1e-8)
        
        # Build KNN patches
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        
        # Sample patch centers
        if len(points) > self.num_patches:
            patch_indices = np.random.choice(len(points), self.num_patches, replace=False)
        else:
            patch_indices = np.arange(len(points))
        
        # Get neighbors for each patch center
        _, neighbor_idx = tree.query(points[patch_indices], k=self.k_neighbors)
        
        # Build input patches
        input_patches = points[neighbor_idx]  # (num_patches, k_neighbors, 3)
        patch_normals = normals[patch_indices]  # (num_patches, 3)
        
        data_dict = {
            'input': torch.from_numpy(input_patches.astype(np.float32)),
            'normal': torch.from_numpy(patch_normals.astype(np.float32)),
            'cloud': torch.from_numpy(points.astype(np.float32)),
            'all_normals': torch.from_numpy(normals.astype(np.float32)),
        }
        
        if self.transform:
            data_dict = self.transform(data_dict)
        
        return data_dict

class Shapes3dDataset(data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, dataset_folder, fields, split=None,
                 categories=None, no_except=True, transform=None):
        ''' Initialization of the the 3D shape dataset.

        Args:
            dataset_folder (str): dataset folder
            fields (dict): dictionary of fields
            split (str): which split is used
            categories (list): list of categories to use
            no_except (bool): no exception
            transform (callable): transformation applied to data points
        '''
        # Attributes
        self.dataset_folder = dataset_folder
        self.fields = fields
        self.no_except = no_except
        self.transform = transform

        # If categories is None, use all subfolders
        if categories is None:
            categories = os.listdir(dataset_folder)
            categories = [c for c in categories
                          if os.path.isdir(os.path.join(dataset_folder, c))]

        # Read metadata file

        metadata_file = os.path.join(dataset_folder, 'metadata.yaml')
 
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                self.metadata = yaml.load(f)
        else:
            self.metadata = {
                c: {'id': c, 'name': 'n/a'} for c in categories
            } 
     
        # Set index
        for c_idx, c in enumerate(categories):
            self.metadata[c]['idx'] = c_idx
        # Get all models
        self.models = []
        for c_idx, c in enumerate(categories):
            subpath = os.path.join(dataset_folder, c)
            if not os.path.isdir(subpath):
                logger.warning('Category %s does not exist in dataset.' % c)

            split_file = os.path.join(subpath, split + '.lst')
            with open(split_file, 'r') as f:
                models_c = f.read().split('\n')
            
            # Filter out empty model names
            models_c = [m for m in models_c if m.strip()]
            self.models += [{'category': c, 'model': m} for m in models_c]
        

    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.models)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        category = self.models[idx]['category']
        model = self.models[idx]['model']
        c_idx = self.metadata[category]['idx']

        
        model_path = os.path.join(self.dataset_folder, category, model)
        data =tsf.GdataKNN(dict(self.fields[0].load(model_path), ** self.fields[1].load(model_path)))
        return data

    def get_model_dict(self, idx):
        return self.models[idx]

    def test_model_complete(self, category, model):
        ''' Tests if model is complete.

        Args:
            model (str): modelname
        '''
        model_path = os.path.join(self.dataset_folder, category, model)
        files = os.listdir(model_path)
        for field_name, field in self.fields.items():
            if not field.check_complete(files):
                logger.warn('Field "%s" is incomplete: %s'
                            % (field_name, model_path))
                return False

        return True


def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''

    batch = list(filter(lambda x: x is not None, batch))

    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)