import os
import logging
from torch.utils import data
import numpy as np
import yaml
import h5py
import torch
from fd import transform as tsf

logger = logging.getLogger(__name__)


class PU1KDataset(data.Dataset):
    """HDF5-based dataset for PU1K and PUGAN data.
    
    Supports loading from single HDF5 file or combining multiple HDF5 files.
    Expected HDF5 structure: 'poisson_256' (input), 'poisson_1024' (ground truth)
    """
    
    def __init__(self, h5_paths, split='train', transform=None, 
                 input_key='poisson_256', gt_key='poisson_1024',
                 num_input_points=256, num_gt_points=1024, k_neighbors=20):
        """
        Args:
            h5_paths: str or list of str - path(s) to HDF5 file(s)
            split: 'train' or 'val' - for train, uses all data; for val, uses last 10%
            transform: optional transformation
            input_key: HDF5 key for input point clouds
            gt_key: HDF5 key for ground truth point clouds
            num_input_points: number of input points to sample
            num_gt_points: number of ground truth points
            k_neighbors: number of nearest neighbors for KNN graph
        """
        self.transform = transform
        self.input_key = input_key
        self.gt_key = gt_key
        self.num_input_points = num_input_points
        self.num_gt_points = num_gt_points
        self.k_neighbors = k_neighbors
        self.split = split
        
        # Handle single path or list of paths
        if isinstance(h5_paths, str):
            h5_paths = [h5_paths]
        
        # Load all data into memory for fast access
        self.inputs = []
        self.gt = []
        
        for h5_path in h5_paths:
            if not os.path.exists(h5_path):
                logger.warning(f"HDF5 file not found: {h5_path}")
                continue
                
            logger.info(f"Loading HDF5 data from: {h5_path}")
            with h5py.File(h5_path, 'r') as f:
                input_data = f[input_key][:]  # (N, 256, 3) or similar
                gt_data = f[gt_key][:]  # (N, 1024, 3) or similar
                
                # Ensure correct shapes
                if input_data.ndim == 3 and gt_data.ndim == 3:
                    self.inputs.append(input_data)
                    self.gt.append(gt_data)
                    logger.info(f"  Loaded {len(input_data)} samples")
                else:
                    logger.warning(f"  Unexpected shape: input {input_data.shape}, gt {gt_data.shape}")
        
        # Concatenate all data
        if self.inputs:
            self.inputs = np.concatenate(self.inputs, axis=0)
            self.gt = np.concatenate(self.gt, axis=0)
        else:
            raise ValueError("No valid HDF5 data loaded")
        
        # Split data (train: first 90%, val: last 10%)
        total_samples = len(self.inputs)
        split_idx = int(total_samples * 0.9)
        
        if split == 'train':
            self.inputs = self.inputs[:split_idx]
            self.gt = self.gt[:split_idx]
        elif split == 'val':
            self.inputs = self.inputs[split_idx:]
            self.gt = self.gt[split_idx:]
        
        logger.info(f"PU1KDataset initialized: {len(self.inputs)} samples ({split} split)")
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        input_pc = self.inputs[idx].copy()  # (256, 3)
        gt_pc = self.gt[idx].copy()  # (1024, 3)
        
        # Data augmentation for training
        if self.split == 'train':
            # Random rotation around Z-axis
            theta = np.random.uniform(0, 2 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rotation = np.array([[cos_t, -sin_t, 0],
                                  [sin_t, cos_t, 0],
                                  [0, 0, 1]], dtype=np.float32)
            input_pc = input_pc @ rotation.T
            gt_pc = gt_pc @ rotation.T
            
            # Random scaling
            scale = np.random.uniform(0.8, 1.2)
            input_pc *= scale
            gt_pc *= scale
            
            # Random jitter (small noise)
            input_pc += np.random.normal(0, 0.002, input_pc.shape).astype(np.float32)
        
        # Normalize to unit sphere
        centroid = input_pc.mean(axis=0)
        input_pc -= centroid
        gt_pc -= centroid
        
        max_dist = np.max(np.sqrt(np.sum(input_pc ** 2, axis=1)))
        if max_dist > 0:
            input_pc /= max_dist
            gt_pc /= max_dist
        
        # Compute distance field: for each input point, compute distance to nearest GT point
        # This gives us the "local resolution" information that the SNN can learn
        from scipy.spatial import cKDTree
        gt_tree = cKDTree(gt_pc)
        distances, _ = gt_tree.query(input_pc, k=1)  # Distance to nearest GT point
        distances = distances.astype(np.float32)
        
        # Sample M neighbors for each input point to form input patches
        # This matches the original fd data format
        M = min(self.k_neighbors, len(input_pc))
        input_tree = cKDTree(input_pc)
        _, neighbor_idx = input_tree.query(input_pc, k=M)
        input_patches = input_pc[neighbor_idx]  # (N, M, 3)
        
        # Return format matching original trainer expectations
        data_dict = {
            'input': torch.from_numpy(input_patches.astype(np.float32)),  # (N, M, 3)
            'len': torch.from_numpy(distances),  # (N,) - distance field values
            'cloud': torch.from_numpy(input_pc.astype(np.float32)),  # (N, 3)
            'points': torch.from_numpy(gt_pc.astype(np.float32)),  # (1024, 3) - GT for reference
        }
        
        if self.transform:
            data_dict = self.transform(data_dict)
        
        return data_dict


class CombinedPU1KDataset(data.Dataset):
    """Dataset that combines PUGAN and PU1K for training."""
    
    def __init__(self, pugan_path, pu1k_path, split='train', **kwargs):
        """
        Args:
            pugan_path: path to PUGAN HDF5 file
            pu1k_path: path to PU1K train HDF5 file
            split: 'train' or 'val'
            **kwargs: additional arguments for PU1KDataset
        """
        paths = []
        if pugan_path and os.path.exists(pugan_path):
            paths.append(pugan_path)
        if pu1k_path and os.path.exists(pu1k_path):
            paths.append(pu1k_path)
        
        if not paths:
            raise ValueError("At least one valid HDF5 path must be provided")
        
        self.dataset = PU1KDataset(paths, split=split, **kwargs)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

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