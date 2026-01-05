import yaml
from fn import field
import torch
import numpy as np
from fn import datacore
import os
import sys
from fn.snn_coder import ImprovedSNNNormalEstimation

def load_config(path, default_path=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    inherit_from = cfg_special.get('inherit_from')
    if inherit_from is not None:
        if not os.path.isabs(inherit_from):
            inherit_from = os.path.join(os.path.dirname(path), inherit_from)
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict()

    update_recursive(cfg, cfg_special)
    set_default_config_values(cfg)
    
    return cfg

def update_recursive(dict1, dict2):
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v

def set_default_config_values(cfg):
    data_defaults = {
        'pointcloud_n': 2048,
        'patch_n': 100,
        'patch_k': 16,
        'pointcloud_noise': 0.005,
        'pointcloud_file': 'pointcloud.npz',
        'fn_file': 'fn.npz',
        'normalize_points': True,
        'normalize_scale': 1.0,
        'use_cache': False,
        'cache_size': 1000
    }
    
    for key, default in data_defaults.items():
        cfg['data'].setdefault(key, default)
    
    model_defaults = {
        'k_values': [20, 20, 16],
        'emb_dims': 1024,
        'time_steps_enc': 8,
        'time_steps_dec': 12,
        'num_heads': 4,
        'd_model': 512,
        'dropout': 0.1,
        'decoder_hidden_dims': [1024, 512, 256],
        'snn_params': {
            'membrane_decay_init': 0.9,
            'threshold_adapt_init': 0.01,
            'refractory_decay_init': 0.5,
            'grad_width': 10.0
        }
    }
    
    for key, default in model_defaults.items():
        cfg['model'].setdefault(key, default)
    
    training_defaults = {
        'batch_size': 8,
        'num_workers': 4,
        'prefetch_factor': 2,
        'lr': 0.001,
        'lr_policy': 'none',
        'lr_decay': 0.95,
        'lr_decay_step': 2000,
        'min_lr': 1e-5,
        'warmup_steps': 0,
        'warmup_factor': 0.1,
        'optimizer': 'adam',
        'weight_decay': 0.0,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'grad_clip': None,
        'grad_clip_type': 'norm',
        'gradient_accumulation': 1,
        'max_iterations': 300000,
        'max_epochs': 500,
        'print_every': 50,
        'visualize_every': 1000,
        'checkpoint_every': 1000,
        'validate_every': 2000,
        'backup_every': 50000,
        'early_stopping': False,
        'patience': 20000,
        'min_delta': 0.0001,
        'use_amp': False,
        'amp_opt_level': 'O1',
        'gradient_checkpointing': False,
        'snn_training': {
            'spike_rate_target': 0.1,
            'spike_rate_weight': 0.01,
            'state_reset_freq': 100
        }
    }
    
    for key, default in training_defaults.items():
        cfg['training'].setdefault(key, default)
    
    loss_defaults = {
        'temperature': 0.1,
        'alpha': 0.1,
        'consistency_weight': 0.15,
        'k_neighbors': 8,
        'norm_regularization': False,
        'norm_weight': 0.01,
        'spike_rate_regularization': False,
        'spike_rate_weight': 0.001
    }
    
    for key, default in loss_defaults.items():
        cfg['loss'].setdefault(key, default)

def get_dataset(mode, cfg):
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }
    split = splits[mode]
    
    # Check if using mesh-based dataset (PU1K)
    use_mesh = cfg['data'].get('use_mesh', False)
    
    if use_mesh:
        mesh_folder = cfg['data'].get('mesh_folder', '')
        if not os.path.exists(mesh_folder):
            raise FileNotFoundError(f"Mesh folder not found: {mesh_folder}")
        
        # Map split names
        split_map = {'train': 'train', 'val': 'val', 'test': 'val'}
        actual_split = split_map.get(split, 'train')
        
        dataset = datacore.PU1KMeshDataset(
            mesh_folder=mesh_folder,
            split=actual_split,
            num_points=cfg['data'].get('pointcloud_n', 512),
            num_patches=cfg['data'].get('patch_n', 64),
            k_neighbors=cfg['data'].get('patch_k', 12),
        )
        print(f"Created {mode} PU1K mesh dataset with {len(dataset)} samples")
        return dataset
    
    # Legacy folder-based dataset
    dataset_folder = cfg['data']['path']
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_folder}")
    
    fields = [
        field.PointCloudField(cfg['data']['pointcloud_file']),
        field.fnField()
    ]

    dataset = datacore.Shapes3dDataset(
        dataset_folder, 
        fields, 
        split=split
    )
    
    print(f"Created {mode} dataset with {len(dataset)} samples")
    return dataset

def get_model(cfg, device=None):
    model_cfg = cfg['model']
    
    required_params = ['k_values', 'emb_dims']
    for param in required_params:
        if param not in model_cfg:
            raise ValueError(f"Missing required model parameter: {param}")
    
    k_values = model_cfg['k_values']
    emb_dims = model_cfg['emb_dims']
    time_steps_enc = model_cfg['time_steps_enc']
    time_steps_dec = model_cfg['time_steps_dec']
    num_heads = model_cfg['num_heads']
    dropout = model_cfg.get('dropout', 0.1)
    use_snn_decoder = model_cfg.get('use_snn_decoder', False)
    decoder_dropout = model_cfg.get('decoder_dropout', 0.1)
    
    snn_params = model_cfg.get('snn_params', {})
    
    model = ImprovedSNNNormalEstimation(
        k_values=k_values,
        emb_dims=emb_dims,
        time_steps_enc=time_steps_enc,
        time_steps_dec=time_steps_dec,
        num_heads=num_heads,
        use_snn_decoder=use_snn_decoder,
        decoder_dropout=decoder_dropout
    )
    
    decoder_type = "SNN" if use_snn_decoder else "Standard"
    print(f"Created model: {cfg['model']['type']}")
    print(f"  - k_values: {k_values}")
    print(f"  - emb_dims: {emb_dims}")
    print(f"  - time_steps: {time_steps_enc}/{time_steps_dec}")
    print(f"  - num_heads: {num_heads}")
    print(f"  - dropout: {dropout}")
    print(f"  - Encoder: SNN-based (temporal dynamics)")
    print(f"  - Decoder: {decoder_type}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    if device is not None:
        model = model.to(device)
        print(f"  - Moved to device: {device}")
    
    return model

def get_optimizer(model, cfg):
    training_cfg = cfg['training']
    
    lr = training_cfg['lr']
    weight_decay = training_cfg['weight_decay']
    betas = tuple(training_cfg['betas'])
    
    snn_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'membrane_decay' in name or 'threshold' in name or 'refractory' in name:
                snn_params.append(param)
            else:
                other_params.append(param)
    
    param_groups = []
    
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': lr,
            'weight_decay': weight_decay
        })
    
    if snn_params:
        param_groups.append({
            'params': snn_params,
            'lr': lr * 0.5,
            'weight_decay': weight_decay * 0.1
        })
    
    optimizer_type = training_cfg['optimizer'].lower()
    
    if optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(
            param_groups if param_groups else model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=training_cfg.get('eps', 1e-8)
        )
    elif optimizer_type == 'adam':
        optimizer = torch.optim.Adam(
            param_groups if param_groups else model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=training_cfg.get('eps', 1e-8)
        )
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(
            param_groups if param_groups else model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=training_cfg.get('momentum', 0.9)
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    print(f"Created {optimizer_type.upper()} optimizer")
    print(f"  - Learning rate: {lr}")
    print(f"  - Weight decay: {weight_decay}")
    
    return optimizer

def get_scheduler(optimizer, cfg):
    training_cfg = cfg['training']
    lr_policy = training_cfg['lr_policy']
    
    if lr_policy == 'none' or lr_policy is None:
        return None
    
    elif lr_policy == 'step':
        lr_decay = training_cfg['lr_decay']
        lr_decay_step = training_cfg['lr_decay_step']
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=lr_decay_step, 
            gamma=lr_decay
        )
        print(f"Created StepLR scheduler: step_size={lr_decay_step}, gamma={lr_decay}")
        
    elif lr_policy == 'cosine':
        max_iterations = training_cfg['max_iterations']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=max_iterations
        )
        print(f"Created CosineAnnealingLR scheduler: T_max={max_iterations}")
        
    elif lr_policy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            patience=training_cfg.get('plateau_patience', 10),
            factor=training_cfg.get('plateau_factor', 0.5),
            min_lr=training_cfg['min_lr']
        )
        print("Created ReduceLROnPlateau scheduler")
        
    else:
        raise ValueError(f"Unsupported LR policy: {lr_policy}")
    
    return scheduler

def setup_seed(cfg):
    seed = cfg.get('hardware', {}).get('seed', 42)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        deterministic = cfg.get('hardware', {}).get('deterministic', False)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = cfg.get('hardware', {}).get('cudnn_benchmark', True)
    
    print(f"Set random seed: {seed}")
    print(f"CUDA deterministic: {cfg.get('hardware', {}).get('deterministic', False)}")

if __name__ == '__main__':
    test_config_path = 'config/fn.yaml'
    
    if os.path.exists(test_config_path):
        cfg = load_config(test_config_path)
        print("Config loaded successfully!")
        print(f"Model type: {cfg['model']['type']}")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = get_model(cfg, device)
        
        dataset = get_dataset('train', cfg)
        print(f"Dataset created with {len(dataset)} samples")
    else:
        print(f"Test config not found: {test_config_path}")