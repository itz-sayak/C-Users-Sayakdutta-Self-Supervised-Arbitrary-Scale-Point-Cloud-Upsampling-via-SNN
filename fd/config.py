import yaml
from fd import field
import torch
from fd import datacore

def load_config(path, default_path=None):
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    inherit_from = cfg_special.get('inherit_from')

    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict()

    update_recursive(cfg, cfg_special)

    return cfg

def update_recursive(dict1, dict2):
    for k, v in dict2.items():
        if isinstance(v, dict):
            dict1[k] = update_recursive(dict1.get(k, {}), v)
        else:
            dict1[k] = v
    return dict1

def get_dataset(mode, cfg):
    # Check if using new HDF5-based dataset
    use_hdf5 = cfg['data'].get('use_hdf5', False)
    
    if use_hdf5:
        hdf5_paths = cfg['data'].get('hdf5_paths', {})
        pugan_path = hdf5_paths.get('pugan', None)
        pu1k_path = hdf5_paths.get('pu1k_train', None)
        
        # Make paths absolute if relative
        data_base = cfg['data'].get('path', 'data')
        if pugan_path and not pugan_path.startswith('/'):
            pugan_path = pugan_path
        if pu1k_path and not pu1k_path.startswith('/'):
            pu1k_path = pu1k_path
        
        split = 'train' if mode == 'train' else 'val'
        
        input_key = cfg['data'].get('hdf5_input_key', 'poisson_256')
        gt_key = cfg['data'].get('hdf5_gt_key', 'poisson_1024')
        num_input = cfg['data'].get('num_input_points', 256)
        num_gt = cfg['data'].get('num_gt_points', 1024)
        k_neighbors = cfg['model'].get('k', 20)
        
        print(f"Loading HDF5 dataset for {mode}:")
        print(f"  PUGAN: {pugan_path}")
        print(f"  PU1K: {pu1k_path}")
        
        dataset = datacore.CombinedPU1KDataset(
            pugan_path=pugan_path,
            pu1k_path=pu1k_path,
            split=split,
            input_key=input_key,
            gt_key=gt_key,
            num_input_points=num_input,
            num_gt_points=num_gt,
            k_neighbors=k_neighbors
        )
        return dataset
    
    # Legacy folder-based dataset
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }
    split = splits[mode]
    dataset_folder = cfg['data']['path']

    fields = [
        field.PointCloudField(cfg['data']['pointcloud_file']),
        field.fdField()   
    ]
    
    dataset = datacore.Shapes3dDataset(dataset_folder, fields, split=split)
    return dataset

def get_model(cfg, device):
    k = cfg['model'].get('k', 20)
    emb_dims = cfg['model'].get('emb_dims', 512)
    time_steps_enc = cfg['model'].get('time_steps_enc', 5)
    time_steps_dec = cfg['model'].get('time_steps_dec', 8)
    
    num_heads = cfg['model'].get('num_heads', 4)
    dropout = cfg['model'].get('dropout', 0.1)
    use_snn_decoder = cfg['model'].get('use_snn_decoder', False)
    
    # New multi-scale parameters
    k_scales = cfg['model'].get('k_scales', [10, 20, 40])
    
    model_type = cfg['model'].get('type', 'enhanced')
    
    from fd import snn_coder
    
    if model_type == 'enhanced':
        try:
            model = snn_coder.EnhancedSNNDistanceEstimation(
                k=k,
                emb_dims=emb_dims,
                time_steps_enc=time_steps_enc,
                time_steps_dec=time_steps_dec,
                num_heads=num_heads,
                dropout=dropout,
                use_snn_decoder=use_snn_decoder,
                k_scales=k_scales  # Pass multi-scale k values
            )
            decoder_type = "SNN" if use_snn_decoder else "Standard"
            print(f"Created EnhancedSNNDistanceEstimation with {num_heads} heads, dropout={dropout}")
            print(f"  - Encoder: SNN-based with EIF in layers 0-1, k_scales={k_scales}")
            print(f"  - Decoder: {decoder_type}")
        except Exception as e:
            print(f"Warning: Could not create enhanced model: {e}")
            print("Falling back to basic model without enhanced features...")
            try:
                model = snn_coder.EnhancedSNNDistanceEstimation(
                    k=k,
                    emb_dims=emb_dims,
                    time_steps_enc=time_steps_enc,
                    time_steps_dec=time_steps_dec
                )
                print("Created EnhancedSNNDistanceEstimation (basic version)")
            except Exception as e2:
                print(f"Error creating basic enhanced model: {e2}")
                raise
    else:
        try:
            model = snn_coder.SNNDistanceEstimation(
                k=k,
                emb_dims=emb_dims,
                time_steps_enc=time_steps_enc,
                time_steps_dec=time_steps_dec
            )
            print("Created SNNDistanceEstimation (legacy version)")
        except AttributeError:
            print("Legacy model not found, using enhanced model instead...")
            model = snn_coder.EnhancedSNNDistanceEstimation(
                k=k,
                emb_dims=emb_dims,
                time_steps_enc=time_steps_enc,
                time_steps_dec=time_steps_dec
            )
            print("Created EnhancedSNNDistanceEstimation (fallback)")
    
    return model.to(device)

def get_data_loaders(cfg, mode='train'):
    dataset = get_dataset(mode, cfg)
    
    batch_size = cfg['training']['batch_size']
    num_workers = cfg['training'].get('num_workers', 4)
    shuffle = (mode == 'train')
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True
    )
    return loader

def get_optimizer(cfg, model):
    optimizer_type = cfg['training']['optimizer']
    learning_rate = cfg['training']['learning_rate']
    weight_decay = cfg['training'].get('weight_decay', 0.0)
    
    use_snn_aware_optimizer = cfg['training'].get('use_snn_aware_optimizer', False)
    
    if use_snn_aware_optimizer:
        snn_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'membrane_decay' in name or 'threshold' in name or 'refractory' in name:
                snn_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = []
        
        if snn_params:
            param_groups.append({
                'params': snn_params,
                'lr': learning_rate * 0.1,
                'weight_decay': weight_decay
            })
        
        param_groups.append({
            'params': other_params,
            'lr': learning_rate,
            'weight_decay': weight_decay
        })
        
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(param_groups)
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=learning_rate,
                momentum=cfg['training'].get('momentum', 0.9),
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(param_groups)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    else:
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=learning_rate,
                momentum=cfg['training'].get('momentum', 0.9),
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    return optimizer

def get_scheduler(cfg, optimizer):
    scheduler_type = cfg['training'].get('scheduler', None)
    
    if scheduler_type == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg['training']['step_size'],
            gamma=cfg['training']['gamma']
        )
    elif scheduler_type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=cfg['training']['factor'],
            patience=cfg['training']['patience']
        )
    else:
        return None

def clip_gradients(optimizer, max_norm):
    if max_norm is not None and max_norm > 0:
        torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], max_norm)