import os
import sys
import numpy as np
import torch
# Ensure project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fd import config, datacore
from fd.trainer import Trainer
from torch.utils.data import Subset, DataLoader

cfg = config.load_config('config/fd.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = config.get_model(cfg, device)
ckpt='out/fd/model_best.pt'
if os.path.exists(ckpt):
    d=torch.load(ckpt,map_location='cpu')
    if 'model_state_dict' in d:
        model.load_state_dict(d['model_state_dict'])
    else:
        try:
            model.load_state_dict(d)
        except:
            pass
model.to(device)
trainer = Trainer(model,None,device=device)

dataset = config.get_dataset('test', cfg)
valid_idxs = []
for i, m in enumerate(dataset.models):
    category = m['category']
    model_id = m['model']
    model_path = os.path.join(cfg['data']['path'], category, model_id)
    pc = os.path.join(model_path, 'pointcloud.npz')
    fdnp = os.path.join(model_path, 'fd.npz')
    try:
        if not (os.path.exists(pc) and os.path.exists(fdnp)):
            continue
        np.load(pc)
        np.load(fdnp)
        valid_idxs.append(i)
    except Exception:
        continue

print(f"Found {len(valid_idxs)} valid test samples out of {len(dataset)}")
sub = Subset(dataset, valid_idxs)
loader = DataLoader(sub, batch_size=8, shuffle=False, collate_fn=datacore.collate_remove_none)
res = trainer.evaluate(loader)
print('FD evaluation result:', res)
