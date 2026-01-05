import os
import sys
import numpy as np
import torch
# Ensure project root is on sys.path so modules like `fn` and `fd` can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fn import config, datacore
from fn.trainer import Trainer
from torch.utils.data import Subset, DataLoader

cfg = config.load_config('config/fn.yaml')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = config.get_model(cfg, device)
# load checkpoint
ckpt='out/fn/model_best.pt'
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
    fnp = os.path.join(model_path, 'fn.npz')
    try:
        if not (os.path.exists(pc) and os.path.exists(fnp)):
            continue
        # attempt load
        np.load(pc)
        np.load(fnp)
        valid_idxs.append(i)
    except Exception:
        continue

print(f"Found {len(valid_idxs)} valid test samples out of {len(dataset)}")
if len(valid_idxs)==0:
    raise SystemExit('No valid test samples')

sub = Subset(dataset, valid_idxs)
loader = DataLoader(sub, batch_size=8, shuffle=False, collate_fn=datacore.collate_remove_none)
res = trainer.evaluate(loader)
print('FN evaluation result:', res)
