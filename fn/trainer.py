import os
from tqdm import tqdm
import torch
from torch.nn import functional as F
import numpy as np
import warnings
from fn.snn_coder import enhanced_angular_loss_with_consistency

class Trainer:
    def __init__(self, model, optimizer, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False,
                 gradient_accumulation=1, use_amp=False, scaler=None,
                 grad_clip=None, grad_clip_type='norm'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample
        
        self.gradient_accumulation = gradient_accumulation
        self.use_amp = use_amp
        self.scaler = scaler
        self.grad_clip = grad_clip
        self.grad_clip_type = grad_clip_type
        self.accumulation_step = 0
        
        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
            
        if device is not None:
            self.model.to(device)

    def cos_sim(self, x1, x2):
        cosine_sim = F.cosine_similarity(x1, x2, dim=-1)
        cosine_sim = torch.clamp(cosine_sim, -1 + 1e-6, 1 - 1e-6)
        angular_error = torch.acos(cosine_sim)
        return torch.rad2deg(angular_error).mean()

    def train_step(self, data):
        self.model.train()
        self.accumulation_step += 1
        
        device_data = {}
        for k, v in data.items():
            if torch.is_tensor(v):
                device_data[k] = v.to(self.device)
            else:
                device_data[k] = v
        
        points = device_data['input'].float()
        gt_normals = device_data['normal'].float()
        
        if torch.isnan(points).any() or torch.isinf(points).any():
            print(f"WARNING: NaN/Inf detected in input points at step {self.accumulation_step}")
            return None, None
        
        if torch.isnan(gt_normals).any() or torch.isinf(gt_normals).any():
            print(f"WARNING: NaN/Inf detected in ground truth normals at step {self.accumulation_step}")
            return None, None
        
        gt_normals = F.normalize(gt_normals, dim=-1)
        
        try:
            if self.use_amp and self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred_normals = self.model(points)
                    
                    if torch.isnan(pred_normals).any() or torch.isinf(pred_normals).any():
                        print(f"WARNING: NaN/Inf in model predictions")
                        return None, None
                    
                    pred_normals = F.normalize(pred_normals, dim=-1)
                    loss, loss_dict = self.model.compute_loss(pred_normals, gt_normals, points)
                
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"WARNING: NaN/Inf in loss value")
                    return None, None
                
                scaled_loss = loss / self.gradient_accumulation
                self.scaler.scale(scaled_loss).backward()
            else:
                pred_normals = self.model(points)
                
                if torch.isnan(pred_normals).any() or torch.isinf(pred_normals).any():
                    print(f"WARNING: NaN/Inf in model predictions")
                    return None, None
                
                pred_normals = F.normalize(pred_normals, dim=-1)
                loss, loss_dict = self.model.compute_loss(pred_normals, gt_normals, points)
                
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"WARNING: NaN/Inf in loss value")
                    return None, None
                
                scaled_loss = loss / self.gradient_accumulation
                scaled_loss.backward()
            
            if self.accumulation_step % self.gradient_accumulation == 0:
                if self.grad_clip is not None:
                    if self.use_amp and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    if self.grad_clip_type == 'norm':
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    else:
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.grad_clip)
                
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                            print(f"WARNING: NaN/Inf in gradient for {name}")
                            self.optimizer.zero_grad()
                            self.accumulation_step = 0
                            return None, None
                
                if self.use_amp and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.accumulation_step = 0
            
            return loss.item(), loss_dict
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"WARNING: OOM in train_step")
                self.optimizer.zero_grad()
                self.accumulation_step = 0
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return None, None
            else:
                print(f"ERROR in train_step: {e}")
                self.optimizer.zero_grad()
                self.accumulation_step = 0
                return None, None
        except Exception as e:
            import traceback
            print(f"ERROR in train_step: {e}")
            traceback.print_exc()
            self.optimizer.zero_grad()
            self.accumulation_step = 0
            return None, None

    def evaluate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_confidence = 0.0
        total_angular_error = 0.0
        num_batches = 0
        
        metrics_dict = {
            'angular_loss': [],
            'consistency_loss': [],
            'confidence': [],
            'angular_error_deg': []
        }

        with torch.no_grad():
            for data in tqdm(val_loader, desc="Evaluating"):
                device_data = {}
                for k, v in data.items():
                    if torch.is_tensor(v):
                        device_data[k] = v.to(self.device)
                    else:
                        device_data[k] = v
                
                points = device_data['input'].float()
                gt_normals = device_data['normal'].float()
                
                if torch.isnan(points).any() or torch.isinf(points).any():
                    print(f"WARNING: NaN/Inf in validation input, skipping batch")
                    continue
                
                gt_normals = F.normalize(gt_normals, dim=-1)
                
                pred_normals = self.model(points)
                pred_normals = F.normalize(pred_normals, dim=-1)
                
                loss, loss_dict = self.model.compute_loss(pred_normals, gt_normals, points)
                
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    print(f"WARNING: NaN/Inf in validation loss, skipping batch")
                    continue
                
                batch_size = pred_normals.shape[0]
                num_patches = pred_normals.shape[1] if pred_normals.dim() == 3 else 1
                
                pred_flat = pred_normals.view(batch_size * num_patches, 3)
                gt_flat = gt_normals.view(batch_size * num_patches, 3)
                angular_error = self.compute_angular_error(pred_flat, gt_flat)
                
                total_loss += loss.item()
                total_angular_error += angular_error.item()
                
                if loss_dict:
                    if 'angular_loss' in loss_dict:
                        metrics_dict['angular_loss'].append(loss_dict['angular_loss'])
                    if 'consistency_loss' in loss_dict:
                        metrics_dict['consistency_loss'].append(loss_dict['consistency_loss'])
                    if 'confidence' in loss_dict:
                        total_confidence += loss_dict['confidence']
                        metrics_dict['confidence'].append(loss_dict['confidence'])
                
                metrics_dict['angular_error_deg'].append(angular_error.item())
                
                num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        avg_confidence = total_confidence / max(num_batches, 1)
        avg_angular_error = total_angular_error / max(num_batches, 1)
        
        avg_metrics = {}
        for key, values in metrics_dict.items():
            if values:
                avg_metrics[key] = np.mean(values)
        
        avg_metrics['angular_error_deg'] = avg_angular_error
        
        return avg_loss, avg_confidence, avg_metrics

    def eval_step(self, data):
        self.model.eval()
        
        device_data = {}
        for k, v in data.items():
            if torch.is_tensor(v):
                device_data[k] = v.to(self.device)
            else:
                device_data[k] = v
        
        points = device_data['input'].float()
        gt_normals = device_data['normal'].float()

        with torch.no_grad():
            pred_normals = self.model(points)
            loss, loss_dict = self.model.compute_loss(pred_normals, gt_normals, points)
            
            confidence = loss_dict.get('confidence', 0.0) if loss_dict else 0.0

        return loss.item(), confidence

    def compute_loss(self, data):
        warnings.warn("compute_loss is deprecated. Use train_step instead.", DeprecationWarning)
        
        self.model.train()
        
        device_data = {}
        for k, v in data.items():
            if torch.is_tensor(v):
                device_data[k] = v.to(self.device)
            else:
                device_data[k] = v
        
        points = device_data['input'].float()
        gt_normals = device_data['normal'].float()
        
        pred_normals = self.model(points)
        loss, _ = self.model.compute_loss(pred_normals, gt_normals, points)
        
        return loss

    def compute_angular_error(self, pred_normals, gt_normals):
        pred_normals_norm = F.normalize(pred_normals, dim=-1)
        gt_normals_norm = F.normalize(gt_normals, dim=-1)
        
        cosine_sim = F.cosine_similarity(pred_normals_norm, gt_normals_norm, dim=-1)
        cosine_sim = torch.clamp(cosine_sim, -1 + 1e-6, 1 - 1e-6)
        
        angular_error_rad = torch.acos(cosine_sim)
        angular_error_deg = torch.rad2deg(angular_error_rad)
        
        return angular_error_deg.mean()
    
    def get_spike_statistics(self):
        if hasattr(self.model, 'get_spike_statistics'):
            return self.model.get_spike_statistics()
        return {}
    
    def reset_states(self):
        if hasattr(self.model, 'reset_states'):
            self.model.reset_states()