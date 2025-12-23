import os
from tqdm import tqdm
import torch
from torch.nn import functional as F

class Trainer():
    def __init__(self, model, optimizer, device=None, input_type='pointcloud',
                 vis_dir=None, threshold=0.5, eval_sample=False):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.input_type = input_type
        self.vis_dir = vis_dir
        self.threshold = threshold
        self.eval_sample = eval_sample

        if vis_dir is not None and not os.path.exists(vis_dir):
            os.makedirs(vis_dir)

    def train_step(self, data):
        self.model.train()
        self.optimizer.zero_grad()
        
        if hasattr(self.model, 'reset_states'):
            self.model.reset_states()
        
        loss, loss_dict = self.compute_loss_with_dict(data)
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), loss_dict

    def evaluate(self, val_loader, return_metrics=False):
        self.model.eval()
        total_loss = 0.0
        total_metrics = {}
        num_batches = 0
        
        if hasattr(self.model, 'reset_states'):
            self.model.reset_states()
        
        for data in tqdm(val_loader):
            eval_loss, batch_metrics = self.eval_step_with_metrics(data)
            total_loss += float(eval_loss)
            
            for key, value in batch_metrics.items():
                if key not in total_metrics:
                    total_metrics[key] = 0.0
                total_metrics[key] += value
            
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        if return_metrics:
            for key in total_metrics:
                total_metrics[key] /= num_batches
            return avg_loss, total_metrics
        
        return avg_loss

    def eval_step(self, data):
        self.model.eval()
        device = self.device

        Xc_rotated = data.get('input').to(device).float()
        gt_distances = data.get('len').to(device).float()

        if gt_distances.dim() == 2 and gt_distances.shape[1] == 1:
            gt_distances = gt_distances.squeeze(-1)
        elif gt_distances.dim() == 3 and gt_distances.shape[2] == 1:
            gt_distances = gt_distances.squeeze(-1)

        with torch.no_grad():
            pred_distances = self.model(Xc_rotated)
            loss, loss_dict = self.model.compute_loss(pred_distances, gt_distances)

        return loss

    def eval_step_with_metrics(self, data):
        device = self.device

        Xc_rotated = data.get('input').to(device).float()
        gt_distances = data.get('len').to(device).float()

        if gt_distances.dim() == 2 and gt_distances.shape[1] == 1:
            gt_distances = gt_distances.squeeze(-1)
        elif gt_distances.dim() == 3 and gt_distances.shape[2] == 1:
            gt_distances = gt_distances.squeeze(-1)

        with torch.no_grad():
            pred_distances = self.model(Xc_rotated)
            loss, loss_dict = self.model.compute_loss(pred_distances, gt_distances)
            
            metrics = self.calculate_metrics(pred_distances, gt_distances, loss_dict)

        return loss, metrics

    def compute_loss(self, data):
        device = self.device

        Xc_rotated = data.get('input').to(device).float()
        gt_distances = data.get('len').to(device).float()

        if gt_distances.dim() == 2 and gt_distances.shape[1] == 1:
            gt_distances = gt_distances.squeeze(-1)
        elif gt_distances.dim() == 3 and gt_distances.shape[2] == 1:
            gt_distances = gt_distances.squeeze(-1)

        pred_distances = self.model(Xc_rotated)
        loss, loss_dict = self.model.compute_loss(pred_distances, gt_distances)

        return loss

    def compute_loss_with_dict(self, data):
        device = self.device

        Xc_rotated = data.get('input').to(device).float()
        gt_distances = data.get('len').to(device).float()

        if gt_distances.dim() == 2 and gt_distances.shape[1] == 1:
            gt_distances = gt_distances.squeeze(-1)
        elif gt_distances.dim() == 3 and gt_distances.shape[2] == 1:
            gt_distances = gt_distances.squeeze(-1)

        pred_distances = self.model(Xc_rotated)
        loss, loss_dict = self.model.compute_loss(pred_distances, gt_distances)

        return loss, loss_dict

    def calculate_metrics(self, pred_distances, gt_distances, loss_dict):
        metrics = loss_dict.copy()
        
        mae = F.l1_loss(pred_distances, gt_distances).item()
        metrics['mae'] = mae
        
        mse = F.mse_loss(pred_distances, gt_distances).item()
        metrics['mse'] = mse
        
        eps = 1e-8
        relative_error = torch.mean(torch.abs(pred_distances - gt_distances) / (gt_distances + eps)).item()
        metrics['relative_error'] = relative_error
        
        return metrics

    def predict(self, data, return_uncertainty=False):
        self.model.eval()
        device = self.device
        
        if hasattr(self.model, 'reset_states'):
            self.model.reset_states()

        Xc_rotated = data.get('input').to(device).float()

        with torch.no_grad():
            if return_uncertainty and hasattr(self.model, 'distance_decoder'):
                pred_distances, uncertainties = self.model(Xc_rotated)
                return pred_distances, uncertainties
            else:
                pred_distances = self.model(Xc_rotated)
                return pred_distances

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def get_learning_rate(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def set_learning_rate(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_model_states(self):
        if hasattr(self.model, 'reset_states'):
            self.model.reset_states()