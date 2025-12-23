import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from functools import lru_cache
from torch.cuda.amp import autocast
import warnings

# Utility Functions 
def square_distance(src, dst):
    """Calculate squared Euclidean distance between points"""
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    """Index points based on indices"""
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def knn(x, k):
    """K-nearest neighbors with caching for repeated queries"""
    B, C, N = x.shape
    k = min(k, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

class KNNCache:
    """Cache for KNN computations to avoid recomputation"""
    def __init__(self, max_size=32):
        self.cache = {}
        self.max_size = max_size
        
    def get_knn(self, xyz, k, block_id=""):
        """Get or compute KNN indices"""
        key = (xyz.shape, k, block_id)
        
        if key not in self.cache:
            xyz_t = xyz.permute(0, 2, 1).contiguous()
            idx = knn(xyz_t, k)
            self.cache[key] = idx
            
            if len(self.cache) > self.max_size:
                del self.cache[next(iter(self.cache))]
        
        return self.cache[key]

# Enhanced MultiTimeConstant LIF Neuron

class MultiTimeConstantLIFNeuron(nn.Module):
    def __init__(self, layer_size=None, membrane_decay_init=0.9, threshold_adapt_init=0.01,
                 refractory_decay_init=0.5, grad_width=10.0, device=None):
        super().__init__()
        self.layer_size = layer_size
        self.membrane_decay_init = membrane_decay_init
        self.threshold_adapt_init = threshold_adapt_init
        self.refractory_decay_init = refractory_decay_init
        self.grad_width = grad_width
        
        
        if layer_size is not None:
            if device is None:
                device = torch.device('cpu')
            self.membrane_decay = nn.Parameter(torch.full((layer_size,), membrane_decay_init, device=device))
            self.threshold_adapt = nn.Parameter(torch.full((layer_size,), threshold_adapt_init, device=device))
            self.refractory_decay = nn.Parameter(torch.full((layer_size,), refractory_decay_init, device=device))
            self.threshold_base = nn.Parameter(torch.ones(layer_size, device=device))
        else:
            self.membrane_decay = None
            self.threshold_adapt = None
            self.refractory_decay = None
            self.threshold_base = None

    def forward(self, x, membrane=None, threshold=None, refractory=None):
        B = x.shape[0]
        C = x.shape[1]

        #initialization
        if self.membrane_decay is None:
            device = x.device
            self.membrane_decay = nn.Parameter(torch.full((C,), self.membrane_decay_init, device=device))
            self.threshold_adapt = nn.Parameter(torch.full((C,), self.threshold_adapt_init, device=device))
            self.refractory_decay = nn.Parameter(torch.full((C,), self.refractory_decay_init, device=device))
            self.threshold_base = nn.Parameter(torch.ones(C, device=device))

        def expand_param(param, x):
            if x.dim() == 2:
                return param.unsqueeze(0).expand_as(x)
            elif x.dim() == 3:
                return param.unsqueeze(0).unsqueeze(-1).expand_as(x)
            elif x.dim() == 4:
                return param.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand_as(x)
            else:
                raise ValueError(f"Unsupported input shape: {x.shape}")

        if membrane is None:
            membrane = torch.zeros_like(x)
        if threshold is None:
            threshold = expand_param(self.threshold_base, x)
        if refractory is None:
            refractory = torch.zeros_like(x)

        membrane_decay = torch.clamp(self.membrane_decay, 0.1, 0.99)
        threshold_adapt = torch.clamp(self.threshold_adapt, 0.001, 0.1)
        refractory_decay = torch.clamp(self.refractory_decay, 0.1, 0.95)
        
        membrane_decay_expanded = expand_param(membrane_decay, x)
        threshold_adapt_expanded = expand_param(threshold_adapt, x)
        refractory_decay_expanded = expand_param(refractory_decay, x)
        threshold_base_expanded = expand_param(self.threshold_base, x)

        x = x * (refractory <= 0).float()
        membrane = membrane * membrane_decay_expanded * (1 - refractory) + x
        spikes = self.spike_function(membrane - threshold)
        membrane = membrane * (1 - spikes)
        refractory = refractory * refractory_decay_expanded + spikes
        threshold = threshold + threshold_adapt_expanded * spikes
        threshold = threshold_base_expanded + (threshold - threshold_base_expanded) * 0.95

        return spikes, membrane, threshold, refractory

    def spike_function(self, x):
        """Spike function with gradient clipping and straight-through estimator"""
        x_clamped = torch.clamp(x, -10.0, 10.0)
        
        
        gaussian = torch.exp(-(x_clamped ** 2) / 2) / np.sqrt(2 * np.pi)
        
    
        sigmoid = torch.sigmoid(self.grad_width * x_clamped)
        
    
        spikes = 0.5 * gaussian + 0.5 * sigmoid
        
        if self.training:
        
            hard_spikes = (x > 0).float()
            spikes = spikes + (hard_spikes - spikes).detach()
        
        return spikes
    
    def get_spike_rate(self, spikes):
        """Compute spike rate for monitoring"""
        if spikes.numel() == 0:
            return 0.0
        return spikes.mean().item()


# SNN State Manager

class SNNStateManager:
    """Manages SNN states across time steps and layers"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset all states"""
        self.states = {}
        self.spike_rates = {}
        
    def get_state(self, layer_name, shape, device, dtype=torch.float32):
        """Get or initialize state for a layer"""
        if layer_name not in self.states:
            self.states[layer_name] = {
                'membrane': torch.zeros(shape, device=device, dtype=dtype),
                'threshold': None,
                'refractory': torch.zeros(shape, device=device, dtype=dtype)
            }
        else:
            # Detach states to prevent backward through multiple batches
            state = self.states[layer_name]
            if state['membrane'] is not None:
                state['membrane'] = state['membrane'].detach()
            if state['refractory'] is not None:
                state['refractory'] = state['refractory'].detach()
            if state['threshold'] is not None:
                state['threshold'] = state['threshold'].detach()
        return self.states[layer_name]
    
    def update_state(self, layer_name, new_state):
        """Update state for a layer"""
        self.states[layer_name] = new_state
    
    def record_spike_rate(self, layer_name, spikes):
        """Record spike rate for monitoring"""
        if layer_name not in self.spike_rates:
            self.spike_rates[layer_name] = []
        self.spike_rates[layer_name].append(spikes.mean().item())
    
    def get_average_spike_rate(self, layer_name):
        """Get average spike rate for a layer"""
        if layer_name in self.spike_rates and self.spike_rates[layer_name]:
            return np.mean(self.spike_rates[layer_name])
        return 0.0
        
#Multi-Head Transformer Block with SNN 


class MultiHeadSNNTransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k, time_steps, num_heads=4, dropout=0.1):
        super().__init__()
        self.k = k
        self.time_steps = time_steps
        self.num_heads = num_heads
        self.d_model = d_model
        self.dropout_rate = dropout
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads
        
    
        self.fc1 = nn.Sequential(
            nn.Conv1d(d_points, d_model, 1),
            nn.BatchNorm1d(d_model),
        )
        self.snn1 = MultiTimeConstantLIFNeuron(d_model)
        
        self.fc2 = nn.Sequential(
            nn.Conv1d(d_model, d_points, 1),
            nn.BatchNorm1d(d_points)
        )
        
        # Position encoding network 
        self.fc_delta = nn.Sequential(
            nn.Conv2d(3, d_model, 1),
            nn.BatchNorm2d(d_model),
        )
        self.snn_delta = MultiTimeConstantLIFNeuron(d_model)
        
        self.fc_delta2 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.BatchNorm2d(d_model),
        )
        self.snn_delta2 = MultiTimeConstantLIFNeuron(d_model)
        
        # Multi-head attention weighting
        self.fc_gamma = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.BatchNorm2d(d_model),
        )
        self.snn_gamma = MultiTimeConstantLIFNeuron(d_model)
        
        self.fc_gamma2 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 1),
            nn.BatchNorm2d(d_model),
        )
        
    
        self.w_qs = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
        )
        self.snn_q = MultiTimeConstantLIFNeuron(d_model)
        
        self.w_ks = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
        )
        self.snn_k = MultiTimeConstantLIFNeuron(d_model)
        
        self.w_vs = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
        )
        self.snn_v = MultiTimeConstantLIFNeuron(d_model)
        

        self.out_proj = nn.Sequential(
            nn.Conv1d(d_model, d_model, 1),
            nn.BatchNorm1d(d_model),
        )
        
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    
        self.knn_cache = KNNCache()
        
        
        self.state_manager = SNNStateManager()

    def forward(self, xyz, features):
        """
        xyz: [B, N, 3] point coordinates
        features: [B, N, C] point features
        """
        B, N, _ = xyz.shape
        
        k_actual = min(self.k, N)
        
        
        state_manager = self.state_manager
        
        
        knn_idx = self.knn_cache.get_knn(xyz, k_actual, f"block_{id(self)}")
        knn_xyz = index_points(xyz, knn_idx) 
        
        pos_diff = xyz[:, :, None, :] - knn_xyz  
        
    
        features = features.permute(0, 2, 1).contiguous() 
        pre = features
        
        # SNN transformation 
        x = self.fc1(features)
        states_1 = [None, None, None]
        for t in range(self.time_steps):
            x, *states_1 = self.snn1(x, *states_1)
        
        q = self.w_qs(x)
        states_q = [None, None, None]
        for t in range(self.time_steps):
            q, *states_q = self.snn_q(q, *states_q)
        
        k = self.w_ks(x)
        states_k = [None, None, None]
        for t in range(self.time_steps):
            k, *states_k = self.snn_k(k, *states_k)
            
        v = self.w_vs(x)
        states_v = [None, None, None]
        for t in range(self.time_steps):
            v, *states_v = self.snn_v(v, *states_v)
        
        q = q.view(B, self.num_heads, self.head_dim, N)
        k = k.view(B, self.num_heads, self.head_dim, N)
        v = v.view(B, self.num_heads, self.head_dim, N)
        
        k = k.permute(0, 1, 3, 2).contiguous()  
        k = k.view(B * self.num_heads, N, self.head_dim)
        
        knn_idx_expanded = knn_idx.unsqueeze(1).expand(-1, self.num_heads, -1, -1)  
        knn_idx_flat = knn_idx_expanded.reshape(B * self.num_heads, N, k_actual) 
        
        k = index_points(k, knn_idx_flat)
        k = k.view(B, self.num_heads, N, k_actual, self.head_dim).permute(0, 1, 4, 2, 3).contiguous()  
        
        v = v.permute(0, 1, 3, 2).contiguous()
        v = v.view(B * self.num_heads, N, self.head_dim)
        v = index_points(v, knn_idx_flat)
        v = v.view(B, self.num_heads, N, k_actual, self.head_dim).permute(0, 1, 4, 2, 3).contiguous() 
        
        pos_enc = self.fc_delta(pos_diff.permute(0, 3, 1, 2).contiguous()) 
        states_delta = [None, None, None]
        for t in range(self.time_steps):
            pos_enc, *states_delta = self.snn_delta(pos_enc, *states_delta)
        
        pos_enc = self.fc_delta2(pos_enc)
        states_delta2 = [None, None, None]
        for t in range(self.time_steps):
            pos_enc, *states_delta2 = self.snn_delta2(pos_enc, *states_delta2)
        
        pos_enc = pos_enc.view(B, self.num_heads, self.head_dim, N, k_actual)  
        
        q_expanded = q.unsqueeze(-1)  
        attn_input = q_expanded - k + pos_enc 
        
       
        attn_input = attn_input.view(B, self.d_model, N, k_actual)  
        
        attn = self.fc_gamma(attn_input)
        states_gamma = [None, None, None]
        for t in range(self.time_steps):
            attn, *states_gamma = self.snn_gamma(attn, *states_gamma)
        
        attn = self.fc_gamma2(attn)
        attn = attn.view(B, self.num_heads, self.head_dim, N, k_actual)  
        attn = F.softmax(attn / np.sqrt(self.head_dim), dim=-1)  
        
        #New
        attn = self.attn_dropout(attn)
        
       
        pos_enc_reshaped = pos_enc  
        v_with_pos = v + pos_enc_reshaped 
        
        res = torch.einsum('bhcnk,bhcnk->bhcn', attn, v_with_pos)  
        
        res = res.view(B, self.d_model, N) 
        
        res = self.out_proj(res)
        res = self.fc2(res) + pre
        
        return res.permute(0, 2, 1).contiguous(), attn

    def reset_states(self):
        """Reset all SNN states (NEW)"""
        self.state_manager.reset()


#Encoder with Multi-Head Transformer Blocks 

class ImprovedSNNEncoder(nn.Module):
    def __init__(self, emb_dims=1024, k_values=[20, 20, 16], time_steps=8, num_heads=4):
        super().__init__()
        self.time_steps = time_steps
        self.k_values = k_values
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
        )
        self.snn_init = MultiTimeConstantLIFNeuron(64)
        
        self.trans1 = MultiHeadSNNTransformerBlock(64, 128, k=k_values[0], time_steps=4, num_heads=num_heads)
        self.trans2 = MultiHeadSNNTransformerBlock(64, 256, k=k_values[1], time_steps=4, num_heads=num_heads)
        self.trans3 = MultiHeadSNNTransformerBlock(64, 512, k=k_values[2], time_steps=4, num_heads=num_heads)
        
        
        self.conv_final = nn.Sequential(
            nn.Conv1d(64*3, emb_dims, 1),
            nn.BatchNorm1d(emb_dims),
        )
        self.snn_final = MultiTimeConstantLIFNeuron(emb_dims)
        
        self.fc_out = nn.Linear(emb_dims, 2048)

    def forward(self, x):
        """
        x: [B, 3, N] or [B, N, 3] or [B, N, M, 3]
        """
        if x.ndim == 4:
            
            B, N, M, _ = x.shape
            x = x.mean(dim=2)  # [B, N, 3] - average over patch n points
            x = x.permute(0, 2, 1).contiguous()  # [B, 3, N] - transpose for convolution
        elif x.ndim == 3:
            
            if x.shape[1] == 3:
                pass
            else:
                x = x.permute(0, 2, 1).contiguous()
        
     
        B, C, N = x.shape
        
   
        xyz = x.permute(0, 2, 1).contiguous()
        
       
        features = self.conv1(x)  # [B, 64, N] - use x which is [B, 3, N]
        states_init = [None, None, None]
        for t in range(self.time_steps):
            features, *states_init = self.snn_init(features, *states_init)
        features = features.permute(0, 2, 1).contiguous()  # [B, N, 64]
        
      
        feat1, _ = self.trans1(xyz, features)  # [B, N, 64]
        feat2, _ = self.trans2(xyz, feat1)     # [B, N, 64]
        feat3, _ = self.trans3(xyz, feat2)     # [B, N, 64]
        
       
        multi_scale = torch.cat([feat1, feat2, feat3], dim=2)  # [B, N, 192]
        
        global_feat = self.conv_final(multi_scale.permute(0, 2, 1))  # [B, emb_dims, N]
        states_final = [None, None, None]
        for t in range(self.time_steps):
            global_feat, *states_final = self.snn_final(global_feat, *states_final)
    
        global_feat = F.adaptive_max_pool1d(global_feat, 1).squeeze(-1)  # [B, emb_dims]
        
      
        out = self.fc_out(global_feat)
        return out


# Decoder

class ImprovedDecoder(nn.Module):
    """Legacy SNN-based decoder (kept for backward compatibility)"""
    def __init__(self, input_dim=2048, output_dim=3, hidden_dims=[1024, 512, 256], time_steps=12):
        super().__init__()
        self.time_steps = time_steps
        
        layers = []
        snns = []
        in_dim = input_dim
        for hid_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hid_dim))
            snns.append(MultiTimeConstantLIFNeuron(hid_dim))
            in_dim = hid_dim
        
        self.layers = nn.ModuleList(layers)
        self.snns = nn.ModuleList(snns)
        
        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)
        self.bn_out = nn.LayerNorm(output_dim)

    def forward(self, x):
        B = x.shape[0]
        
        for layer, snn in zip(self.layers, self.snns):
            x = layer(x)
            states = [None, None, None]
            for t in range(self.time_steps):
                x, *states = snn(x, *states)
        
        x = self.fc_out(x)
        x = self.bn_out(x)
        x = F.normalize(x, dim=1)
        
        return x


class StandardNormalDecoder(nn.Module):
    """
    Standard (non-SNN) decoder for normal estimation.
    SNN processing is handled entirely in the encoder; this decoder uses
    conventional neural network layers for final normal prediction.
    """
    def __init__(self, input_dim=2048, output_dim=3, hidden_dims=[1024, 512, 256], dropout=0.1):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        for i, hid_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, hid_dim))
            layers.append(nn.BatchNorm1d(hid_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hid_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output projection with residual connection from intermediate
        self.fc_out = nn.Linear(hidden_dims[-1], output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

    def forward(self, x):
        """Forward pass through standard decoder"""
        x = self.mlp(x)
        x = self.fc_out(x)
        x = self.norm_out(x)
        # Normalize to unit vectors for normal estimation
        x = F.normalize(x, dim=1)
        return x
    
    def reset_states(self):
        """No states to reset in standard decoder"""
        pass
        
# Normal Consistency Loss

def normal_consistency_loss(pred_normals, xyz, k=8, weight=0.1):
    """
    Enforce that neighboring points should have similar normals
    
    Args:
        pred_normals: [B, 3] predicted normals
        xyz: [B, N, 3] point cloud coordinates
        k: number of nearest neighbors
        weight: weight for this loss term
    """
    B, N, _ = xyz.shape
    
    dists = square_distance(xyz, xyz)  
    knn_idx = dists.argsort()[:, :, 1:k+1]  
    
    if pred_normals.shape[0] == B and pred_normals.shape[1] == 3:
        pred_normals_expanded = pred_normals.unsqueeze(1).expand(B, N, 3)
    else:
        pred_normals_expanded = pred_normals.view(B, N, 3)
    
    neighbor_normals = index_points(pred_normals_expanded, knn_idx)  
    
    center_normals = pred_normals_expanded.unsqueeze(2) 
    cosine_sim = F.cosine_similarity(center_normals, neighbor_normals, dim=-1)  
    
    consistency_loss = (1 - cosine_sim).mean()
    
    return weight * consistency_loss

# Angular Loss

def enhanced_angular_loss_with_consistency(pred_normals, gt_normals, xyz=None, 
                                          temperature=0.1, alpha=0.1, 
                                          consistency_weight=0.15, k_neighbors=8):
    """
    Combined loss: angular loss + normal consistency
    
    Args:
        pred_normals: [B, 3] or [B, N, 3] predicted normals
        gt_normals: [B, 3] or [B, N, 3] ground truth normals
        xyz: [B, N, 3] point cloud (needed for consistency loss)
        temperature: temperature for confidence weighting
        alpha: weight for confidence regularization
        consistency_weight: weight for normal consistency loss
        k_neighbors: number of neighbors for consistency loss
    """
    if pred_normals.dim() == 3:
        B, N, _ = pred_normals.shape
        pred_normals_flat = pred_normals.view(B * N, 3)
        gt_normals_flat = gt_normals.view(B * N, 3)
    else:
        pred_normals_flat = pred_normals
        gt_normals_flat = gt_normals
    
    cosine_sim = F.cosine_similarity(pred_normals_flat, gt_normals_flat, dim=1)
    angular_error = torch.acos(torch.clamp(cosine_sim, -1 + 1e-6, 1 - 1e-6))
    confidence = torch.sigmoid(angular_error.detach() / temperature)
    weighted_loss = angular_error * confidence
    confidence_reg = alpha * (confidence - 0.5)**2
    
    base_loss = (weighted_loss + confidence_reg).mean()

    if xyz is not None and consistency_weight > 0:
        consistency = normal_consistency_loss(pred_normals, xyz, k=k_neighbors, weight=consistency_weight)
        total_loss = base_loss + consistency
        return total_loss, confidence.mean()
    
    return base_loss, confidence.mean()


class ImprovedSNNNormalEstimation(nn.Module):
    """
    Normal estimation model with SNN in encoder only.
    
    Architecture:
        - Encoder: ImprovedSNNEncoder (with SNN temporal dynamics and transformer blocks)
        - Decoder: StandardNormalDecoder (conventional MLP, no SNN)
    
    This design concentrates the bio-inspired spiking computation in the
    feature extraction stage, while using efficient standard layers for
    the final normal prediction task.
    """
    def __init__(self, k_values=[20, 20, 16], emb_dims=1024, 
                 time_steps_enc=8, time_steps_dec=12, num_heads=4,
                 use_snn_decoder=False, decoder_dropout=0.1):
        super().__init__()
        self.use_snn_decoder = use_snn_decoder
        
        # SNN-based encoder for temporal feature extraction
        self.encoder = ImprovedSNNEncoder(
            emb_dims=emb_dims, 
            k_values=k_values, 
            time_steps=time_steps_enc,
            num_heads=num_heads
        )
        
        # Choose decoder type
        if use_snn_decoder:
            # Legacy mode: SNN in both encoder and decoder
            self.decoder = ImprovedDecoder(
                input_dim=2048, 
                output_dim=3, 
                time_steps=time_steps_dec
            )
        else:
            # Default: SNN only in encoder, standard decoder
            self.decoder = StandardNormalDecoder(
                input_dim=2048,
                output_dim=3,
                hidden_dims=[1024, 512, 256],
                dropout=decoder_dropout
            )

    def forward(self, point_cloud):
        """
        Forward pass with SNN encoder and standard decoder.
        
        Input: point_cloud - [B, 3, N] or [B, N, 3] or [B, N, M, 3]
        Output: normals - [B, 3] or [B, N, 3] depending on input
        
        For 4D input [B, N, M, 3] (batch of patches):
            - Process each patch independently
            - Return [B, N, 3] normals (one per patch)
        """
        if point_cloud.ndim == 4:
            # [B, N, M, 3] - B batches, N patches, M neighbors, 3 coords
            B, N, M, C = point_cloud.shape
            # Reshape to process all patches together
            point_cloud_flat = point_cloud.view(B * N, M, C)  # [B*N, M, 3]
            
            # SNN-based feature extraction
            features = self.encoder(point_cloud_flat)  # [B*N, 2048]
            # Standard decoding for normal prediction
            normals = self.decoder(features)  # [B*N, 3]
            
            # Reshape back to [B, N, 3]
            normals = normals.view(B, N, 3)
            return normals
        else:
            # [B, N, 3] or [B, 3, N] - standard case
            features = self.encoder(point_cloud)
            normals = self.decoder(features)
            return normals

    def compute_loss(self, pred_normals, gt_normals, xyz=None, 
                    consistency_weight=0.15, k_neighbors=8):
        """
        Compute angular loss with optional consistency regularization.
        
        Returns: (loss_tensor, loss_dict)
        """
        # Handle 4D xyz input [B, N, M, 3] - extract center points
        if xyz is not None and xyz.ndim == 4:
            # Take the mean of each patch as the center point
            xyz = xyz.mean(dim=2)  # [B, N, 3]
        
        loss, confidence = enhanced_angular_loss_with_consistency(
            pred_normals, gt_normals, xyz,
            consistency_weight=consistency_weight,
            k_neighbors=k_neighbors
        )
        
        loss_dict = {
            'total_loss': loss.item() if torch.is_tensor(loss) else loss,
            'confidence': confidence.item() if torch.is_tensor(confidence) else confidence
        }
        
        return loss, loss_dict
    
    def reset_states(self):
        """Reset SNN states in encoder (decoder has no states if standard)"""
        # Reset encoder transformer block states
        if hasattr(self.encoder, 'trans1'):
            self.encoder.trans1.reset_states()
        if hasattr(self.encoder, 'trans2'):
            self.encoder.trans2.reset_states()
        if hasattr(self.encoder, 'trans3'):
            self.encoder.trans3.reset_states()
        
        # Reset decoder states only if using SNN decoder
        if self.use_snn_decoder and hasattr(self.decoder, 'reset_states'):
            self.decoder.reset_states()

class TrainingUtilities:
    """Optional utilities for training - doesn't affect model I/O"""
    
    @staticmethod
    def monitor_gradients(model, writer=None, step=0):
        """Monitor gradient statistics (optional)"""
        grad_stats = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.data
                grad_stats[f'grad_norm/{name}'] = grad.norm().item()
                
                if writer is not None:
                    writer.add_scalar(f'grad_norm/{name}', grad.norm().item(), step)
        
        return grad_stats
    
    @staticmethod
    def create_optimizer(model, lr=1e-3, weight_decay=1e-4):
        """Create optimizer with parameters grouped by type"""
        param_groups = []
        
        # Group parameters by type for different learning rates
        snn_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if 'membrane_decay' in name or 'threshold' in name or 'refractory' in name:
                snn_params.append(param)
            else:
                other_params.append(param)
        
        if snn_params:
            param_groups.append({'params': snn_params, 'lr': lr * 0.1})
        if other_params:
            param_groups.append({'params': other_params, 'lr': lr})
        
        return torch.optim.AdamW(param_groups, lr=lr, weight_decay=weight_decay)
    
    @staticmethod
    def create_scheduler(optimizer, warmup_steps=1000, total_steps=10000):
        """Create learning rate scheduler with warmup"""
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)