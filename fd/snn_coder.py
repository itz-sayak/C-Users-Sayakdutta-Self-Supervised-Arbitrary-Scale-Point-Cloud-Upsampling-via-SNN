import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
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
    B, C, N = x.shape
    k = min(k, N)
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx

class KNNCache:
    def __init__(self, max_size=32):
        self.cache = {}
        self.max_size = max_size
        
    def get_knn(self, xyz, k, block_id=""):
        key = (xyz.shape, k, block_id)
        
        if key not in self.cache:
            xyz_t = xyz.permute(0, 2, 1).contiguous()
            idx = knn(xyz_t, k)
            self.cache[key] = idx
            
            if len(self.cache) > self.max_size:
                del self.cache[next(iter(self.cache))]
        
        return self.cache[key]

def get_graph_feature(x, k=20, idx=None):
    B, C, N = x.size()
    k = min(k, N)
    
    if idx is None:
        idx = knn(x, k=k)
    
    idx_base = torch.arange(0, B, device=x.device).view(-1, 1, 1) * N
    idx = (idx + idx_base).view(-1)
    
    x = x.transpose(2, 1).contiguous()
    x_flat = x.view(B * N, C)
    neighbors = x_flat[idx, :].view(B, N, k, C)
    x_expanded = x.unsqueeze(2).expand(-1, N, k, -1)
    
    feature = torch.cat((neighbors - x_expanded, neighbors), dim=-1)
    return feature.permute(0, 3, 1, 2).contiguous()

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
        x_clamped = torch.clamp(x, -10.0, 10.0)
        
        gaussian = torch.exp(-(x_clamped ** 2) / 2) / np.sqrt(2 * np.pi)
        sigmoid = torch.sigmoid(self.grad_width * x_clamped)
        
        spikes = 0.5 * gaussian + 0.5 * sigmoid
        
        if self.training:
            hard_spikes = (x > 0).float()
            spikes = spikes + (hard_spikes - spikes).detach()
        
        return spikes

class SNNStateManager:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.states = {}
        self.spike_rates = {}
        
    def get_state(self, layer_name, shape, device, dtype=torch.float32):
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
        self.states[layer_name] = new_state
    
    def record_spike_rate(self, layer_name, spikes):
        if layer_name not in self.spike_rates:
            self.spike_rates[layer_name] = []
        self.spike_rates[layer_name].append(spikes.mean().item())

class TemporalIntegration(nn.Module):
    def __init__(self, time_steps, feature_dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(time_steps))
        self.softmax = nn.Softmax(dim=0)

    def forward(self, temporal_features):
        weights = self.softmax(self.weights)
        return torch.einsum('t, tbf -> bf', weights, temporal_features)

class EnhancedTemporalSNN_DGCNN_fd(nn.Module):
    def __init__(self, k=20, emb_dims=512, time_steps=5, num_scales=3):
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.time_steps = time_steps
        
        self.knn_cache = KNNCache()
        
        self.conv_blocks = nn.ModuleList()
        self.snn_blocks = nn.ModuleList()
        
        self.conv_blocks.append(nn.Sequential(
            nn.Conv2d(6, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2)
        ))
        self.snn_blocks.append(MultiTimeConstantLIFNeuron(64))
        
        in_channels = 64
        out_channels_list = [128, 256, 512]
        
        for out_channels in out_channels_list:
            self.conv_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels*2, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            ))
            self.snn_blocks.append(MultiTimeConstantLIFNeuron(out_channels))
            in_channels = out_channels
        
        self.multi_scale_conv = nn.Sequential(
            nn.Conv1d(64 + 128 + 256 + 512, emb_dims, 1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(0.2)
        )
        
        self.snn_fc = MultiTimeConstantLIFNeuron(emb_dims)
        self.temporal_integration = TemporalIntegration(time_steps, emb_dims)
        
        self.state_manager = SNNStateManager()

    def forward(self, x):
        if x.dim() == 3 and x.shape[1] != 3:
            x = x.transpose(1, 2).contiguous()
        
        B, _, M = x.shape
        device = x.device
        
        if not self.training:
            self.state_manager.reset()
        
        xyz = x.permute(0, 2, 1).contiguous()
        knn_idx = self.knn_cache.get_knn(xyz, min(self.k, M), "encoder")
        
        temporal_features = []
        
        for t in range(self.time_steps):
            multi_scale_features = []
            
            x_current = x
            for block_idx, (conv_block, snn_block) in enumerate(zip(self.conv_blocks, self.snn_blocks)):
                x_graph = get_graph_feature(x_current, k=min(self.k, M))
                
                x_conv = conv_block(x_graph).max(dim=-1)[0]
                
                state_name = f"block_{block_idx}"
                state = self.state_manager.get_state(state_name, x_conv.shape, device)
                
                x_conv, membrane, threshold, refractory = snn_block(
                    x_conv, 
                    membrane=state['membrane'],
                    threshold=state['threshold'],
                    refractory=state['refractory']
                )
                
                self.state_manager.update_state(state_name, {
                    'membrane': membrane,
                    'threshold': threshold,
                    'refractory': refractory
                })
                
                multi_scale_features.append(x_conv)
                x_current = x_conv
            
            multi_scale_cat = torch.cat(multi_scale_features, dim=1)
            aggregated = self.multi_scale_conv(multi_scale_cat)
            
            pooled = F.adaptive_max_pool1d(aggregated, 1).squeeze(-1)
            temporal_features.append(pooled)
        
        temporal_features = torch.stack(temporal_features, dim=0)
        x = self.temporal_integration(temporal_features)
        
        state_name = "final"
        state = self.state_manager.get_state(state_name, x.shape, device)
        x, *_ = self.snn_fc(x, 
                           membrane=state['membrane'],
                           threshold=state['threshold'],
                           refractory=state['refractory'])
        
        return x
    
    def reset_states(self):
        self.state_manager.reset()

class EnhancedSNNResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim, time_steps=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.snn = MultiTimeConstantLIFNeuron(out_dim)
        self.time_steps = time_steps
        
        self.res_proj = None
        if in_dim != out_dim:
            self.res_proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
            
        self.state_manager = SNNStateManager()

    def forward(self, x):
        residual = x
        x = self.fc(x)
        
        state_name = f"res_block_{id(self)}"
        state = self.state_manager.get_state(state_name, x.shape, x.device)
        
        membrane, threshold, refractory = state['membrane'], state['threshold'], state['refractory']
        for _ in range(self.time_steps):
            x, membrane, threshold, refractory = self.snn(x, membrane, threshold, refractory)
        
        self.state_manager.update_state(state_name, {
            'membrane': membrane,
            'threshold': threshold,
            'refractory': refractory
        })
        
        if self.res_proj is not None:
            residual = self.res_proj(residual)
        
        return x + residual
    
    def reset_states(self):
        self.state_manager.reset()

class EnhancedSpikingSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4, time_steps=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.time_steps = time_steps
        
        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim * 3),
            nn.BatchNorm1d(dim * 3)
        )
        
        self.snn_q = MultiTimeConstantLIFNeuron(dim)
        self.snn_k = MultiTimeConstantLIFNeuron(dim)
        self.snn_v = MultiTimeConstantLIFNeuron(dim)
        self.snn_out = MultiTimeConstantLIFNeuron(dim)
        
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        
        self.attn_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.state_manager = SNNStateManager()

    def forward(self, x):
        B = x.shape[0]
        
        state_name = f"attention_{id(self)}"
        states = self.state_manager.get_state(state_name, (B, self.dim), x.device)
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv
        
        q_state = self.state_manager.get_state(f"{state_name}_q", q.shape, x.device)
        k_state = self.state_manager.get_state(f"{state_name}_k", k.shape, x.device)
        v_state = self.state_manager.get_state(f"{state_name}_v", v.shape, x.device)
        
        for _ in range(self.time_steps):
            q, q_state['membrane'], q_state['threshold'], q_state['refractory'] = \
                self.snn_q(q, q_state['membrane'], q_state['threshold'], q_state['refractory'])
            
            k, k_state['membrane'], k_state['threshold'], k_state['refractory'] = \
                self.snn_k(k, k_state['membrane'], k_state['threshold'], k_state['refractory'])
            
            v, v_state['membrane'], v_state['threshold'], v_state['refractory'] = \
                self.snn_v(v, v_state['membrane'], v_state['threshold'], v_state['refractory'])
        
        q = q.view(B, self.num_heads, self.head_dim)
        k = k.view(B, self.num_heads, self.head_dim)
        v = v.view(B, self.num_heads, self.head_dim)
        
        attn = torch.einsum('bhd,bhd->bh', q, k) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        
        out = torch.einsum('bh,bhd->bhd', attn, v).reshape(B, -1)
        
        out_state = self.state_manager.get_state(f"{state_name}_out", out.shape, x.device)
        for _ in range(self.time_steps):
            out, out_state['membrane'], out_state['threshold'], out_state['refractory'] = \
                self.snn_out(out, out_state['membrane'], out_state['threshold'], out_state['refractory'])
        
        out = self.to_out(out)
        
        return x + out
    
    def reset_states(self):
        self.state_manager.reset()

class EnhancedSpikingDistanceDecoder(nn.Module):
    """Legacy SNN-based decoder (kept for backward compatibility)"""
    def __init__(self, input_dim=1024, hidden_dims=[512, 256, 128, 64], 
                 time_steps=8, num_heads=4, dropout=0.1):
        super().__init__()
        self.time_steps = time_steps
        
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0])
        )
        
        self.snn_layers = nn.ModuleList([
            EnhancedSNNResidualBlock(hidden_dims[i], hidden_dims[i+1], time_steps=4)
            for i in range(len(hidden_dims)-1)
        ])
        
        self.spiking_attention = EnhancedSpikingSelfAttention(
            hidden_dims[-1], num_heads=num_heads, time_steps=4, dropout=dropout
        )
        
        self.fc_hidden = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        self.fc_distance = nn.Linear(32, 1)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        B = x.shape[0]
        
        x = self.fc_in(x)
        
        for layer in self.snn_layers:
            x = layer(x)
        
        x = self.spiking_attention(x)
        
        x_hidden = self.fc_hidden(x)
        
        distance = self.fc_distance(x_hidden)
        distance = self.activation(distance)
        
        return distance.squeeze(-1)
    
    def reset_states(self):
        for layer in self.snn_layers:
            layer.reset_states()
        self.spiking_attention.reset_states()


class StandardDistanceDecoder(nn.Module):
    """
    Standard (non-SNN) decoder for distance estimation.
    SNN processing is handled entirely in the encoder; this decoder uses
    conventional neural network layers for final regression.
    """
    def __init__(self, input_dim=512, hidden_dims=[256, 128, 64], 
                 dropout=0.1, use_attention=True, num_heads=4):
        super().__init__()
        self.use_attention = use_attention
        
        # Input projection
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU()
        )
        
        # Residual MLP blocks (standard, no SNN)
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.residual_blocks.append(
                StandardResidualBlock(hidden_dims[i], hidden_dims[i+1], dropout=dropout)
            )
        
        # Optional standard self-attention (non-spiking)
        if use_attention:
            self.attention = StandardSelfAttention(
                hidden_dims[-1], num_heads=num_heads, dropout=dropout
            )
        
        # Output layers
        self.fc_hidden = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.fc_distance = nn.Linear(32, 1)
        self.activation = nn.ReLU()  # Ensure non-negative distances

    def forward(self, x):
        """Forward pass through standard decoder"""
        x = self.fc_in(x)
        
        for block in self.residual_blocks:
            x = block(x)
        
        if self.use_attention:
            x = self.attention(x)
        
        x = self.fc_hidden(x)
        distance = self.fc_distance(x)
        distance = self.activation(distance)
        
        return distance.squeeze(-1)
    
    def reset_states(self):
        """No states to reset in standard decoder"""
        pass


class StandardResidualBlock(nn.Module):
    """Standard residual block without SNN components"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        
        self.res_proj = None
        if in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.fc(x)
        
        if self.res_proj is not None:
            residual = self.res_proj(residual)
        
        return self.activation(out + residual)


class StandardSelfAttention(nn.Module):
    """Standard multi-head self-attention without SNN components"""
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """x: [B, dim]"""
        B = x.shape[0]
        residual = x
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = qkv
        
        # Reshape for multi-head attention
        q = q.view(B, self.num_heads, self.head_dim)
        k = k.view(B, self.num_heads, self.head_dim)
        v = v.view(B, self.num_heads, self.head_dim)
        
        # Attention scores
        attn = torch.einsum('bhd,bhd->bh', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        out = torch.einsum('bh,bhd->bhd', attn, v).reshape(B, -1)
        out = self.to_out(out)
        
        return self.norm(out + residual)

def enhanced_distance_loss(pred_distances, gt_distances, reduction='mean', beta=0.1):
    base_loss = F.smooth_l1_loss(pred_distances, gt_distances, reduction=reduction, beta=beta)
    
    return base_loss

class EnhancedSNNDistanceEstimation(nn.Module):
    """
    Distance estimation model with SNN in encoder only.
    
    Architecture:
        - Encoder: EnhancedTemporalSNN_DGCNN_fd (with SNN temporal dynamics)
        - Decoder: StandardDistanceDecoder (conventional MLP, no SNN)
    
    This design concentrates the bio-inspired spiking computation in the
    feature extraction stage, while using efficient standard layers for
    the final regression task.
    """
    def __init__(self, k=20, emb_dims=512, time_steps_enc=5, time_steps_dec=8,
                 num_heads=4, dropout=0.1, use_snn_decoder=False):
        super().__init__()
        self.use_snn_decoder = use_snn_decoder
        
        # SNN-based encoder for temporal feature extraction
        self.encoder = EnhancedTemporalSNN_DGCNN_fd(
            k=k, 
            emb_dims=emb_dims, 
            time_steps=time_steps_enc
        )
        
        # Standard (non-SNN) decoder - SNN processing done in encoder
        if use_snn_decoder:
            # Legacy mode: SNN in both encoder and decoder
            self.distance_decoder = EnhancedSpikingDistanceDecoder(
                input_dim=emb_dims,
                time_steps=time_steps_dec,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            # Default: SNN only in encoder, standard decoder
            self.distance_decoder = StandardDistanceDecoder(
                input_dim=emb_dims,
                hidden_dims=[256, 128, 64],
                dropout=dropout,
                use_attention=True,
                num_heads=num_heads
            )

    def forward(self, Xc_rotated):
        if Xc_rotated.dim() == 4:
            B, N, M, _ = Xc_rotated.shape
            Xc_rotated = Xc_rotated.view(B * N, M, 3)
            flatten = True
        else:
            flatten = False
            B, M, _ = Xc_rotated.shape

        # SNN-based feature extraction
        features = self.encoder(Xc_rotated)
        
        # Standard decoding for distance regression
        distances = self.distance_decoder(features)
        
        if flatten:
            distances = distances.view(B, N)
        
        return distances

    def compute_loss(self, pred_distances, gt_distances, reduction='mean', beta=0.1):
        total_loss = enhanced_distance_loss(
            pred_distances, gt_distances,
            reduction=reduction, beta=beta
        )
        
        loss_dict = {
            'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
            'distance_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss
        }
        
        if not torch.is_tensor(total_loss):
            total_loss = torch.tensor(total_loss, device=pred_distances.device, requires_grad=True)
        
        return total_loss, loss_dict
    
    def reset_states(self):
        """Reset SNN states in encoder (decoder has no states)"""
        self.encoder.reset_states()
        if self.use_snn_decoder:
            self.distance_decoder.reset_states()