from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class InstancePerChannelNorm(nn.Module):

    def __init__(self, num_channels: int, affine: bool=True, eps: float=1e-05):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_channels, affine=affine, eps=eps, track_running_stats=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class StochasticDepth(nn.Module):

    def __init__(self, drop_prob: float=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x * mask / keep_prob

class TemporalBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_factor: int=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.pool = nn.MaxPool1d(pool_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.act(self.bn(self.conv(x))))

class ECGNextBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_factor: int=2, expand_ratio: int=2):
        super().__init__()
        expand_ch = in_channels * expand_ratio
        self.dw_conv = nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.pw_expand = nn.Conv1d(in_channels, expand_ch, 1)
        self.act = nn.GELU()
        self.pw_project = nn.Conv1d(expand_ch, out_channels, 1)
        self.pool = nn.MaxPool1d(pool_factor)
        self.residual_proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dw_conv(x)
        h = h.permute(0, 2, 1)
        h = self.norm(h)
        h = h.permute(0, 2, 1)
        h = self.pw_expand(h)
        h = self.act(h)
        h = self.pw_project(h)
        h = h + self.residual_proj(x)
        return self.pool(h)

class MultiScaleECGNextBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: tuple, pool_factor: int=2, expand_ratio: int=2):
        super().__init__()
        assert all((k % 2 == 1 for k in kernel_sizes)), 'All kernel sizes must be odd for exact same-padding.'
        expand_ch = in_channels * expand_ratio
        self.dw_branches = nn.ModuleList([nn.Conv1d(in_channels, in_channels, k, padding=k // 2, groups=in_channels) for k in kernel_sizes])
        self.norm = nn.LayerNorm(in_channels)
        self.pw_expand = nn.Conv1d(in_channels, expand_ch, 1)
        self.act = nn.GELU()
        self.pw_project = nn.Conv1d(expand_ch, out_channels, 1)
        self.pool = nn.MaxPool1d(pool_factor)
        self.residual_proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = sum((branch(x) for branch in self.dw_branches))
        h = h.permute(0, 2, 1)
        h = self.norm(h)
        h = h.permute(0, 2, 1)
        h = self.pw_expand(h)
        h = self.act(h)
        h = self.pw_project(h)
        h = h + self.residual_proj(x)
        return self.pool(h)

class GatedMultiScaleECGNextBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_sizes: tuple, pool_factor: int=2, expand_ratio: int=2, drop_path_prob: float=0.0):
        super().__init__()
        assert all((k % 2 == 1 for k in kernel_sizes)), 'All kernel sizes must be odd for exact same-padding.'
        n_branches = len(kernel_sizes)
        expand_ch = in_channels * expand_ratio
        gate_hidden = max(4, in_channels // 8)
        self.dw_branches = nn.ModuleList([nn.Conv1d(in_channels, in_channels, k, padding=k // 2, groups=in_channels) for k in kernel_sizes])
        self.gate = nn.Sequential(nn.Linear(in_channels, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, n_branches))
        self.norm = nn.LayerNorm(in_channels)
        self.pw_expand = nn.Conv1d(in_channels, expand_ch, 1)
        self.act = nn.GELU()
        self.pw_project = nn.Conv1d(expand_ch, out_channels, 1)
        self.pool = nn.MaxPool1d(pool_factor)
        self.residual_proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.drop_path = StochasticDepth(drop_path_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outs = [branch(x) for branch in self.dw_branches]
        branch_stack = torch.stack(branch_outs, dim=1)
        gap = x.mean(dim=-1)
        weights = F.softmax(self.gate(gap), dim=-1)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        h = (branch_stack * weights).sum(dim=1)
        h = h.permute(0, 2, 1)
        h = self.norm(h)
        h = h.permute(0, 2, 1)
        h = self.pw_expand(h)
        h = self.act(h)
        h = self.pw_project(h)
        h = self.drop_path(h) + self.residual_proj(x)
        return self.pool(h)

class BottleneckDropKeyAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int=4, bottleneck_ratio: int=8, drop_key_ratio: float=0.6):
        super().__init__()
        assert dim % bottleneck_ratio == 0, f'dim={dim} must be divisible by bottleneck_ratio={bottleneck_ratio}'
        self.num_heads = num_heads
        self.drop_key_ratio = drop_key_ratio
        inner_dim = dim // bottleneck_ratio
        assert inner_dim % num_heads == 0, f'inner_dim={inner_dim} must be divisible by num_heads={num_heads}'
        self.head_dim = inner_dim // num_heads
        self.scale = self.head_dim ** (-0.5)
        self.input_bottleneck = nn.Linear(dim, inner_dim)
        self.q_proj = nn.Linear(inner_dim, inner_dim)
        self.k_proj = nn.Linear(inner_dim, inner_dim)
        self.v_proj = nn.Linear(inner_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, inner_dim)
        self.output_bottleneck = nn.Linear(inner_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (B, L, D) = x.shape
        h = self.input_bottleneck(x)
        Q = self.q_proj(h)
        K = self.k_proj(h)
        V = self.v_proj(h)

        def split_heads(t):
            return t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        Q = split_heads(Q)
        K = split_heads(K)
        V = split_heads(V)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if self.training and self.drop_key_ratio > 0.0:
            mask = torch.bernoulli(torch.ones_like(scores) * self.drop_key_ratio).bool()
            scores = scores.masked_fill(mask, -1000000000000.0)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, L, -1)
        out = self.out_proj(out)
        out = self.output_bottleneck(out)
        return out

class MultiScalePooling(nn.Module):

    def __init__(self, pool_sizes: tuple=(1, 2, 3, 4)):
        super().__init__()
        self.pool_sizes = pool_sizes
        self.total_size = sum(pool_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = [F.adaptive_avg_pool1d(x, k) for k in self.pool_sizes]
        return torch.cat(pooled, dim=-1)

class SLIMLayer(nn.Module):

    def __init__(self, dim: int, num_heads: int=4, bottleneck_ratio: int=8, drop_key_ratio: float=0.6, ffn_expand: int=4, dropout: float=0.1, drop_path_prob: float=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BottleneckDropKeyAttention(dim, num_heads=num_heads, bottleneck_ratio=bottleneck_ratio, drop_key_ratio=drop_key_ratio)
        self.drop_attn = nn.Dropout(dropout)
        self.drop_path_attn = StochasticDepth(drop_path_prob)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ffn_expand), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * ffn_expand, dim), nn.Dropout(dropout))
        self.drop_path_ffn = StochasticDepth(drop_path_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path_attn(self.drop_attn(self.attn(self.norm1(x))))
        x = x + self.drop_path_ffn(self.ffn(self.norm2(x)))
        return x

class SLIMBlock(nn.Module):

    def __init__(self, dim: int, num_heads: int=4, bottleneck_ratio: int=8, drop_key_ratio: float=0.6, pool_sizes: tuple=(1, 2, 3, 4), ffn_expand: int=4, dropout: float=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = BottleneckDropKeyAttention(dim, num_heads=num_heads, bottleneck_ratio=bottleneck_ratio, drop_key_ratio=drop_key_ratio)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * ffn_expand), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * ffn_expand, dim), nn.Dropout(dropout))
        self.msp = MultiScalePooling(pool_sizes)
        self.out_dim = dim * sum(pool_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.ffn(self.norm2(x))
        x = x.permute(0, 2, 1)
        x = self.msp(x)
        return x.flatten(1)

class SLIMEncoder(nn.Module):

    def __init__(self, dim: int, seq_len: int=200, n_layers: int=2, num_heads: int=4, bottleneck_ratio: int=8, drop_key_ratio: float=0.6, pool_sizes: tuple=(1, 2, 3, 4), ffn_expand: int=4, dropout: float=0.1, drop_path_rate: float=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        dpr = [x.item() for x in torch.linspace(0.0, drop_path_rate, n_layers)]
        self.layers = nn.ModuleList([SLIMLayer(dim=dim, num_heads=num_heads, bottleneck_ratio=bottleneck_ratio, drop_key_ratio=drop_key_ratio, ffn_expand=ffn_expand, dropout=dropout, drop_path_prob=dpr[i]) for i in range(n_layers)])
        self.final_norm = nn.LayerNorm(dim)
        self.msp = MultiScalePooling(pool_sizes)
        self.out_dim = dim * sum(pool_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = x + self.pos_embed[:, :x.size(1), :]
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        x = x.permute(0, 2, 1)
        x = self.msp(x)
        return x.flatten(1)
VARIANTS = {'lite': dict(tb1=16, tb2=16, eb1=32, eb2=32, eb3=32, slim=32), 'base': dict(tb1=32, tb2=32, eb1=64, eb2=64, eb3=64, slim=64), 'plus': dict(tb1=48, tb2=48, eb1=96, eb2=96, eb3=96, slim=96), 'max': dict(tb1=64, tb2=64, eb1=128, eb2=128, eb3=128, slim=128)}

class EEGNextFormer(nn.Module):

    def __init__(self, n_channels: int=7, n_samples: int=6400, n_classes: int=3, variant: str='base', dropout: float=0.3, pool_sizes: tuple=(1, 2, 3, 4), drop_path_rate: float=0.1, slim_layers: int=2):
        super().__init__()
        assert variant in VARIANTS, f'variant must be one of {list(VARIANTS.keys())}'
        cfg = VARIANTS[variant]
        tb1 = cfg['tb1']
        tb2 = cfg['tb2']
        eb1 = cfg['eb1']
        eb2 = cfg['eb2']
        eb3 = cfg['eb3']
        slim = cfg['slim']
        dpr_backbone = [drop_path_rate * (i + 1) / 3 for i in range(3)]
        self.norm_input = InstancePerChannelNorm(n_channels, affine=True)
        self.temporal1 = TemporalBlock(n_channels, tb1, kernel_size=7, pool_factor=2)
        self.temporal2 = TemporalBlock(tb1, tb2, kernel_size=7, pool_factor=2)
        self.ecgnext1 = GatedMultiScaleECGNextBlock(tb2, eb1, kernel_sizes=(7, 31, 63), pool_factor=2, drop_path_prob=dpr_backbone[0])
        self.ecgnext2 = GatedMultiScaleECGNextBlock(eb1, eb2, kernel_sizes=(7, 15, 31), pool_factor=2, drop_path_prob=dpr_backbone[1])
        self.ecgnext3 = GatedMultiScaleECGNextBlock(eb2, eb3, kernel_sizes=(5, 7, 15), pool_factor=2, drop_path_prob=dpr_backbone[2])
        self.proj_to_slim = nn.Sequential(nn.Conv1d(eb3, slim, 1), nn.GELU()) if eb3 != slim else nn.Identity()
        slim_seq_len = n_samples // 32
        self.slim_encoder = SLIMEncoder(dim=slim, seq_len=slim_seq_len, n_layers=slim_layers, num_heads=4, bottleneck_ratio=8, drop_key_ratio=0.6, pool_sizes=pool_sizes, ffn_expand=4, dropout=dropout, drop_path_rate=drop_path_rate)
        msp_out_dim = slim * sum(pool_sizes)
        hidden_dim = max(msp_out_dim // 2, n_classes * 8)
        self.classifier = nn.Sequential(nn.LayerNorm(msp_out_dim), nn.Dropout(dropout), nn.Linear(msp_out_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout * 0.5), nn.Linear(hidden_dim, n_classes))
        self.variant = variant
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.msp_out_dim = msp_out_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm, nn.InstanceNorm1d)):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm_input(x)
        x = self.temporal1(x)
        x = self.temporal2(x)
        x = self.ecgnext1(x)
        x = self.ecgnext2(x)
        x = self.ecgnext3(x)
        x = self.proj_to_slim(x)
        x = self.slim_encoder(x)
        x = self.classifier(x)
        return x

    def get_model_info(self):
        n_params = sum((p.numel() for p in self.parameters()))
        slim_seq = self.slim_encoder.pos_embed.shape[1]
        n_layers = len(self.slim_encoder.layers)
        print(f'\nEEGNextFormer ({self.variant.upper()} variant) — Professor Edition')
        print(f'  Input channels          : {self.n_channels}')
        print(f'  Output classes          : {self.n_classes}')
        print(f'  SLIM seq length         : {slim_seq} positions × 125ms = 25s context')
        print(f'  SLIM transformer layers : {n_layers} (with learnable positional embed)')
        print(f'  MSP output dim          : {self.msp_out_dim}')
        print(f'  Classifier hidden dim   : {max(self.msp_out_dim // 2, self.n_classes * 8)}')
        print(f'  Total params            : {n_params:,}')
        return n_params
