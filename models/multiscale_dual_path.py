from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
DEFAULT_KERNEL_SIZES = [3, 7, 15, 31, 63, 128]

class _SepConv1d(nn.Module):

    def __init__(self, ni, no, kernel, stride, padding):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=padding, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class SepConv1d(nn.Module):

    def __init__(self, ni, no, kernel, stride, pad, drop=None, bn=True, activ=lambda : nn.PReLU()):
        super().__init__()
        assert drop is None or 0.0 < drop < 1.0
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if bn:
            layers.append(nn.BatchNorm1d(no))
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Flatten(nn.Module):

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.reshape(x.size(0), -1)
        return x.reshape(-1)

class MultiScaleConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes=None, stride=1, drop=None, bn=True, fusion='concat'):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = DEFAULT_KERNEL_SIZES
        self.kernel_sizes = kernel_sizes
        self.fusion = fusion
        self.n_branches = len(kernel_sizes)
        self.branches = nn.ModuleList()
        for k in kernel_sizes:
            pad = k // 2
            branch = nn.Sequential(_SepConv1d(in_channels, out_channels, k, stride, pad), nn.PReLU(), nn.BatchNorm1d(out_channels) if bn else nn.Identity(), nn.Dropout(drop) if drop else nn.Identity())
            self.branches.append(branch)
        if fusion == 'concat':
            self.out_channels = out_channels * self.n_branches
        elif fusion == 'sum':
            self.out_channels = out_channels
        elif fusion == 'attention':
            self.out_channels = out_channels
            self.branch_attention = nn.Sequential(nn.AdaptiveAvgPool1d(1), Flatten(), nn.Linear(out_channels * self.n_branches, self.n_branches), nn.Softmax(dim=1))
        else:
            raise ValueError(f'Unknown fusion mode: {fusion}')

    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        min_t = min((b.shape[-1] for b in branch_outputs))
        branch_outputs = [b[..., :min_t] for b in branch_outputs]
        if self.fusion == 'concat':
            return torch.cat(branch_outputs, dim=1)
        elif self.fusion == 'sum':
            return sum(branch_outputs)
        else:
            stacked = torch.stack(branch_outputs, dim=1)
            concat_for_att = torch.cat(branch_outputs, dim=1)
            att_weights = self.branch_attention(concat_for_att)
            att_weights = att_weights.unsqueeze(-1).unsqueeze(-1)
            return (stacked * att_weights).sum(dim=1)

class MultiScaleBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes=None, stride=1, drop=0.3, fusion='concat'):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = DEFAULT_KERNEL_SIZES
        self.ms_conv = MultiScaleConv1d(in_channels, out_channels, kernel_sizes, stride, drop, bn=True, fusion=fusion)
        self.actual_out_channels = self.ms_conv.out_channels
        if in_channels != self.actual_out_channels or stride != 1:
            self.residual = nn.Sequential(nn.Conv1d(in_channels, self.actual_out_channels, 1, stride), nn.BatchNorm1d(self.actual_out_channels))
        else:
            self.residual = nn.Identity()
        self.activation = nn.PReLU()

    def forward(self, x):
        identity = self.residual(x)
        out = self.ms_conv(x)
        if out.shape[-1] != identity.shape[-1]:
            identity = F.adaptive_avg_pool1d(identity, out.shape[-1])
        return self.activation(out + identity)

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channels, reduced_channels, bias=False), nn.ReLU(inplace=True), nn.Linear(reduced_channels, channels, bias=False), nn.Sigmoid())

    def forward(self, x):
        (b, c, _) = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class ECABlock(nn.Module):

    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        if k_size is None:
            k_size = int(abs((math.log(channels, 2) + 1) / 2))
            k_size = k_size if k_size % 2 else k_size + 1
        self.k_size = max(k_size, 3)
        padding = self.k_size // 2
        self.conv = nn.Conv1d(1, 1, kernel_size=self.k_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        (b, c, _) = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.conv(y.unsqueeze(1)).squeeze(1)
        y = self.sigmoid(y).view(b, c, 1)
        return x * y.expand_as(x)

class ChannelAttention1D(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(nn.Conv1d(channels, reduced_channels, 1, bias=False), nn.ReLU(), nn.Conv1d(reduced_channels, channels, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention1D(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        (max_out, _) = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(combined))

class CBAM1D(nn.Module):

    def __init__(self, channels, reduction=4, kernel_size=7):
        super().__init__()
        self.channel_att = ChannelAttention1D(channels, reduction)
        self.spatial_att = SpatialAttention1D(kernel_size)

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x

class SpectralAttention(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(channels, reduced_channels, bias=False), nn.ReLU(inplace=True), nn.Linear(reduced_channels, channels, bias=False), nn.Sigmoid())
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        (b, c, _) = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return self.gamma * (x * y.expand_as(x)) + x

class PerBandTemporalAttention(nn.Module):

    def __init__(self, channels, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.attention = nn.MultiheadAttention(channels, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        (attn_out, _) = self.attention(x_t, x_t, x_t)
        x_t = self.norm(x_t + attn_out)
        return x_t.transpose(1, 2)

class EnhancedTemporalAttention(nn.Module):

    def __init__(self, channels, n_heads=8, dropout=0.1, max_len=6500):
        super().__init__()
        self.channels = channels
        self.n_heads = n_heads
        self.register_buffer('pos_encoding', self._get_positional_encoding(max_len, channels))
        self.attention = nn.MultiheadAttention(channels, n_heads, dropout=dropout, batch_first=True)
        self.temporal_conv = nn.Conv1d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _get_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, x):
        (B, C, T) = x.shape
        x_t = x.transpose(1, 2)
        x_t = x_t + self.pos_encoding[:, :T, :]
        (attn_out, _) = self.attention(x_t, x_t, x_t)
        x_t = self.norm1(x_t + self.dropout(attn_out))
        x_local = self.temporal_conv(x_t.transpose(1, 2))
        x_t = self.norm2(x_t + self.dropout(x_local.transpose(1, 2)))
        return x_t.transpose(1, 2)

class MultiHeadClassifier(nn.Module):

    def __init__(self, in_features, n_classes, n_heads=3, dropout=0.3):
        super().__init__()
        self.n_heads = n_heads
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(in_features, 128), nn.PReLU(), nn.BatchNorm1d(128), nn.Dropout(dropout), nn.Linear(128, 64), nn.PReLU(), nn.BatchNorm1d(64), nn.Dropout(dropout / 2), nn.Linear(64, n_classes)) for _ in range(n_heads)])

    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        return torch.mean(torch.stack(outputs), dim=0)

class MultiScaleDualPath7ChannelEEGNet(nn.Module):

    def __init__(self, n_channels: int=7, n_samples: int=5000, n_classes: int=3, dropout: float=0.5, attention_type: str='cbam', kernel_sizes: list[int] | None=None, fusion: str='concat', enable_band_attention: bool=False, enable_early_attention: bool=False, enable_mid_attention: bool=False, enable_enhanced_temporal: bool=True) -> None:
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = DEFAULT_KERNEL_SIZES
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.attention_type = attention_type
        self.kernel_sizes = kernel_sizes
        self.fusion = fusion
        self.enable_band_attention = enable_band_attention
        self.enable_early_attention = enable_early_attention
        self.enable_mid_attention = enable_mid_attention
        self.enable_enhanced_temporal = enable_enhanced_temporal
        self.n_fft_bins = n_samples // 2 + 1
        reduced_kernels = [k for k in kernel_sizes if k <= 63]
        small_kernels = [k for k in kernel_sizes if k <= 31]
        if enable_band_attention:
            self.raw_input_band_cbam = CBAM1D(channels=n_channels, reduction=1, kernel_size=31)
            self.fft_input_band_cbam = CBAM1D(channels=n_channels, reduction=1, kernel_size=31)
        else:
            self.raw_input_band_cbam = nn.Identity()
            self.fft_input_band_cbam = nn.Identity()
        self.raw_ms_block1 = MultiScaleBlock(n_channels, 32, kernel_sizes=kernel_sizes, stride=2, drop=dropout, fusion=fusion)
        ms1_out = self.raw_ms_block1.actual_out_channels
        self.raw_early_cbam = CBAM1D(channels=ms1_out, reduction=2, kernel_size=15) if enable_early_attention else nn.Identity()
        self.raw_ms_block2 = MultiScaleBlock(ms1_out, 64, kernel_sizes=reduced_kernels, stride=4, drop=dropout, fusion=fusion)
        ms2_out = self.raw_ms_block2.actual_out_channels
        self.raw_ms_block3 = MultiScaleBlock(ms2_out, 128, kernel_sizes=small_kernels, stride=4, drop=dropout, fusion=fusion)
        ms3_out = self.raw_ms_block3.actual_out_channels
        self.raw_mid_cbam = CBAM1D(channels=ms3_out, reduction=4, kernel_size=11) if enable_mid_attention else nn.Identity()
        self.raw_final = nn.Sequential(SepConv1d(ms3_out, 256, kernel=8, stride=4, pad=4, drop=None, bn=True))
        self.raw_attention = self._get_attention(256, attention_type)
        self.raw_temporal_attn = EnhancedTemporalAttention(channels=256, n_heads=8, dropout=dropout) if enable_enhanced_temporal else PerBandTemporalAttention(channels=256, n_heads=4, dropout=dropout)
        self._raw_features_size = self._calc_raw_features()
        self.raw_dense = nn.Sequential(Flatten(), nn.Dropout(dropout), nn.Linear(self._raw_features_size, 256), nn.PReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout), nn.Linear(256, 128), nn.PReLU(), nn.BatchNorm1d(128))
        self.fft_ms_block1 = MultiScaleBlock(n_channels, 32, kernel_sizes=kernel_sizes, stride=2, drop=dropout, fusion=fusion)
        fft_ms1_out = self.fft_ms_block1.actual_out_channels
        self.fft_early_cbam = CBAM1D(channels=fft_ms1_out, reduction=2, kernel_size=15) if enable_early_attention else nn.Identity()
        self.fft_ms_block2 = MultiScaleBlock(fft_ms1_out, 64, kernel_sizes=reduced_kernels, stride=2, drop=dropout, fusion=fusion)
        fft_ms2_out = self.fft_ms_block2.actual_out_channels
        self.fft_ms_block3 = MultiScaleBlock(fft_ms2_out, 128, kernel_sizes=small_kernels, stride=4, drop=dropout, fusion=fusion)
        fft_ms3_out = self.fft_ms_block3.actual_out_channels
        self.fft_mid_cbam = CBAM1D(channels=fft_ms3_out, reduction=4, kernel_size=11) if enable_mid_attention else nn.Identity()
        self.fft_final = nn.Sequential(SepConv1d(fft_ms3_out, 256, kernel=8, stride=4, pad=4, drop=None, bn=True))
        self.fft_attention = SpectralAttention(channels=256, reduction=4)
        self.fft_temporal_attn = EnhancedTemporalAttention(channels=256, n_heads=8, dropout=dropout) if enable_enhanced_temporal else PerBandTemporalAttention(channels=256, n_heads=4, dropout=dropout)
        self._fft_features_size = self._calc_fft_features()
        self.fft_dense = nn.Sequential(Flatten(), nn.Dropout(dropout), nn.Linear(self._fft_features_size, 256), nn.PReLU(), nn.BatchNorm1d(256), nn.Dropout(dropout), nn.Linear(256, 128), nn.PReLU(), nn.BatchNorm1d(128))
        self.classifier = MultiHeadClassifier(256, n_classes, n_heads=3, dropout=dropout)

    @staticmethod
    def _get_attention(channels: int, attention_type: str) -> nn.Module:
        if attention_type == 'se':
            return SEBlock(channels, reduction=4)
        if attention_type == 'eca':
            return ECABlock(channels, k_size=3)
        if attention_type == 'cbam':
            return CBAM1D(channels, reduction=4, kernel_size=7)
        if attention_type == 'spectral':
            return SpectralAttention(channels, reduction=4)
        return nn.Identity()

    def _calc_raw_features(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, self.n_channels, self.n_samples)
            x = self.raw_input_band_cbam(x)
            x = self.raw_ms_block1(x)
            x = self.raw_early_cbam(x)
            x = self.raw_ms_block2(x)
            x = self.raw_ms_block3(x)
            x = self.raw_mid_cbam(x)
            x = self.raw_final(x)
            x = self.raw_attention(x)
            x = self.raw_temporal_attn(x)
            return x.numel()

    def _calc_fft_features(self) -> int:
        with torch.no_grad():
            x = torch.zeros(1, self.n_channels, self.n_fft_bins)
            x = self.fft_input_band_cbam(x)
            x = self.fft_ms_block1(x)
            x = self.fft_early_cbam(x)
            x = self.fft_ms_block2(x)
            x = self.fft_ms_block3(x)
            x = self.fft_mid_cbam(x)
            x = self.fft_final(x)
            x = self.fft_attention(x)
            x = self.fft_temporal_attn(x)
            return x.numel()

    @staticmethod
    def _compute_fft(x: torch.Tensor) -> torch.Tensor:
        return torch.abs(torch.fft.rfft(x, dim=-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        raw = self.raw_input_band_cbam(x)
        raw = self.raw_ms_block1(raw)
        raw = self.raw_early_cbam(raw)
        raw = self.raw_ms_block2(raw)
        raw = self.raw_ms_block3(raw)
        raw = self.raw_mid_cbam(raw)
        raw = self.raw_final(raw)
        raw = self.raw_attention(raw)
        raw = self.raw_temporal_attn(raw)
        raw_features = self.raw_dense(raw)
        x_fft = self._compute_fft(x)
        fft = self.fft_input_band_cbam(x_fft)
        fft = self.fft_ms_block1(fft)
        fft = self.fft_early_cbam(fft)
        fft = self.fft_ms_block2(fft)
        fft = self.fft_ms_block3(fft)
        fft = self.fft_mid_cbam(fft)
        fft = self.fft_final(fft)
        fft = self.fft_attention(fft)
        fft = self.fft_temporal_attn(fft)
        fft_features = self.fft_dense(fft)
        combined = torch.cat([raw_features, fft_features], dim=1)
        return self.classifier(combined)
