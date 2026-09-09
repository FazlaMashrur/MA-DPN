from __future__ import annotations
import torch
from torch import nn

class EEGNetSmall(nn.Module):

    def __init__(self, n_channels: int, n_samples: int, n_classes: int, dropout_rate: float=0.25, F1: int=8, D: int=2, F2: int=16, temporal_kernel: int=64, separable_kernel: int=16):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.temporal = nn.Sequential(nn.Conv2d(1, F1, kernel_size=(1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False), nn.BatchNorm2d(F1))
        self.depthwise = nn.Sequential(nn.Conv2d(F1, F1 * D, kernel_size=(n_channels, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d(kernel_size=(1, 4)), nn.Dropout(dropout_rate))
        self.separable = nn.Sequential(nn.Conv2d(F1 * D, F1 * D, kernel_size=(1, separable_kernel), padding=(0, separable_kernel // 2), groups=F1 * D, bias=False), nn.Conv2d(F1 * D, F2, kernel_size=(1, 1), bias=False), nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d(kernel_size=(1, 8)), nn.Dropout(dropout_rate))
        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_samples)
            n_features = self._forward_features(dummy).flatten(1).shape[1]
        self.classifier = nn.Linear(n_features, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        return self.classifier(torch.flatten(x, start_dim=1))
