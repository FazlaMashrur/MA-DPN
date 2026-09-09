from __future__ import annotations
from typing import Any
from torch import nn
from .eegnet import EEGNetSmall
from .multiscale_dual_path import MultiScaleDualPath7ChannelEEGNet
IN_HOUSE_MODELS = {'eegnet', 'eegnet_small', 'multiscale_dual_path', 'majority_class', 'eegnextformer'}
BRAINDECODE_MODEL = 'braindecode'
FOUNDATION_MODELS = {'sleepfm', 'eegpt', 'reve', 'reve_1ch', 'labram_1ch', 'biot_1ch', 'bendr_1ch', 'reve_7ch', 'labram_7ch', 'biot_7ch', 'bendr_7ch'}

def build_model(model_cfg: dict[str, Any], n_channels: int, n_samples: int, n_classes: int) -> nn.Module:
    name = model_cfg.get('name', 'eegnet_small')
    kwargs = {key: value for (key, value) in model_cfg.items() if key != 'name'}
    if name in {'eegnet', 'eegnet_small'}:
        return EEGNetSmall(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes, **kwargs)
    if name == 'multiscale_dual_path':
        return MultiScaleDualPath7ChannelEEGNet(n_channels=n_channels, n_samples=n_samples, n_classes=n_classes, dropout=float(kwargs.get('dropout', 0.5)), attention_type=str(kwargs.get('attention_type', 'cbam')), kernel_sizes=list(kwargs.get('kernel_sizes', [3, 7, 15, 31, 63, 128])), fusion=str(kwargs.get('fusion', 'concat')), enable_band_attention=bool(kwargs.get('enable_band_attention', False)), enable_early_attention=bool(kwargs.get('enable_early_attention', False)), enable_mid_attention=bool(kwargs.get('enable_mid_attention', False)), enable_enhanced_temporal=bool(kwargs.get('enable_enhanced_temporal', True)))
    if name == 'braindecode':
        from .braindecode_arm import BraindecodeArm
        return BraindecodeArm(arch=str(kwargs['arch']), n_channels=n_channels, n_samples=n_samples, n_classes=n_classes, sfreq=float(kwargs.get('sfreq', 256.0)), pretrained=bool(kwargs.get('pretrained', False)), freeze_backbone=bool(kwargs.get('freeze_backbone', False)), extra=kwargs.get('extra'))
    if name in FOUNDATION_MODELS:
        from ..foundation import build_foundation_model
        return build_foundation_model(name, n_channels=n_channels, n_samples=n_samples, n_classes=n_classes, **kwargs)
    raise ValueError(f'Unknown model name: {name}. Known: {sorted(IN_HOUSE_MODELS | FOUNDATION_MODELS | {BRAINDECODE_MODEL})}')

def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum((p.numel() for p in model.parameters()))
    trainable = sum((p.numel() for p in model.parameters() if p.requires_grad))
    return {'total': int(total), 'trainable': int(trainable), 'frozen': int(total - trainable)}
