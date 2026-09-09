from .eegnet import EEGNetSmall
from .multiscale_dual_path import MultiScaleDualPath7ChannelEEGNet
from .eegnextformer import EEGNextFormer
from .registry import build_model, count_parameters, IN_HOUSE_MODELS, FOUNDATION_MODELS
__all__ = ['EEGNetSmall', 'MultiScaleDualPath7ChannelEEGNet', 'EEGNextFormer', 'build_model', 'count_parameters', 'IN_HOUSE_MODELS', 'FOUNDATION_MODELS']
