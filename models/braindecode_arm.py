from __future__ import annotations
from typing import Any
import torch
from torch import nn
REFERENCE_ARCHS = {'ShallowFBCSPNet', 'Deep4Net', 'EEGConformer', 'EEGNetv4', 'EEGNeX', 'EEGITNet', 'EEGTCNet', 'ATCNet', 'TSception', 'SincShallowNet', 'EEGSimpleConv', 'SPARCNet', 'CTNet', 'MSVTNet', 'FBCNet', 'TCN', 'SleepStagerChambon2018', 'SleepStagerBlanco2020', 'USleep', 'AttnSleep', 'DeepSleepNet', 'ContraWR', 'SCCNet', 'EEGInceptionERP'}
PRETRAINED_ARCHS = {'Labram': 'pretrained on TUEG and related corpora', 'BIOT': 'pretrained biosignal transformer', 'BENDR': 'pretrained contrastive EEG representation', 'REVE': 'brain-bzh/reve-base', 'SignalJEPA': 'self-supervised joint embedding predictive architecture'}

class BraindecodeArm(nn.Module):

    def __init__(self, arch: str, n_channels: int, n_samples: int, n_classes: int, sfreq: float=256.0, pretrained: bool=False, freeze_backbone: bool=False, extra: dict[str, Any] | None=None):
        super().__init__()
        try:
            import braindecode.models as bdm
        except ImportError as exc:
            raise ImportError(f'The {arch} arm needs braindecode. Use the braindecode_falza interpreter: <CONDA_ENV_PYTHON>') from exc
        if not hasattr(bdm, arch):
            raise ValueError(f'braindecode has no model named {arch!r}')
        cls = getattr(bdm, arch)
        kwargs: dict[str, Any] = {'n_chans': n_channels, 'n_outputs': n_classes, 'n_times': n_samples, 'sfreq': sfreq}
        kwargs.update(extra or {})
        if pretrained:
            if arch not in PRETRAINED_ARCHS:
                raise ValueError(f'{arch} has no pretrained weights registered. Set pretrained=False or pick one of: ' + ', '.join(sorted(PRETRAINED_ARCHS)))
            self.net = cls.from_pretrained(**kwargs)
        else:
            self.net = self._construct(cls, kwargs)
        self.arch = arch
        self.braindecode_pretrained = bool(pretrained)
        self.frozen_backbone = bool(freeze_backbone)
        if freeze_backbone:
            self._freeze()

    @staticmethod
    def _construct(cls, kwargs: dict[str, Any]) -> nn.Module:
        attempt = dict(kwargs)
        for _ in range(len(kwargs)):
            try:
                return cls(**attempt)
            except TypeError as exc:
                msg = str(exc)
                dropped = None
                for key in ('sfreq', 'n_times', 'chs_info', 'input_window_seconds'):
                    if key in attempt and key in msg:
                        dropped = key
                        break
                if dropped is None:
                    raise
                attempt.pop(dropped)
        return cls(**attempt)

    def _freeze(self) -> None:
        for p in self.net.parameters():
            p.requires_grad = False
        for attr in ('final_layer', 'head', 'classifier', 'fc', 'clf', 'output'):
            mod = getattr(self.net, attr, None)
            if mod is not None and isinstance(mod, nn.Module):
                for p in mod.parameters():
                    p.requires_grad = True
                return
        last = [m for m in self.net.modules() if any((True for _ in m.parameters(recurse=False)))]
        if last:
            for p in last[-1].parameters(recurse=False):
                p.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        out = self.net(x)
        if out.dim() == 3:
            out = out.mean(dim=-1)
        return out
