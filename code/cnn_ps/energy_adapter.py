"""Minimal energy conditioning of the existing boundary adapter's fine path.

Only the pre-GELU fine projection changes: acoustic_linear(h) + energy_linear(e).
The base adapter still owns every coarse, attention, gate and boundary operation.
"""
from __future__ import annotations

import torch
from torch import nn

from .dual_resolution import DualResolutionAdapter
from .train_four import ARMS


class EnergyBoundaryAdapter(DualResolutionAdapter):
    """Original boundary adapter plus optional zero-initialized energy input.

Raw energy5[T,5] is a fixed waveform feature. Normalization statistics come
from clean train only. Actual positive duration, not nonzero feature content,
defines valid frames. The control bypasses the new projection exactly.
"""
    def __init__(self, mean, std, energy_enabled):
        super().__init__(torch.zeros(256), torch.ones(256), **ARMS['boundary'])
        # Construct this only after all common layers, preserving their draws.
        self.energy_enabled = bool(energy_enabled)
        self.energy_projection = nn.Linear(5, self.hidden, bias=False)
        nn.init.zeros_(self.energy_projection.weight)
        self.energy_projection.weight.requires_grad_(self.energy_enabled)
        mean, std = torch.as_tensor(mean, dtype=torch.float32), torch.as_tensor(std, dtype=torch.float32)
        if (mean.shape != (5,) or std.shape != (5,) or not torch.isfinite(mean).all()
                or not torch.isfinite(std).all() or (std < 1e-4).any()):
            raise ValueError('Energy normalization needs finite mean/std[5], std>=1e-4')
        self.register_buffer('energy_mean', mean.detach().clone())
        self.register_buffer('energy_std', std.detach().clone())

    def normalize_energy(self, energy, weights):
        if (weights.ndim != 1 or energy.shape != (len(weights), 5)
                or energy.device != weights.device or weights.device != self.energy_mean.device):
            raise ValueError('Expected raw energy[T,5] on the duration-mask device')
        valid = weights > 0
        clean = torch.where(valid[:, None], energy.detach().to(self.energy_mean), self.energy_mean[None, :])
        if not torch.isfinite(clean).all():
            raise ValueError('Non-finite energy in an actual frame')
        return ((clean-self.energy_mean)/self.energy_std).clamp(-10., 10.) * valid[:, None]

    def forward(self, h, z, weights, energy):
        if energy.shape != (len(weights), 5):
            raise ValueError('Expected aligned raw energy[T,5]')
        condition = self.energy_projection(self.normalize_energy(energy, weights)) if self.energy_enabled else None
        return super().forward(h, z, weights, fine_condition=condition)

    def common_state_dict(self):
        return {key:value for key,value in self.state_dict().items()
                if key not in ('energy_mean', 'energy_std', 'energy_projection.weight')}


def make_energy_adapter(enabled, normalization, seed, device):
    # Do not advance CPU/CUDA RNG used by the shared backbone's dropout.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        module = EnergyBoundaryAdapter(normalization['mean'], normalization['std'], enabled)
    return module.to(device)
