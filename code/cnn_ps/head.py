"""Small residual frame classifier on a frozen CNN-BiGRU representation.

The eight-frame reduction is fixed uniform pooling, not learned attention.
The discarded initializer below preserves the random-number sequence used by
the historical experiment, without retaining its unused pooling network.
"""
from pathlib import Path

import torch
from torch import nn


class ResidualFrameHead(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.register_buffer('context_mean', torch.zeros(256))
        self.register_buffer('context_scale', torch.ones(256))
        # Historical initialization instantiated an unused gate first. Its
        # random draws matter when reproducing the original seed-based fits.
        discarded = nn.Sequential(nn.Linear(260, self.hidden_size), nn.Tanh(),
                                  nn.Linear(self.hidden_size, 1))
        del discarded
        self.frame_residual = nn.Sequential(
            nn.Linear(256, self.hidden_size), nn.Tanh(), nn.Linear(self.hidden_size, 1))
        nn.init.zeros_(self.frame_residual[-1].weight)
        nn.init.zeros_(self.frame_residual[-1].bias)

    def forward(self, native_logits, context, return_native=False):
        if native_logits.ndim != 2 or native_logits.shape[1] != 8:
            raise ValueError('native_logits must have shape [blocks,8]')
        if context.shape != (*native_logits.shape, 256):
            raise ValueError('context must have shape [blocks,8,256]')
        native_logits, context = native_logits.float(), context.float()
        normalized = ((context-self.context_mean) /
                      self.context_scale.clamp_min(1e-4)).clamp(-10, 10)
        fine = native_logits+self.frame_residual(normalized).squeeze(-1)
        weights = torch.full_like(fine, 1/8)
        coarse = (weights*fine).sum(1)
        return (coarse, weights, fine) if return_native else coarse


def load_head(path, device='cpu'):
    """Load a new checkpoint or the historical fixed-uniform b_model.pt.

    Checkpoints are pickle-based PyTorch artifacts: load only trusted files.
    Historical adaptive-pooling checkpoints are deliberately rejected.
    """
    stored = torch.load(Path(path), map_location='cpu', weights_only=False)
    if stored.get('fine_aux', True) is not True or stored.get('freeze_gate', True) is not True:
        raise ValueError('Checkpoint is not a residual head with fixed uniform pooling')
    model = ResidualFrameHead(stored['hidden_size'])
    state = stored['state_dict']
    expected = set(model.state_dict())
    ignored = {'logit_mean', 'logit_scale', 'position',
               'gate.0.weight', 'gate.0.bias', 'gate.2.weight', 'gate.2.bias'}
    unexpected = set(state)-expected-ignored
    if unexpected:
        raise ValueError(f'Unexpected checkpoint fields: {sorted(unexpected)}')
    model.load_state_dict({key: state[key] for key in expected}, strict=True)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError('Nonfinite head checkpoint')
    return model.to(device).eval()
