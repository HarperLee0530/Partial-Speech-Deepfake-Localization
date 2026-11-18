"""Original project LogMel, with only a dimension adapter for the SAL backend."""
import torch
from torch import nn
from cnn import LogMel, ModelConfig


class LogMelEncoder(nn.Module):
    def __init__(self, hid_dim=1024):
        super().__init__()
        self.out_dim = hid_dim
        self.frontend = LogMel(ModelConfig())
        self.input_norm = nn.LayerNorm(80)
        self.projection = nn.Linear(80, hid_dim)

    def extract_feat(self, waveform):
        # Invoke the original frontend verbatim, including its edge padding.
        features = torch.cat([self.frontend(x) for x in waveform], dim=0)
        return self.projection(self.input_norm(features))
