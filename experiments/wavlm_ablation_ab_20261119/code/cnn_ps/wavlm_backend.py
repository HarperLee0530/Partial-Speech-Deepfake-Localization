"""Explicit learned coordinate projection before the unchanged CNN backbone.

WavLM coordinates are not frequency bins: this is an architectural adaptation,
not a claim of a feature-only, parameter-matched ablation.
"""
import torch
from .backbone import CNNBiGRU

class ProjectedCNN(torch.nn.Module):
    def __init__(self, config, arm, seed):
        super().__init__()
        if arm not in ('mel_control','wavlm_frozen'):
            raise ValueError(arm)
        self.config=config
        self.cnn=CNNBiGRU(config)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed+771)
            self.project=(torch.nn.Sequential(torch.nn.LayerNorm(768),torch.nn.Linear(768,80))
                          if arm=='wavlm_frozen' else torch.nn.Identity())
        self.arm=arm
    def forward(self,features,lengths,return_features=False):
        expected=768 if self.arm=='wavlm_frozen' else 80
        if features.shape[-1]!=expected:
            raise ValueError('Wrong feature dimensions')
        mask=torch.arange(features.shape[2],device=features.device)[None,:]<lengths.to(features.device)[:,None]
        safe=torch.where(mask[:,None,:,None],features,torch.zeros_like(features))
        return self.cnn(self.project(safe),lengths,return_features=return_features)
