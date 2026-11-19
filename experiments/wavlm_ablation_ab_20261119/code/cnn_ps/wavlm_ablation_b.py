"""Frozen-encoder positional interventions, with the complete A0 backend.

Only instance methods are changed; installed transformers and checkpoint
parameters remain untouched. These are pretrained-model interventions, not
claims about architectures pretrained from scratch without these components.
"""
from pathlib import Path
from types import MethodType
import numpy as np
import torch
import transformers
from transformers.models.wavlm import modeling_wavlm as hf
from . import wavlm_ablation_a as a
from .dual_features import FeatureData
from .io import read_json, sha256
from .wavlm_frontend import REVISION
from .wavlm_multilayer import FrozenMultiWavLM

INTERVENTIONS = dict(B0_full='original', B1_no_conv_position='no_conv_position',
                     B2_position_gate_one='gate_one', B3_no_relative_bias='no_relative_bias')
ARMS = {arm: dict(a.ARMS['A0_full'], encoder_intervention=mode)
        for arm, mode in INTERVENTIONS.items()}
NEW_ARMS = tuple(list(ARMS)[1:])
CACHE_SCHEMA = 'ps_frozen_wavlm_positional_cache_v1'
MODEL_SHA256 = '3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51'


def encoder_sources():
    return dict(transformers_version=transformers.__version__, torch_version=torch.__version__,
        transformers_wavlm_sha256=sha256(hf.__file__),
        source_sha256={name: sha256(Path(__file__).with_name(name)) for name in
            ('wavlm_ablation_b.py', 'cache_wavlm_ablation_b.py', 'wavlm_frontend.py',
             'wavlm_multilayer.py', 'augmentation.py')})


def zero_conv_position(self, hidden_states):
    # Positional convolution returns B,T,H; replacing with Identity would add
    # hidden_states again and would NOT remove the positional term.
    return torch.zeros_like(hidden_states)


def ungated_attention(self, hidden_states, attention_mask=None, position_bias=None,
                      output_attentions=False, index=0):
    batch, length, _ = hidden_states.shape
    if self._b_mode == 'no_relative_bias':
        position_bias = hidden_states.new_zeros(batch*self.num_heads, length, length)
    elif position_bias is None:
        bias = self.compute_bias(length, length)
        position_bias = bias.unsqueeze(0).repeat(batch, 1, 1, 1).view(
            batch*self.num_heads, length, length)
    # Gate is exactly one in B2. B3 skips both relative bias and its gate.
    result, weights = self.torch_multi_head_self_attention(
        hidden_states, attention_mask, position_bias, output_attentions)
    return result, weights, position_bias


def configure_encoder(model, arm):
    if arm not in ARMS:
        raise ValueError(f'Unknown B arm: {arm}')
    if hasattr(model, '_b_arm'):
        raise ValueError('An encoder instance cannot be configured twice')
    if transformers.__version__ != '4.51.3' or not isinstance(model.encoder, hf.WavLMEncoder):
        raise ValueError('Validated implementation requires transformers 4.51.3 and WavLMEncoder')
    mode = INTERVENTIONS[arm]
    if mode == 'no_conv_position':
        model.encoder.pos_conv_embed.forward = MethodType(zero_conv_position, model.encoder.pos_conv_embed)
    elif mode in ('gate_one', 'no_relative_bias'):
        for layer in model.encoder.layers:
            layer.attention._b_mode = mode
            layer.attention.forward = MethodType(ungated_attention, layer.attention)
    model._b_arm = arm
    model.eval().requires_grad_(False)
    return model


class FrozenPositionalWavLM(FrozenMultiWavLM):
    def __init__(self, folder, arm, device='cuda'):
        if sha256(Path(folder)/'pytorch_model.bin') != MODEL_SHA256:
            raise ValueError('Pretrained WavLM weights differ from the declared baseline')
        super().__init__(folder, device)
        configure_encoder(self.encoder, arm)
        self.arm = arm
        self.eval()

    def train(self, mode=True):
        # An accidental parent.train() must never activate dropout/SpecAugment.
        return super().train(False)


class PositionalCNN(a.AblationCNN):
    def __init__(self, config, arm, seed):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(config, 'A0_full', seed)
        self.ablation_arm = arm


make_adapter = a.make_adapter
training_loss = a.training_loss


def loss_weights(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return a.loss_weights('A0_full')


def validate_cache_metadata(metadata, arm, reference_sha, split, recordings, channels):
    expected = dict(schema=CACHE_SCHEMA, status='complete', arm=arm,
        encoder_intervention=INTERVENTIONS[arm], layers=[6, 9, 12], frozen=True,
        reference_sha256=reference_sha, split=split, recordings=recordings,
        channels=channels, dtype='float32', revision=REVISION,
        model_sha256=MODEL_SHA256, source_unchanged=True, encoder_sources=encoder_sources())
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f'B cache provenance mismatch: {key}')


class PositionalData(FeatureData):
    def __init__(self, reference, cache, arm, **kwargs):
        super().__init__(reference, **kwargs)
        self.ssl_metadata = m = read_json(Path(cache)/'complete.json')
        validate_cache_metadata(m, arm, sha256(self.folder/'complete.json'),
                                self.split, len(self), self.channels)
        path = Path(cache)/'features.npy'
        if sha256(path) != m['features_sha256']:
            raise ValueError('B feature bytes changed')
        self.ssl = np.load(path, mmap_mode='r', allow_pickle=False)
        expected = (self.channels, int(self.offsets[-1]), 3, 768)
        if self.ssl.shape != expected or self.ssl.dtype != np.float32 or m['shape'] != list(expected):
            raise ValueError('B feature grid/dtype mismatch')

    def get_record(self, index, channel=0):
        row = super().get_record(index, channel)
        start, end = self.offsets[index:index+2]
        row['features'] = torch.from_numpy(np.array(self.ssl[channel, start:end], copy=True)
                                          .reshape(end-start, 2304)).unsqueeze(0)
        return row
