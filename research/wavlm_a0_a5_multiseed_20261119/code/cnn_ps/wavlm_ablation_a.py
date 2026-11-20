"""Predeclared A0--A10 ablations; independent architectural and loss switches.

Instantiate the full adapter before disabling parameters, preserving every
shared initialization and CPU/CUDA RNG. Disabled parameters are frozen and
bypassed, not silently trained through another path.
"""
import numpy as np
import torch
from torch.nn import functional as F
from .dual_resolution import DualResolutionAdapter
from .dual_features import FeatureData
from .wavlm_multilayer import MultiLayerData
from .wavlm_backend import ProjectedCNN
from .finetune import objective
from .train_dual import boundary_targets, weighted_bce

BASE_OPTIONS = dict(variant='fusion', normalize_mode='layernorm', learned_pool=True,
                    boundary_head=True, relative_position=False, reliability_gate=False)
DEFAULT = dict(feature_mode='learned', learned_pool=True, attention=True, gate=True,
               boundary=True, consistency=True, coarse_aux=True)
CHANGES = {
    'A0_full': {},
    'A1_layer12': dict(feature_mode='layer12'),
    'A2_layer6': dict(feature_mode='layer6'),
    'A3_layer9': dict(feature_mode='layer9'),
    'A4_mean': dict(feature_mode='mean'),
    'A5_fixed_pool': dict(learned_pool=False),
    'A6_no_attention': dict(attention=False),
    'A7_fixed_gate': dict(gate=False),
    'A8_no_boundary': dict(boundary=False),
    'A9_no_consistency': dict(consistency=False),
    'A10_no_coarse_aux': dict(coarse_aux=False),
}
ARMS = {name: dict(DEFAULT, **change) for name, change in CHANGES.items()}
NEW_ARMS = tuple(list(ARMS)[2:])


def loss_weights(arm):
    option = ARMS[arm]
    return dict(main20=1., main160=1., coarse_aux=.5 if option['coarse_aux'] else 0.,
                boundary=.1 if option['boundary'] else 0.,
                consistency=.05 if option['consistency'] else 0.)


class AblationCNN(ProjectedCNN):
    def __init__(self, config, arm, seed):
        super().__init__(config, 'wavlm_frozen', seed)
        self.ablation_arm, self.feature_mode = arm, ARMS[arm]['feature_mode']
        if self.feature_mode == 'learned':
            self.layer_logits = torch.nn.Parameter(torch.zeros(3))

    def layer_weights(self):
        if self.feature_mode == 'learned':
            return self.layer_logits.softmax(0)
        if self.feature_mode == 'mean':
            return self.project[1].weight.new_full((3,), 1/3)
        values = [float(self.feature_mode == f'layer{i}') for i in (6, 9, 12)]
        return self.project[1].weight.new_tensor(values)

    def forward(self, features, lengths, return_features=False):
        multiple = self.feature_mode in ('learned', 'mean')
        if features.shape[-1] != (2304 if multiple else 768):
            raise ValueError('Incorrect ablation feature dimensions')
        if multiple:
            states = features.reshape(*features.shape[:-1], 3, 768)
            features = (states * self.layer_weights()[:, None]).sum(-2)
        return super().forward(features, lengths, return_features=return_features)


class AblationData(MultiLayerData):
    def __init__(self, reference, cache, arm, **kwargs):
        super().__init__(reference, cache, arm='fusion6_9_12', **kwargs)
        self.feature_mode = ARMS[arm]['feature_mode']

    def get_record(self, index, channel=0):
        row = FeatureData.get_record(self, index, channel)
        a, b = self.offsets[index:index+2]
        selected = self.ssl[channel, a:b]
        if self.feature_mode.startswith('layer'):
            layer_index = (6, 9, 12).index(int(self.feature_mode[5:]))
            selected = selected[:, layer_index, :]
        row['features'] = torch.from_numpy(np.array(selected, copy=True).reshape(b-a, -1)).unsqueeze(0)
        return row


class AblationAdapter(DualResolutionAdapter):
    def __init__(self, options):
        super().__init__(torch.zeros(256), torch.ones(256), **BASE_OPTIONS)
        self.learned_pool = options['learned_pool']
        self.boundary_head = options['boundary']
        self.attention_enabled, self.gate_enabled = options['attention'], options['gate']
        disabled = []
        if not self.learned_pool:
            disabled += ['coarse_downsample']
        if not self.boundary_head:
            disabled += ['boundary_output']
        if not self.attention_enabled:
            disabled += ['query_projection', 'key_projection', 'value_projection', 'attention_output']
        if not self.gate_enabled:
            disabled += ['gate_network']
        self.disabled_modules = disabled
        for name in disabled:
            getattr(self, name).requires_grad_(False)

    def forward(self, h, z, weights):
        normalized, base, weights, valid = self._inputs(h, z, weights)
        fine_features = F.gelu(self.fine_projection(normalized)) * valid[:, None]
        fine_residual = self.fine_output(fine_features).squeeze(-1) * valid
        fine_logits = base + fine_residual
        coarse_base, coarse_weights = self.masked_pool(base, weights)
        coarse_valid = coarse_weights > 0
        result = dict(fine_logits=fine_logits, fine_residual=fine_residual,
                      coarse_base=coarse_base, coarse_weights=coarse_weights,
                      valid_mask=valid, coarse_valid_mask=coarse_valid)
        boundary_logits = None
        if self.boundary_head:
            boundary_logits = self.boundary_output(fine_features).squeeze(-1) * valid
            result['boundary_logits'] = boundary_logits
        coarse_input = self._coarse_input(normalized, weights)
        coarse_features = F.gelu(self.coarse_projection(coarse_input)) * coarse_valid[:, None]
        coarse_features = self.coarse_conv(coarse_features.transpose(0, 1).unsqueeze(0))
        coarse_features = F.gelu(coarse_features.squeeze(0).transpose(0, 1)) * coarse_valid[:, None]
        coarse_residual = self.coarse_output(coarse_features).squeeze(-1) * coarse_valid
        coarse_logits = coarse_base + coarse_residual
        repeated_residual = coarse_residual.repeat_interleave(self.block_size) * valid
        result.update(coarse_logits=coarse_logits, coarse_residual=coarse_residual)
        if self.attention_enabled:
            context, attention, indices, allowed = self._local_attention(fine_features, coarse_features, valid, coarse_valid)
            attention_residual = self.attention_output(context).squeeze(-1) * valid
            result.update(attention_weights=attention, attention_indices=indices, attention_mask=allowed)
        else:
            attention_residual = torch.zeros_like(base)
        fused_coarse_residual = repeated_residual + attention_residual
        if self.gate_enabled:
            fine_probability = torch.sigmoid(fine_logits)
            coarse_probability = torch.sigmoid(coarse_base.repeat_interleave(self.block_size) + fused_coarse_residual)
            gate_inputs = self._gate_inputs(fine_probability, coarse_probability, normalized, valid, boundary_logits)
            gate = torch.sigmoid(self.gate_network(gate_inputs).squeeze(-1)) * valid
            result['gate_inputs'] = gate_inputs
        else:
            gate = torch.full_like(base, .5) * valid
        result.update(final_logits=base + (1 - gate) * fine_residual + gate * fused_coarse_residual,
                      gate=gate, attention_residual=attention_residual,
                      fused_coarse_residual=fused_coarse_residual)
        return result


def make_adapter(options, seed, device):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        adapter = AblationAdapter(options)
    return adapter.to(device)


def training_loss(final, outputs, batch, device, weights):
    main = objective(final, batch, device)
    auxiliary, boundary, consistency = (main.new_zeros(()) for _ in range(3))
    if weights['coarse_aux']:
        all_coarse = torch.cat([r['coarse_logits'] for r in outputs])
        targets = torch.cat([batch['labels'][i, :len(r['coarse_logits'])] for i, r in enumerate(outputs)]).to(device)
        duration = torch.cat([batch['weights'][i, :len(r['coarse_logits'])] for i, r in enumerate(outputs)]).to(device)
        auxiliary = weighted_bce(all_coarse, targets, duration)
    if weights['boundary'] or weights['consistency']:
        boundary_logits, boundary_labels, boundary_weights = [], [], []
        same_numerator, same_denominator = main.new_zeros(()), main.new_zeros(())
        for i, result in enumerate(outputs):
            n = len(result['final_logits'])
            y = batch['fine_labels'][i, :n].to(device)
            w = batch['fine_weights'][i, :n].to(device)
            if weights['boundary']:
                target, pair = boundary_targets(y, w)
                boundary_logits.append(result['boundary_logits'])
                boundary_labels.append(target)
                boundary_weights.append(pair)
            if weights['consistency']:
                inside = torch.minimum(w[1:], w[:-1]) * (y[1:] == y[:-1])
                probability = result['final_logits'].sigmoid()
                same_numerator = same_numerator + ((probability[1:] - probability[:-1]).square() * inside).sum()
                same_denominator = same_denominator + inside.sum()
        if weights['boundary']:
            boundary = weighted_bce(torch.cat(boundary_logits), torch.cat(boundary_labels), torch.cat(boundary_weights))
        if weights['consistency']:
            consistency = same_numerator / same_denominator.clamp_min(1e-8)
    total = main + weights['coarse_aux'] * auxiliary + weights['boundary'] * boundary + weights['consistency'] * consistency
    return total, dict(main=float(main.detach()), coarse_aux=float(auxiliary.detach()),
                       boundary=float(boundary.detach()), consistency=float(consistency.detach()))
