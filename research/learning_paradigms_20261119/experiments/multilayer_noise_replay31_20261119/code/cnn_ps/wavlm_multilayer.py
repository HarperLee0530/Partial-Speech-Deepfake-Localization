"""Frozen intermediate WavLM states and a minimal convex layer mixture.

Hidden state 0 is the transformer input; states 6/9/12 are transformer outputs.
Only the three global mixture logits are added to the existing backend.
"""
from pathlib import Path
import argparse
import time
import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from .augmentation import transform
from .dual_features import FeatureData
from .io import read_json, save_json, sha256
from .wavlm_backend import ProjectedCNN
from .wavlm_frontend import FrozenWavLM, REVISION

LAYERS = (6, 9, 12)
ARMS = ('layer12', 'fusion6_9_12')
SCHEMA = 'ps_frozen_wavlm_multilayer_cache_v1'


class FrozenMultiWavLM(FrozenWavLM):
    @torch.inference_mode()
    def forward(self, wave, storage_frames):
        wave = torch.as_tensor(wave, dtype=torch.float32, device=self.device)
        if (wave.ndim != 1 or not wave.numel() or not torch.isfinite(wave).all()
                or storage_frames < 1 or len(wave) > storage_frames * 320):
            raise ValueError('Invalid waveform or storage length')
        wave = F.pad(wave, (40, storage_frames * 320 - len(wave) + 40))
        self.encoder.eval()
        output = self.encoder(wave[None], output_hidden_states=True)
        if len(output.hidden_states) != 13:
            raise ValueError('Expected twelve transformer layers plus input')
        if not torch.equal(output.hidden_states[12], output.last_hidden_state):
            raise AssertionError('Layer 12 differs from original last_hidden_state')
        states = torch.stack([output.hidden_states[i][0] for i in LAYERS], dim=1)
        if states.shape != (storage_frames, len(LAYERS), 768) or not torch.isfinite(states).all():
            raise RuntimeError('Invalid intermediate features')
        return states


class LayerFusionCNN(ProjectedCNN):
    def __init__(self, config, arm, seed):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(config, 'wavlm_frozen', seed)
        self.layer_arm = arm
        # Constant initialization consumes no RNG and preserves CNN/dropout seeds.
        if arm == 'fusion6_9_12':
            self.layer_logits = torch.nn.Parameter(torch.zeros(len(LAYERS)))

    def layer_weights(self):
        return (self.layer_logits.softmax(0) if self.layer_arm == 'fusion6_9_12'
                else next(self.parameters()).new_tensor([0., 0., 1.]))

    def forward(self, features, lengths, return_features=False):
        expected = 768 if self.layer_arm == 'layer12' else 768 * len(LAYERS)
        if features.shape[-1] != expected:
            raise ValueError('Unexpected selected layer feature dimensions')
        if self.layer_arm == 'fusion6_9_12':
            states = features.reshape(*features.shape[:-1], len(LAYERS), 768)
            features = (states * self.layer_weights()[:, None]).sum(dim=-2)
        return super().forward(features, lengths, return_features=return_features)


def build_cache(reference, model_folder, output, device='cuda', frozen_plan=None):
    reference, model_folder, output = map(Path, (reference, model_folder, output))
    data = FeatureData(reference, verify_hashes=True)
    if data.split == 'eval':
        if frozen_plan is None:
            raise ValueError('Eval features require the completed frozen training plan')
        from .train_wavlm_multilayer import freeze_eval_plan
        freeze_eval_plan(Path(frozen_plan).parent, list(ARMS))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    model = FrozenMultiWavLM(model_folder, device)
    protocol = dict(schema=SCHEMA, split=data.split, layers=list(LAYERS), frozen=True,
        model_id='microsoft/wavlm-base-plus', revision=REVISION,
        model_sha256=sha256(model_folder/'pytorch_model.bin'),
        reference=str(reference), reference_sha256=sha256(reference/'complete.json'),
        recordings=len(data), channels=data.channels, dtype='float32',
        source_sha256=sha256(__file__), test_read=data.split == 'eval',
        normalization='none; unchanged official preprocessor',
        grid='stride320/receptive400/sidepad40; unchanged full recording storage grid')
    save_json(output/'protocol.json', protocol)
    shape = (data.channels, int(data.offsets[-1]), len(LAYERS), 768)
    array = np.lib.format.open_memmap(output/'features.npy', mode='w+', dtype='float32', shape=shape)
    start = last = time.monotonic()
    for channel in range(data.channels):
        for i, row in enumerate(data.records):
            wave, sr = sf.read(row['audio'], dtype='float32')
            if sr != 16000 or wave.ndim != 1 or len(wave)/sr != data.durations[i]:
                raise ValueError('Audio identity/grid mismatch')
            a, b = map(int, data.offsets[i:i+2])
            array[channel, a:b] = model(transform(wave, row['uid'], channel), b-a).cpu().numpy()
            if time.monotonic() - last >= 20:
                progress = dict(split=data.split, channel=channel, recordings=i+1,
                    per_channel=len(data), elapsed_seconds=time.monotonic()-start)
                save_json(output/'progress.json', progress)
                print(progress, flush=True)
                last = time.monotonic()
        array.flush()
    complete = dict(protocol, status='complete', shape=list(shape),
        features_sha256=sha256(output/'features.npy'), elapsed_seconds=time.monotonic()-start,
        source_unchanged=sha256(__file__) == protocol['source_sha256'])
    save_json(output/'complete.json', complete)
    return complete


class MultiLayerData(FeatureData):
    def __init__(self, reference, cache, arm='fusion6_9_12', **kwargs):
        super().__init__(reference, **kwargs)
        if arm not in ARMS:
            raise ValueError(arm)
        self.layer_arm = arm
        self.ssl_metadata = m = read_json(Path(cache)/'complete.json')
        if (m['schema'] != SCHEMA or m['status'] != 'complete' or m['layers'] != list(LAYERS)
                or m['reference_sha256'] != sha256(self.folder/'complete.json')
                or m['recordings'] != len(self) or m['channels'] != self.channels
                or not m['source_unchanged'] or not m['frozen']):
            raise ValueError('Invalid multilayer cache provenance')
        path = Path(cache)/'features.npy'
        if sha256(path) != m['features_sha256']:
            raise ValueError('Multilayer feature hash mismatch')
        self.ssl = np.load(path, mmap_mode='r', allow_pickle=False)
        if self.ssl.shape != (self.channels, int(self.offsets[-1]), len(LAYERS), 768):
            raise ValueError('Multilayer feature grid mismatch')

    def get_record(self, index, channel=0):
        row = super().get_record(index, channel)
        a, b = self.offsets[index:index+2]
        feature = self.ssl[channel, a:b]
        if self.layer_arm == 'layer12':
            feature = feature[:, -1, :]
        row['features'] = torch.from_numpy(np.array(feature, copy=True).reshape(b-a, -1)).unsqueeze(0)
        return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--model-folder', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--frozen-plan')
    args = parser.parse_args()
    from .train_wavlm import setup
    setup(666, 2)
    print(build_cache(**vars(args)), flush=True)
