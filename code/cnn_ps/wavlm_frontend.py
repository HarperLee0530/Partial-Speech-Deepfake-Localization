"""Frozen WavLM last-layer features, aligned to PS 20 ms cells.

The convolutional encoder has stride 320 and receptive field 400 samples.
40 samples on each side put its centres at 159.5 + 320*t samples.
This is an offline contextual representation, not an 80-bin frequency image.
"""
from pathlib import Path
import argparse
import time
import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F
from transformers import WavLMModel
from .augmentation import transform
from .dual_features import FeatureData
from .io import read_json, save_json, sha256

REVISION = '4c66d4806a428f2e922ccfa1a962776e232d487b'
SCHEMA = 'ps_frozen_wavlm_last_v1'

class FrozenWavLM(torch.nn.Module):
    def __init__(self, folder, device='cuda'):
        super().__init__()
        prep = read_json(Path(folder)/'preprocessor_config.json')
        if prep['do_normalize'] or prep['sampling_rate'] != 16000:
            raise ValueError('Unexpected official waveform preprocessing')
        self.encoder, info = WavLMModel.from_pretrained(
            str(folder), local_files_only=True, output_loading_info=True)
        if any(info.values()):
            raise ValueError(f'Incomplete WavLM checkpoint: {info}')
        jump, receptive = 1, 1
        for k, s in zip(self.encoder.config.conv_kernel, self.encoder.config.conv_stride):
            receptive += (k-1)*jump
            jump *= s
        if (jump, receptive, self.encoder.config.hidden_size) != (320, 400, 768):
            raise ValueError('Unexpected feature grid')
        self.encoder.to(device).eval().requires_grad_(False)
        self.device = device

    @torch.inference_mode()
    def forward(self, wave, storage_frames):
        wave = torch.as_tensor(wave, dtype=torch.float32, device=self.device)
        if (wave.ndim != 1 or not wave.numel() or not torch.isfinite(wave).all()
                or storage_frames < 1 or len(wave) > storage_frames*320):
            raise ValueError('Invalid waveform or storage length')
        # Single full recording; no cross-record attention or batch padding.
        wave = F.pad(wave, (40, storage_frames*320-len(wave)+40))
        self.encoder.eval()
        feature = self.encoder(wave[None]).last_hidden_state[0]
        if feature.shape != (storage_frames, 768) or not torch.isfinite(feature).all():
            raise RuntimeError('Invalid WavLM features')
        return feature

def verify_frozen_plan(frozen_plan):
    """Revalidate the exact completed single-arm or paired plan before eval access."""
    from .train_wavlm import freeze_eval_plan, SCHEMA as TRAIN_SCHEMA
    path = Path(frozen_plan)
    if path.name != 'frozen_eval_plan.json':
        raise ValueError('Expected the canonical frozen_eval_plan.json')
    plan = read_json(path)
    if plan.get('schema') != TRAIN_SCHEMA or not isinstance(plan.get('arms'), dict):
        raise ValueError('Invalid frozen WavLM training plan')
    return freeze_eval_plan(path.parent, list(plan['arms']))


def build_cache(reference, model_folder, output, device='cuda', limit=None, frozen_plan=None):
    reference, model_folder, output = map(Path, (reference, model_folder, output))
    if read_json(reference/'complete.json')['split']=='eval':
        if frozen_plan is None:
            raise ValueError('Eval requires a verified frozen training plan')
        verify_frozen_plan(frozen_plan)
    data = FeatureData(reference, verify_hashes=True)
    if output.exists():
        raise FileExistsError(output)
    if limit is not None and limit < 1:
        raise ValueError('Positive probe limit required')
    count = len(data) if limit is None else min(limit, len(data))
    output.mkdir(parents=True)
    model = FrozenWavLM(model_folder, device)
    protocol = dict(schema=SCHEMA, split=data.split, model_id='microsoft/wavlm-base-plus',
        revision=REVISION, model_sha256=sha256(model_folder/'pytorch_model.bin'),
        reference=str(reference), reference_sha256=sha256(reference/'complete.json'),
        feature_dim=768, layer='last_hidden_state', frozen=True, dtype='float32',
        channels=data.channels, recordings=count, probe_only=limit is not None,
        normalization='none, official preprocessor_config',
        grid='stride320, receptive400, sidepad40; full recording padded to official storage grid',
        source_sha256=sha256(__file__), device=device, test_read=data.split=='eval')
    save_json(output/'protocol.json', protocol)
    total = int(data.offsets[count])
    array = np.lib.format.open_memmap(output/'features.npy', mode='w+', dtype='float32',
                                    shape=(data.channels,total,768))
    start, last = time.monotonic(), time.monotonic()
    for channel in range(data.channels):
        for i in range(count):
            row = data.records[i]
            wave, sr = sf.read(row['audio'], dtype='float32')
            if sr != 16000 or wave.ndim != 1 or len(wave)/sr != data.durations[i]:
                raise ValueError('Audio identity/grid mismatch')
            a,b = map(int,data.offsets[i:i+2])
            changed = transform(wave,row['uid'],channel)
            feature = model(changed,b-a).cpu().numpy()
            array[channel,a:b] = feature
            if time.monotonic()-last > 20:
                progress = dict(channel=channel, recordings=i+1, per_channel=count,
                    elapsed_seconds=time.monotonic()-start)
                save_json(output/'progress.json',progress)
                print(progress,flush=True)
                last=time.monotonic()
        array.flush()
    complete = dict(protocol,status='complete',elapsed_seconds=time.monotonic()-start,
        features_sha256=sha256(output/'features.npy'),shape=list(array.shape),
        source_unchanged=sha256(__file__)==protocol['source_sha256'],
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else None)
    save_json(output/'complete.json',complete)
    print(complete,flush=True)
    return complete

class WavLMData(FeatureData):
    def __init__(self, reference, cache, **kwargs):
        super().__init__(reference, **kwargs)
        self.ssl_metadata=read_json(Path(cache)/'complete.json')
        m=self.ssl_metadata
        if (m['schema']!=SCHEMA or m['status']!='complete' or m['probe_only']
            or m['reference_sha256']!=sha256(self.folder/'complete.json')
            or m['recordings']!=len(self) or m['channels']!=self.channels
            or not m['source_unchanged']):
            raise ValueError('Invalid WavLM cache provenance')
        if sha256(Path(cache)/'features.npy')!=m['features_sha256']:
            raise ValueError('WavLM feature hash mismatch')
        self.ssl=np.load(Path(cache)/'features.npy',mmap_mode='r')
        if self.ssl.shape!=(self.channels,int(self.offsets[-1]),768):
            raise ValueError('WavLM grid mismatch')
    def get_record(self,index,channel=0):
        row=super().get_record(index,channel)
        a,b=self.offsets[index:index+2]
        row['features']=torch.from_numpy(np.array(self.ssl[channel,a:b],copy=True)).unsqueeze(0)
        return row

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reference',required=True)
    p.add_argument('--model-folder',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    p.add_argument('--limit',type=int)
    p.add_argument('--frozen-plan')
    args=p.parse_args()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    build_cache(**vars(args))
