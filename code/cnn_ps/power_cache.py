"""Immutable STFT-power caches for fixed vs learnable filterbank experiments.

Waveform channels, storage padding and actual-duration labels are bound to the
existing LogMel cache. No learned features, statistics or predictions are cached.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from .augmentation import transform, waveform_sha256, NOISE_SEED
from .backbone import LogMel, ModelConfig
from .dual_features import FeatureData, frontend_config
from .finetune import fine_collate
from .io import load_manifest, read_json, save_json, sha256
from .learnable_filterbank import LearnableFilterbank, PowerSpectrogram


SCHEMA = 'cnn_stft_power_cache_v1'
ALIGNMENT_KEYS = ('split', 'recordings', 'channels', 'uids', 'offsets', 'lengths20',
                  'lengths160', 'storage_lengths20', 'durations', 'uid_order_sha256')


def definition():
    return dict(sample_rate=16000, hop_length=320, n_fft=512, win_length=400,
                frequency_bins=257, power=2., window='periodic Hann', center=False,
                side_padding=96, layout='channels, concatenated stored20 frames, frequency',
                storage='Full transformed waveform padded to official N160*2560 before STFT',
                tail='Stored tail power retained; actual-duration masks define supervision',
                normalization='None; linear float32 power before filterbank and logarithm')


@torch.inference_mode()
def build_power_cache(frontend_cache, output, threads=1):
    frontend_cache, output = Path(frontend_cache).resolve(), Path(output).resolve()
    if threads <= 0:
        raise ValueError('threads must be positive')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError('Power cache output must be empty; no overwrite or implicit resume')
    torch.set_num_threads(threads)
    dataset = FeatureData(frontend_cache, verify_hashes=True)
    source = dataset.metadata
    if source['frontend_config'] != frontend_config(ModelConfig()) or source.get('noise_seed') != NOISE_SEED:
        raise ValueError('Power cache requires the original 16k/20ms frontend and channel transforms')
    manifest = Path(source['manifest'])
    if sha256(manifest) != source['manifest_sha256'] or load_manifest(manifest, expected_split=dataset.split) != dataset.records:
        raise ValueError('Original manifest changed')
    source_sha = sha256(frontend_cache/'complete.json')
    code_hashes = {name:sha256(Path(__file__).with_name(name)) for name in
                  ('power_cache.py', 'learnable_filterbank.py', 'backbone.py', 'augmentation.py')}
    protocol = dict(schema=SCHEMA, **{key:source[key] for key in ALIGNMENT_KEYS},
                    frontend_cache=str(frontend_cache), frontend_complete_sha256=source_sha,
                    manifest=str(manifest), manifest_sha256=source['manifest_sha256'],
                    feature_definition=definition(), source_sha256=code_hashes,
                    noise_seed=NOISE_SEED, augmentation=source['augmentation'],
                    trained=False, cnn_inference=False, normalization_fitted=False,
                    threshold_selection=False, eval_access=dataset.split=='eval',
                    created_utc=datetime.now(timezone.utc).isoformat())
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/'protocol.json', protocol)
    shape = (dataset.channels, int(dataset.offsets[-1]), 257)
    array = np.lib.format.open_memmap(output/'power.npy', mode='w+', dtype='float32', shape=shape)
    config = ModelConfig(**source['frontend_config'])
    original, extractor = LogMel(config).eval(), PowerSpectrogram(config).eval()
    fixed = LearnableFilterbank(config, learnable=False).eval()
    waveform_hashes, channel_hashes = [], []
    completed = exact_projection = 0
    max_error = 0.
    started = last = time.monotonic()
    try:
        for index, record in enumerate(dataset.records):
            wave, sr = sf.read(record['audio'], dtype='float32')
            if sr != 16000 or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
                raise ValueError(f"{record['uid']}: expected finite nonempty 16kHz mono audio")
            a, b = map(int, dataset.offsets[index:index+2])
            actual = (len(wave)+319)//320
            if len(wave)/sr != dataset.durations[index] or actual != dataset.lengths20[index] or len(wave) > (b-a)*320:
                raise ValueError(f"{record['uid']}: waveform/grid mismatch")
            expected_weights = np.zeros(b-a, dtype=np.float32)
            expected_weights[:actual] = np.maximum(0., np.minimum((np.arange(actual)+1)*.02,len(wave)/sr)-np.arange(actual)*.02)
            if not np.array_equal(expected_weights, dataset.arrays['weights20'][a:b]):
                raise ValueError('Actual-duration labels or weights changed')
            waveform_hashes.append(waveform_sha256(wave))
            view_hashes = []
            for channel in range(dataset.channels):
                changed = transform(wave, record['uid'], channel)
                view_hashes.append(waveform_sha256(changed))
                padded = F.pad(torch.from_numpy(changed), (0,(b-a)*320-len(wave)))
                reference = original(padded)[0].numpy()
                if not np.array_equal(reference, dataset.arrays['features'][channel,a:b]):
                    raise ValueError(f"{record['uid']}/{channel}: original LogMel cache cannot be reproduced")
                power = extractor(padded)[0].contiguous()
                if power.shape != (b-a,257) or power.dtype != torch.float32 or not torch.isfinite(power).all() or (power<0).any():
                    raise ValueError('Invalid STFT power')
                regenerated = fixed(power).numpy()
                error = float(np.max(np.abs(regenerated.astype(np.float64)-reference)))
                max_error = max(max_error,error)
                exact_projection += int(np.array_equal(regenerated,reference))
                if not np.allclose(regenerated,reference,atol=2e-5,rtol=2e-6):
                    raise ValueError(f"{record['uid']}/{channel}: initial fixed projection differs, max {error}")
                array[channel,a:b] = power.numpy()
                completed += 1
            channel_hashes.append(view_hashes)
            if time.monotonic()-last >= 20:
                progress = dict(split=dataset.split, recordings=index+1, recording_views=completed,
                                total_recordings=len(dataset), elapsed_seconds=time.monotonic()-started)
                save_json(output/'progress.json',progress)
                print(progress,flush=True)
                last=time.monotonic()
        array.flush()
        if sha256(frontend_cache/'complete.json') != source_sha or sha256(manifest) != source['manifest_sha256']:
            raise ValueError('Input metadata changed during extraction')
        if any(sha256(Path(__file__).with_name(name)) != expected for name,expected in code_hashes.items()):
            raise ValueError('Source code changed during extraction')
        complete = dict(protocol,status='complete', completed_recording_views=completed,
                        logmel_exact_matches=completed, fixed_projection_exact_matches=exact_projection,
                        fixed_projection_max_absolute_error=max_error,
                        fixed_projection_tolerance=dict(atol=2e-5,rtol=2e-6),
                        waveform_sha256=waveform_hashes,channel_waveform_sha256=channel_hashes,
                        array=dict(file='power.npy',shape=list(shape),dtype='float32',sha256=sha256(output/'power.npy')),
                        elapsed_seconds=time.monotonic()-started)
        save_json(output/'complete.json',complete)
        save_json(output/'progress.json',dict(status='complete',split=dataset.split,recording_views=completed))
        return complete
    except Exception as exc:
        save_json(output/'failure.json',dict(status='failed',recording_views=completed,
                   error_type=type(exc).__name__,error=str(exc)))
        raise
    finally:
        array.flush()


class PowerFeatureData(FeatureData):
    """Add immutable power to the existing labels, channel order and grid."""
    def __init__(self,frontend_cache,power_cache,split=None,seed=666,config=None,verify_hashes=False):
        super().__init__(frontend_cache,split=split,seed=seed,config=config,verify_hashes=verify_hashes)
        self.power_folder=Path(power_cache)
        self.power_metadata=read_json(self.power_folder/'complete.json')
        m=self.power_metadata
        if (m.get('schema')!=SCHEMA or m.get('status')!='complete'
                or m.get('frontend_complete_sha256')!=sha256(self.folder/'complete.json')
                or m.get('feature_definition')!=definition()
                or any(m.get(key)!=self.metadata[key] for key in ALIGNMENT_KEYS)
                or m.get('completed_recording_views')!=self.channels*len(self)):
            raise ValueError('Power cache differs from the original waveform/grid identity')
        self.power_array=np.load(self.power_folder/'power.npy',mmap_mode='r',allow_pickle=False)
        shape=(self.channels,int(self.offsets[-1]),257)
        info=m['array']
        if (self.power_array.shape!=shape or self.power_array.dtype!=np.float32
                or info.get('file')!='power.npy' or info['shape']!=list(shape) or info['dtype']!='float32'):
            raise ValueError('Invalid power cache shape/type')
        if verify_hashes and sha256(self.power_folder/'power.npy')!=info['sha256']:
            raise ValueError('Power cache array changed')

    def get_record(self,index,channel=0):
        row=super().get_record(index,channel)
        a,b=map(int,self.offsets[index:index+2])
        row['power_features']=torch.from_numpy(np.array(self.power_array[channel,a:b],dtype=np.float32,copy=True)).unsqueeze(0)
        return row


def power_collate(rows):
    batch=fine_collate(rows)
    maximum=int(batch['lengths'].max())
    tensors=[]
    for row in rows:
        power=row['power_features']
        if power.shape!=(1,len(row['fine_weights']),257) or not torch.isfinite(power).all() or (power<0).any():
            raise ValueError('Expected finite nonnegative power[1,storage20,257]')
        tensors.append(F.pad(power,(0,0,0,maximum-power.shape[1])))
    batch['power_features']=torch.stack(tensors)
    return batch


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frontend-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=1)
    result=build_power_cache(**vars(parser.parse_args(argv)))
    print({key:result[key] for key in ('status','split','recordings','channels','fixed_projection_max_absolute_error')},flush=True)


if __name__=='__main__':
    main()
