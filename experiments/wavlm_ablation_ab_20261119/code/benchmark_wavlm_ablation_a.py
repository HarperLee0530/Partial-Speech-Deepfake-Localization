"""Sequential, raw-audio inference verification and latency for completed A arms."""
from pathlib import Path
import time
import numpy as np
import soundfile as sf
import torch
from cnn_ps.backbone import ModelConfig
from cnn_ps.io import read_json, save_json, sha256
from cnn_ps.predict_mel import waveform_batch
from cnn_ps.train_dual import forward_batch
from cnn_ps.train_wavlm_ablation_a import setup
from cnn_ps.wavlm_ablation_a import ARMS, AblationCNN, make_adapter
from cnn_ps.wavlm_multilayer import FrozenMultiWavLM

CODE = Path(__file__).resolve().parent
ROOT = CODE.parents[1]/'handoff_audit/ablation_a_20261118'
REFERENCE = CODE.parents[1]/'handoff_audit/multilayer_20261117/full'


@torch.inference_mode()
def benchmark():
    comparison = read_json(ROOT/'all_comparison.json')
    if comparison['status'] != 'complete':
        raise ValueError('Complete training and evaluation must precede benchmarking')
    setup(666, 4)
    device = 'cuda'
    metadata = read_json(REFERENCE/'reference/dev/complete.json')
    # Fixed choice independent of scores. Decode outside the timed region.
    indices = np.random.default_rng(20261118).choice(len(metadata['records']), 32, replace=False).tolist()
    if 0 not in indices:
        indices[0] = 0
    indices.sort()
    samples = [(i, *sf.read(metadata['records'][i]['audio'], dtype='float32')) for i in indices]
    encoder = FrozenMultiWavLM(CODE.parent/'data/models/wavlm-base-plus', device)
    if any(p.requires_grad for p in encoder.parameters()):
        raise AssertionError('Encoder must remain frozen')
    timings, checks = {}, {}
    for arm, options in ARMS.items():
        folder = ROOT/'training'/arm
        if sha256(folder/'best.pt') != comparison['frozen']['arms'][arm]['complete']['checkpoint_sha256']:
            raise ValueError('Checkpoint changed after freezing')
        payload = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        if payload['threshold'] != read_json(folder/'dev_metrics.json')['threshold']:
            raise ValueError('Checkpoint threshold differs from frozen dev calibration')
        config = dict(payload['model_config'])
        config['channels'] = tuple(config['channels'])
        model = AblationCNN(ModelConfig(**config), arm, 666).to(device)
        adapter = make_adapter(options, 666, device)
        model.load_state_dict(payload['backbone_state_dict'], strict=True)
        adapter.load_state_dict(payload['adapter_state_dict'], strict=True)
        active = sum(p.numel() for module in (model, adapter) for p in module.parameters() if p.requires_grad)
        model.eval().requires_grad_(False)
        adapter.eval().requires_grad_(False)

        def frontend(wave):
            states = encoder(wave, len(wave)//320)
            mode = options['feature_mode']
            selected = (states[:, (6, 9, 12).index(int(mode[5:])), :]
                        if mode.startswith('layer') else states.flatten(-2))
            return selected.unsqueeze(0).cpu()

        def predict(wave, sr):
            batch, duration, frames = waveform_batch(wave, sr, frontend)
            final, _ = forward_batch(model, adapter, batch, device)
            return final[0, :frames].float().cpu().numpy().copy(), duration

        for _, wave, sr in samples[:3]:
            predict(wave, sr)
        times, durations, maximum, unequal, changed = [], [], 0., 0, 0
        with np.load(folder/'dev_scores20.npz', allow_pickle=False) as scores:
            archived = scores['scores20']
            offsets = np.asarray(metadata['offsets'], dtype=np.int64)
            np.testing.assert_array_equal(scores['uids'], np.asarray(metadata['uids']))
            if archived.size != offsets[-1]:
                raise ValueError('Archived score/grid mismatch')
            for index, wave, sr in samples:
                torch.cuda.synchronize()
                started = time.perf_counter()
                predicted, duration = predict(wave, sr)
                torch.cuda.synchronize()
                times.append(time.perf_counter()-started)
                durations.append(duration)
                old = archived[offsets[index]:offsets[index]+len(predicted)]
                delta = float(np.abs(predicted-old).max())
                maximum = max(maximum, delta)
                unequal += int(np.count_nonzero(predicted != old))
                threshold = np.float64(payload['threshold'])
                changed += int(np.count_nonzero((predicted.astype('float64') >= threshold) !=
                                               (old.astype('float64') >= threshold)))
        if maximum > 1e-4 or changed:
            raise AssertionError(f'Raw audio/cache predictions differ: {arm}, {maximum}, decisions={changed}')
        checks[arm] = dict(recordings=len(samples), max_absolute_score_difference=maximum,
                           unequal_score_frames=unequal, differing_decisions=changed,
                           threshold_exact=payload['threshold'] == read_json(folder/'dev_metrics.json')['threshold'])
        timings[arm] = dict(recordings=len(samples), mean_ms=1000*float(np.mean(times)),
            p50_ms=1000*float(np.median(times)), p95_ms=1000*float(np.percentile(times, 95)),
            real_time_factor=sum(times)/sum(durations), total_audio_seconds=sum(durations),
            active_trainable_backend_parameters=active, inactive_adapter_modules=adapter.disabled_modules,
            seconds_per_recording=times)
        print(arm, checks[arm], {k: timings[arm][k] for k in ('mean_ms', 'real_time_factor')}, flush=True)
        del model, adapter, payload
        torch.cuda.empty_cache()
    save_json(ROOT/'single_audio_verification.json', dict(status='passed', split='dev',
        sample_indices=indices, uids=[metadata['uids'][i] for i in indices], models=checks))
    save_json(ROOT/'inference_benchmark.json', dict(status='complete', device=torch.cuda.get_device_name(),
        threads=4, sample_seed=20261118, warmup_recordings=3, split='dev', sample_indices=indices,
        frozen_encoder_parameters=sum(p.numel() for p in encoder.parameters()), models=timings,
        scope='Sequential complete raw-waveform inference; includes encoder, CPU/GPU transfers and backend; excludes disk decoding and model loading',
        caveat='Single timing pass on 32 fixed dev recordings; all variants execute the entire frozen encoder, including layer6/9 variants; not encoder truncation'))


if __name__ == '__main__':
    benchmark()
