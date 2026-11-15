"""Single-record offline prediction for the paired boundary/energy5 experiment.

Both arms load their entire CNN, adapter, embedded train-clean energy statistics
and dev-selected threshold. No labels, feature cache or normalization file is
needed. The default candidate checkpoint may not yet have been published; use
--checkpoint to inspect an existing completed dev checkpoint in that case.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .backbone import CNNBiGRU, LogMel, ModelConfig
from .energy_cache import NORMALIZATION_SCHEMA
from .energy_features import FEATURE_NAMES, extract
from .io import save_json, sha256
from .predict_main import waveform_batch, decode_events
from .train_energy import ARMS, BASE_OPTIONS, SCHEMA, forward_batch, make_energy_adapter
from .train_four import model_config


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / 'weights' / 'boundary_energy5_seed666.pt'


def _json_values(value):
    # torch checkpoints retain tuples; the original JSON protocol uses lists.
    return json.loads(json.dumps(value, allow_nan=False))


def load_energy_checkpoint(checkpoint, device='cpu'):
    """Load only the declared paired architecture and preserve its dev cutoff."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError('Energy checkpoint is not available; pass --checkpoint for an existing dev-selected .pt file')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    required = ('schema', 'protocol', 'model_config', 'adapter_options', 'energy_enabled',
                'backbone_state_dict', 'adapter_state_dict', 'energy_normalization',
                'threshold', 'dev_metrics', 'epoch')
    if not isinstance(payload, dict) or any(key not in payload for key in required):
        raise ValueError('Expected a complete ps_boundary_energy5_v1 checkpoint')
    protocol, dev, normalization = payload['protocol'], payload['dev_metrics'], payload['energy_normalization']
    if not isinstance(protocol, dict) or not isinstance(dev, dict) or not isinstance(normalization, dict):
        raise ValueError('Missing checkpoint protocol, dev metrics or embedded normalization')
    arm = protocol.get('arm')
    if (payload['schema'] != SCHEMA or protocol.get('schema') != SCHEMA or arm not in ARMS
            or type(payload['energy_enabled']) is not bool or type(protocol.get('energy_enabled')) is not bool
            or payload['energy_enabled'] is not ARMS[arm] or protocol['energy_enabled'] is not ARMS[arm]
            or payload['adapter_options'] != BASE_OPTIONS or protocol.get('adapter_options') != BASE_OPTIONS
            or not isinstance(payload['adapter_state_dict'], dict)):
        raise ValueError('Checkpoint arm, energy flag or adapter architecture is inconsistent')
    if (normalization.get('schema') != NORMALIZATION_SCHEMA
            or normalization.get('fit_split') != 'train' or normalization.get('fit_channel') != 0
            or normalization.get('feature_names') != list(FEATURE_NAMES)
            or normalization.get('padded_frames_in_statistics') is not False
            or normalization.get('valid_train_frames', 0) <= 0
            or _json_values(protocol.get('energy_normalization')) != _json_values(normalization)):
        raise ValueError('Expected matching embedded normalization fitted on actual clean training frames')
    mean, std = np.asarray(normalization.get('mean'), dtype=np.float64), np.asarray(normalization.get('std'), dtype=np.float64)
    if (mean.shape != (5,) or std.shape != (5,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or (std < 1e-4).any()):
        raise ValueError('Embedded energy mean/std must contain five finite values; std>=1e-4')
    expected_config = model_config().to_dict()
    if (_json_values(payload['model_config']) != _json_values(expected_config)
            or _json_values(protocol.get('model_config')) != _json_values(expected_config)):
        raise ValueError('Checkpoint CNN/frontend configuration differs from this experiment')
    threshold = float(payload['threshold'])
    if (not np.isfinite(threshold) or threshold != dev.get('threshold')
            or not np.isfinite(dev.get('real_fpr_percent', np.nan))
            or not 0 <= dev['real_fpr_percent'] <= 5 + 1e-10):
        raise ValueError('Missing or inconsistent dev-selected threshold/FPR')
    config = ModelConfig(**payload['model_config'])
    model = CNNBiGRU(config)
    model.load_state_dict(payload['backbone_state_dict'], strict=True)
    model.to(device).eval().requires_grad_(False)
    adapter = make_energy_adapter(payload['energy_enabled'], normalization, protocol['seed'], device)
    adapter.load_state_dict(payload['adapter_state_dict'], strict=True)
    if (not torch.equal(adapter.energy_mean.cpu(), torch.as_tensor(mean, dtype=torch.float32))
            or not torch.equal(adapter.energy_std.cpu(), torch.as_tensor(std, dtype=torch.float32))):
        raise ValueError('Adapter normalization buffers disagree with embedded statistics')
    if not payload['energy_enabled'] and torch.count_nonzero(adapter.energy_projection.weight):
        raise ValueError('Control checkpoint must retain its disabled zero energy projection')
    adapter.eval().requires_grad_(False)
    return model, adapter, LogMel(config).eval(), payload, threshold


def energy_waveform_batch(wave, sample_rate, frontend):
    """Use the acoustic grid, extracting energy before storage-tail zero padding."""
    batch, duration, actual20 = waveform_batch(wave, sample_rate, frontend)
    energy = extract(wave, sample_rate)
    if energy.shape != (actual20, 5):
        raise ValueError('Raw energy and actual acoustic frames disagree')
    storage20 = int(batch['lengths'][0])
    batch['energy_features'] = torch.zeros((1, storage20, 5), dtype=torch.float32)
    batch['energy_features'][0, :actual20] = torch.from_numpy(energy)
    return batch, duration, actual20


@torch.inference_mode()
def predict(audio, output, checkpoint=DEFAULT_CHECKPOINT, device='cpu', threads=4):
    if threads <= 0 or device not in ('cpu', 'cuda'):
        raise ValueError('threads must be positive and device must be cpu or cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Prediction output already exists: {output}')
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    wave, sample_rate = sf.read(audio, dtype='float32')
    model, adapter, frontend, payload, threshold = load_energy_checkpoint(checkpoint, device)
    batch, duration, actual20 = energy_waveform_batch(wave, sample_rate, frontend)
    final, _ = forward_batch(model, adapter, batch, device)
    scores20 = final[0, :actual20].float().cpu().numpy().copy()
    detected = decode_events(scores20, threshold, duration)
    result = dict(audio=str(Path(audio).resolve()), duration_seconds=duration, sample_rate=sample_rate,
                  checkpoint_sha256=sha256(checkpoint), checkpoint_schema=SCHEMA,
                  arm=payload['protocol']['arm'], energy_enabled=payload['energy_enabled'],
                  energy_feature_names=list(FEATURE_NAMES), energy_normalization_source='Embedded checkpoint statistics fitted on train clean actual frames',
                  selected_epoch=int(payload['epoch']), threshold=threshold,
                  threshold_source='Checkpoint dev genuine-duration FPR<=5%, selected by dev event F1',
                  score_kind='native20 fake-positive raw logits, not calibrated probabilities',
                  threshold_decision_rule='float64(score) >= float64(threshold)',
                  interval_resolution_seconds=.02, native20_actual_frames=actual20,
                  stored20_frames=int(batch['lengths'][0]), scores20=scores20.tolist(),
                  predicted_fake_intervals_seconds=detected.tolist(),
                  postprocessing='Consecutive positive frames only; no smoothing, gap filling or short-event removal',
                  inference_mode='Offline full-recording CNN-BiGRU plus complete boundary/energy adapter',
                  precision_note='Regenerated scores may differ slightly across devices/library versions; archived scores define the original comparison.')
    save_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    result = predict(**vars(args))
    print(dict(output=str(args.output), arm=result['arm'], actual20_frames=result['native20_actual_frames'],
               events=len(result['predicted_fake_intervals_seconds']), threshold=result['threshold']))


if __name__ == '__main__':
    main()
