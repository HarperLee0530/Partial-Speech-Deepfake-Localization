"""Offline native20 prediction using the complete boundary main-method checkpoint.

No labels, feature cache or training-run directory are needed. The threshold is
read from the checkpoint's completed dev selection; inference never calibrates it.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from .backbone import CNNBiGRU, LogMel, ModelConfig
from .io import save_json, sha256
from .metrics160 import intervals
from .train_dual import forward_batch
from .train_four import ARMS, SCHEMA, make_adapter, model_config


DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[2] / 'weights' / 'cnn_boundary_ps_eventf1_seed666.pt'


def load_main_checkpoint(checkpoint, device='cpu'):
    """Load the CNN and adapter together, preserving the selected dev threshold."""
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    required = ('schema', 'protocol', 'model_config', 'adapter_options',
                'backbone_state_dict', 'adapter_state_dict', 'threshold', 'dev_metrics', 'epoch')
    if not isinstance(payload, dict) or any(key not in payload for key in required):
        raise ValueError('Expected a complete ps_four_random_cnn_v1 boundary checkpoint')
    protocol, dev = payload['protocol'], payload['dev_metrics']
    if (payload['schema'] != SCHEMA or not isinstance(protocol, dict)
            or protocol.get('arm') != 'boundary' or protocol.get('schema') != SCHEMA
            or payload['adapter_options'] != ARMS['boundary']
            or protocol.get('adapter_options') != ARMS['boundary']
            or not isinstance(dev, dict) or not isinstance(payload['adapter_state_dict'], dict)):
        raise ValueError('This entry accepts only the boundary main-method configuration')
    fields = dict(payload['model_config'])
    fields['channels'] = tuple(fields['channels'])
    config = ModelConfig(**fields)
    if config != model_config():
        raise ValueError('Main-method CNN/frontend configuration does not match')
    threshold = float(payload['threshold'])
    if (not np.isfinite(threshold) or threshold != dev.get('threshold')
            or not np.isfinite(dev.get('real_fpr_percent', np.nan))
            or not 0 <= dev['real_fpr_percent'] <= 5 + 1e-10):
        raise ValueError('Missing or inconsistent dev-selected threshold/FPR')
    model = CNNBiGRU(config)
    model.load_state_dict(payload['backbone_state_dict'], strict=True)
    model.to(device).eval().requires_grad_(False)
    adapter = make_adapter(payload['adapter_options'], protocol['seed'], device)
    adapter.load_state_dict(payload['adapter_state_dict'], strict=True)
    adapter.eval().requires_grad_(False)
    return model, adapter, LogMel(config).eval(), payload, threshold


def waveform_batch(wave, sample_rate, frontend):
    """Build the same padded grid and duration masks as clean FineData, without labels."""
    wave = np.asarray(wave, dtype=np.float32)
    if sample_rate != 16000 or wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
        raise ValueError('Expected finite, nonempty mono audio sampled at 16 kHz')
    duration = len(wave) / sample_rate
    blocks, actual20 = (len(wave) + 2559) // 2560, (len(wave) + 319) // 320
    stored20 = blocks * 8
    padded = F.pad(torch.from_numpy(wave.copy()), (0, blocks * 2560 - len(wave)))
    fine_weights = np.zeros(stored20, dtype=np.float32)
    starts = np.arange(actual20) * .02
    fine_weights[:actual20] = np.maximum(0., np.minimum((np.arange(actual20) + 1) * .02, duration) - starts)
    coarse_starts = np.arange(blocks) * .16
    coarse_weights = np.maximum(0., np.minimum((np.arange(blocks) + 1) * .16, duration) - coarse_starts).astype(np.float32)
    batch = dict(features=frontend(padded).unsqueeze(0), lengths=torch.tensor([stored20]),
                 fine_weights=torch.from_numpy(fine_weights).unsqueeze(0),
                 weights=torch.from_numpy(coarse_weights).unsqueeze(0))
    return batch, duration, actual20


def decode_events(scores20, threshold, duration):
    """Float64 comparison preserves a dev cut just above a tied float32 score."""
    scores = np.asarray(scores20, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError('Expected finite one-dimensional native20 logits and threshold')
    return intervals(scores >= np.float64(threshold), duration, step=.02)


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
    model, adapter, frontend, payload, threshold = load_main_checkpoint(checkpoint, device)
    batch, duration, actual20 = waveform_batch(wave, sample_rate, frontend)
    final, _ = forward_batch(model, adapter, batch, device)
    scores20 = final[0, :actual20].float().cpu().numpy().copy()
    detected = decode_events(scores20, threshold, duration)
    result = dict(audio=str(Path(audio).resolve()), duration_seconds=duration, sample_rate=sample_rate,
                  checkpoint_sha256=sha256(checkpoint), checkpoint_schema=SCHEMA, arm='boundary',
                  selected_epoch=int(payload['epoch']), threshold=threshold,
                  threshold_source='Checkpoint dev genuine-duration FPR<=5%, selected by dev event F1',
                  score_kind='native20 fake-positive raw logits, not calibrated probabilities',
                  threshold_decision_rule='float64(score) >= float64(threshold)',
                  interval_resolution_seconds=.02, native20_actual_frames=actual20,
                  stored20_frames=int(batch['lengths'][0]), scores20=scores20.tolist(),
                  predicted_fake_intervals_seconds=detected.tolist(),
                  postprocessing='Consecutive positive frames only; no smoothing, gap filling or short-event removal',
                  inference_mode='Offline full-recording CNN-BiGRU plus complete boundary adapter',
                  precision_note='Regenerated scores may differ slightly across devices/library versions; archived scores define the original comparison.')
    save_json(output, result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    result = predict(**vars(args))
    print(dict(output=str(args.output), actual20_frames=result['native20_actual_frames'],
               events=len(result['predicted_fake_intervals_seconds']), threshold=result['threshold']))


if __name__ == '__main__':
    main()
