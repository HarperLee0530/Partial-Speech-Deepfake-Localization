"""Score one mono 16kHz waveform with an original or fine-tuned CNN-BiGRU.

Outputs are raw fake-positive logits, not calibrated probabilities. The chosen
threshold applies to --resolution only; 20ms and 160ms reporting thresholds are
generally different. Raw feature regeneration may differ slightly from archived
scores across hardware, batch composition and library versions. Specify
--resolution 0.02 and the checkpoint's dev-calibrated threshold for native 20 ms
reporting; the default 0.16 resolution is retained for historical callers.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from .backbone import load_backbone
from .io import save_json, sha256
from .metrics160 import intervals


@torch.inference_mode()
def predict(audio, checkpoint, output, threshold=None, resolution=.16,
            device='cpu', threads=4):
    explicit_threshold = threshold is not None
    threshold = 0. if threshold is None else float(threshold)
    if resolution not in (.02, .16) or not np.isfinite(threshold) or threads <= 0:
        raise ValueError('Resolution must be .02 or .16, threshold finite, threads positive')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Prediction output already exists: {output}')
    wave, sample_rate = sf.read(audio, dtype='float32')
    if sample_rate != 16000 or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError('Expected finite nonempty mono 16kHz waveform')
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model, frontend = load_backbone(checkpoint, device=device)
    duration = len(wave)/sample_rate
    blocks = (len(wave)+2559)//2560
    waveform = F.pad(torch.from_numpy(wave), (0, blocks*2560-len(wave)))
    features = frontend(waveform).unsqueeze(0).to(device)
    native = model(features, torch.tensor([blocks*8]))
    fine_storage = native[0].cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    scores160 = fine_storage.reshape(-1, 8).mean(1)
    actual20 = (len(wave)+319)//320
    scores20 = fine_storage[:actual20]
    selected = scores20 if resolution == .02 else scores160
    detected = intervals(selected.astype(np.float64) >= threshold, duration, step=resolution)
    result = dict(audio=str(Path(audio).resolve()), duration_seconds=duration, sample_rate=sample_rate,
                  checkpoint_sha256=sha256(checkpoint),
                  score_kind='raw fake-positive logits; these are not calibrated probabilities',
                  threshold=threshold, interval_resolution_seconds=resolution,
                  threshold_source='Caller-supplied threshold; dev calibration must match the checkpoint and resolution' if explicit_threshold else 'Default raw-logit0; not dev-calibrated',
                  scores20=scores20.tolist(), scores160=scores160.tolist(),
                  predicted_fake_intervals_seconds=detected.tolist(),
                  native20_actual_frames=int(actual20), stored20_frames=int(len(fine_storage)),
                  padding='Zero-pad waveform to full160ms; mean all8 stored logits; clip reported intervals to actual duration',
                  precision_note='Raw regeneration may differ slightly across devices, batch composition and library versions; use archived caches for exact replay.')
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--resolution', type=float, choices=(.02, .16), default=.16)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    predict(**vars(args))


if __name__ == '__main__':
    main()
