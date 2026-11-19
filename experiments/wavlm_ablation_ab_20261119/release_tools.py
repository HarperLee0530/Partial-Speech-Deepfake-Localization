"""Portable archive verification, summaries and single-audio A/B inference.

Default operation is verification only. This file never starts training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def checkpoint(group, arm):
    if group == 'B' and arm == 'B0_full':
        return ROOT/'weights/A/A0_full/best.pt'
    return ROOT/'weights'/group/arm/'best.pt'


def verify():
    manifest = read(ROOT/'release_manifest.json')
    for name, value in manifest['files'].items():
        path = (ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != value['sha256']:
            raise ValueError(f'Archive changed: {name}')
    for group in ('A', 'B'):
        comparison = read(ROOT/'reports'/group/'all_comparison.json')
        for arm in comparison['models']:
            folder = ROOT/'reports'/group/'training'/arm
            complete = read(folder/'complete.json')
            if digest(checkpoint(group, arm)) != complete['checkpoint_sha256']:
                raise ValueError(f'Wrong best weight: {group}/{arm}')
            for name, key in [('protocol.json', 'protocol_sha256'), ('dev_metrics.json', 'dev_metrics_sha256')]:
                if digest(folder/name) != complete[key]:
                    raise ValueError(f'Frozen artifact changed: {group}/{arm}/{name}')
            protocol = read(folder/'protocol.json')
            for name, expected in protocol['source_sha256'].items():
                if digest(ROOT/'code/cnn_ps'/name) != expected:
                    raise ValueError(f'Training code changed: {name}')
            if read(folder/'eval_metrics.json') != comparison['models'][arm]:
                raise ValueError('Evaluation summary mismatch')
    print(f"VERIFIED: {len(manifest['files'])} files; 14 distinct best weights; B0 aliases A0.")


def summarize():
    for group in ('A', 'B'):
        models = read(ROOT/'reports'/group/'all_comparison.json')['models']
        baseline = models['A0_full' if group == 'A' else 'B0_full']['event_f1_iou05_percent']
        for arm, value in models.items():
            score = value['event_f1_iou05_percent']
            print(f"{arm:25} F1={score:.4f}% delta={score-baseline:+.4f}pp FPR={value['real_duration_fpr_percent']:.4f}%")


def predict(args):
    sys.path.insert(0, str(ROOT/'code'))
    import numpy as np
    import soundfile as sf
    import torch
    from cnn_ps.backbone import ModelConfig
    from cnn_ps.io import save_json
    from cnn_ps.predict_mel import waveform_batch, decode_events
    from cnn_ps.train_dual import forward_batch
    from cnn_ps.train_wavlm_ablation_a import setup
    if args.group == 'A':
        from cnn_ps.wavlm_ablation_a import ARMS, AblationCNN as Model, make_adapter
        from cnn_ps.wavlm_multilayer import FrozenMultiWavLM
        encoder_class = lambda: FrozenMultiWavLM(args.model_folder, args.device)
    else:
        from cnn_ps.wavlm_ablation_b import ARMS, PositionalCNN as Model, make_adapter, FrozenPositionalWavLM
        encoder_class = lambda: FrozenPositionalWavLM(args.model_folder, args.arm, args.device)
    if args.arm not in ARMS:
        raise ValueError('Arm does not belong to the selected group')
    if args.output.exists():
        raise FileExistsError(args.output)
    setup(666, args.threads)
    path = checkpoint(args.group, args.arm)
    complete = read(ROOT/'reports'/args.group/'training'/args.arm/'complete.json')
    if digest(path) != complete['checkpoint_sha256']:
        raise ValueError('Checkpoint hash mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=False)
    config = dict(payload['model_config']); config['channels'] = tuple(config['channels'])
    model = Model(ModelConfig(**config), args.arm, payload['protocol']['seed']).to(args.device)
    adapter = make_adapter(ARMS[args.arm], payload['protocol']['seed'], args.device)
    model.load_state_dict(payload['backbone_state_dict'], strict=True)
    adapter.load_state_dict(payload['adapter_state_dict'], strict=True)
    model.eval().requires_grad_(False); adapter.eval().requires_grad_(False)
    encoder = encoder_class()
    def frontend(wave):
        states = encoder(wave, len(wave)//320)
        mode = ARMS[args.arm]['feature_mode']
        features = (states[:, (6, 9, 12).index(int(mode[5:])), :]
                    if mode.startswith('layer') else states.flatten(-2))
        return features.unsqueeze(0).cpu()
    with torch.inference_mode():
        wave, sr = sf.read(args.audio, dtype='float32')
        batch, duration, frames = waveform_batch(wave, sr, frontend)
        result, _ = forward_batch(model, adapter, batch, args.device)
        scores = result[0, :frames].float().cpu().numpy()
    threshold = float(payload['threshold'])
    if threshold != complete['threshold']:
        raise ValueError('Threshold differs from frozen calibration')
    save_json(args.output, dict(group=args.group, arm=args.arm, duration_seconds=duration,
        threshold=threshold, checkpoint_sha256=digest(path), scores20=scores.tolist(),
        predicted_fake_intervals_seconds=decode_events(scores, threshold, duration).tolist(),
        note='Frozen dev threshold; 20ms logits; no smoothing or test calibration'))
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command')
    subs.add_parser('verify'); subs.add_parser('summarize')
    p = subs.add_parser('predict')
    p.add_argument('--group', choices=('A', 'B'), required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--model-folder', type=Path, required=True)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    p.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.command in (None, 'verify'): verify()
    elif args.command == 'summarize': summarize()
    else: predict(args)


if __name__ == '__main__':
    main()
