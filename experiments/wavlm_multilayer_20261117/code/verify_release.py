"""Verify the release, pinned source bytes, selected model and paired bootstrap."""
from pathlib import Path
import json
import torch
from cnn_ps.io import sha256, read_json
from report_multilayer_experiment import paired_bootstrap


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = read_json(root/'release_manifest.json')
    for name, info in manifest['files'].items():
        path = root/name
        if path.stat().st_size != info['bytes'] or sha256(path) != info['sha256']:
            raise ValueError(f'Release file changed: {name}')
    checkpoint = root/'weights/best.pt'
    expected = '160a4b3cacbee48398d11f218e3ca2dff76615670ae05255f6775119dd37b50e'
    if sha256(checkpoint) != expected:
        raise ValueError('Selected checkpoint does not match measured experiment')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    for name, digest in payload['protocol']['source_sha256'].items():
        if sha256(root/'code/cnn_ps'/name) != digest:
            raise ValueError(f'Training source hash mismatch: {name}')
    if (payload['protocol']['arm'] != 'fusion6_9_12' or payload['epoch'] != 18
            or payload['threshold'] != -0.17905920743942258):
        raise ValueError('Unexpected selected model or threshold')
    folder = root/'reports/training'
    a = read_json(folder/'layer12/eval_per_recording.json')
    b = read_json(folder/'fusion6_9_12/eval_per_recording.json')
    recomputed = paired_bootstrap(a, b)
    archived = read_json(root/'reports/paired_bootstrap.json')
    if recomputed != archived:
        raise ValueError('Paired bootstrap differs from archived results')
    for key, values in recomputed['metrics'].items():
        x = read_json(folder/'layer12/eval_metrics.json')[key]
        y = read_json(folder/'fusion6_9_12/eval_metrics.json')[key]
        if abs((y-x)-values['delta_pp']) > 1e-9:
            raise ValueError(f'Metric delta disagrees with recording counts: {key}')
    print(json.dumps(dict(status='passed', files=len(manifest['files']),
        checkpoint_sha256=expected, training_source_hashes='matched',
        paired_bootstrap='reproduced exactly'), indent=2))


if __name__ == '__main__':
    main()
