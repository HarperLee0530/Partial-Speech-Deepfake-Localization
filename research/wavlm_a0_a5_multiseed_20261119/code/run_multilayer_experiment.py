"""Run the isolated, full fixed-split layer12 versus layer-fusion experiment.

Existing source and published checkpoints are never overwritten. Completed
stages are reused; incomplete output directories cause a visible error.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parent
REPO = CODE.parent
DEFAULT_OUTPUT = REPO.parent/'handoff_audit'/'multilayer_20261117'/'full'


def run(output, prepare_only=False):
    from cnn_ps.io import save_json, read_json, sha256
    from cnn_ps.train_wavlm import model_config
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output/'logs'
    logs.mkdir(exist_ok=True)
    started = time.monotonic()

    def stage(name, arguments, complete):
        if complete.exists():
            print(f'Reusing completed stage {name}', flush=True)
            return
        save_json(output/'status.json', dict(phase=name, status='running',
                  elapsed_seconds=time.monotonic()-started))
        print(f'START {name}', flush=True)
        with (logs/f'{name}.log').open('a', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, '-u', '-m', *map(str, arguments)],
                                    cwd=CODE, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode or not complete.exists():
            save_json(output/'status.json', dict(phase=name, status='failed', returncode=result.returncode))
            raise RuntimeError(f'{name} failed; see {logs/name}.log')
        print(f'COMPLETE {name}', flush=True)

    config = output/'model_config.json'
    if not config.exists():
        save_json(config, model_config().to_dict())
    manifests, labels = output/'manifests', output/'labels20'
    stage('manifests', ['cnn_ps.manifests', 'reduced', '--reduced-root', REPO/'data/reduced',
                      '--output', manifests], manifests/'eval.json')
    stage('labels20', ['cnn_ps.manifests', 'labels20', '--archive',
                      REPO/'data/downloads/database_segment_labels_v1.2.tar.gz',
                      '--output', labels], labels/'provenance.json')
    for split in ('train', 'dev'):
        stage(f'reference_{split}', ['cnn_ps.dual_features', '--model-config', config,
              '--manifest', manifests/f'{split}.json', '--labels20', labels/f'{split}.npz',
              '--output', output/'reference'/split, '--split', split,
              '--channels', 3 if split == 'train' else 1, '--cpu-threads', 2],
              output/'reference'/split/'complete.json')
    if prepare_only:
        save_json(output/'status.json', dict(phase='prepared', status='complete'))
        return
    for split in ('train', 'dev'):
        stage(f'layers_{split}', ['cnn_ps.wavlm_multilayer', '--reference', output/'reference'/split,
              '--model-folder', REPO/'data/models/wavlm-base-plus', '--output', output/'layers'/split],
              output/'layers'/split/'complete.json')
    training = output/'training'
    for arm in ('layer12', 'fusion6_9_12'):
        stage(f'train_{arm}', ['cnn_ps.train_wavlm_multilayer', '--arm', arm,
              '--train-cache', output/'reference/train', '--dev-cache', output/'reference/dev',
              '--train-ssl', output/'layers/train', '--dev-ssl', output/'layers/dev',
              '--output', training, '--epochs', 20, '--batch-size', 8, '--seed', 666, '--threads', 4],
              training/arm/'complete.json')
    stage('freeze', ['cnn_ps.train_wavlm_multilayer', '--freeze-only', '--output', training],
          training/'frozen_eval_plan.json')
    stage('reference_eval', ['cnn_ps.dual_features', '--model-config', config,
          '--manifest', manifests/'eval.json', '--labels20', labels/'eval.npz',
          '--output', output/'reference/eval', '--split', 'eval', '--channels', 1, '--cpu-threads', 2],
          output/'reference/eval/complete.json')
    stage('layers_eval', ['cnn_ps.wavlm_multilayer', '--reference', output/'reference/eval',
          '--model-folder', REPO/'data/models/wavlm-base-plus', '--output', output/'layers/eval',
          '--frozen-plan', training/'frozen_eval_plan.json'], output/'layers/eval/complete.json')
    stage('evaluate', ['cnn_ps.train_wavlm_multilayer', '--evaluate', '--output', training,
          '--eval-cache', output/'reference/eval', '--eval-ssl', output/'layers/eval'],
          training/'comparison.json')
    save_json(output/'status.json', dict(phase='evaluated', status='complete',
              elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    run(args.output, args.prepare_only)
