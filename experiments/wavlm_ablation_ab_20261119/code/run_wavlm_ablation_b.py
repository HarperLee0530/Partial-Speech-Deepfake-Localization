"""B-group workflow. Default: print plan only. Training requires --execute."""
import argparse
import ctypes
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
from cnn_ps.io import read_json, save_json, sha256
from cnn_ps.wavlm_ablation_b import ARMS, NEW_ARMS, encoder_sources

CODE = Path(__file__).resolve().parent
WORKSPACE = CODE.parents[1]
REFERENCE = WORKSPACE/'handoff_audit/multilayer_20261117/full'
BASELINE = WORKSPACE/'handoff_audit/ablation_a_20261118/training/A0_full'
DEFAULT_OUTPUT = WORKSPACE/'handoff_audit/ablation_b_20261119'


def baseline_manifest():
    complete = read_json(BASELINE/'complete.json')
    for name, key in [('best.pt', 'checkpoint_sha256'), ('protocol.json', 'protocol_sha256'),
                      ('dev_scores20.npz', 'dev_scores20_sha256'), ('dev_metrics.json', 'dev_metrics_sha256')]:
        if sha256(BASELINE/name) != complete[key]:
            raise ValueError(f'A0 baseline changed: {name}')
    if complete['trained_epochs'] != 20 or complete['updates'] != 15880:
        raise ValueError('A0 baseline budget mismatch')
    protocol = read_json(BASELINE/'protocol.json')
    for name, digest in protocol['source_sha256'].items():
        if sha256(CODE/'cnn_ps'/name) != digest:
            raise ValueError(f'A0 source changed: {name}')
    return complete


def discard_owned_cache(output, arm, split):
    """Delete only this run's exact generated cache after archiving its manifest."""
    base = (output/'cache').resolve()
    target = (base/arm/split).resolve()
    if (arm not in NEW_ARMS or split not in ('train', 'dev', 'eval')
            or not base.is_relative_to(output.resolve()) or not target.is_relative_to(base)):
        raise ValueError('Refusing deletion outside the declared B cache')
    if not target.exists():
        return
    complete = read_json(target/'complete.json')
    if complete['arm'] != arm or complete['split'] != split or complete['status'] != 'complete':
        raise ValueError('Refusing to delete an unrecognized/incomplete cache')
    save_json(output/'cache_manifests'/arm/(split+'.json'), complete)
    shutil.rmtree(target)


def run(output, discard_caches=False):
    output = Path(output).resolve()
    if output.exists() and not (output/'experiment_plan.json').exists():
        raise FileExistsError('Output exists without a B experiment plan')
    baseline = baseline_manifest()
    plan = dict(arms=ARMS, reference='A0 full backend, not A5', seed=666, epochs=20,
        batch_size=8, threads=4, baseline=baseline, encoder_sources=encoder_sources(),
        trainer_sha256=sha256(CODE/'cnn_ps/train_wavlm_ablation_b.py'),
        selection='dev duration FPR<=5%, best dev event F1, earliest tie, epoch0 eligible',
        test_previously_inspected=True, discard_generated_caches=discard_caches)
    if (output/'experiment_plan.json').exists() and read_json(output/'experiment_plan.json') != plan:
        raise ValueError('Cannot change an existing B experiment plan')
    output.mkdir(parents=True, exist_ok=True)
    save_json(output/'experiment_plan.json', plan)
    (output/'logs').mkdir(exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING='utf-8')

    def command(label, *args):
        if encoder_sources() != plan['encoder_sources'] or sha256(CODE/'cnn_ps/train_wavlm_ablation_b.py') != plan['trainer_sha256']:
            raise ValueError('Source changed during run')
        save_json(output/'status.json', dict(status='running', phase=label))
        with (output/'logs'/(label+'.log')).open('a', encoding='utf-8') as log:
            subprocess.run([sys.executable, '-u', *map(str, args)], cwd=CODE, env=env,
                stdout=log, stderr=subprocess.STDOUT, check=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)

    def cache(arm, split):
        folder = output/'cache'/arm/split
        if (folder/'complete.json').exists():
            return # Data loader validates provenance and feature digest before use.
        args = ['-m', 'cnn_ps.cache_wavlm_ablation_b', '--execute', '--arm', arm, '--split', split,
                '--reference', REFERENCE/'reference'/split, '--model-folder', CODE.parent/'data/models/wavlm-base-plus',
                '--output', folder]
        if split == 'eval':
            args += ['--frozen-plan', output/'training/frozen_eval_plan.json']
        command('cache_'+arm+'_'+split, *args)

    api = None
    try:
        if os.name == 'nt':
            api = ctypes.windll.kernel32.SetThreadExecutionState
            api.argtypes, api.restype = [ctypes.c_uint], ctypes.c_uint
            if not api(0x80000001):
                raise ctypes.WinError()
        save_json(output/'awake_status.json', dict(state='active', pid=os.getpid()))
        for arm in NEW_ARMS:
            if not (output/'training'/arm/'complete.json').exists():
                for split in ('train', 'dev'):
                    cache(arm, split)
                command('train_'+arm, '-m', 'cnn_ps.train_wavlm_ablation_b', '--execute', '--arm', arm,
                    '--train-cache', REFERENCE/'reference/train', '--dev-cache', REFERENCE/'reference/dev',
                    '--train-ssl', output/'cache'/arm/'train', '--dev-ssl', output/'cache'/arm/'dev',
                    '--output', output/'training', '--epochs', '20', '--seed', '666', '--batch-size', '8', '--threads', '4')
            if discard_caches:
                for split in ('train', 'dev'):
                    discard_owned_cache(output, arm, split)
        from cnn_ps.train_wavlm_ablation_b import freeze_eval_plan
        frozen = freeze_eval_plan(output/'training', list(NEW_ARMS))
        baseline_manifest()
        baseline_protocol = read_json(BASELINE/'protocol.json')
        for arm in NEW_ARMS:
            protocol = read_json(output/'training'/arm/'protocol.json')
            for key in ('seed', 'epochs', 'batch_size', 'model_config', 'learning_rate', 'weight_decay',
                        'gradient_clip', 'losses', 'train_uids_sha256', 'dev_uids_sha256',
                        'train_cache_metadata_sha256', 'dev_cache_metadata_sha256', 'initial_backbone_sha256'):
                if protocol[key] != baseline_protocol[key]:
                    raise ValueError(f'B/A0 comparison conditions differ: {arm}/{key}')
        save_json(output/'frozen_all_arms.json', dict(baseline=baseline, arms=frozen))
        for arm in NEW_ARMS:
            if not (output/'training'/arm/'eval_metrics.json').exists():
                cache(arm, 'eval')
                command('eval_'+arm, '-m', 'cnn_ps.train_wavlm_ablation_b', '--execute', '--evaluate',
                    '--evaluate-arm', arm, '--eval-cache', REFERENCE/'reference/eval',
                    '--eval-ssl', output/'cache', '--output', output/'training')
            if discard_caches:
                discard_owned_cache(output, arm, 'eval')
        baseline_output = output/'training/B0_full'
        baseline_output.mkdir(exist_ok=True)
        for name in ('best.pt', 'protocol.json', 'complete.json', 'dev_metrics.json', 'dev_scores20.npz', 'eval_metrics.json', 'eval_per_recording.json'):
            target = baseline_output/name
            if target.exists() and sha256(target) != sha256(BASELINE/name):
                raise ValueError('Archived baseline differs from verified A0')
            if not target.exists():
                shutil.copyfile(BASELINE/name, target)
        save_json(output/'all_comparison.json', dict(status='complete', frozen=frozen, baseline=baseline,
            models={arm: read_json(output/'training'/arm/'eval_metrics.json') for arm in ARMS}))
        command('benchmark', CODE/'benchmark_wavlm_ablation_b.py', '--output', output)
        command('report', CODE/'report_wavlm_ablation_b.py', '--output', output)
        save_json(output/'status.json', dict(status='complete', phase='reported'))
    except BaseException as exc:
        save_json(output/'status.json', dict(status='failed', error=repr(exc)))
        raise
    finally:
        if api is not None:
            api(0x80000000)
        save_json(output/'awake_status.json', dict(state='released', pid=os.getpid(),
                  updated_utc=datetime.now(timezone.utc).isoformat()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--discard-caches', action='store_true', help='Delete only generated B caches after use; retain manifests')
    args = parser.parse_args(argv)
    if not args.execute:
        print('PLAN ONLY. No training, cache generation, evaluation or output directory creation.')
        print('Reference: A0 full; new arms:', ', '.join(NEW_ARMS))
        print('Sequential cache/train per arm, freeze all, sequential eval, then report.')
        print('Budget: seed666, 20 epochs, batch8. Add --execute only when training is authorized.')
        return
    run(args.output, args.discard_caches)


if __name__ == '__main__':
    main()
