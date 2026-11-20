"""Run the approved A0--A10 matrix in isolated files and processes.

Two independent GPU workers overlap compute without sharing model/RNG state.
No eval artifact is opened until every training arm is complete and frozen.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from cnn_ps.io import read_json, save_json, sha256
from cnn_ps.wavlm_ablation_a import ARMS, NEW_ARMS, loss_weights

CODE = Path(__file__).resolve().parent
WORKSPACE = CODE.parents[1]
ROOT = WORKSPACE/'handoff_audit/ablation_a_20261118'
REFERENCE = WORKSPACE/'handoff_audit/multilayer_20261117/full'
REFERENCES = {'A0_full': 'fusion6_9_12', 'A1_layer12': 'layer12'}


def conditions(protocol):
    return {key: protocol[key] for key in ('seed', 'epochs', 'batch_size', 'model_config',
        'learning_rate', 'weight_decay', 'gradient_clip', 'train_uids_sha256',
        'dev_uids_sha256', 'train_recordings', 'dev_recordings',
        'train_cache_metadata_sha256', 'dev_cache_metadata_sha256', 'initial_backbone_sha256')}


def freeze_all():
    from cnn_ps.train_wavlm_ablation_a import freeze_eval_plan
    fresh = freeze_eval_plan(ROOT/'training', list(NEW_ARMS))
    verified = read_json(ROOT/'reference_reuse_verification.json')
    if verified['status'] != 'passed':
        raise ValueError('Reference equivalence must pass')
    result, common = {}, None
    for arm in ARMS:
        folder = ROOT/'training'/arm
        complete, protocol = read_json(folder/'complete.json'), read_json(folder/'protocol.json')
        for filename, field in [('best.pt', 'checkpoint_sha256'), ('protocol.json', 'protocol_sha256'),
                                ('dev_scores20.npz', 'dev_scores20_sha256'), ('dev_metrics.json', 'dev_metrics_sha256')]:
            if sha256(folder/filename) != complete[field]:
                raise ValueError(f'Changed artifact: {arm}/{filename}')
        for name, digest in protocol['source_sha256'].items():
            if sha256(CODE/'cnn_ps'/name) != digest:
                raise ValueError(f'Training source changed: {name}')
        if complete['trained_epochs'] != 20 or complete['updates'] != 15880:
            raise ValueError('Unequal training budget')
        if protocol['losses'] != loss_weights(arm):
            raise ValueError('Losses differ from the approved matrix')
        same = conditions(protocol)
        if common is not None and same != common:
            raise ValueError(f'Unpaired training conditions: {arm}')
        common = same
        if arm in REFERENCES:
            if complete != verified['models'][arm]['source_complete']:
                raise ValueError('Reference artifact differs from verified source')
        result[arm] = dict(complete=complete, kind='reused_reference' if arm in REFERENCES else 'new_training',
                           options=ARMS[arm], loss_weights=loss_weights(arm))
    for key in ('initial_projection_sha256', 'initial_adapter_sha256'):
        if len({read_json(ROOT/'training'/arm/'protocol.json')[key] for arm in NEW_ARMS}) != 1:
            raise ValueError(f'Common initialization differs: {key}')
    plan = dict(status='frozen', arms=result, common_conditions=common,
                test_threshold_selection=False, eval_previously_inspected=True,
                reference_verification_sha256=sha256(ROOT/'reference_reuse_verification.json'))
    path = ROOT/'frozen_all_arms.json'
    if path.exists() and read_json(path) != plan:
        raise ValueError('Frozen all-arm plan cannot be replaced')
    if not path.exists():
        save_json(path, plan)
    return plan


def run(workers):
    if workers not in (1, 2):
        raise ValueError('Use one or two independent workers')
    ROOT.mkdir(parents=True, exist_ok=True)
    logs = ROOT/'logs'
    logs.mkdir(exist_ok=True)
    source_hashes = {name: sha256(CODE/'cnn_ps'/name) for name in
                     ('wavlm_ablation_a.py', 'train_wavlm_ablation_a.py')}
    plan = dict(arms=ARMS, seed=666, epochs=20, batch_size=8, threads_per_worker=4,
                workers=workers, reference_root=str(REFERENCE), new_arms=list(NEW_ARMS),
                losses={arm: loss_weights(arm) for arm in ARMS}, source_sha256=source_hashes,
                primary_metric='event F1 at IoU0.5',
                selection='dev duration-FPR<=5%; maximum dev event F1; epoch0 eligible; earliest tie',
                calibration='separate dev threshold per model; never test calibration',
                references='Reuse A0/A1 only after exact CPU/CUDA step and full-dev parity checks',
                latency='After all training, sequential full-audio inference on fixed dev sample; no GPU contention',
                uncertainty='paired recording bootstrap; single training seed; exploratory comparisons',
                eval_previously_inspected=True)
    plan_path = ROOT/'experiment_plan.json'
    if plan_path.exists() and read_json(plan_path) != plan:
        raise ValueError('Predeclared plan changed')
    if not plan_path.exists():
        save_json(plan_path, plan)
    if read_json(ROOT/'reference_reuse_verification.json')['status'] != 'passed':
        raise ValueError('Missing reference equivalence verification')
    for arm, previous in REFERENCES.items():
        folder = ROOT/'training'/arm
        if not folder.exists():
            folder.mkdir(parents=True)
            for name in ('best.pt', 'complete.json', 'protocol.json', 'history.json', 'dev_metrics.json', 'dev_scores20.npz'):
                shutil.copyfile(REFERENCE/'training'/previous/name, folder/name)
            save_json(folder/'reuse.json', dict(source=str(REFERENCE/'training'/previous),
                      verified_by=str(ROOT/'reference_reuse_verification.json')))
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    creation = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    pending = [arm for arm in NEW_ARMS if not (ROOT/'training'/arm/'complete.json').exists()]
    running, completed = {}, [arm for arm in NEW_ARMS if arm not in pending]
    started = time.monotonic()
    try:
        if os.name == 'nt':
            api = ctypes.windll.kernel32.SetThreadExecutionState
            api.argtypes, api.restype = [ctypes.c_uint], ctypes.c_uint
            if not api(0x80000001):
                raise ctypes.WinError()
            save_json(ROOT/'awake_status.json', dict(state='active', pid=os.getpid(),
                      started_utc=datetime.now(timezone.utc).isoformat(), release='when orchestration process exits'))
        while pending or running:
            while pending and len(running) < workers:
                arm = pending.pop(0)
                if (ROOT/'training'/arm).exists():
                    raise FileExistsError(f'Incomplete arm needs an explicit clean retry: {arm}')
                if any(sha256(CODE/'cnn_ps'/name) != digest for name, digest in source_hashes.items()):
                    raise ValueError('Training source changed while the experiment was running')
                arguments = [sys.executable, '-u', '-m', 'cnn_ps.train_wavlm_ablation_a', '--arm', arm,
                    '--train-cache', str(REFERENCE/'reference/train'), '--dev-cache', str(REFERENCE/'reference/dev'),
                    '--train-ssl', str(REFERENCE/'layers/train'), '--dev-ssl', str(REFERENCE/'layers/dev'),
                    '--output', str(ROOT/'training'), '--epochs', '20', '--batch-size', '8', '--seed', '666', '--threads', '4']
                log = (logs/f'{arm}.log').open('w', encoding='utf-8')
                process = subprocess.Popen(arguments, cwd=CODE, stdout=log, stderr=subprocess.STDOUT,
                                           env=env, creationflags=creation)
                running[arm] = (process, log)
                print(f'START {arm} pid={process.pid}', flush=True)
            for arm, (process, log) in list(running.items()):
                if process.poll() is not None:
                    log.close()
                    if process.returncode or not (ROOT/'training'/arm/'complete.json').exists():
                        raise RuntimeError(f'{arm} failed; inspect {logs/arm}.log')
                    completed.append(arm)
                    del running[arm]
                    print(f'COMPLETE {arm}', flush=True)
            save_json(ROOT/'status.json', dict(phase='training', status='running',
                active={a: p.pid for a, (p, _) in running.items()}, pending=pending, completed=completed,
                elapsed_seconds=time.monotonic()-started))
            if pending or running:
                time.sleep(10)
        frozen = freeze_all()
        save_json(ROOT/'status.json', dict(phase='evaluation', status='running', elapsed_seconds=time.monotonic()-started))
        args = [sys.executable, '-u', '-m', 'cnn_ps.train_wavlm_ablation_a', '--evaluate',
                '--arms', *NEW_ARMS, '--eval-cache', str(REFERENCE/'reference/eval'),
                '--eval-ssl', str(REFERENCE/'layers/eval'), '--output', str(ROOT/'training'), '--threads', '4']
        if not (ROOT/'training/comparison.json').exists():
            with (logs/'evaluation.log').open('w', encoding='utf-8') as log:
                subprocess.run(args, cwd=CODE, stdout=log, stderr=subprocess.STDOUT, check=True,
                               env=env, creationflags=creation)
        for arm, previous in REFERENCES.items():
            for name in ('eval_metrics.json', 'eval_per_recording.json', 'eval_scores20.npz'):
                source, target = REFERENCE/'training'/previous/name, ROOT/'training'/arm/name
                if target.exists() and sha256(source) != sha256(target):
                    raise ValueError(f'Reused test artifact changed: {target}')
                if not target.exists():
                    shutil.copyfile(source, target)
        save_json(ROOT/'all_comparison.json', dict(status='complete', frozen=frozen,
                  models={arm: read_json(ROOT/'training'/arm/'eval_metrics.json') for arm in ARMS}))
        for script in ('benchmark_wavlm_ablation_a.py', 'report_wavlm_ablation_a.py'):
            save_json(ROOT/'status.json', dict(phase=script, status='running', elapsed_seconds=time.monotonic()-started))
            with (logs/(script+'.log')).open('w', encoding='utf-8') as log:
                subprocess.run([sys.executable, '-u', str(CODE/script)], cwd=CODE,
                               stdout=log, stderr=subprocess.STDOUT, check=True, env=env, creationflags=creation)
        save_json(ROOT/'status.json', dict(status='complete', phase='reported',
                  elapsed_seconds=time.monotonic()-started))
    except BaseException as error:
        save_json(ROOT/'failure.json', dict(type=type(error).__name__, error=str(error),
                  active={a: p.pid for a, (p, _) in running.items()}))
        raise
    finally:
        # Other active arms may still be finishing if one fails; keep their logs.
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            save_json(ROOT/'awake_status.json', dict(state='released', pid=os.getpid(),
                      updated_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    run(parser.parse_args().workers)
