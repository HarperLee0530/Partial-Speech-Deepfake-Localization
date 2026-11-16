"""Paired fixed/learnable Mel frontends on the unchanged boundary CNN model.

Training reads train/dev only. Both arms use the same cached STFT powers and
the same frontend arithmetic; only the 80 frequency-shift parameters differ
in requires_grad. Freeze both completed arms before opening any eval cache.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .backbone import CNNBiGRU, ModelConfig
from .finetune import save_scores
from .io import manifest_hash, read_json, save_json, sha256
from .learnable_filterbank import LearnableFilterbank
from .metrics20 import evaluate_dev
from .power_cache import PowerFeatureData, power_collate
from .report_native20 import evaluate_report
from .train_dual import forward_batch as boundary_forward, training_loss
from .train_four import (model_config, setup, state_digest, TrainingView,
                         is_better, _cpu_state, make_adapter, ARMS as ORIGINAL_ARMS)


ARMS = {'fixed_mel': False, 'learnable_mel': True}
BASE_OPTIONS = ORIGINAL_ARMS['boundary']
SCHEMA = 'ps_boundary_learnable_filterbank_v1'
RUNTIME_SOURCES = ('train_filterbank.py', 'learnable_filterbank.py', 'power_cache.py',
                   'train_four.py', 'train_dual.py', 'dual_resolution.py', 'backbone.py',
                   'finetune.py', 'dual_features.py', 'dual_cache.py', 'augmentation.py',
                   'metrics20.py', 'metrics160.py', 'report_native20.py', 'io.py')


def json_native(value):
    return json.loads(json.dumps(value, allow_nan=False))


def dictionary_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def shared_frontend_config(frontend):
    # The treatment flag is separately recorded; the numerical design is shared.
    return {key: value for key, value in json_native(frontend.config_dict()).items()
            if key not in ('learnable', 'frontend_learnable')}


@torch.no_grad()
def frontend_diagnostics(frontend, initial_frequencies):
    # Canonical CPU diagnostics avoid CPU/CUDA tanh rounding differences when
    # a GPU-trained checkpoint is later independently inspected on CPU.
    canonical = LearnableFilterbank(frontend.config, learnable=False)
    canonical.load_state_dict(_cpu_state(frontend), strict=True)
    current = canonical.frequencies_hz().detach().to(torch.float64)
    original = torch.as_tensor(initial_frequencies, dtype=torch.float64)
    if current.shape != original.shape or current.numel() != 82:
        raise ValueError('Expected 82 ordered endpoints/centers')
    shifts = current[1:-1] - original[1:-1]
    parameters = torch.cat([p.detach().cpu().reshape(-1) for p in frontend.parameters()])
    return dict(frequencies_hz=current.tolist(), center_shift_hz=shifts.tolist(),
                max_abs_center_shift_hz=float(shifts.abs().max()),
                mean_abs_center_shift_hz=float(shifts.abs().mean()),
                nonzero_parameters=int(torch.count_nonzero(parameters)),
                raw_parameter_l2=float(torch.linalg.vector_norm(parameters)),
                minimum_node_gap_hz=float(torch.diff(current).min()),
                diagnostic_device='cpu; training projection remains on the declared device')


def forward_batch(model, adapter, frontend, batch, device):
    power = batch['power_features'].to(device)
    lengths = torch.as_tensor(batch['lengths'], dtype=torch.long, device='cpu')
    if (power.ndim != 4 or power.shape[1] != 1 or power.shape[-1] != 257
            or len(lengths) != power.shape[0] or (lengths <= 0).any()
            or (lengths > power.shape[2]).any()):
        raise ValueError('Expected aligned [B,1,T,257] powers and storage lengths')
    mask = torch.arange(power.shape[2], device=power.device)[None, :] < lengths.to(power.device)[:, None]
    # Batch-only padding may contain arbitrary values. The padded waveform tail
    # inside each storage length remains, exactly as in the original main model.
    power = torch.where(mask[:, None, :, None], power, torch.zeros_like(power))
    features = frontend(power)
    return boundary_forward(model, adapter, dict(batch, features=features), device)


@torch.inference_mode()
def infer(model, adapter, frontend, dataset, device, progress=None):
    model.eval()
    adapter.eval()
    frontend.eval()
    records, stored = [], []
    metadata = dict(uids=dataset.uids, lengths=[], lengths20=[], durations=[])
    last = time.monotonic()
    for index, batch in enumerate(DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=power_collate)):
        scores, _ = forward_batch(model, adapter, frontend, batch, device)
        uid, duration = batch['uids'][0], float(batch['durations'][0])
        n = int(batch['lengths'][0])
        score = scores[0, :n].float().cpu().numpy().copy()
        if not np.isfinite(score).all():
            raise ValueError('Nonfinite inference score')
        records.append(dict(uid=uid, duration=duration, labels20=dataset.fine[uid], scores20=score))
        stored.append(score)
        metadata['lengths'].append(n // 8)
        metadata['lengths20'].append(len(dataset.fine[uid]))
        metadata['durations'].append(duration)
        if progress and time.monotonic() - last >= 25:
            event = dict(progress[1], recordings=index + 1, recordings_total=len(dataset))
            save_json(progress[0], event)
            print(event, flush=True)
            last = time.monotonic()
    return records, np.concatenate(stored), metadata


def train_arm(args, train, dev):
    arm = args.arm
    if arm not in ARMS or args.epochs < 0 or args.batch_size <= 0:
        raise ValueError('Invalid arm, epoch count or batch size')
    if (train.split != 'train' or train.channels != 3 or dev.split != 'dev'
            or dev.channels != 1 or set(train.uids) & set(dev.uids)):
        raise ValueError('Require three-view train and disjoint clean dev')
    output = Path(args.output) / arm
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite arm directory: {output}')
    setup(args.seed, args.threads)
    model = CNNBiGRU(model_config()).to(args.device)
    adapter = make_adapter(BASE_OPTIONS, args.seed, args.device)
    frontend = LearnableFilterbank(model.config, learnable=ARMS[arm]).to(args.device)
    initial_backbone, initial_adapter, initial_frontend = map(state_digest, (model, adapter, frontend))
    initial_frequencies = frontend.frequencies_hz().detach().cpu().tolist()
    parameters = [p for module in (model, adapter, frontend) for p in module.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
    view = TrainingView(train, clean_only=False)
    loader = DataLoader(view, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=power_collate, generator=torch.Generator().manual_seed(args.seed))
    protocol = dict(schema=SCHEMA, arm=arm, adapter_options=BASE_OPTIONS,
        frontend_learnable=ARMS[arm], frontend_config=json_native(frontend.config_dict()),
        shared_frontend_config=shared_frontend_config(frontend),
        seed=args.seed, epochs=args.epochs, batch_size=args.batch_size,
        model_config=model.config.to_dict(),
        initialization='New random identical CNN/boundary; both frontends start at the same fixed Mel matrix',
        initial_backbone_sha256=initial_backbone, initial_adapter_sha256=initial_adapter,
        initial_frontend_sha256=initial_frontend, initial_frequencies_hz=initial_frequencies,
        pretrained_weights_used=False, external_normalization_used=False,
        all_backbone_parameters_trainable=True, learning_rate=1e-3, weight_decay=1e-4,
        optimizer='AdamW', gradient_clip=5.,
        frontend_optimizer='Same AdamW learning rate/weight decay and joint global clip as the backend',
        train_cache=str(train.folder), dev_cache=str(dev.folder),
        train_power_cache=str(train.power_folder), dev_power_cache=str(dev.power_folder),
        train_cache_metadata_sha256=sha256(train.folder/'complete.json'),
        dev_cache_metadata_sha256=sha256(dev.folder/'complete.json'),
        train_power_metadata_sha256=sha256(train.power_folder/'complete.json'),
        dev_power_metadata_sha256=sha256(dev.power_folder/'complete.json'),
        train_recordings=len(train), dev_recordings=len(dev),
        train_uids_sha256=manifest_hash(train.uids), dev_uids_sha256=manifest_hash(dev.uids),
        backbone_parameters=sum(p.numel() for p in model.parameters()),
        adapter_parameters=sum(p.numel() for p in adapter.parameters()),
        frontend_parameters=sum(p.numel() for p in frontend.parameters()),
        trainable_frontend_parameters=sum(p.numel() for p in frontend.parameters() if p.requires_grad),
        trainable_parameters=sum(p.numel() for p in parameters),
        channels='Balanced per-UID clean/noise20dB/EQ assignment, rotated by epoch',
        data_order='Independent DataLoader generator with the same seed in both arms',
        losses=dict(main20=1., main160=1., coarse_aux=.5, boundary=.1, consistency=.05),
        final160_pool='Mean of eight stored logits; actual-duration coarse loss; storage-tail logits retained',
        power_projection='Both arms project the identical cached float32 STFT powers on the training/inference device',
        selection='Clean dev genuine-duration FPR<=5%, then maximum dev event F1@IoU0.5; epoch0 eligible; earliest tie',
        test_threshold_selection=False, eval_read=False,
        comparison_scope='Only frontend frequency-knot training is enabled in treatment; GPU projection may differ slightly from historical CPU LogMel caches',
        source_sha256={name: sha256(Path(__file__).with_name(name)) for name in RUNTIME_SOURCES})
    output.mkdir(parents=True)
    (output/'epochs').mkdir()
    save_json(output/'protocol.json', protocol)
    best, history, updates = None, [], 0
    started = time.monotonic()
    for epoch in range(args.epochs + 1):
        tick, losses, components, last_frontend_gradient = time.monotonic(), [], [], None
        if epoch:
            view.epoch = epoch
            for module in (model, adapter, frontend):
                module.train()
            last = time.monotonic()
            for index, batch in enumerate(loader):
                optimizer.zero_grad(set_to_none=True)
                final, outputs = forward_batch(model, adapter, frontend, batch, args.device)
                loss, parts = training_loss(final, outputs, batch, args.device, True)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                gradients = [p.grad.detach().flatten() for p in frontend.parameters() if p.grad is not None]
                last_frontend_gradient = float(torch.linalg.vector_norm(torch.cat(gradients))) if gradients else 0.
                torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                losses.append(float(loss.detach()))
                components.append(parts)
                if time.monotonic() - last >= 25:
                    progress = dict(arm=arm, phase='train', epoch=epoch, batches=index+1,
                                    batches_total=len(loader), updates=updates,
                                    loss=float(np.mean(losses)), elapsed_seconds=time.monotonic()-tick)
                    save_json(output/'progress.json', progress)
                    print(progress, flush=True)
                    last = time.monotonic()
        train_seconds = time.monotonic() - tick
        event = dict(arm=arm, phase='dev', epoch=epoch, updates=updates)
        save_json(output/'progress.json', event)
        records, scores, metadata = infer(model, adapter, frontend, dev, args.device,
                                          progress=(output/'progress.json', event))
        metrics = evaluate_dev(records, max_real_fpr=.05)
        metrics.pop('selection_score_j_percent', None)
        if metrics['real_fpr_percent'] > 5 + 1e-10:
            raise AssertionError('Development threshold exceeded the duration-FPR budget')
        value = float(metrics['event_f1_iou05_percent'])
        selected = dict(epoch=epoch, threshold=float(metrics['threshold']), event_f1_iou05_percent=value)
        diagnostics = frontend_diagnostics(frontend, initial_frequencies)
        prefix = output/'epochs'/f'epoch_{epoch:02d}'
        payload = dict(schema=SCHEMA, backbone_state_dict=_cpu_state(model),
                       adapter_state_dict=_cpu_state(adapter), adapter_options=BASE_OPTIONS,
                       frontend_state_dict=_cpu_state(frontend), frontend_config=json_native(frontend.config_dict()),
                       frontend_learnable=ARMS[arm], frontend_diagnostics=diagnostics,
                       model_config=model.config.to_dict(), protocol=protocol, dev_metrics=metrics, **selected)
        torch.save(payload, prefix.with_suffix('.pt'))
        save_scores(prefix.with_name(prefix.name+'_scores20.npz'), scores, metadata)
        save_json(prefix.with_name(prefix.name+'_metrics.json'), metrics)
        archive_hashes = dict(checkpoint_sha256=sha256(prefix.with_suffix('.pt')),
                              scores20_sha256=sha256(prefix.with_name(prefix.name+'_scores20.npz')),
                              metrics_sha256=sha256(prefix.with_name(prefix.name+'_metrics.json')))
        if is_better(value, best):
            best = selected
            shutil.copyfile(prefix.with_suffix('.pt'), output/'best.pt')
            shutil.copyfile(prefix.with_name(prefix.name+'_scores20.npz'), output/'dev_scores20.npz')
            save_json(output/'dev_metrics.json', metrics)
        row = dict(arm=arm, epoch=epoch, updates=updates,
            loss=float(np.mean(losses)) if losses else None,
            loss_components={key: float(np.mean([item[key] for item in components])) for key in components[0]} if components else {},
            train_seconds=train_seconds, total_seconds=time.monotonic()-tick, best_epoch=best['epoch'],
            frontend_diagnostics=diagnostics, last_frontend_gradient_norm=last_frontend_gradient,
            archive_hashes=archive_hashes,
            **{key: metrics[key] for key in ('threshold', 'event_f1_iou05_percent',
                'short_event_recall_le320ms_percent', 'real_fpr_percent')})
        history.append(row)
        save_json(output/'history.json', history)
        save_json(output/'progress.json', dict(phase='epoch_complete', **row))
        print({key: row[key] for key in ('arm', 'epoch', 'updates', 'loss', 'train_seconds',
              'total_seconds', 'best_epoch', 'threshold', 'event_f1_iou05_percent', 'real_fpr_percent')}, flush=True)
    complete = dict(status='complete', schema=SCHEMA, arm=arm, **best,
        checkpoint='best.pt', checkpoint_sha256=sha256(output/'best.pt'),
        initial_backbone_sha256=initial_backbone, initial_adapter_sha256=initial_adapter,
        initial_frontend_sha256=initial_frontend, last_backbone_sha256=state_digest(model),
        last_frontend_sha256=state_digest(frontend), frontend_learnable=ARMS[arm],
        final_frontend_diagnostics=frontend_diagnostics(frontend, initial_frequencies),
        trained_epochs=args.epochs, updates=updates, archived_epochs=args.epochs+1,
        history_sha256=sha256(output/'history.json'), protocol_sha256=sha256(output/'protocol.json'), eval_read=False,
        dev_scores20_sha256=sha256(output/'dev_scores20.npz'), dev_metrics_sha256=sha256(output/'dev_metrics.json'),
        total_seconds=time.monotonic()-started)
    save_json(output/'complete.json', complete)
    return complete


def _check_payload(payload, protocol, epoch, metrics):
    enabled = protocol['frontend_learnable']
    if (payload.get('schema') != SCHEMA or type(payload.get('frontend_learnable')) is not bool
            or payload['frontend_learnable'] != enabled or payload['adapter_options'] != BASE_OPTIONS
            or json_native(payload['protocol']) != protocol or json_native(payload['model_config']) != protocol['model_config']
            or payload['frontend_config'] != protocol['frontend_config'] or payload['dev_metrics'] != metrics
            or payload['epoch'] != epoch or payload['threshold'] != metrics['threshold']
            or payload['event_f1_iou05_percent'] != metrics['event_f1_iou05_percent']
            or not np.isfinite(payload['threshold']) or not 0 <= metrics['real_fpr_percent'] <= 5+1e-10
            or 'selection_score_j_percent' in metrics):
        raise ValueError('Checkpoint schema, dev metrics, epoch or fixed threshold disagree')
    for key in ('backbone_state_dict', 'adapter_state_dict', 'frontend_state_dict'):
        if not isinstance(payload[key], dict) or any(not torch.isfinite(t).all() for t in payload[key].values()):
            raise ValueError('Nonfinite or missing checkpoint state')
    frontend = LearnableFilterbank(ModelConfig(**payload['model_config']), learnable=enabled)
    frontend.load_state_dict(payload['frontend_state_dict'], strict=True)
    if (json_native(frontend.config_dict()) != protocol['frontend_config']
            or shared_frontend_config(frontend) != protocol['shared_frontend_config']
            or frontend_diagnostics(frontend, protocol['initial_frequencies_hz']) != payload['frontend_diagnostics']):
        raise ValueError('Frontend configuration or saved frequency diagnostics disagree')
    if not enabled and state_digest(frontend) != protocol['initial_frontend_sha256']:
        raise ValueError('Fixed frontend changed during training')
    if epoch == 0:
        for key, expected in (('backbone_state_dict', 'initial_backbone_sha256'),
                              ('adapter_state_dict', 'initial_adapter_sha256'),
                              ('frontend_state_dict', 'initial_frontend_sha256')):
            if dictionary_digest(payload[key]) != protocol[expected]:
                raise ValueError('Epoch-zero initialized tensor digest changed')


def _score_archive(path, protocol):
    with np.load(path, allow_pickle=False) as source:
        values = {key: source[key].copy() for key in source.files}
    required = {'scores20', 'uids', 'lengths', 'lengths20', 'durations'}
    if set(values) != required:
        raise ValueError('Unexpected score archive fields')
    n = protocol['dev_recordings']
    if (any(values[key].shape != (n,) for key in ('uids', 'lengths', 'lengths20', 'durations'))
            or manifest_hash(values['uids'].tolist()) != protocol['dev_uids_sha256']
            or values['scores20'].shape != (int(values['lengths'].sum())*8,)
            or (values['lengths'] <= 0).any() or (values['lengths20'] <= 0).any()
            or (values['lengths20'] > 8*values['lengths']).any()
            or not np.isfinite(values['durations']).all() or (values['durations'] <= 0).any()
            or not np.isfinite(values['scores20']).all()):
        raise ValueError('Score archive UID/grid/duration validation failed')
    return values


def freeze_eval_plan(output):
    """Check all epoch archives and shared conditions before any eval access."""
    output = Path(output)
    frozen, shared, epoch_zero = {}, None, None
    checked_cache_paths = set()
    for arm, enabled in ARMS.items():
        folder = output/arm
        complete, protocol = read_json(folder/'complete.json'), read_json(folder/'protocol.json')
        if (complete.get('status') != 'complete' or complete.get('schema') != SCHEMA
                or complete.get('arm') != arm or complete.get('frontend_learnable') is not enabled
                or protocol.get('schema') != SCHEMA or protocol.get('arm') != arm
                or protocol.get('frontend_learnable') is not enabled or complete.get('eval_read') is not False):
            raise ValueError(f'Incomplete or mismatched training arm: {arm}')
        for filename, key in (('best.pt', 'checkpoint_sha256'), ('protocol.json', 'protocol_sha256'),
                              ('history.json', 'history_sha256'), ('dev_scores20.npz', 'dev_scores20_sha256'),
                              ('dev_metrics.json', 'dev_metrics_sha256')):
            if sha256(folder/filename) != complete[key]:
                raise ValueError(f'Selected artifact changed: {arm}/{filename}')
        history = read_json(folder/'history.json')
        per_epoch = int(np.ceil(protocol['train_recordings']/protocol['batch_size']))
        if (len(history) != protocol['epochs']+1 or complete['archived_epochs'] != len(history)
                or complete['trained_epochs'] != protocol['epochs']
                or complete['updates'] != per_epoch*protocol['epochs']
                or [row['epoch'] for row in history] != list(range(len(history)))):
            raise ValueError(f'Incomplete training budget or history: {arm}')
        best, grid = None, None
        for row in history:
            epoch = row['epoch']
            prefix = folder/'epochs'/f'epoch_{epoch:02d}'
            files = (prefix.with_suffix('.pt'), prefix.with_name(prefix.name+'_scores20.npz'),
                     prefix.with_name(prefix.name+'_metrics.json'))
            for path, key in zip(files, ('checkpoint_sha256', 'scores20_sha256', 'metrics_sha256')):
                if sha256(path) != row['archive_hashes'][key]:
                    raise ValueError(f'Epoch archive changed: {path}')
            payload = torch.load(files[0], map_location='cpu', weights_only=False)
            metrics = read_json(files[2])
            _check_payload(payload, protocol, epoch, metrics)
            archive = _score_archive(files[1], protocol)
            current_grid = {key: value for key, value in archive.items() if key != 'scores20'}
            if grid is not None and any(not np.array_equal(value, grid[key]) for key, value in current_grid.items()):
                raise ValueError('Development grids changed between epochs')
            grid = current_grid
            if (row['updates'] != per_epoch*epoch or row['frontend_diagnostics'] != payload['frontend_diagnostics']
                    or any(row[key] != metrics[key] for key in ('threshold', 'event_f1_iou05_percent',
                           'short_event_recall_le320ms_percent', 'real_fpr_percent'))):
                raise ValueError('Epoch history disagrees with archived metrics or update budget')
            if is_better(row['event_f1_iou05_percent'], best):
                best = row
            if row['best_epoch'] != best['epoch']:
                raise ValueError('History did not use earliest maximum dev event F1')
            if epoch == 0:
                if epoch_zero is not None and any(not np.array_equal(value, epoch_zero[key]) for key, value in archive.items()):
                    raise ValueError('Paired epoch-zero scores or grids differ')
                epoch_zero = archive
        if (best['epoch'] != complete['epoch'] or best['threshold'] != complete['threshold']
                or best['event_f1_iou05_percent'] != complete['event_f1_iou05_percent']
                or best['archive_hashes']['checkpoint_sha256'] != complete['checkpoint_sha256']
                or best['archive_hashes']['scores20_sha256'] != complete['dev_scores20_sha256']
                or best['archive_hashes']['metrics_sha256'] != complete['dev_metrics_sha256']):
            raise ValueError(f'Selected artifacts are not the earliest best dev epoch: {arm}')
        for filename, value in protocol['source_sha256'].items():
            if sha256(Path(__file__).with_name(filename)) != value:
                raise ValueError(f'Runtime source changed since training: {filename}')
        for split in ('train', 'dev'):
            acoustic, power = Path(protocol[split+'_cache']), Path(protocol[split+'_power_cache'])
            if (sha256(acoustic/'complete.json') != protocol[split+'_cache_metadata_sha256']
                    or sha256(power/'complete.json') != protocol[split+'_power_metadata_sha256']):
                raise ValueError(f'Paired input cache metadata changed: {arm}/{split}')
            key = (str(acoustic), str(power), split)
            if key not in checked_cache_paths:
                verified = PowerFeatureData(acoustic, power, split=split, seed=protocol['seed'],
                                            config=ModelConfig(**protocol['model_config']), verify_hashes=True)
                if (len(verified) != protocol[split+'_recordings']
                        or manifest_hash(verified.uids) != protocol[split+'_uids_sha256']):
                    raise ValueError('Frozen cache recording identities changed')
                checked_cache_paths.add(key)
                del verified
        conditions = {key: protocol[key] for key in (
            'seed', 'epochs', 'batch_size', 'model_config', 'learning_rate', 'weight_decay', 'gradient_clip', 'losses',
            'train_uids_sha256', 'dev_uids_sha256', 'train_recordings', 'dev_recordings',
            'train_cache_metadata_sha256', 'dev_cache_metadata_sha256', 'train_power_metadata_sha256',
            'dev_power_metadata_sha256', 'source_sha256', 'shared_frontend_config', 'initial_frequencies_hz',
            'initial_backbone_sha256', 'initial_adapter_sha256', 'initial_frontend_sha256')}
        if shared is not None and shared != conditions:
            raise ValueError('Arms differ in paired initialization, input, losses, budget or source')
        shared = conditions
        frozen[arm] = complete
    plan = dict(schema=SCHEMA, arms=frozen, calibration='Completed dev calibration only',
                test_threshold_selection=False, eval_previously_inspected=True,
                epoch_zero_scores_identical=True, all_epoch_archives_verified=True, paired_conditions=shared)
    target = output/'frozen_eval_plan.json'
    if target.exists():
        if read_json(target) != plan:
            raise ValueError('An existing frozen plan differs and cannot be replaced')
    else:
        save_json(target, plan)
    return frozen


def evaluate_arms(args, dataset, frozen):
    """Use the two frozen dev thresholds without calibration on eval."""
    if set(frozen) != set(ARMS) or dataset.split != 'eval' or dataset.channels != 1:
        raise ValueError('Both frozen arms and a clean eval cache are required')
    results = {}
    # Refuse partial reruns before mutating either arm's evaluation files.
    for arm in ARMS:
        folder = Path(args.output)/arm
        if any((folder/name).exists() for name in ('eval_metrics.json', 'eval_scores20.npz', 'eval_per_recording.json')):
            raise FileExistsError(f'Refusing to overwrite evaluation artifacts: {arm}')
    for arm, enabled in ARMS.items():
        folder = Path(args.output)/arm
        if sha256(folder/'best.pt') != frozen[arm]['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed before evaluation')
        payload = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        _check_payload(payload, json_native(payload['protocol']), frozen[arm]['epoch'], payload['dev_metrics'])
        config = ModelConfig(**payload['model_config'])
        model = CNNBiGRU(config).to(args.device)
        model.load_state_dict(payload['backbone_state_dict'], strict=True)
        adapter = make_adapter(BASE_OPTIONS, payload['protocol']['seed'], args.device)
        adapter.load_state_dict(payload['adapter_state_dict'], strict=True)
        frontend = LearnableFilterbank(config, learnable=enabled).to(args.device)
        frontend.load_state_dict(payload['frontend_state_dict'], strict=True)
        event = dict(arm=arm, phase='eval')
        records, scores, metadata = infer(model, adapter, frontend, dataset, args.device,
                                          progress=(folder/'eval_progress.json', event))
        report = evaluate_report(records, threshold=float(frozen[arm]['threshold']))
        save_scores(folder/'eval_scores20.npz', scores, metadata)
        save_json(folder/'eval_per_recording.json', report.pop('per_recording'))
        save_json(folder/'eval_metrics.json', report)
        save_json(folder/'eval_progress.json', dict(event, status='complete', recordings=len(dataset)))
        results[arm] = report
        print(dict(arm=arm, event_f1=report['event_f1_iou05_percent'],
                   real_fpr=report['real_duration_fpr_percent']), flush=True)
    save_json(Path(args.output)/'comparison.json', dict(status='complete', schema=SCHEMA,
        metric_protocol='ps_native20_event_v2', models=results, frozen=frozen,
        eval_cache=str(dataset.folder), eval_cache_metadata_sha256=sha256(dataset.folder/'complete.json'),
        eval_power_cache=str(dataset.power_folder), eval_power_cache_metadata_sha256=sha256(dataset.power_folder/'complete.json'),
        test_threshold_selection=False, eval_previously_inspected=True))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(ARMS))
    for name in ('train-cache', 'dev-cache', 'train-power', 'dev-power', 'eval-cache', 'eval-power'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--freeze-only', action='store_true')
    mode.add_argument('--evaluate', action='store_true')
    args = parser.parse_args(argv)
    if args.epochs < 0 or args.batch_size <= 0 or args.threads <= 0:
        parser.error('epochs must be nonnegative; batch-size/threads must be positive')
    training_inputs = (args.train_cache, args.dev_cache, args.train_power, args.dev_power)
    eval_inputs = (args.eval_cache, args.eval_power)
    if args.evaluate or args.freeze_only:
        if args.arm or any(training_inputs) or (args.freeze_only and any(eval_inputs)):
            parser.error('Freeze/evaluate cannot accept training inputs; freeze cannot accept eval inputs')
        if args.evaluate and not all(eval_inputs):
            parser.error('Evaluation requires both --eval-cache and --eval-power')
    elif not args.arm or not all(training_inputs) or any(eval_inputs):
        parser.error('Training requires one arm and train/dev acoustic+power caches, with no eval inputs')
    setup(args.seed, args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; explicitly choose --device cpu')
    if args.evaluate or args.freeze_only:
        frozen = freeze_eval_plan(args.output)
        if not args.freeze_only:
            dataset = PowerFeatureData(args.eval_cache, args.eval_power, split='eval', seed=args.seed,
                                       config=model_config(), verify_hashes=True)
            evaluate_arms(args, dataset, frozen)
    else:
        train = PowerFeatureData(args.train_cache, args.train_power, split='train', seed=args.seed,
                                 config=model_config(), verify_hashes=True)
        dev = PowerFeatureData(args.dev_cache, args.dev_power, split='dev', seed=args.seed,
                               config=model_config(), verify_hashes=True)
        train_arm(args, train, dev)


if __name__ == '__main__':
    main()
