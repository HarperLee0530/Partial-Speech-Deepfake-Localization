"""Paired boundary vs boundary+energy5 training; dev-only event-F1 selection.

Both arms restart from the same random CNN/boundary parameters. The treatment
adds only a zero-initialized 5-to-32 input to the existing fine projection.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .backbone import CNNBiGRU, ModelConfig
from .energy_adapter import make_energy_adapter
from .energy_cache import EnergyFeatureData, energy_collate, NORMALIZATION_SCHEMA
from .energy_features import FEATURE_NAMES
from .finetune import save_scores
from .io import manifest_hash, read_json, save_json, sha256
from .metrics20 import evaluate_dev
from .report_native20 import evaluate_report
from .train_dual import training_loss
from .train_four import model_config, setup, state_digest, TrainingView, is_better, _cpu_state, ARMS as ORIGINAL_ARMS

ARMS = {'boundary_control':False, 'boundary_energy5':True}
BASE_OPTIONS = ORIGINAL_ARMS['boundary']
SCHEMA = 'ps_boundary_energy5_v1'


def dictionary_digest(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_normalization(train):
    path = train.energy_folder/'normalization.json'
    value = read_json(path)
    if (train.energy_metadata['normalization']['sha256'] != sha256(path)
            or value.get('schema') != NORMALIZATION_SCHEMA
            or value.get('fit_split') != 'train' or value.get('fit_channel') != 0
            or value.get('feature_names') != list(FEATURE_NAMES)
            or value.get('frontend_complete_sha256') != sha256(train.folder/'complete.json')
            or value.get('energy_sha256') != train.energy_metadata['array']['sha256']
            or value.get('valid_train_frames', 0) <= 0
            or value.get('padded_frames_in_statistics') is not False):
        raise ValueError('Energy normalization must be fitted to this clean training cache only')
    return value, path


def forward_batch(model, adapter, batch, device):
    native, context = model(batch['features'].to(device), batch['lengths'], return_features=True)
    outputs, corrected = [], []
    for i, length in enumerate(batch['lengths'].tolist()):
        result = adapter(context[i, :length], native[i, :length],
                         batch['fine_weights'][i, :length].to(device),
                         batch['energy_features'][i, :length].to(device))
        outputs.append(result)
        corrected.append(F.pad(result['final_logits'], (0, native.shape[1]-length)))
    return torch.stack(corrected), outputs


@torch.inference_mode()
def infer(model, adapter, dataset, device):
    model.eval()
    adapter.eval()
    records, stored = [], []
    metadata = dict(uids=dataset.uids, lengths=[], lengths20=[], durations=[])
    for batch in DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=energy_collate):
        scores, _ = forward_batch(model, adapter, batch, device)
        uid, duration = batch['uids'][0], float(batch['durations'][0])
        n = int(batch['lengths'][0])
        score = scores[0, :n].float().cpu().numpy().copy()
        if not np.isfinite(score).all():
            raise ValueError('Nonfinite inference score')
        records.append(dict(uid=uid, duration=duration, labels20=dataset.fine[uid], scores20=score))
        stored.append(score)
        metadata['lengths'].append(n//8)
        metadata['lengths20'].append(len(dataset.fine[uid]))
        metadata['durations'].append(duration)
    return records, np.concatenate(stored), metadata


def train_arm(args, train, dev):
    arm = args.arm
    if arm not in ARMS or args.epochs < 0 or args.batch_size <= 0:
        raise ValueError("Invalid arm, epoch count or batch size")
    if dev.split != "dev" or dev.channels != 1 or set(train.uids) & set(dev.uids):
        raise ValueError("Require clean dev and disjoint train/dev UIDs")
    output = Path(args.output) / arm
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite arm directory: {output}")
    setup(args.seed, args.threads)
    model = CNNBiGRU(model_config()).to(args.device)
    initial_digest = state_digest(model)
    normalization, normalization_file = load_normalization(train)
    adapter = make_energy_adapter(ARMS[arm], normalization, args.seed, args.device)
    initial_adapter = state_digest(adapter)
    initial_common_adapter = dictionary_digest(adapter.common_state_dict())
    parameters = [p for p in list(model.parameters()) + list(adapter.parameters()) if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-3, weight_decay=1e-4)
    view = TrainingView(train, clean_only=False)
    loader = DataLoader(view, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=energy_collate, generator=torch.Generator().manual_seed(args.seed))
    boundary_enabled = True
    protocol = dict(schema=SCHEMA, arm=arm, adapter_options=BASE_OPTIONS, energy_enabled=ARMS[arm], seed=args.seed,
        epochs=args.epochs, batch_size=args.batch_size, model_config=model.config.to_dict(),
        initialization="New random CNN; identical initialization seed and tensor values across both arms",
        initial_backbone_sha256=initial_digest, initial_adapter_sha256=initial_adapter,
        initial_common_adapter_sha256=initial_common_adapter, pretrained_weights_used=False,
        external_normalization_used=True, adapter_normalization="trainable per-frame LayerNorm256",
        energy_normalization=normalization, normalization_file=str(normalization_file),
        normalization_sha256=sha256(normalization_file),
        energy_feature_names=normalization['feature_names'], energy_projection='5->32, no bias, zero initialized, before fine GELU',
        energy_mask='Actual positive fine duration; valid zero feature vectors retained',
        energy_clip=[-10., 10.], energy_projection_parameters=160,
        initial_energy_weight_nonzero=int(torch.count_nonzero(adapter.energy_projection.weight)),
        all_backbone_parameters_trainable=True, learning_rate=1e-3, weight_decay=1e-4,
        optimizer="AdamW", gradient_clip=5.,
        train_cache=str(args.train_cache), dev_cache=str(args.dev_cache),
        train_energy_cache=str(args.train_energy), dev_energy_cache=str(args.dev_energy),
        train_energy_metadata_sha256=sha256(train.energy_folder/'complete.json'),
        dev_energy_metadata_sha256=sha256(dev.energy_folder/'complete.json'),
        train_cache_metadata_sha256=sha256(train.folder / "complete.json") if hasattr(train, "folder") else None,
        dev_cache_metadata_sha256=sha256(dev.folder / "complete.json") if hasattr(dev, "folder") else None,
        train_recordings=len(train), dev_recordings=len(dev),
        train_uids_sha256=manifest_hash(train.uids), dev_uids_sha256=manifest_hash(dev.uids),
        backbone_parameters=sum(p.numel() for p in model.parameters()),
        adapter_parameters=sum(p.numel() for p in adapter.parameters()),
        trainable_adapter_parameters=sum(p.numel() for p in adapter.parameters() if p.requires_grad),
        trainable_parameters=sum(p.numel() for p in parameters),
        channels="Balanced per-UID clean/noise20dB/EQ assignment, rotated by epoch",
        data_order="Independent DataLoader generator with the same seed in every arm",
        losses=dict(main20=1., main160=1., coarse_aux=.5, boundary=.1, consistency=.05),
        final160_pool="Mean of eight stored logits; actual-duration coarse loss; storage-tail logits retained",
        selection="Clean dev genuine-duration FPR<=5%, then maximum dev event F1@IoU0.5; epoch0 eligible; earliest tie",
        test_threshold_selection=False, eval_read=False,
        comparison_scope="Same boundary architecture and loss; only valid energy5 conditioning is enabled in the treatment. Not the historical frozen-CNN prosody experiment.",
        source_sha256={name: sha256(Path(__file__).with_name(name)) for name in
                       ("train_energy.py", "energy_adapter.py", "energy_cache.py", "energy_features.py", "train_four.py", "train_dual.py", "dual_resolution.py", "backbone.py", "finetune.py", "dual_features.py", "augmentation.py", "metrics20.py", "report_native20.py")})
    output.mkdir(parents=True)
    (output / "epochs").mkdir()
    save_json(output / "protocol.json", protocol)
    best, history, updates = None, [], 0
    started = time.monotonic()
    for epoch in range(args.epochs + 1):
        tick, losses, components = time.monotonic(), [], []
        if epoch:
            view.epoch = epoch
            model.train()
            if adapter is not None:
                adapter.train()
            last = time.monotonic()
            for index, batch in enumerate(loader):
                optimizer.zero_grad(set_to_none=True)
                final, outputs = forward_batch(model, adapter, batch, args.device)
                loss, parts = training_loss(final, outputs, batch, args.device, boundary_enabled)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                optimizer.step()
                updates += 1
                losses.append(float(loss.detach()))
                components.append(parts)
                if time.monotonic() - last >= 25:
                    progress = dict(arm=arm, phase="train", epoch=epoch, batches=index+1,
                                    batches_total=len(loader), updates=updates,
                                    loss=float(np.mean(losses)), elapsed_seconds=time.monotonic()-tick)
                    save_json(output / "progress.json", progress)
                    print(progress, flush=True)
                    last = time.monotonic()
        train_seconds = time.monotonic() - tick
        save_json(output / "progress.json", dict(arm=arm, phase="dev", epoch=epoch, updates=updates))
        records, scores, metadata = infer(model, adapter, dev, args.device)
        metrics = evaluate_dev(records, max_real_fpr=.05)
        metrics.pop("selection_score_j_percent", None)
        if metrics["real_fpr_percent"] > 5 + 1e-10:
            raise AssertionError("Development threshold exceeded the declared duration-FPR budget")
        value = float(metrics["event_f1_iou05_percent"])
        selected = dict(epoch=epoch, threshold=float(metrics["threshold"]), event_f1_iou05_percent=value)
        prefix = output / "epochs" / f"epoch_{epoch:02d}"
        payload = dict(schema=SCHEMA, backbone_state_dict=_cpu_state(model),
                       adapter_state_dict=_cpu_state(adapter), adapter_options=BASE_OPTIONS, energy_enabled=ARMS[arm],
                       model_config=model.config.to_dict(), energy_normalization=normalization, protocol=protocol, dev_metrics=metrics, **selected)
        torch.save(payload, prefix.with_suffix(".pt"))
        save_scores(prefix.with_name(prefix.name + "_scores20.npz"), scores, metadata)
        save_json(prefix.with_name(prefix.name + "_metrics.json"), metrics)
        if is_better(value, best):
            best = selected
            shutil.copyfile(prefix.with_suffix(".pt"), output / "best.pt")
            shutil.copyfile(prefix.with_name(prefix.name + "_scores20.npz"), output / "dev_scores20.npz")
            save_json(output / "dev_metrics.json", metrics)
        row = dict(arm=arm, epoch=epoch, updates=updates,
            loss=float(np.mean(losses)) if losses else None,
            loss_components={key:float(np.mean([item[key] for item in components])) for key in components[0]} if components else {},
            train_seconds=train_seconds, total_seconds=time.monotonic()-tick, best_epoch=best["epoch"],
            **{key: metrics[key] for key in ("threshold", "event_f1_iou05_percent",
                "short_event_recall_le320ms_percent", "real_fpr_percent")})
        history.append(row)
        save_json(output / "history.json", history)
        save_json(output / "progress.json", dict(phase="epoch_complete", **row))
        print(row, flush=True)
    complete = dict(status="complete", schema=SCHEMA, arm=arm, **best,
        checkpoint="best.pt", checkpoint_sha256=sha256(output / "best.pt"),
        initial_backbone_sha256=initial_digest, last_backbone_sha256=state_digest(model),
        initial_adapter_sha256=initial_adapter, initial_common_adapter_sha256=initial_common_adapter,
        energy_enabled=ARMS[arm], final_energy_weight_nonzero=int(torch.count_nonzero(adapter.energy_projection.weight)),
        trained_epochs=args.epochs, updates=updates, archived_epochs=args.epochs+1,
        protocol_sha256=sha256(output / "protocol.json"), eval_read=False,
        dev_scores20_sha256=sha256(output / "dev_scores20.npz"),
        dev_metrics_sha256=sha256(output / "dev_metrics.json"),
        total_seconds=time.monotonic()-started)
    save_json(output / "complete.json", complete)
    return complete


def freeze_eval_plan(output):
    """Freeze both completed arms before any evaluation cache is opened."""
    output = Path(output)
    frozen, shared, epoch_zero = {}, None, None
    for arm, enabled in ARMS.items():
        folder = output/arm
        complete, protocol = read_json(folder/'complete.json'), read_json(folder/'protocol.json')
        if (complete.get('status') != 'complete' or complete.get('schema') != SCHEMA
                or complete.get('arm') != arm or complete.get('energy_enabled') != enabled):
            raise ValueError(f'Incomplete or mismatched training arm: {arm}')
        for filename, key in (('best.pt', 'checkpoint_sha256'), ('protocol.json', 'protocol_sha256'),
                              ('dev_scores20.npz', 'dev_scores20_sha256'), ('dev_metrics.json', 'dev_metrics_sha256')):
            if sha256(folder/filename) != complete[key]:
                raise ValueError(f'Selected artifact changed: {arm}/{filename}')
        payload = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        metrics = read_json(folder/'dev_metrics.json')
        if (payload.get('schema') != SCHEMA or payload['energy_enabled'] != enabled
                or payload['adapter_options'] != BASE_OPTIONS
                or json_native(payload['protocol']) != protocol
                or payload['energy_normalization'] != protocol['energy_normalization']
                or payload['dev_metrics'] != metrics
                or payload['epoch'] != complete['epoch']
                or payload['event_f1_iou05_percent'] != complete['event_f1_iou05_percent']
                or payload['threshold'] != complete['threshold'] or payload['threshold'] != metrics['threshold']
                or not np.isfinite(payload['threshold']) or not 0 <= metrics['real_fpr_percent'] <= 5+1e-10
                or complete['trained_epochs'] != protocol['epochs']):
            raise ValueError(f'Weight, protocol, selection or threshold disagree: {arm}')
        history = read_json(folder/'history.json')
        expected_updates = int(np.ceil(protocol['train_recordings']/protocol['batch_size']))*protocol['epochs']
        if (len(history) != protocol['epochs']+1 or complete['archived_epochs'] != len(history)
                or complete['updates'] != expected_updates
                or [row['epoch'] for row in history] != list(range(len(history)))):
            raise ValueError(f'Incomplete training budget or history: {arm}')
        best = None
        for row in history:
            if is_better(row['event_f1_iou05_percent'], best):
                best = row
            if not (folder/'epochs'/f"epoch_{row['epoch']:02d}.pt").is_file():
                raise ValueError(f'Missing epoch archive: {arm}')
        if (best['epoch'] != complete['epoch'] or best['threshold'] != complete['threshold']
                or best['event_f1_iou05_percent'] != complete['event_f1_iou05_percent']
                or sha256(folder/'epochs'/f"epoch_{best['epoch']:02d}.pt") != complete['checkpoint_sha256']):
            raise ValueError(f'Earliest maximum dev event-F1 selection changed: {arm}')
        for filename, value in protocol['source_sha256'].items():
            if sha256(Path(__file__).with_name(filename)) != value:
                raise ValueError(f'Runtime source changed since training: {filename}')
        if sha256(protocol['normalization_file']) != protocol['normalization_sha256']:
            raise ValueError('Clean-training energy normalization changed')
        for prefix, folder_key in (('train_cache', 'train_cache'), ('dev_cache', 'dev_cache'),
                                   ('train_energy', 'train_energy_cache'), ('dev_energy', 'dev_energy_cache')):
            if sha256(Path(protocol[folder_key])/'complete.json') != protocol[prefix+'_metadata_sha256']:
                raise ValueError(f'Paired input cache changed: {arm}/{prefix}')
        conditions = {key:protocol[key] for key in (
            'seed', 'epochs', 'batch_size', 'model_config', 'learning_rate', 'weight_decay', 'gradient_clip', 'losses',
            'train_uids_sha256', 'dev_uids_sha256', 'train_recordings', 'dev_recordings',
            'train_cache_metadata_sha256', 'dev_cache_metadata_sha256', 'train_energy_metadata_sha256',
            'dev_energy_metadata_sha256', 'energy_normalization', 'normalization_sha256', 'source_sha256',
            'initial_backbone_sha256', 'initial_adapter_sha256', 'initial_common_adapter_sha256')}
        if shared is not None and shared != conditions:
            raise ValueError('Arms differ in paired initialization, data, normalization, losses, budget or source')
        shared = conditions
        with np.load(folder/'epochs'/'epoch_00_scores20.npz', allow_pickle=False) as archive:
            zero = {key:archive[key].copy() for key in archive.files}
        if epoch_zero is not None and (set(zero) != set(epoch_zero)
                or any(not np.array_equal(zero[key], epoch_zero[key]) for key in zero)):
            raise ValueError('Paired epoch-zero development scores or grids differ')
        epoch_zero = zero
        frozen[arm] = complete
    plan = dict(schema=SCHEMA, arms=frozen, calibration='Completed dev calibration only',
                test_threshold_selection=False, eval_previously_inspected=True,
                epoch_zero_scores_identical=True, paired_conditions=shared)
    target = output/'frozen_eval_plan.json'
    if target.exists():
        if read_json(target) != plan:
            raise ValueError('An existing frozen plan differs and cannot be replaced')
    else:
        save_json(target, plan)
    return frozen


def json_native(value):
    # Checkpoint config tuples and their JSON list representations are equivalent.
    import json
    return json.loads(json.dumps(value, allow_nan=False))


def evaluate_arms(args, dataset, frozen):
    """Use frozen dev thresholds; never call a calibration helper on eval."""
    if set(frozen) != set(ARMS) or dataset.split != 'eval' or dataset.channels != 1:
        raise ValueError('Both frozen arms and a clean evaluation cache are required')
    results = {}
    for arm in ARMS:
        folder = Path(args.output)/arm
        if (folder/'eval_metrics.json').exists():
            raise FileExistsError(f'Refusing to overwrite evaluation: {arm}')
        if sha256(folder/'best.pt') != frozen[arm]['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed before evaluation')
        payload = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        model = CNNBiGRU(ModelConfig(**payload['model_config'])).to(args.device)
        model.load_state_dict(payload['backbone_state_dict'], strict=True)
        adapter = make_energy_adapter(payload['energy_enabled'], payload['energy_normalization'],
                                      payload['protocol']['seed'], args.device)
        adapter.load_state_dict(payload['adapter_state_dict'], strict=True)
        records, scores, metadata = infer(model, adapter, dataset, args.device)
        report = evaluate_report(records, threshold=float(frozen[arm]['threshold']))
        save_scores(folder/'eval_scores20.npz', scores, metadata)
        save_json(folder/'eval_per_recording.json', report.pop('per_recording'))
        save_json(folder/'eval_metrics.json', report)
        results[arm] = report
        print(dict(arm=arm, event_f1=report['event_f1_iou05_percent'],
                   real_fpr=report['real_duration_fpr_percent']), flush=True)
    save_json(Path(args.output)/'comparison.json', dict(status='complete', schema=SCHEMA,
        metric_protocol='ps_native20_event_v2', models=results, frozen=frozen,
        eval_cache=str(dataset.folder), eval_cache_metadata_sha256=sha256(dataset.folder/'complete.json'),
        eval_energy_cache=str(dataset.energy_folder),
        eval_energy_cache_metadata_sha256=sha256(dataset.energy_folder/'complete.json'),
        test_threshold_selection=False, eval_previously_inspected=True))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=tuple(ARMS))
    for name in ('train-cache', 'dev-cache', 'train-energy', 'dev-energy', 'eval-cache', 'eval-energy'):
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
    training_inputs = (args.train_cache, args.dev_cache, args.train_energy, args.dev_energy)
    eval_inputs = (args.eval_cache, args.eval_energy)
    if args.evaluate or args.freeze_only:
        if args.arm or any(training_inputs) or (args.freeze_only and any(eval_inputs)):
            parser.error('Freeze/evaluate cannot accept training inputs; freeze cannot accept eval inputs')
        if args.evaluate and not all(eval_inputs):
            parser.error('Evaluation requires both --eval-cache and --eval-energy')
    elif not args.arm or not all(training_inputs) or any(eval_inputs):
        parser.error('Training needs one arm and train/dev acoustic+energy caches, with no eval inputs')
    setup(args.seed, args.threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; explicitly choose --device cpu')
    if args.evaluate or args.freeze_only:
        frozen = freeze_eval_plan(args.output)
        if not args.freeze_only:
            dataset = EnergyFeatureData(args.eval_cache, args.eval_energy, split='eval', seed=args.seed,
                                        config=model_config(), verify_hashes=True)
            evaluate_arms(args, dataset, frozen)
    else:
        train = EnergyFeatureData(args.train_cache, args.train_energy, split='train', seed=args.seed,
                                  config=model_config(), verify_hashes=True)
        dev = EnergyFeatureData(args.dev_cache, args.dev_energy, split='dev', seed=args.seed,
                                config=model_config(), verify_hashes=True)
        train_arm(args, train, dev)


if __name__ == '__main__':
    main()
