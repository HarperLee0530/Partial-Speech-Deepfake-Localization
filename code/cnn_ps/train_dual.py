"""Paired end-to-end PS experiments, with dev-only event-F1 selection.

All arms start at the same CNN checkpoint and receive the same waveform views,
recording order, number of updates and main duration-weighted 20/160 ms losses.
The augmentation and original backbone arithmetic are unchanged. Extra losses
are part of the explicitly declared architectural treatments, not hidden changes.
Evaluation is a separate command, after every arm's checkpoint is frozen.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .backbone import load_backbone
from .finetune import fine_collate, objective as baseline_objective, save_scores
from .io import save_json, read_json, sha256, manifest_hash
from .metrics20 import evaluate_dev
from .report_native20 import evaluate_report


ARMS = {
    'baseline': None,
    'dual_fusion': dict(learned_pool=False, boundary_head=False,
                        relative_position=False, reliability_gate=False),
    'boundary_dual': dict(learned_pool=True, boundary_head=True,
                          relative_position=False, reliability_gate=False),
    'boundary_position': dict(learned_pool=True, boundary_head=True,
                              relative_position=True, reliability_gate=False),
    'boundary_reliability': dict(learned_pool=True, boundary_head=True,
                                 relative_position=False, reliability_gate=True),
    'boundary_both': dict(learned_pool=True, boundary_head=True,
                          relative_position=True, reliability_gate=True),
}


def boundary_targets(labels, weights):
    """A transition belongs to frame t; no invented transition at recording start."""
    if labels.ndim != 1 or weights.shape != labels.shape:
        raise ValueError('Expected one aligned recording')
    target, pair = torch.zeros_like(labels), torch.zeros_like(weights)
    target[1:] = (labels[1:] != labels[:-1]).to(labels.dtype)
    pair[1:] = torch.minimum(weights[1:], weights[:-1])
    target = torch.where(pair > 0, target, torch.zeros_like(target))
    return target, pair


def within_segment_consistency(logits, labels, weights):
    """Only adjacent valid cells in the same truth segment can be regularized."""
    if logits.ndim != 1 or logits.shape != labels.shape or weights.shape != labels.shape:
        raise ValueError('Expected aligned one-dimensional logits, labels and weights')
    pair = torch.minimum(weights[1:], weights[:-1]) * (labels[1:] == labels[:-1])
    probabilities = logits.sigmoid()
    return ((probabilities[1:] - probabilities[:-1]).square() * pair).sum() / pair.sum().clamp_min(1e-8)


def weighted_bce(logits, labels, weights):
    valid = weights > 0
    if not valid.any():
        return logits.sum() * 0
    return (F.binary_cross_entropy_with_logits(logits[valid], labels[valid], reduction='none')
            * weights[valid]).sum() / weights[valid].sum()


def forward_batch(model, adapter, batch, device):
    native, context = model(batch['features'].to(device), batch['lengths'], return_features=True)
    if adapter is None:
        return native, []
    outputs, corrected = [], []
    for i, length in enumerate(batch['lengths'].tolist()):
        result = adapter(context[i, :length], native[i, :length], batch['fine_weights'][i, :length].to(device))
        outputs.append(result)
        corrected.append(F.pad(result['final_logits'], (0, native.shape[1] - length)))
    return torch.stack(corrected), outputs


def training_loss(final, outputs, batch, device, boundary_enabled):
    # Exactly the original objective: final160 is mean of eight stored logits;
    # only its BCE is weighted by actual coarse-bin duration. The independent
    # coarse branch instead masks tails while forming its feature tokens.
    main = baseline_objective(final, batch, device)
    auxiliary = main.new_zeros(())
    boundary = main.new_zeros(())
    consistency = main.new_zeros(())
    if outputs:
        all_coarse = torch.cat([r['coarse_logits'] for r in outputs])
        coarse_targets = torch.cat([batch['labels'][i, :len(r['coarse_logits'])] for i, r in enumerate(outputs)]).to(device)
        coarse_weights = torch.cat([batch['weights'][i, :len(r['coarse_logits'])] for i, r in enumerate(outputs)]).to(device)
        auxiliary = weighted_bce(all_coarse, coarse_targets, coarse_weights)
    if boundary_enabled:
        boundary_logits, boundary_labels, boundary_weights = [], [], []
        same_numerator, same_denominator = main.new_zeros(()), main.new_zeros(())
        for i, r in enumerate(outputs):
            n = len(r['final_logits'])
            y = batch['fine_labels'][i, :n].to(device)
            w = batch['fine_weights'][i, :n].to(device)
            target, pair = boundary_targets(y, w)
            boundary_logits.append(r['boundary_logits'])
            boundary_labels.append(target)
            boundary_weights.append(pair)
            inside = torch.minimum(w[1:], w[:-1]) * (y[1:] == y[:-1])
            probability = r['final_logits'].sigmoid()
            same_numerator = same_numerator + ((probability[1:] - probability[:-1]).square() * inside).sum()
            same_denominator = same_denominator + inside.sum()
        boundary = weighted_bce(torch.cat(boundary_logits), torch.cat(boundary_labels), torch.cat(boundary_weights))
        consistency = same_numerator / same_denominator.clamp_min(1e-8)
    total = main + .5 * auxiliary + .1 * boundary + .05 * consistency
    return total, dict(main=float(main.detach()), coarse_aux=float(auxiliary.detach()),
                       boundary=float(boundary.detach()), consistency=float(consistency.detach()))


@torch.inference_mode()
def infer(model, adapter, dataset, device):
    model.eval()
    if adapter is not None:
        adapter.eval()
    records, stored = [], []
    metadata = dict(uids=dataset.uids, lengths=[], lengths20=[], durations=[])
    # Single-record inference exactly preserves the established reference path.
    for batch in DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=fine_collate):
        scores, _ = forward_batch(model, adapter, batch, device)
        uid, duration = batch['uids'][0], float(batch['durations'][0])
        n = int(batch['lengths'][0])
        score = scores[0, :n].float().cpu().numpy().copy()
        if not np.isfinite(score).all():
            raise ValueError('Non-finite inference score')
        records.append(dict(uid=uid, duration=duration, labels20=dataset.fine[uid], scores20=score))
        stored.append(score)
        metadata['lengths'].append(n // 8)
        metadata['lengths20'].append(len(dataset.fine[uid]))
        metadata['durations'].append(duration)
    return records, np.concatenate(stored), metadata


def setup(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_adapter(normalization, options, seed, device):
    if options is None:
        return None
    from .dual_resolution import DualResolutionAdapter
    # Adapter construction must not advance the backbone's dropout RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        with np.load(normalization, allow_pickle=False) as stats:
            module = DualResolutionAdapter(stats['mean'], stats['std'], variant='fusion', **options)
    return module.to(device)


def train_arm(args, arm, train, dev):
    setup(args.seed, args.threads)
    output = args.output / arm
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    output.mkdir(parents=True)
    model, _ = load_backbone(args.checkpoint, device=args.device)
    model.requires_grad_(True)
    adapter = make_adapter(args.normalization, ARMS[arm], args.seed, args.device)
    parameters = list(model.parameters()) + ([] if adapter is None else list(adapter.parameters()))
    groups = [dict(params=model.parameters(), lr=1e-4)]
    if adapter is not None:
        groups.append(dict(params=adapter.parameters(), lr=1e-3))
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    train.epoch = 0
    loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        collate_fn=fine_collate, generator=torch.Generator().manual_seed(args.seed))
    protocol = dict(arm=arm, options=ARMS[arm], seed=args.seed, epochs=args.epochs,
        batch_size=args.batch_size, backbone_lr=1e-4, adapter_lr=1e-3,
        weight_decay=1e-4, gradient_clip=5, all_backbone_parameters_trainable=True,
        original_checkpoint=str(args.checkpoint), original_checkpoint_sha256=sha256(args.checkpoint),
        train_cache=str(args.train_cache), dev_cache=str(args.dev_cache),
        train_uids_sha256=manifest_hash(train.uids), dev_uids_sha256=manifest_hash(dev.uids),
        normalization=str(args.normalization), normalization_sha256=sha256(args.normalization),
        train_recordings=len(train), dev_recordings=len(dev),
        backbone_parameters=sum(p.numel() for p in model.parameters()),
        added_parameters=0 if adapter is None else sum(p.numel() for p in adapter.parameters()),
        source_sha256=sha256(__file__),
        model_source_sha256=sha256(Path(__file__).with_name('dual_resolution.py')),
        selection='clean dev actual20 duration FPR<=5%; maximize dev event F1; epoch0 eligible; earliest ties',
        losses=dict(main20=1., main160=1., coarse_aux=0. if adapter is None else .5,
                    boundary=.1 if ARMS[arm] and ARMS[arm]['boundary_head'] else 0.,
                    within_segment=.05 if ARMS[arm] and ARMS[arm]['boundary_head'] else 0.),
        coarse_main_pool='unweighted mean8 stored logits, identical to retained CNN training',
        augmentation='unchanged deterministic clean/noise20dB/EQ, balanced UID assignment and epoch rotation',
        comparison_scope='dual modules include auxiliary coarse supervision; boundary package combines three changes',
        teammate_original_code_available=False, eval_read=False)
    save_json(output / 'protocol.json', protocol)
    initial_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best, history = None, []
    start = time.monotonic()
    for epoch in range(args.epochs + 1):
        tick, losses, components = time.monotonic(), [], []
        if epoch:
            train.epoch = epoch
            model.train()
            if adapter is not None:
                adapter.train()
            last = time.monotonic()
            for index, batch in enumerate(loader):
                optimizer.zero_grad(set_to_none=True)
                final, outputs = forward_batch(model, adapter, batch, args.device)
                loss, parts = training_loss(final, outputs, batch, args.device,
                                            bool(ARMS[arm] and ARMS[arm]['boundary_head']))
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                components.append(parts)
                if time.monotonic() - last >= 25:
                    progress = dict(arm=arm, phase='train', epoch=epoch, batches=index+1,
                                    batches_total=len(loader), loss=float(np.mean(losses)),
                                    elapsed=time.monotonic()-tick)
                    save_json(output / 'progress.json', progress)
                    print(progress, flush=True)
                    last = time.monotonic()
        train_seconds = time.monotonic() - tick
        save_json(output / 'progress.json', dict(arm=arm, phase='dev', epoch=epoch))
        records, scores, metadata = infer(model, adapter, dev, args.device)
        metric = evaluate_dev(records, max_real_fpr=.05)
        metric.pop('selection_score_j_percent', None)
        value = metric['event_f1_iou05_percent']
        if epoch == 0:
            save_json(output / 'epoch_zero_dev.json', metric)
            save_scores(output / 'epoch_zero_scores20.npz', scores, metadata)
        if best is None or value > best['event_f1_iou05_percent'] + 1e-10:
            best = dict(epoch=epoch, threshold=metric['threshold'], event_f1_iou05_percent=value)
            payload = dict(backbone_state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                adapter_state_dict=None if adapter is None else {k:v.detach().cpu().clone() for k,v in adapter.state_dict().items()},
                model_config=model.config.to_dict(), protocol=protocol, dev_metrics=metric, **best)
            torch.save(payload, output / 'best.pt')
            save_scores(output / 'dev_scores20.npz', scores, metadata)
            save_json(output / 'dev_metrics.json', metric)
        row = dict(arm=arm, epoch=epoch, loss=float(np.mean(losses)) if losses else None,
            loss_components={k:float(np.mean([c[k] for c in components])) for k in components[0]} if components else {},
            train_seconds=train_seconds, total_seconds=time.monotonic()-tick,
            best_epoch=best['epoch'], threshold=metric['threshold'], event_f1_iou05_percent=value,
            short_event_recall_le320ms_percent=metric['short_event_recall_le320ms_percent'],
            real_fpr_percent=metric['real_fpr_percent'])
        history.append(row)
        save_json(output / 'history.json', history)
        save_json(output / 'progress.json', dict(phase='epoch_complete', **row))
        print(row, flush=True)
    changed = [k for k,v in model.state_dict().items() if not torch.equal(v.detach().cpu(), initial_state[k])]
    save_json(output / 'complete.json', dict(status='complete', **best,
        checkpoint='best.pt', checkpoint_sha256=sha256(output/'best.pt'),
        backbone_changed_tensors=changed, trained_epochs=args.epochs,
        total_seconds=time.monotonic()-start, eval_read=False))


def freeze_eval_plan(args):
    # Freeze every selected checkpoint/threshold before touching test records.
    frozen = {}
    for arm in args.arms:
        path = args.output / arm
        complete = read_json(path/'complete.json')
        if complete['status'] != 'complete' or sha256(path/'best.pt') != complete['checkpoint_sha256']:
            raise ValueError('Completed checkpoint hash mismatch')
        frozen[arm] = complete
    save_json(args.output/'frozen_eval_plan.json', dict(arms=frozen, calibration='dev only', eval_previously_inspected=True))
    return frozen


def evaluate_arms(args, dataset, frozen):
    from .backbone import CNNBiGRU, ModelConfig
    results = {}
    for arm in args.arms:
        path = args.output / arm
        if (path/'eval_metrics.json').exists():
            raise FileExistsError('Refusing to replace completed evaluation')
        payload = torch.load(path/'best.pt', map_location='cpu', weights_only=False)
        model = CNNBiGRU(ModelConfig(**payload['model_config'])).to(args.device)
        model.load_state_dict(payload['backbone_state_dict'], strict=True)
        adapter = make_adapter(args.normalization, ARMS[arm], args.seed, args.device)
        if adapter is not None:
            adapter.load_state_dict(payload['adapter_state_dict'], strict=True)
        records, scores, metadata = infer(model, adapter, dataset, args.device)
        threshold = float(frozen[arm]['threshold'])
        if float(payload['threshold']) != threshold:
            raise ValueError('Frozen threshold mismatch')
        report = evaluate_report(records, threshold)
        save_scores(path/'eval_scores20.npz', scores, metadata)
        save_json(path/'eval_per_recording.json', report.pop('per_recording'))
        save_json(path/'eval_metrics.json', report)
        results[arm] = report
        print(dict(arm=arm, event_f1=report['event_f1_iou05_percent'],
                   short_recall=report['short_event_recall_le320ms_percent'],
                   real_fpr=report['real_fpr_percent']), flush=True)
    save_json(args.output/'comparison.json', dict(protocol='ps_native20_event_v2',seed=args.seed,
        status='complete', models=results, frozen=frozen, eval_previously_inspected=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--normalization', type=Path, required=True)
    parser.add_argument('--train-cache', type=Path)
    parser.add_argument('--dev-cache', type=Path)
    parser.add_argument('--eval-cache', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', choices=list(ARMS), default=list(ARMS))
    parser.add_argument('--seed', type=int, default=666)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    from .dual_features import FeatureData
    setup(args.seed, args.threads)
    args.output.mkdir(parents=True, exist_ok=True)
    from .backbone import ModelConfig
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ModelConfig(**source['model_config'])
    del source
    if args.evaluate:
        frozen = freeze_eval_plan(args)
        dataset = FeatureData(args.eval_cache, seed=args.seed, split='eval', config=config, verify_hashes=True)
        evaluate_arms(args, dataset, frozen)
    else:
        if args.eval_cache:
            parser.error('Training cannot receive an eval cache')
        train = FeatureData(args.train_cache, seed=args.seed, split='train', config=config, verify_hashes=True)
        dev = FeatureData(args.dev_cache, seed=args.seed, split='dev', config=config, verify_hashes=True)
        if train.channels != 3 or dev.channels != 1:
            raise ValueError('Paired experiment requires three train channels and clean dev')
        if set(train.uids) & set(dev.uids):
            raise ValueError('Train/dev overlap')
        for arm in args.arms:
            train_arm(args, arm, train, dev)


if __name__ == '__main__':
    main()
