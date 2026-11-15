"""Train the residual head with clean input or three rotating channel views.

Only original PS train/dev caches are accepted. The frozen backbone is already
encoded in the caches. No generator groups, event-length weights, class
balancing or additional grouped supervision are used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from .cached_data import CachedData
from .head import ResidualFrameHead, load_head


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fit_original_statistics(train):
    """Use original train views only, including historical 8N storage cells."""
    total = 0
    sums = torch.zeros(256, dtype=torch.float64)
    squares = torch.zeros(256, dtype=torch.float64)
    for start in range(0, len(train), 2048):
        context = train.context[start:start+2048].reshape(-1, 256).double()
        sums += context.sum(0)
        squares += context.square().sum(0)
        total += len(context)
    return dict(context_mean=(sums/total).float(),
                context_scale=(squares/total-(sums/total).square()).clamp_min(1e-8).sqrt().float())


def initialize_model(seed, statistics, hidden_size=32):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ResidualFrameHead(hidden_size).float()
    with torch.no_grad():
        for name in ('context_mean', 'context_scale'):
            getattr(model, name).copy_(statistics[name])
    return model


def objective(model, batch):
    coarse, _, native = model(batch['native'], batch['context'], return_native=True)
    weights, fine_weights = batch['weights'].float(), batch['fine_weights'].float()
    coarse_loss = F.binary_cross_entropy_with_logits(coarse, batch['labels'].float(), reduction='none')
    fine_loss = F.binary_cross_entropy_with_logits(native, batch['fine_labels'].float(), reduction='none')
    return ((coarse_loss*weights).sum()/weights.sum().clamp_min(1e-8)
            + (fine_loss*fine_weights).sum()/fine_weights.sum().clamp_min(1e-8))


@torch.inference_mode()
def predict_native_scores(model, data, batch_size=2048):
    model.eval()
    values = []
    for start in range(0, len(data), batch_size):
        batch = data.batch(slice(start, start+batch_size))
        _, _, native = model(batch['native'], batch['context'], return_native=True)
        values.append(native.cpu().numpy().reshape(-1))
    result = np.concatenate(values).astype(np.float32, copy=False)
    if len(result) != len(data)*8 or not np.isfinite(result).all():
        raise ValueError('Incomplete or nonfinite native predictions')
    return result


class ChannelViews:
    def __init__(self, folder, train):
        self.folder = Path(folder).resolve()
        self.meta = read_json(self.folder/'complete.json')
        if self.meta['uids'] != train.uids or self.meta['lengths20'] != train.lengths20:
            raise ValueError('Channel data order/actual lengths differ from original training data')
        if self.meta.get('split', 'train') != 'train':
            raise ValueError('Channel views must belong to the train split')
        self.native = torch.from_numpy(np.load(self.folder/'native_channels.npy', mmap_mode='c', allow_pickle=False))
        self.context = torch.from_numpy(np.load(self.folder/'context_channels.npy', mmap_mode='c', allow_pickle=False))
        if self.native.shape != (3, len(train), 8) or self.context.shape != (3, len(train), 8, 256):
            raise ValueError('Expected three complete channel views of the training blocks')
        for start in range(0, len(train), 2048):
            sl = slice(start, start+2048)
            np.testing.assert_array_equal(self.native[0, sl], train.native[sl])
            np.testing.assert_array_equal(self.context[0, sl], train.context[sl])
            if not torch.isfinite(self.native[:, sl]).all() or not torch.isfinite(self.context[:, sl]).all():
                raise ValueError('Nonfinite channel cache')
        self.record_id = torch.repeat_interleave(torch.arange(len(train.uids)), torch.tensor(train.lengths))

    def assign_channels(self, seed, recording_count):
        assignment = np.arange(recording_count)%3
        np.random.default_rng(seed+20261113).shuffle(assignment)
        return torch.from_numpy(assignment)

    def batch(self, train, index, epoch, assignment):
        batch = train.batch(index)
        channels = (assignment[self.record_id[index]]+epoch-1)%3
        batch['native'] = self.native[channels, index]
        batch['context'] = self.context[channels, index]
        return batch


def train_one(train, dev, views, statistics, baseline, *, output, arm, seed):
    from .metrics20 import evaluate_dev

    folder = Path(output)/f'{arm}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=False)
    checkpoint = folder/'b_model.pt'
    model = initialize_model(seed, statistics)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    assignment = views.assign_channels(seed, len(train.uids)) if views is not None else None
    best = None
    best_scores = None
    history = []
    started = time.monotonic()
    for epoch in range(9):
        tick = time.monotonic()
        order_hash = None
        losses = []
        if epoch:
            model.train()
            order = torch.randperm(len(train), generator=generator)
            order_hash = hashlib.sha256(order.numpy().tobytes()).hexdigest()
            for index in order.split(2048):
                batch = (views.batch(train, index, epoch, assignment) if arm == 'aug_erm'
                         else train.batch(index))
                optimizer.zero_grad(set_to_none=True)
                loss = objective(model, batch)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                losses.append(float(loss.detach()))
        scores = predict_native_scores(model, dev)
        metric = baseline if epoch == 0 else evaluate_dev(dev.prediction_records(scores), max_real_fpr=.05)
        if epoch == 0:
            np.testing.assert_array_equal(scores, dev.native.numpy().reshape(-1))
        selection = metric['selection_score_j_percent']
        if best is None or selection > best['selection_score_j_percent']+1e-10:
            best = dict(epoch=epoch, threshold=metric['threshold'], selection_score_j_percent=selection)
            best_scores = scores.copy()
            temporary = checkpoint.with_suffix('.pt.tmp')
            torch.save(dict(state_dict=model.state_dict(), hidden_size=32,
                            fine_aux=True, freeze_gate=True, format='cnn_ps_residual_head_v1',
                            arm=arm, seed=seed, dev_metrics=metric, **best), temporary)
            temporary.replace(checkpoint)
            save_json(folder/'dev_metrics.json', metric)
        row = dict(epoch=epoch,
                   **{key: metric[key] for key in ('threshold', 'selection_score_j_percent',
                                                  'duration_f1_percent', 'event_f1_iou05_percent')},
                   loss_mean=float(np.mean(losses)) if losses else None,
                   seconds=time.monotonic()-tick, sampling_order_sha256=order_hash,
                   best_epoch=best['epoch'])
        history.append(row)
        save_json(folder/'history.json', history)
        save_json(folder/'progress.json', dict(arm=arm, seed=seed, **row))
        print(json.dumps(dict(arm=arm, seed=seed, epoch=epoch, selection_score_j_percent=selection)), flush=True)
    np.testing.assert_array_equal(predict_native_scores(load_head(checkpoint), dev), best_scores)
    np.savez_compressed(folder/'dev_scores20.npz', scores20=best_scores)
    complete = dict(status='complete', arm=arm, seed=seed, **best,
                    checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
                    reload_exact=True, trained_epochs=8, elapsed_seconds=time.monotonic()-started,
                    eval_read=False,
                    sampling_order_sha256=[row['sampling_order_sha256'] for row in history[1:]])
    save_json(folder/'complete.json', complete)
    return complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-cache', type=Path, required=True)
    parser.add_argument('--dev-cache', type=Path, required=True)
    parser.add_argument('--channel-cache', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arms', nargs='+', choices=('aug_erm', 'clean_erm'), default=['aug_erm'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[666, 667, 668])
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    if len(set(args.arms)) != len(args.arms) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Arms and seeds must not contain duplicates')
    if 'aug_erm' in args.arms and args.channel_cache is None:
        parser.error('--channel-cache is required for aug_erm')
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Output must be new or empty; existing results are not overwritten: {args.output}')
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    train, dev = CachedData(args.train_cache), CachedData(args.dev_cache)
    if train.role != 'train' or dev.role != 'dev':
        raise ValueError('Training only accepts caches explicitly marked train and dev')
    if set(train.uids) & set(dev.uids):
        raise ValueError('Train/dev recording IDs overlap')
    views = ChannelViews(args.channel_cache, train) if 'aug_erm' in args.arms else None
    statistics = fit_original_statistics(train)
    from .metrics20 import evaluate_dev
    baseline = evaluate_dev(dev.prediction_records(dev.native.numpy().reshape(-1)), max_real_fpr=.05)
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output/'epoch_zero_dev.json', baseline)
    protocol = dict(arms=args.arms, seeds=args.seeds, epochs=8, batch_size=2048,
                    learning_rate=.001, weight_decay=1e-4, gradient_clip=5,
                    backbone='Frozen original CNN-BiGRU', train_cache=str(train.folder),
                    dev_cache=str(dev.folder), train_metadata_sha256=sha256(train.folder/'complete.json'),
                    dev_metadata_sha256=sha256(dev.folder/'complete.json'),
                    channel_cache=str(views.folder) if views is not None else None,
                    channel_metadata_sha256=sha256(views.folder/'complete.json') if views is not None else None,
                    channels=['original', 'white_noise_20db', 'mild_eq'],
                    exposure='One view per recording per epoch, deterministic three-channel rotation',
                    loss='duration-weighted 160ms BCE + actual20ms duration-weighted BCE; both coefficients 1',
                    selection='PS dev only: real-duration FPR <= 5%, J=.5 durationF1+.5 eventF1; epoch0 eligible, earlier ties win',
                    normalization='Original train context only, including all historical 8N storage cells',
                    trainable_parameters=8257, init='Historical random draws preserved; unused gate not retained',
                    source_sha256=sha256(__file__), eval_read=False)
    save_json(args.output/'training_protocol.json', protocol)
    runs = []
    for seed in args.seeds:
        for arm in args.arms:
            runs.append(train_one(train, dev, views, statistics, baseline,
                                  output=args.output, arm=arm, seed=seed))
            save_json(args.output/'training_progress.json', dict(completed_runs=len(runs), runs=runs))
    save_json(args.output/'complete.json', dict(status='complete', count=len(runs), runs=runs))
    save_json(args.output/'training_complete.json', dict(status='complete', count=len(runs), runs=runs))


if __name__ == '__main__':
    main()
