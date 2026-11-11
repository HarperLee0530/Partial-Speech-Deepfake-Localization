"""Small MLP for expected modification gain; frozen SAL is never instantiated.

Inputs: candidate acoustic summaries/resampled score and before/after masks;
optional GT-free word categories. Outputs: one real-valued gain per edit.
Assumptions: fixed feature schema saved in checkpoint. Usage: train(...), predict.
Limitations: no acoustic embeddings; temporal pooling can hide very short valleys.
Semantic features use transcript categories only, not outcome-derived priors.
"""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from .core import Candidate, Config, OPS, Record, apply, frame_span, generate
from .metrics import gain

CATEGORIES = ('negation', 'quantity', 'time', 'person', 'place', 'organization', 'other')
FEATURE_VERSION = 'acoustic_v1_semantic_categories_v1'
ACOUSTIC_DIM = 6+14+48
SEMANTIC_DIM = 10


def semantic_features(record: Record, region: tuple[float, float]) -> np.ndarray:
    """Missing/failed alignment => zeros plus explicit availability=0.

    Unknown words use category other, not an inferred spoof probability. Reject
    retrospective/prior fields rather than silently forwarding target statistics.
    """
    out = np.zeros(SEMANTIC_DIM, dtype=np.float32)
    if record.words is None:
        return out
    previous = 0.
    total = confidence = confidence_duration = 0.
    for word in record.words:
        allowed = {'text', 'start', 'end', 'confidence', 'normalized', 'types', 'primary_type', 'tag_sources', 'pos', 'lemma'}
        if set(word)-allowed:
            raise ValueError('D requires transcript fields only; outcome/prior fields forbidden')
        try:
            start, end = float(word['start']), float(word['end'])
            conf = word.get('confidence')
            if not np.isfinite([start, end]).all() or start < previous-1e-6 or start < 0 or end <= start or end > record.seconds+1e-6:
                return np.zeros_like(out)
            if conf is not None and (not np.isfinite(conf) or not 0 <= conf <= 1):
                return np.zeros_like(out)
            previous = end
        except (KeyError, TypeError, ValueError):
            return np.zeros_like(out)
        category = word.get('primary_type', 'other')
        if category not in CATEGORIES:
            return np.zeros_like(out)
        overlap = max(0., min(end, region[1])-max(start, region[0]))
        out[CATEGORIES.index(category)] += overlap
        total += overlap
        if conf is not None:
            confidence += overlap*conf
            confidence_duration += overlap
    width = region[1]-region[0]
    if total:
        out[:7] /= total
        out[7] = 1.
        out[8] = min(1., total/width)
        out[9] = confidence/confidence_duration if confidence_duration else 0.
    return out


def features(record: Record, candidate: Candidate, semantic: bool = False) -> np.ndarray:
    start, end = candidate.region
    a, b = frame_span(start, end, record.dt, len(record.scores))
    local = record.scores[a:b]
    if not len(local):
        raise ValueError('Empty candidate region')
    before, after = candidate.before, candidate.after
    gap = before[1][0]-before[0][1] if len(before) == 2 else 0.
    summary = [local.mean(), local.max(), local.min(), local.std(), local[0], local[-1], end-start,
               sum(e-s for s, e in before), sum(e-s for s, e in after), len(before), len(after),
               len(record.original), len(apply(record, candidate)), gap]
    # Fixed 16-point representation includes acoustic scores and binary masks.
    points = start+(np.arange(16)+.5)*(end-start)/16
    idx = np.minimum((points/record.dt).astype(int), len(record.scores)-1)
    masks = [np.array([any(s <= t < e for s, e in intervals) for t in points], dtype=float) for intervals in (before, after)]
    x = np.r_[[float(candidate.operation == op) for op in OPS], summary, record.scores[idx], *masks].astype(np.float32)
    if semantic:
        x = np.r_[x, semantic_features(record, candidate.region)].astype(np.float32)
    if not np.isfinite(x).all() or len(x) != ACOUSTIC_DIM+(SEMANTIC_DIM if semantic else 0):
        raise ValueError('Invalid feature schema')
    return x


class GainMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def training_arrays(records: list[Record], cfg: Config, semantic: bool) -> tuple[np.ndarray, np.ndarray]:
    if not records:
        raise ValueError('Empty partition')
    xs, ys = [], []
    for r in records:
        if r.truth is None:
            raise ValueError('Training/validation labels unavailable')
        for c in generate(r, cfg):
            xs.append(features(r, c, semantic))
            ys.append(gain(r, c, cfg))
    return np.stack(xs), np.asarray(ys, dtype=np.float32)


def train(train_records: list[Record], dev_records: list[Record], cfg: Config, output: Path,
          semantic: bool = False, seed: int = 888, epochs: int = 40, patience: int = 6,
          batch_size: int = 256, learning_rate: float = .001, hidden: int = 64) -> dict:
    """CPU deterministic first version, early stopping on independent dev loss.

    Keeps all positive/negative/zero/KEEP labels. Output must be a fresh directory.
    Only correction-training IDs fit feature normalization and gradient weights.
    """
    if epochs < 1 or patience < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError('Invalid training parameters')
    if any(r.split != 'train' for r in train_records) or any(r.split == 'test' for r in dev_records):
        raise ValueError('Official test cannot train/select model')
    if {r.uid for r in train_records} & {r.uid for r in dev_records} or {r.source_group for r in train_records} & {r.source_group for r in dev_records}:
        raise ValueError('Training/validation source leakage')
    if semantic and (not any(r.words for r in train_records) or not any(r.words for r in dev_records)):
        raise ValueError('D requires actual training and validation transcripts')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    x, y = training_arrays(train_records, cfg, semantic)
    vx, vy = training_arrays(dev_records, cfg, semantic)
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-5)
    x, vx = torch.from_numpy((x-mean)/std), torch.from_numpy((vx-mean)/std)
    y, vy = torch.from_numpy(y), torch.from_numpy(vy)
    model = GainMLP(x.shape[1], hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.SmoothL1Loss()
    output.mkdir(parents=True, exist_ok=False)
    best, stale = float('inf'), 0
    history = []
    metadata = dict(config=asdict(cfg), semantic=semantic, feature_version=FEATURE_VERSION,
                    input_dim=x.shape[1], hidden=hidden, seed=seed, epochs=epochs, patience=patience,
                    batch_size=batch_size, learning_rate=learning_rate, train_ids=[r.uid for r in train_records],
                    validation_ids=[r.uid for r in dev_records], train_groups=sorted({r.source_group for r in train_records}),
                    validation_groups=sorted({r.source_group for r in dev_records}),
                    torch_version=str(torch.__version__), numpy_version=np.__version__,
                    label_counts={k:int(v) for k, v in [('positive', (y>0).sum()), ('negative', (y<0).sum()), ('zero', (y==0).sum())]})
    for epoch in range(epochs):
        model.train()
        total = 0.
        for batch in torch.randperm(len(x)).split(batch_size):
            optimizer.zero_grad()
            loss = loss_fn(model(x[batch]), y[batch])
            loss.backward()
            optimizer.step()
            total += loss.item()*len(batch)
        model.eval()
        with torch.inference_mode():
            val = loss_fn(model(vx), vy).item()
        history.append(dict(epoch=epoch+1, train_loss=total/len(x), validation_loss=val))
        if val < best:
            best, stale = val, 0
            torch.save(dict(state_dict=model.state_dict(), mean=mean.tolist(), std=std.tolist(), metadata=metadata), output/'model.pt')
        else:
            stale += 1
        with (output/'training.jsonl').open('a', encoding='utf8') as stream:
            stream.write(json.dumps(history[-1])+'\n')
        if stale >= patience:
            break
    (output/'config.json').write_text(json.dumps(metadata, indent=2), encoding='utf8')
    return dict(best_validation_loss=best, epochs_completed=len(history), training_candidates=len(x), validation_candidates=len(vx))


class Predictor:
    def __init__(self, path: Path, cfg: Config, semantic: bool) -> None:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.meta = checkpoint['metadata']
        if self.meta['feature_version'] != FEATURE_VERSION or self.meta['semantic'] != semantic or self.meta['config'] != asdict(cfg):
            raise ValueError('Checkpoint feature/config mismatch; do not silently change candidate or risk settings')
        self.semantic = semantic
        self.mean = np.asarray(checkpoint['mean'], dtype=np.float32)
        self.std = np.asarray(checkpoint['std'], dtype=np.float32)
        self.model = GainMLP(self.meta['input_dim'], self.meta['hidden'])
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.eval()

    def predict(self, record: Record, pool: list[Candidate]) -> list[float]:
        x = np.stack([features(record, c, self.semantic) for c in pool])
        with torch.inference_mode():
            values = self.model(torch.from_numpy((x-self.mean)/self.std)).tolist()
        # KEEP is known exactly, not whatever regression error the network makes.
        return [0. if c.operation == 'KEEP' else float(v) for c, v in zip(pool, values)]
