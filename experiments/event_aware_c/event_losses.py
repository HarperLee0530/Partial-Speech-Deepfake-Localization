"""Event-normalized auxiliary objectives on the existing native 20 ms logits."""
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F


@dataclass
class Regions:
    rows: torch.Tensor
    frames: torch.Tensor
    weights: torch.Tensor
    truth: torch.Tensor | None

    def gather(self, logits):
        device = logits.device
        return logits[self.rows.to(device), self.frames.to(device)], self.weights.to(device)


def pack(spans, weights, labels=None):
    if not spans:
        return None
    width = max(end - start for _, start, end in spans)
    frames = np.zeros((len(spans), width), dtype=np.int64)
    durations = np.zeros(frames.shape, dtype=np.float32)
    truth = np.zeros_like(durations) if labels is not None else None
    for i, (row, start, end) in enumerate(spans):
        n = end - start
        frames[i, :n] = np.arange(start, end)
        durations[i, :n] = weights[row, start:end]
        if truth is not None:
            truth[i, :n] = labels[row, start:end]
    return Regions(torch.tensor([row for row, _, _ in spans])[:, None],
                   torch.from_numpy(frames), torch.from_numpy(durations),
                   torch.from_numpy(truth) if truth is not None else None)


def prepare_regions(labels, weights, *, c1, c2):
    """Collate ground-truth intervals on CPU, with no forward-pass dependency."""
    labels = labels.numpy()
    weights = weights.numpy()
    windows, gaps = [], []
    for row, (label, weight) in enumerate(zip(labels, weights)):
        valid_end = np.count_nonzero(weight > 0)
        if not (np.all(weight[:valid_end] > 0) and np.all(weight[valid_end:] == 0)):
            raise ValueError("Expected contiguous valid durations followed by zero padding")
        edges = np.flatnonzero(np.diff(np.r_[False, label[:valid_end] == 1, False]))
        events = list(zip(edges[::2], edges[1::2]))
        for i, (start, end) in enumerate(events):
            previous = events[i - 1][1] if i else 0
            left_limit = (previous + start + 1) // 2 if i else 0
            right_limit = (end + events[i + 1][0] + 1) // 2 if i + 1 < len(events) else valid_end
            if c1:
                windows.append((row, max(left_limit, start - 4), min(right_limit, end + 4)))
            if c2 and i:
                gaps.append((row, previous, start))
    return pack(windows, weights, labels), pack(gaps, weights)


def event_losses(logits, regions):
    """Compute the two mean losses with one device-side tensor operation per region type."""
    windows, gaps = regions
    overlap = separation = logits.new_zeros(())
    if windows is not None:
        values, weight = windows.gather(logits)
        p = values.sigmoid()
        truth = windows.truth.to(logits.device)
        intersection = (weight * p * truth).sum(dim=1)
        union = (weight * (truth + (1 - truth) * p)).sum(dim=1)
        overlap = (1 - intersection / union).mean()
    if gaps is not None:
        values, weight = gaps.gather(logits)
        temperature = .25
        log_weight = (weight / weight.sum(dim=1, keepdim=True)).log()
        soft_minimum = -temperature * torch.logsumexp(log_weight - values / temperature, dim=1)
        separation = F.softplus(soft_minimum + 1).mean()
    return overlap, separation
