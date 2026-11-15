"""Original frozen features and actual labels, with no generator metadata.

Required files are six .npy arrays and complete.json. Both the published
portable preparation command and the historical original-cache directories
use this schema. Extra historical event/splice arrays are never loaded.
"""
import json
from pathlib import Path

import numpy as np
import torch


ARRAY_NAMES = ('native', 'context', 'labels', 'weights', 'fine_labels', 'fine_weights')


class CachedData:
    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.meta = json.loads((self.folder/'complete.json').read_text(encoding='utf-8-sig'))
        self.role = self.meta.get('split')
        for name in ARRAY_NAMES:
            array = np.load(self.folder/f'{name}.npy', mmap_mode='c', allow_pickle=False)
            setattr(self, name, torch.from_numpy(array))
        for name in ('uids', 'lengths', 'lengths20', 'durations'):
            setattr(self, name, self.meta[name])
        self.offsets = np.r_[0, np.cumsum(self.lengths, dtype=np.int64)]
        self._validate()

    def _validate(self):
        n = len(self.labels)
        shapes = dict(native=(n, 8), context=(n, 8, 256), labels=(n,), weights=(n,),
                      fine_labels=(n, 8), fine_weights=(n, 8))
        if not n or len(self.uids) != len(set(self.uids)):
            raise ValueError('Empty cache or duplicate recording IDs')
        if not (len(self.lengths) == len(self.lengths20) == len(self.durations) == len(self.uids)):
            raise ValueError('Recording metadata lengths disagree')
        if int(self.offsets[-1]) != n:
            raise ValueError('Block offsets do not cover cache arrays')
        for uid, blocks, fine, duration in zip(self.uids, self.lengths, self.lengths20, self.durations):
            if int(blocks) != blocks or int(fine) != fine or not 0 < fine <= blocks*8:
                raise ValueError(f'Invalid frame counts for {uid}')
            if not np.isfinite(duration) or duration <= 0 or abs(np.ceil(duration/.02)-fine) > 1:
                raise ValueError(f'Invalid actual duration for {uid}')
        for name, shape in shapes.items():
            if tuple(getattr(self, name).shape) != shape:
                raise ValueError(f'{name} must have shape {shape}')
        # Bounded checks avoid materializing the large float16 context matrix.
        for start in range(0, n, 2048):
            batch = self.batch(slice(start, start+2048))
            for name, values in batch.items():
                if not torch.isfinite(values).all():
                    raise ValueError(f'Nonfinite cache array: {name}')
            for name in ('labels', 'fine_labels'):
                values = batch[name]
                if not ((values == 0) | (values == 1)).all():
                    raise ValueError(f'{name} must use 0=real, 1=fake')
            if (batch['weights'] < 0).any() or (batch['fine_weights'] < 0).any():
                raise ValueError('Negative duration weights')
        if self.fine_weights.sum() <= 0 or self.weights.sum() <= 0:
            raise ValueError('Cache contains no actual duration')

    def __len__(self):
        return len(self.labels)

    def batch(self, index):
        return {name: getattr(self, name)[index] for name in ARRAY_NAMES}

    def prediction_records(self, flat_native_scores):
        scores = np.asarray(flat_native_scores).reshape(-1)
        if len(scores) != len(self)*8 or not np.isfinite(scores).all():
            raise ValueError('Native scores do not align with prepared blocks')
        labels = self.fine_labels.numpy().reshape(-1)
        rows = []
        for i, uid in enumerate(self.uids):
            start = int(self.offsets[i])*8
            stop = start+int(self.lengths20[i])
            rows.append(dict(uid=uid, labels20=labels[start:stop], scores20=scores[start:stop],
                             duration=self.durations[i]))
        return rows
