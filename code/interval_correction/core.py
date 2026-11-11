"""Shared candidate generation and deterministic single-round selection.

Purpose: provide identical GT-free candidate pools to B/C/D.
Inputs: uncalibrated softmax spoof scores and original half-open intervals.
Outputs: immutable candidates, corrected intervals, operation traces.
Assumptions: seconds from audio start; scored horizon may be shorter than audio.
Usage: generate(record, Config()); select(record, candidates, utilities).
Limitations: finite local proposals cannot recover all missed events. Defaults
are illustrative development settings, not tuned or validated thresholds.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import math
from typing import Sequence
import numpy as np

Interval = tuple[float, float]
OPS = ('KEEP', 'DELETE', 'INSERT', 'SPLIT', 'MERGE', 'ADJUST')


@dataclass(frozen=True)
class Config:
    main_threshold: float = .5
    low_threshold: float = .3
    valley_threshold: float = .2
    merge_gap: float = .2
    boundary_radius: float = .32
    minimum_duration: float = .02
    delete_confidence: float = .6
    isolation_gap: float = .4
    merge_support: float = .35
    insert_support: float = .4
    minimum_gain: float = .01
    tiou: float = .5
    short_duration: float = .32
    risk_miss: float = 1.
    risk_fp: float = 1.
    risk_fp_seconds: float = .1
    risk_localization: float = 1.

    def __post_init__(self) -> None:
        if not all(math.isfinite(float(v)) for v in asdict(self).values()):
            raise ValueError('Nonfinite configuration')
        if not 0 < self.valley_threshold < self.low_threshold < self.main_threshold < 1:
            raise ValueError('Require valley < low < main within (0,1)')
        for name in ('delete_confidence', 'merge_support', 'insert_support', 'tiou'):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(name)
        for name in ('merge_gap', 'boundary_radius', 'minimum_duration', 'isolation_gap',
                     'minimum_gain', 'short_duration', 'risk_miss', 'risk_fp',
                     'risk_fp_seconds', 'risk_localization'):
            if getattr(self, name) < 0:
                raise ValueError(name)
        if self.minimum_duration <= 0:
            raise ValueError('minimum_duration must be positive')


def canonical(intervals: Sequence[Interval], horizon: float) -> tuple[Interval, ...]:
    """Clip, sort and union overlaps; touching events remain distinct."""
    if not math.isfinite(horizon) or horizon < 0:
        raise ValueError('Invalid horizon')
    values = []
    for start, end in intervals:
        if not all(map(math.isfinite, (start, end))) or end <= start:
            raise ValueError('Intervals require finite start < end')
        start, end = max(0., float(start)), min(horizon, float(end))
        if end > start:
            values.append((start, end))
    out: list[Interval] = []
    for start, end in sorted(values):
        if out and start < out[-1][1] - 1e-9:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return tuple(out)


def frame_span(start: float, end: float, dt: float, n: int) -> tuple[int, int]:
    """Frame cells with nonzero overlap with [start,end), robust at grid edges."""
    if dt <= 0 or not math.isfinite(dt) or end < start:
        raise ValueError('Invalid grid/span')
    return max(0, min(n, math.floor(start/dt+1e-9))), max(0, min(n, math.ceil(end/dt-1e-9)))


def runs(mask: np.ndarray, dt: float, horizon: float) -> tuple[Interval, ...]:
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return canonical([(float(a*dt), float(b*dt)) for a, b in edges.reshape(-1, 2)], horizon)


@dataclass
class Record:
    uid: str
    scores: np.ndarray
    dt: float
    seconds: float
    original: tuple[Interval, ...]
    source_group: str
    split: str
    truth: tuple[Interval, ...] | None = None
    words: list[dict] | None = None

    def __post_init__(self) -> None:
        self.scores = np.asarray(self.scores, dtype=np.float64)
        if self.scores.ndim != 1 or not len(self.scores) or not np.isfinite(self.scores).all():
            raise ValueError('Require nonempty finite 1D score array')
        if (self.scores < 0).any() or (self.scores > 1).any():
            raise ValueError('Scores must be explicit spoof softmax values')
        if not math.isfinite(self.dt) or self.dt <= 0 or not math.isfinite(self.seconds) or self.seconds <= 0:
            raise ValueError('Invalid time grid')
        if (len(self.scores)-1)*self.dt >= self.seconds + 1e-8:
            raise ValueError('Grid contains wholly padded frames')
        if not self.uid or not self.source_group or self.split not in ('train', 'dev', 'test'):
            raise ValueError('Require uid, source_group and train/dev/test split')
        self.original = canonical(self.original, self.horizon)
        if self.truth is not None:
            self.truth = canonical(self.truth, self.horizon)

    @property
    def horizon(self) -> float:
        return min(self.seconds, len(self.scores)*self.dt)


def support(record: Record, region: Interval) -> float:
    a, b = frame_span(*region, record.dt, len(record.scores))
    if b <= a:
        return 0.
    left = np.arange(a, b)*record.dt
    weights = np.maximum(0., np.minimum(left+record.dt, min(region[1], record.horizon)) - np.maximum(left, region[0]))
    return float(np.average(record.scores[a:b], weights=weights)) if weights.sum() else 0.


@dataclass(frozen=True)
class Candidate:
    operation: str
    region: Interval
    source_indices: tuple[int, ...]
    before: tuple[Interval, ...]
    after: tuple[Interval, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def generate(record: Record, cfg: Config) -> list[Candidate]:
    """Proposals use only scores/original intervals, never Record.truth/words."""
    original, horizon = record.original, record.horizon
    pool: list[Candidate] = [Candidate('KEEP', (0., horizon), (), original, original)]
    seen = set()

    def add(op: str, ids: tuple[int, ...], new: Sequence[Interval], region: Interval) -> None:
        after = canonical(new, horizon)
        if after == tuple(original[i] for i in ids) or (op != 'DELETE' and not after):
            return
        if any(e-s < cfg.minimum_duration-1e-9 for s, e in after):
            return
        # A local edit must not silently consume a different original event.
        if any(max(s, a) < min(e, b)-1e-9 for s, e in after
               for j, (a, b) in enumerate(original) if j not in ids):
            return
        region = (max(0., region[0]), min(horizon, region[1]))
        key = (op, ids, after)
        if region[1] > region[0] and key not in seen:
            seen.add(key)
            pool.append(Candidate(op, region, ids, tuple(original[i] for i in ids), after))

    for i, (start, end) in enumerate(original):
        add('DELETE', (i,), (), (start, end))
        a, b = frame_span(start, end, record.dt, len(record.scores))
        # Internal low valleys propose splitting; no GT or semantic inputs.
        for lo, hi in runs(record.scores[a:b] <= cfg.valley_threshold, record.dt, (b-a)*record.dt):
            lo, hi = lo+a*record.dt, hi+a*record.dt
            if start < lo and hi < end:
                add('SPLIT', (i,), ((start, lo), (hi, end)), (start, end))
        # Both expansion and shrinkage proposals, constrained by boundary radius.
        for threshold in (cfg.low_threshold, cfg.main_threshold, cfg.delete_confidence):
            for lo, hi in runs(record.scores >= threshold, record.dt, horizon):
                if max(lo, start) < min(hi, end) and abs(lo-start) <= cfg.boundary_radius+1e-9 and abs(hi-end) <= cfg.boundary_radius+1e-9:
                    add('ADJUST', (i,), ((lo, hi),), (min(start, lo), max(end, hi)))
        if i+1 < len(original):
            nxt = original[i+1]
            if nxt[0]-end <= cfg.merge_gap+1e-9:
                add('MERGE', (i, i+1), ((start, nxt[1]),), (start, nxt[1]))
    # Includes proposals whose maximum never crosses the main threshold.
    for start, end in runs(record.scores >= cfg.low_threshold, record.dt, horizon):
        if not any(max(start, a) < min(end, b)-1e-9 for a, b in original):
            add('INSERT', (), ((start, end),), (start, end))
    return pool


def apply(record: Record, candidate: Candidate) -> tuple[Interval, ...]:
    if candidate.operation == 'KEEP':
        return record.original
    return canonical([x for i, x in enumerate(record.original) if i not in candidate.source_indices]+list(candidate.after), record.horizon)


def conflict(a: Candidate, b: Candidate) -> bool:
    return bool(set(a.source_indices) & set(b.source_indices)) or max(a.region[0], b.region[0]) < min(a.region[1], b.region[1])-1e-9


def select(record: Record, candidates: list[Candidate], utilities: Sequence[float], minimum: float = 0.) -> tuple[tuple[Interval, ...], list[Candidate]]:
    """Greedy fixed-baseline edits; conflicting edits rejected; KEEP utility is 0."""
    if len(candidates) != len(utilities) or not np.isfinite(utilities).all():
        raise ValueError('Require one finite utility per candidate')
    chosen: list[Candidate] = []
    for i in sorted(range(len(candidates)), key=lambda j: (-utilities[j], j)):
        c = candidates[i]
        if c.operation != 'KEEP' and utilities[i] > max(0., minimum) and not any(conflict(c, p) for p in chosen):
            chosen.append(c)
    removed = {i for c in chosen for i in c.source_indices}
    after = canonical([x for i, x in enumerate(record.original) if i not in removed]+[x for c in chosen for x in c.after], record.horizon)
    return after, chosen


def rule_utilities(record: Record, pool: list[Candidate], cfg: Config) -> list[float]:
    """Independent evidence heuristics; utilities are not learned risk gains."""
    values = []
    for c in pool:
        utility = 0.
        if c.operation == 'DELETE':
            i = c.source_indices[0]
            start, end = c.before[0]
            gaps = ([start-record.original[i-1][1]] if i else []) + ([record.original[i+1][0]-end] if i+1 < len(record.original) else [])
            if (not gaps or min(gaps) >= cfg.isolation_gap) and support(record, (start, end)) < cfg.delete_confidence:
                utility = cfg.delete_confidence-support(record, (start, end))
        elif c.operation == 'MERGE':
            gap = (c.before[0][1], c.before[1][0])
            evidence = support(record, gap) if gap[1] > gap[0] else 1.
            if evidence >= cfg.merge_support:
                utility = evidence-cfg.merge_support+1e-6
        elif c.operation == 'SPLIT':
            gap = (c.after[0][1], c.after[1][0])
            evidence = support(record, gap)
            if evidence <= cfg.valley_threshold:
                utility = cfg.valley_threshold-evidence+1e-6
        elif c.operation == 'INSERT':
            evidence = support(record, c.after[0])
            if evidence >= cfg.insert_support:
                utility = evidence-cfg.insert_support+1e-6
        elif c.operation == 'ADJUST':
            # Local surrogate rewards spoof-positive cells and removes negative cells.
            def score(intervals: tuple[Interval, ...]) -> float:
                return sum((e-s)*(support(record, (s, e))-cfg.main_threshold) for s, e in intervals)
            utility = max(0., score(c.after)-score(c.before))
        values.append(utility)
    return values
