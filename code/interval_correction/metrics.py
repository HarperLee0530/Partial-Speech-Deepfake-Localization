"""Research event metrics, risk labels and explicitly offline oracle diagnostics.

Inputs: half-open second intervals; GT only for supervision/evaluation.
Outputs: one-to-one tIoU metrics, risk and candidate coverage/upper bounds.
Assumptions: same saved grid for truth/prediction. Usage: evaluate(pred,gt,cfg).
Limitations: not official SAL frame EER/F1. Matching follows visualization's
maximum-cardinality-then-summed-tIoU Hungarian rule at a fixed threshold.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
from scipy.optimize import linear_sum_assignment
from .core import Candidate, Config, Interval, Record, apply, canonical, conflict


def match(prediction: tuple[Interval, ...], truth: tuple[Interval, ...], threshold: float) -> tuple[np.ndarray, list[tuple[int, int]]]:
    p, t = np.asarray(prediction).reshape(-1, 2), np.asarray(truth).reshape(-1, 2)
    overlap = np.maximum(0., np.minimum(p[:, None, 1], t[None, :, 1])-np.maximum(p[:, None, 0], t[None, :, 0]))
    union = (p[:, 1]-p[:, 0])[:, None]+(t[:, 1]-t[:, 0])[None, :]-overlap
    iou = overlap/union
    if not len(p) or not len(t):
        return iou, []
    eligible = iou >= threshold-1e-8
    reward = np.where(eligible, min(len(p), len(t))+1+iou, 0.)
    rr, cc = linear_sum_assignment(-reward)
    return iou, [(int(i), int(j)) for i, j in zip(rr, cc) if eligible[i, j]]


def evaluate(prediction: tuple[Interval, ...], truth: tuple[Interval, ...], cfg: Config) -> dict:
    iou, pairs = match(prediction, truth, cfg.tiou)
    tp = len(pairs)
    precision = tp/len(prediction) if prediction else 0.
    recall = tp/len(truth) if truth else 0.
    # Count duration outside the union of ALL truth, including matched predictions.
    # Overlapping truth is canonicalized solely for duration, not event matching.
    horizon = max([e for _, e in (*prediction, *truth)], default=0.)
    union_truth = canonical(truth, horizon)
    fp_seconds = sum(e-s-sum(max(0., min(e, b)-max(s, a)) for a, b in union_truth) for s, e in canonical(prediction, horizon))
    short_ids = {i for i, (s, e) in enumerate(truth) if e-s <= cfg.short_duration+1e-9}
    short_tp = sum(j in short_ids for _, j in pairs)
    tiou_sum = sum(float(iou[i, j]) for i, j in pairs)
    return dict(event_precision=precision, event_recall=recall,
                event_f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                matched_event_tiou=tiou_sum/tp if tp else None, tiou_sum=tiou_sum,
                matched_events=tp, predicted_events=len(prediction), true_events=len(truth),
                missed_events=len(truth)-tp, false_positive_events=len(prediction)-tp,
                false_positive_duration=max(0., fp_seconds), short_events=len(short_ids), short_matched=short_tp,
                short_event_recall=short_tp/len(short_ids) if short_ids else None)


def risk(prediction: tuple[Interval, ...], truth: tuple[Interval, ...], cfg: Config) -> float:
    m = evaluate(prediction, truth, cfg)
    return (cfg.risk_miss*m['missed_events'] + cfg.risk_fp*m['false_positive_events']
            + cfg.risk_fp_seconds*m['false_positive_duration']
            + cfg.risk_localization*(m['matched_events']-m['tiou_sum']))


def gain(record: Record, candidate: Candidate, cfg: Config) -> float:
    if record.truth is None:
        raise ValueError('Gain labels require training/offline GT')
    return risk(record.original, record.truth, cfg)-risk(apply(record, candidate), record.truth, cfg)


def diagnostics(record: Record, pool: list[Candidate], cfg: Config, exact_limit: int = 14) -> dict:
    """GT-only diagnostics; oracle results MUST NOT be reported as predictions.

    Coverage: fraction of GT events matchable by any original/proposed interval.
    Oracle: exact best compatible subset for small pools; otherwise risk(before)
    is a valid but loose relaxed upper bound because all risk terms are >= 0.
    Single-edit oracle is separately a feasible lower bound on best subset gain.
    """
    if record.truth is None:
        raise ValueError('Offline diagnostics require GT')
    proposed = tuple(record.original)+tuple(x for c in pool if c.operation != 'KEEP' for x in c.after)
    covered = 0
    for event in record.truth:
        iou, _ = match(proposed, (event,), cfg.tiou)
        covered += int(bool(iou.size and np.max(iou) >= cfg.tiou-1e-8))
    gains = [gain(record, c, cfg) for c in pool]
    upper = risk(record.original, record.truth, cfg)
    candidates = [c for c in pool if c.operation != 'KEEP']
    exact = len(candidates) <= exact_limit
    if exact:
        best = 0.
        for size in range(1, len(candidates)+1):
            for subset in combinations(candidates, size):
                if any(conflict(a, b) for a, b in combinations(subset, 2)):
                    continue
                removed = {i for c in subset for i in c.source_indices}
                after = canonical([x for i, x in enumerate(record.original) if i not in removed]+[x for c in subset for x in c.after], record.horizon)
                best = max(best, upper-risk(after, record.truth, cfg))
        upper = best
    return dict(candidate_covered_events=covered, candidate_truth_events=len(record.truth),
                candidate_coverage=covered/len(record.truth) if record.truth else None,
                offline_oracle_single_edit_gain=max(gains, default=0.),
                offline_oracle_improvement_upper_bound=upper,
                offline_oracle_bound_kind='exact_compatible_subset' if exact else 'relaxed_nonnegative_risk_bound',
                oracle_is_prediction=False)


def aggregate(rows: list[dict]) -> dict:
    keys = ('matched_events', 'predicted_events', 'true_events', 'missed_events', 'false_positive_events',
            'false_positive_duration', 'short_events', 'short_matched', 'tiou_sum')
    out = {key: sum(row[key] for row in rows) for key in keys}
    tp, p, t = out['matched_events'], out['predicted_events'], out['true_events']
    out.update(event_precision=tp/p if p else 0., event_recall=tp/t if t else 0.,
               event_f1=2*tp/(p+t) if p+t else 0., matched_event_tiou=out['tiou_sum']/tp if tp else None,
               short_event_recall=out['short_matched']/out['short_events'] if out['short_events'] else None)
    return out
