"""Native-20ms, duration and event metrics for the fixed PS experiment.

Positive means fake. Prediction is float64(score) >= float64(threshold).
Reference labels must cover actual duration; tail padding contributes nothing.
The threshold helper uses only genuine-labelled durations in its supplied dev
records. It never accesses a split/file or optimizes a test operating point.
"""
import math

import numpy as np


STEP = .02
IOU_THRESHOLD = .5
BIN_EDGES = (.08, .16, .32, .64)
BIN_NAMES = ('le80ms', 'gt80_le160ms', 'gt160_le320ms', 'gt320_le640ms', 'gt640ms')
from .metrics160 import match_events


def decision_mask(scores20, threshold):
    """Use this for external saved masks: a float32 comparison can round a cut."""
    if not np.isfinite(float(threshold)):
        raise ValueError('Threshold must be finite')
    return np.asarray(scores20, dtype=np.float64) >= np.float64(threshold)


def _prepare(records):
    prepared, seen = [], set()
    for record in records:
        uid = str(record['uid'])
        if uid in seen:
            raise ValueError(f'Duplicate recording uid: {uid}')
        seen.add(uid)
        y = np.asarray(record['labels20'])
        s = np.asarray(record['scores20'])
        duration = float(record['duration'])
        if y.ndim != 1 or s.ndim != 1 or len(s) < len(y):
            raise ValueError(f'{uid}: 1D scores must cover all supplied 20ms labels')
        if not np.isfinite(duration) or duration < 0.:
            raise ValueError(f'{uid}: invalid recording duration')
        if len(y)*STEP < duration-1e-9:
            raise ValueError(f'{uid}: genuine 20ms labels do not cover recording duration')
        if not np.isin(y, (0, 1)).all():
            raise ValueError(f'{uid}: labels must be binary with 1=fake')
        start = np.arange(len(y), dtype=np.float64)*STEP
        widths = np.maximum(0., np.minimum(start+STEP, duration)-start)
        active = widths > 0.
        y = y[active].astype(bool)
        s = np.asarray(s[:len(active)][active], dtype=np.float64)
        widths = widths[active]
        if not np.isfinite(s).all():
            raise ValueError(f'{uid}: non-finite score within actual recording')
        if not math.isclose(float(widths.sum()), duration, rel_tol=0., abs_tol=1e-8):
            raise ValueError(f'{uid}: native grid/duration coverage mismatch')
        prepared.append(dict(uid=uid, labels=y, scores=s, widths=widths, duration=duration))
    if not prepared:
        raise ValueError('At least one recording is required')
    return prepared


def _intervals(mask, duration):
    flags = np.asarray(mask, dtype=bool)
    change = np.diff(np.r_[False, flags, False].astype(np.int8))
    return np.asarray([(float(a*STEP), float(min(b*STEP, duration)))
                       for a,b in zip(np.flatnonzero(change==1), np.flatnonzero(change==-1))
                       if a*STEP < duration], dtype=np.float64).reshape(-1,2)


def _percent(numerator, denominator):
    return float(100*numerator/denominator) if denominator else 0.


def _bin_index(duration):
    for i, edge in enumerate(BIN_EDGES):
        if duration <= edge+1e-9:
            return i
    return len(BIN_EDGES)


def _evaluate(prepared, threshold):
    if not np.isfinite(float(threshold)):
        raise ValueError('Threshold must be finite')
    cm = np.zeros(4, dtype=np.float64)  # TP FP FN TN in seconds
    event_tp = event_pred = event_truth = 0
    bins_truth = np.zeros(5, dtype=np.int64)
    bins_matched = np.zeros(5, dtype=np.int64)
    full_real = dict(recordings=0, false_alarm_recordings=0, audio_seconds=0., fp_seconds=0., predicted_events=0)
    per_recording = []
    for r in prepared:
        y, s, w, d = r['labels'], r['scores'], r['widths'], r['duration']
        p = decision_mask(s, threshold)
        seconds = np.array([w[p&y].sum(), w[p&~y].sum(), w[~p&y].sum(), w[~p&~y].sum()], dtype=float)
        cm += seconds
        ps, ts = _intervals(p,d), _intervals(y,d)
        matches, _ = match_events(ps, ts, threshold=IOU_THRESHOLD)
        reference_bins = [_bin_index(b-a) for a,b in ts]
        truth_counts = np.bincount(reference_bins, minlength=5)
        matched_counts = np.bincount([reference_bins[t] for _,t,_ in matches], minlength=5)
        bins_truth += truth_counts; bins_matched += matched_counts
        event_tp += len(matches); event_pred += len(ps); event_truth += len(ts)
        is_full_real = not bool(y.any()) and d > 0.
        if is_full_real:
            full_real['recordings'] += 1
            full_real['false_alarm_recordings'] += int(bool(p.any()))
            full_real['audio_seconds'] += d
            full_real['fp_seconds'] += float(seconds[1])
            full_real['predicted_events'] += len(ps)
        per_recording.append(dict(uid=r['uid'], duration=d, active_20ms_frames=len(y),
            tp_seconds=float(seconds[0]), fp_seconds=float(seconds[1]),
            fn_seconds=float(seconds[2]), tn_seconds=float(seconds[3]),
            event_tp=len(matches), event_pred=len(ps), event_truth=len(ts),
            unmatched_fp_events=len(ps)-len(matches), missed_events=len(ts)-len(matches),
            short_tp=int(matched_counts[:3].sum()), short_truth=int(truth_counts[:3].sum()),
            event_bin_truth=truth_counts.tolist(), event_bin_matched=matched_counts.tolist(),
            fully_genuine=is_full_real, fully_genuine_false_alarm=is_full_real and bool(p.any())))
    tp, fp, fn, tn = map(float, cm)
    audio = math.fsum(r['duration'] for r in prepared)
    short_tp, short_truth = int(bins_matched[:3].sum()), int(bins_truth[:3].sum())
    unmatched = event_pred-event_tp
    full_real.update(false_alarm_recording_rate_percent=_percent(full_real['false_alarm_recordings'],full_real['recordings']),
        duration_fpr_percent=_percent(full_real['fp_seconds'],full_real['audio_seconds']),
        unmatched_fp_events_per_hour=(3600*full_real['predicted_events']/full_real['audio_seconds']
                                     if full_real['audio_seconds'] else 0.))
    result = dict(recordings=len(prepared), frames=sum(len(r['labels']) for r in prepared),
        threshold=float(threshold), resolution_seconds=STEP, positive_label='fake',
        duration_f1_percent=_percent(2*tp,2*tp+fp+fn),
        duration_precision_percent=_percent(tp,tp+fp), duration_recall_percent=_percent(tp,tp+fn),
        duration_accuracy_percent=_percent(tp+tn,tp+fp+fn+tn),
        real_fpr_percent=_percent(fp,fp+tn), spoof_fnr_percent=_percent(fn,tp+fn),
        confusion_seconds_tp_fp_fn_tn=[tp,fp,fn,tn], fake_seconds=tp+fn, real_seconds=fp+tn,
        fp_seconds=fp, fn_seconds=fn, audio_seconds=audio,
        event_f1_iou05_percent=_percent(2*event_tp,event_pred+event_truth),
        event_precision_percent=_percent(event_tp,event_pred), event_recall_percent=_percent(event_tp,event_truth),
        event_counts=dict(matched=event_tp,predicted=event_pred,truth=event_truth,
                          unmatched_predictions=unmatched,missed_truth=event_truth-event_tp),
        short_event_recall_le320ms_percent=_percent(short_tp,short_truth),
        short_event_counts=dict(matched=short_tp,truth=short_truth),
        unmatched_fp_events_per_hour=(3600*unmatched/audio if audio else 0.),
        fully_genuine=full_real, per_recording=per_recording,
        metric_note='Duration metrics use clipped real 20ms reference; event TP is maximum-cardinality one-to-one IoU>=0.5 matching, then total-IoU tie-break.',
        fp_event_rate_denominator='All evaluated audio duration; an unmatched predicted event is an FP event even if it partially overlaps a true fake event.')
    result['event_duration_bins'] = {name:dict(truth=int(n),matched=int(m),recall_percent=_percent(m,n))
                                   for name,n,m in zip(BIN_NAMES,bins_truth,bins_matched)}
    for i, edge in enumerate((80,160,320,640)):
        result[f'event_recall_le{edge}ms_percent'] = _percent(int(bins_matched[:i+1].sum()),int(bins_truth[:i+1].sum()))
    result['selection_score_j_percent'] = .5*result['duration_f1_percent']+.5*result['event_f1_iou05_percent']
    return result


def evaluate(records, threshold=0.0):
    """One native20 evaluation; no threshold fitting or parameter selection."""
    return _evaluate(_prepare(records), threshold)


def _choose_threshold(prepared, max_real_fpr):
    target = float(max_real_fpr)
    if not np.isfinite(target) or not 0. <= target <= 1.:
        raise ValueError('max_real_fpr must be a fraction in [0,1]')
    real_scores = np.concatenate([r['scores'][~r['labels']] for r in prepared])
    real_widths = np.concatenate([r['widths'][~r['labels']] for r in prepared])
    if not len(real_scores):
        raise ValueError('Development threshold requires genuine-labelled duration')
    order = np.argsort(-real_scores, kind='stable')
    sorted_scores, sorted_widths = real_scores[order],real_widths[order]
    starts = np.r_[0,np.flatnonzero(sorted_scores[1:]!=sorted_scores[:-1])+1]
    group_scores = sorted_scores[starts]
    group_widths = np.add.reduceat(sorted_widths,starts)
    real_seconds = float(real_widths.sum()); budget=target*real_seconds
    cumulative = np.cumsum(group_widths,dtype=np.float64)
    # Never split a tied-score group. No positive-budget numerical tolerance is
    # added: a borderline rounding case is handled conservatively by exclusion.
    forbidden = np.flatnonzero(cumulative > budget)
    if target == 1. or not len(forbidden):
        threshold = float(min(float(r['scores'].min()) for r in prepared if len(r['scores'])))
        boundary = None; excluded_tie_seconds=0.; included_groups=len(group_scores)
    else:
        at = int(forbidden[0]); boundary=float(group_scores[at])
        threshold = float(np.nextafter(np.float64(boundary),np.float64(np.inf)))
        if not np.isfinite(threshold):
            raise ValueError('Cannot represent a finite threshold above the maximum input score')
        excluded_tie_seconds=float(group_widths[at]); included_groups=at
    fp_seconds = float(real_widths[decision_mask(real_scores,threshold)].sum())
    measured = fp_seconds/real_seconds
    if measured > target+1e-12:
        raise AssertionError('Chosen development threshold violates the duration FPR budget')
    return dict(threshold=threshold,max_real_fpr=target,nominal_max_real_fpr_percent=100*target,
        real_fpr_percent=100*measured,fp_seconds=fp_seconds,real_seconds=real_seconds,
        budget_seconds=budget,included_real_score_groups=included_groups,total_real_score_groups=len(group_scores),
        excluded_boundary_score=boundary,excluded_boundary_tie_seconds=excluded_tie_seconds,
        selection_source='Genuine-labelled durations in supplied development recordings only; includes genuine regions of partial fakes.',
        tie_rule='Exclude the entire first score-tie group that would exceed the budget; score>=float64 threshold.',
        score_comparison_dtype='float64')


def choose_threshold(records, max_real_fpr=.05):
    """Most permissive real-score boundary satisfying the dev duration FPR cap.

    No event matching is performed. Uses all genuine-labelled regions, including
    genuine regions inside partially fake recordings, rather than whole-real
    recordings only. Use decision_mask for masks outside this module.
    """
    return _choose_threshold(_prepare(records),max_real_fpr)


def evaluate_dev(records, max_real_fpr=.05):
    """One threshold selection and exactly one event-evaluation pass per epoch.

    Caller selects epoch by selection_score_j_percent, keeping earlier epochs on
    ties. The selected threshold must be frozen before evaluating test records.
    """
    prepared=_prepare(records)
    choice=_choose_threshold(prepared,max_real_fpr)
    result=_evaluate(prepared,choice['threshold'])
    if abs(result['real_fpr_percent']-choice['real_fpr_percent'])>1e-8:
        raise AssertionError('Threshold and metric duration FPR disagree')
    result['threshold_selection']=choice
    return result
