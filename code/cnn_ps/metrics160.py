"""PS 160 ms comparison: fake-positive frame metrics and 1:1 event matching."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_curve

STEP = 0.16


def intervals(mask, duration, step=STEP):
    edges = np.diff(np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8))
    return np.array([(a * step, min(b * step, float(duration)))
                     for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
                     if a * step < float(duration)], dtype=float).reshape(-1, 2)


def match_events(pred, truth, threshold=0.5):
    """Maximum-cardinality 1:1 matching above threshold, then maximum summed IoU."""
    pred = np.asarray(pred, dtype=float).reshape(-1, 2)
    truth = np.asarray(truth, dtype=float).reshape(-1, 2)
    if not len(pred) or not len(truth):
        return [], np.zeros((len(pred), len(truth)))
    overlap = np.maximum(0, np.minimum(pred[:, None, 1], truth[None, :, 1])
                         - np.maximum(pred[:, None, 0], truth[None, :, 0]))
    union = (pred[:, 1]-pred[:, 0])[:, None] + (truth[:, 1]-truth[:, 0])[None, :] - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap), where=union > 0)
    eligible = iou >= threshold - 1e-10
    # The bounded IoU tie-break cannot outweigh one extra eligible match.
    benefit = eligible.astype(float) + eligible * iou / (min(iou.shape) + 1)
    rr, cc = linear_sum_assignment(-benefit)
    return [(int(r), int(c), float(iou[r, c])) for r, c in zip(rr, cc) if eligible[r, c]], iou


def choose_threshold(labels, scores):
    """Exact dev frame-spoof-F1 optimum, resolving tied scores together."""
    y = np.asarray(labels, dtype=np.uint8)
    s = np.asarray(scores, dtype=np.float64)
    assert len(s) == len(y) and np.isfinite(s).all()
    order = np.argsort(-s, kind='stable')
    sy, ss = y[order], s[order]
    last = np.r_[ss[1:] != ss[:-1], True]
    n = np.flatnonzero(last) + 1
    tp = np.cumsum(sy, dtype=np.int64)[last]
    fp, fn = n-tp, int(y.sum())-tp
    f1 = 2*tp / np.maximum(2*tp+fp+fn, 1)
    best = int(np.argmax(f1))
    return float(ss[n[best]-1]), float(100*f1[best])


def evaluate(record_outputs, threshold=0.0):
    """Outputs: iterable {uid,labels,scores,mask(optional),duration}."""
    cm = np.zeros(4, dtype=np.int64)  # TP, FP, FN, TN, positive=fake
    all_y, all_s = [], []
    event_tp = event_pred = event_truth = short_tp = short_truth = 0
    fake_seconds = fn_seconds = fp_seconds = audio_seconds = 0.0
    per_recording = []
    for r in record_outputs:
        y = np.asarray(r['labels'], dtype=bool)
        s = np.asarray(r['scores'], dtype=float)
        p = np.asarray(r['mask'], dtype=bool) if 'mask' in r else s >= threshold
        assert y.shape == s.shape == p.shape and np.isfinite(s).all()
        d = float(r['duration'])
        w = np.maximum(0, np.minimum((np.arange(len(y))+1)*STEP, d)-np.arange(len(y))*STEP)
        counts = np.array([(p & y).sum(), (p & ~y).sum(), (~p & y).sum(), (~p & ~y).sum()])
        cm += counts
        ps = intervals(p, d)
        truth20 = np.asarray(r['labels20'], dtype=bool) if 'labels20' in r else None
        ts = intervals(truth20, d, .02) if truth20 is not None else intervals(y, d)
        matches, _ = match_events(ps, ts)
        short = (ts[:, 1]-ts[:, 0]) <= .32 + 1e-8
        st = sum(bool(short[t]) for _, t, _ in matches)
        event_tp += len(matches); event_pred += len(ps); event_truth += len(ts)
        short_tp += st; short_truth += int(short.sum())
        fn = float(w[~p & y].sum()); fp = float(w[p & ~y].sum())
        fn_seconds += fn; fp_seconds += fp; fake_seconds += float(w[y].sum()); audio_seconds += d
        all_y.append(y); all_s.append(s)
        per_recording.append(dict(uid=r['uid'], tp=int(counts[0]), fp=int(counts[1]),
            fn=int(counts[2]), tn=int(counts[3]), event_tp=len(matches), event_pred=len(ps),
            event_truth=len(ts), short_tp=st, short_truth=int(short.sum()), fn_seconds=fn,
            fp_seconds=fp, duration=d))
    tp, fp, fn, tn = map(int, cm)
    y, s = np.concatenate(all_y), np.concatenate(all_s)
    fpr, tpr, _ = roc_curve(y, s, drop_intermediate=False)
    i = int(np.argmin(np.abs(fpr-(1-tpr))))
    ratio = lambda a, b: 100*a/b if b else 0.0
    return dict(recordings=len(per_recording), frames=len(y), threshold=float(threshold),
        spoof_f1_percent=ratio(2*tp, 2*tp+fp+fn), accuracy_percent=ratio(tp+tn,len(y)),
        spoof_precision_percent=ratio(tp,tp+fp), spoof_recall_percent=ratio(tp,tp+fn),
        real_fpr_percent=ratio(fp,fp+tn), eer_percent=float(50*(fpr[i]+1-tpr[i])),
        confusion_tp_fp_fn_tn=[tp,fp,fn,tn], event_f1_iou05_percent=ratio(2*event_tp,event_pred+event_truth),
        event_precision_percent=ratio(event_tp,event_pred), event_recall_percent=ratio(event_tp,event_truth),
        event_counts=dict(matched=event_tp,predicted=event_pred,truth=event_truth),
        short_event_recall_le320ms_percent=ratio(short_tp,short_truth),
        short_event_counts=dict(matched=short_tp,truth=short_truth),
        fake_seconds=fake_seconds, fn_seconds=fn_seconds, fp_seconds=fp_seconds,
        audio_seconds=audio_seconds, per_recording=per_recording)
