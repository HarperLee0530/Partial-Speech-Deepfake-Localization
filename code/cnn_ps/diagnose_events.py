"""Read-only native20 event-error descriptions at a supplied frozen threshold.

These are overlap/geometry diagnostics, not inferred causes of model errors.
No threshold fitting, smoothing, checkpoint selection or model execution occurs.
"""
import numpy as np

from . import metrics20
from .metrics160 import match_events


def diagnose_events(records, threshold):
    """Describe unmatched predictions using the exact evaluation matching.

    Input records have uid, labels20 (1=fake), scores20 and duration. Counts
    are pooled over recordings. A prediction touching a truth endpoint has
    zero overlap. Multiple predictions overlapping one truth are explicitly
    labelled fragmentation *candidates*: overlap alone does not prove cause.
    """
    records = list(records)
    if any(not np.isfinite(float(r['duration'])) or float(r['duration']) <= 0 for r in records):
        raise ValueError('Recording duration must be positive and finite')
    prepared = metrics20._prepare(records)
    totals = dict(matched=0, predicted=0, truth=0, unmatched_predictions=0,
                  missed_truth=0, unmatched_pred_no_truth_overlap=0,
                  unmatched_pred_with_truth_overlap=0,
                  unmatched_pred_overlap_below_iou_threshold=0,
                  unmatched_pred_eligible_but_unmatched=0,
                  truth_with_multiple_overlapping_predictions=0,
                  prediction_with_multiple_overlapping_truths=0)
    per_recording = []
    for record in prepared:
        predicted = metrics20._intervals(metrics20.decision_mask(record['scores'], threshold), record['duration'])
        truth = metrics20._intervals(record['labels'], record['duration'])
        matches, iou = match_events(predicted, truth, threshold=metrics20.IOU_THRESHOLD)
        overlap = np.maximum(0., np.minimum(predicted[:, None, 1], truth[None, :, 1])
                             - np.maximum(predicted[:, None, 0], truth[None, :, 0]))
        matched_pred = {p for p, _, _ in matches}
        matched_truth = {t for _, t, _ in matches}
        details = []
        for p, interval in enumerate(predicted):
            if p in matched_pred:
                continue
            overlapping = np.flatnonzero(overlap[p] > 0.)
            detail = dict(prediction_index=p, interval_seconds=interval.tolist(),
                          overlapping_truth_indices=overlapping.tolist(),
                          category='no_truth_overlap' if not len(overlapping) else 'truth_overlap_but_unmatched')
            if len(overlapping):
                t = int(overlapping[np.argmax(iou[p, overlapping])])
                detail.update(best_overlap_truth_index=t, best_overlap_iou=float(iou[p, t]),
                              has_iou_eligible_truth=bool((iou[p] >= metrics20.IOU_THRESHOLD - 1e-10).any()),
                              start_offset_seconds=float(interval[0] - truth[t, 0]),
                              end_offset_seconds=float(interval[1] - truth[t, 1]),
                              duration_ratio_to_truth=float((interval[1] - interval[0]) / (truth[t, 1] - truth[t, 0])))
            details.append(detail)
        fragmentation = [dict(truth_index=t, interval_seconds=truth[t].tolist(),
                              overlapping_prediction_indices=np.flatnonzero(overlap[:, t] > 0.).tolist())
                         for t in range(len(truth)) if (overlap[:, t] > 0.).sum() >= 2]
        merging = [dict(prediction_index=p, interval_seconds=predicted[p].tolist(),
                       overlapping_truth_indices=np.flatnonzero(overlap[p] > 0.).tolist())
                   for p in range(len(predicted)) if (overlap[p] > 0.).sum() >= 2]
        no_overlap = sum(d['category'] == 'no_truth_overlap' for d in details)
        eligible = sum(d.get('has_iou_eligible_truth', False) for d in details)
        counts = dict(matched=len(matches), predicted=len(predicted), truth=len(truth),
                      unmatched_predictions=len(details), missed_truth=len(truth) - len(matches),
                      unmatched_pred_no_truth_overlap=no_overlap,
                      unmatched_pred_with_truth_overlap=len(details) - no_overlap,
                      unmatched_pred_overlap_below_iou_threshold=len(details) - no_overlap - eligible,
                      unmatched_pred_eligible_but_unmatched=eligible,
                      truth_with_multiple_overlapping_predictions=len(fragmentation),
                      prediction_with_multiple_overlapping_truths=len(merging))
        for key, value in counts.items():
            totals[key] += value
        per_recording.append(dict(uid=record['uid'], counts=counts,
                                  matches=[dict(prediction_index=p, truth_index=t, iou=value)
                                           for p, t, value in matches],
                                  unmatched_predictions=details,
                                  missed_truth_indices=[t for t in range(len(truth)) if t not in matched_truth],
                                  fragmentation_candidates=fragmentation, merging_candidates=merging))
    return dict(threshold=float(threshold), resolution_seconds=metrics20.STEP,
                iou_threshold=metrics20.IOU_THRESHOLD, counts=totals,
                per_recording=per_recording,
                diagnostic_note='Overlap multiplicity and boundary offsets are descriptive proxies, not causal attribution. '
                                'A partially overlapping unmatched event still counts as an event false positive. '
                                'Matching is maximum-cardinality one-to-one, then maximum summed IoU; no rematching is used.')
