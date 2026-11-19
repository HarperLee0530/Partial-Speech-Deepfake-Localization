"""Fixed-threshold native20 reporting with explicit undefined-value semantics.

This is a scoring adapter, not a checkpoint/threshold-selection protocol.
Existing checkpoints can be reported without claiming they were selected by
the future v2 event-F1 rule. No input record or legacy metric module is changed.
"""
import numpy as np
from sklearn.metrics import roc_curve

from . import metrics20


def evaluate_report(records, threshold):
    """Evaluate fake-positive native20 scores at a caller-frozen threshold.

    Records use the existing uid/labels20/scores20/duration schema. Duration
    must be positive; scores and labels outside actual duration are excluded.
    Legacy field names and per_recording remain available. Undefined ratios
    become None, with reasons in undefined_metrics. Supplemental frame F1/EER
    count each active20 cell once, including a final fractional cell; duration
    metrics instead weight that cell by its true width.
    """
    records = list(records)
    for record in records:
        duration = float(record['duration'])
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"{record.get('uid', '<unknown>')}: duration must be positive and finite")
    result = metrics20.evaluate(records, threshold=threshold)
    # A fixed-checkpoint report must not suggest that the historical composite
    # J was used for selection, nor silently switch the checkpoint to event F1.
    result.pop('selection_score_j_percent', None)
    result['metric_protocol_version'] = 'ps_native20_event_v2'
    result['evaluation_only'] = True
    result['selection_note'] = 'Fixed supplied checkpoint and threshold; no checkpoint or threshold selection is performed here.'
    undefined = {}

    def ratio(container, key, numerator, denominator, path=None):
        path = path or key
        if denominator == 0:
            container[key] = None
            undefined[path] = 'zero_denominator'
        else:
            container[key] = float(100*numerator/denominator)

    tp, fp, fn, tn = result['confusion_seconds_tp_fp_fn_tn']
    ratio(result, 'duration_f1_percent', 2*tp, 2*tp+fp+fn)
    ratio(result, 'duration_precision_percent', tp, tp+fp)
    ratio(result, 'duration_recall_percent', tp, tp+fn)
    ratio(result, 'real_fpr_percent', fp, fp+tn)
    ratio(result, 'spoof_fnr_percent', fn, tp+fn)
    result['real_duration_fpr_percent'] = result['real_fpr_percent']
    if result['real_duration_fpr_percent'] is None:
        undefined['real_duration_fpr_percent'] = 'zero_denominator'

    counts = result['event_counts']
    matched, predicted, truth = (counts[key] for key in ('matched', 'predicted', 'truth'))
    ratio(result, 'event_precision_percent', matched, predicted)
    ratio(result, 'event_recall_percent', matched, truth)
    ratio(result, 'event_f1_iou05_percent', 2*matched, predicted+truth)
    short = result['short_event_counts']
    ratio(result, 'short_event_recall_le320ms_percent', short['matched'], short['truth'])
    cumulative_matched = cumulative_truth = 0
    for i, name in enumerate(metrics20.BIN_NAMES):
        bucket = result['event_duration_bins'][name]
        ratio(bucket, 'recall_percent', bucket['matched'], bucket['truth'],
              path=f'event_duration_bins.{name}.recall_percent')
        cumulative_matched += bucket['matched']
        cumulative_truth += bucket['truth']
        if i < 4:
            key = f'event_recall_le{(80, 160, 320, 640)[i]}ms_percent'
            ratio(result, key, cumulative_matched, cumulative_truth)

    genuine = result['fully_genuine']
    ratio(genuine, 'false_alarm_recording_rate_percent', genuine['false_alarm_recordings'],
          genuine['recordings'], path='fully_genuine.false_alarm_recording_rate_percent')
    ratio(genuine, 'duration_fpr_percent', genuine['fp_seconds'], genuine['audio_seconds'],
          path='fully_genuine.duration_fpr_percent')
    if genuine['audio_seconds'] == 0:
        genuine['unmatched_fp_events_per_hour'] = None
        undefined['fully_genuine.unmatched_fp_events_per_hour'] = 'zero_denominator'

    # Reuse precisely the actual-duration grid convention of metrics20._prepare,
    # but count each active cell once for these explicitly frame-based metrics.
    all_labels, all_scores = [], []
    for record in records:
        labels = np.asarray(record['labels20'])
        scores = np.asarray(record['scores20'])
        start = np.arange(len(labels), dtype=np.float64)*metrics20.STEP
        widths = np.maximum(0., np.minimum(start+metrics20.STEP, float(record['duration']))-start)
        active = widths > 0
        all_labels.append(labels[active].astype(bool))
        all_scores.append(np.asarray(scores[:len(labels)][active], dtype=np.float64))
    labels, scores = np.concatenate(all_labels), np.concatenate(all_scores)
    prediction = metrics20.decision_mask(scores, threshold)
    frame_counts = [int((prediction & labels).sum()), int((prediction & ~labels).sum()),
                    int((~prediction & labels).sum()), int((~prediction & ~labels).sum())]
    check_frames = sum(frame_counts)
    if check_frames != result['frames']:
        raise AssertionError('Supplemental and duration metrics used different active20 cells')
    frame_tp, frame_fp, frame_fn, _ = frame_counts
    result['active_20ms_frames'] = check_frames
    result['frame_confusion_tp_fp_fn_tn'] = frame_counts
    ratio(result, 'frame_f1_percent', 2*frame_tp, 2*frame_tp+frame_fp+frame_fn)
    if len(np.unique(labels)) < 2:
        result['eer_percent'] = None
        undefined['eer_percent'] = 'single_class'
    else:
        fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
        crossing = int(np.argmin(np.abs(fpr-(1-tpr))))
        result['eer_percent'] = float(50*(fpr[crossing]+1-tpr[crossing]))
    result['supplementary_native20'] = dict(
        duration_f1_percent=result['duration_f1_percent'],
        frame_confusion_tp_fp_fn_tn=frame_counts.copy(),
        frame_f1_percent=result['frame_f1_percent'],
        eer_percent=result['eer_percent'],
        eer_rule='fake_positive_frame_ROC_nearest_crossing_mean_FPR_FNR',
        frame_rule='Each positive-duration20ms cell counts once, including the fractional final cell.')
    for key in ('duration_f1_percent', 'frame_f1_percent', 'eer_percent'):
        if result['supplementary_native20'][key] is None:
            undefined[f'supplementary_native20.{key}'] = undefined[key]
    result['undefined_metrics'] = undefined
    return result
