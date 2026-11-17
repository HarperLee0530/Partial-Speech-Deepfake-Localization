"""Native20 report boundaries, with no data/checkpoint or legacy-source dependency."""
import json
import unittest

import numpy as np

from cnn_ps.report_native20 import evaluate_report


class Native20ReportTests(unittest.TestCase):
    def test_two_truth_two_predictions_example(self):
        truth = np.zeros(200, dtype=np.uint8)
        truth[50:60] = 1
        truth[100:130] = 1
        scores = np.full(200, -1., dtype=np.float32)
        scores[49:59] = 1
        scores[150:160] = 1
        result = evaluate_report([dict(uid='toy', labels20=truth, scores20=scores, duration=4.)], 0.)
        for name, expected in [('event_precision_percent', 50.), ('event_recall_percent', 50.),
                               ('event_f1_iou05_percent', 50.), ('short_event_recall_le320ms_percent', 100.),
                               ('real_duration_fpr_percent', 6.875), ('duration_f1_percent', 30.)]:
            self.assertAlmostEqual(result[name], expected, places=9)
        self.assertEqual(result['event_counts']['matched'], 1)
        self.assertEqual(result['short_event_counts'], dict(matched=1, truth=1))
        self.assertEqual(result['frame_confusion_tp_fp_fn_tn'], [9, 11, 31, 149])
        self.assertEqual(result['fully_genuine']['recordings'], 0)
        self.assertIsNone(result['fully_genuine']['false_alarm_recording_rate_percent'])
        self.assertNotIn('selection_score_j_percent', result)
        self.assertTrue(result['evaluation_only'])
        json.dumps(result, allow_nan=False)

    def test_empty_prediction_with_true_event_is_not_undefined_f1(self):
        result = evaluate_report([dict(uid='missed', labels20=[0, 1], scores20=[-1., -1.], duration=.04)], 0.)
        self.assertIsNone(result['event_precision_percent'])
        self.assertEqual(result['event_recall_percent'], 0.)
        self.assertEqual(result['event_f1_iou05_percent'], 0.)
        self.assertEqual(result['duration_f1_percent'], 0.)
        self.assertEqual(result['frame_f1_percent'], 0.)
        self.assertIsNone(result['duration_precision_percent'])
        self.assertEqual(result['short_event_recall_le320ms_percent'], 0.)

    def test_all_real_without_predictions_has_na_event_f1(self):
        result = evaluate_report([dict(uid='normal', labels20=[0, 0], scores20=[-1., -1.], duration=.04)], 0.)
        for key in ('event_precision_percent', 'event_recall_percent', 'event_f1_iou05_percent',
                    'duration_f1_percent', 'frame_f1_percent', 'short_event_recall_le320ms_percent',
                    'eer_percent'):
            self.assertIsNone(result[key], key)
        self.assertEqual(result['fully_genuine']['recordings'], 1)
        self.assertEqual(result['fully_genuine']['false_alarm_recording_rate_percent'], 0.)
        self.assertEqual(result['real_duration_fpr_percent'], 0.)
        self.assertEqual(result['undefined_metrics']['eer_percent'], 'single_class')
        self.assertEqual(result['undefined_metrics']['event_f1_iou05_percent'], 'zero_denominator')
        self.assertIsNone(result['event_duration_bins']['le80ms']['recall_percent'])
        json.dumps(result, allow_nan=False)

    def test_all_real_false_alarm_has_defined_zero_event_f1(self):
        result = evaluate_report([dict(uid='alarm', labels20=[0, 0], scores20=[1., -1.], duration=.04)], 0.)
        self.assertEqual(result['event_precision_percent'], 0.)
        self.assertIsNone(result['event_recall_percent'])
        self.assertEqual(result['event_f1_iou05_percent'], 0.)
        self.assertEqual(result['fully_genuine']['false_alarm_recording_rate_percent'], 100.)
        self.assertEqual(result['real_duration_fpr_percent'], 50.)

    def test_all_fake_has_no_real_duration_denominator(self):
        result = evaluate_report([dict(uid='fake', labels20=[1, 1], scores20=[1., 1.], duration=.04)], 0.)
        self.assertEqual(result['event_f1_iou05_percent'], 100.)
        self.assertIsNone(result['real_fpr_percent'])
        self.assertIsNone(result['real_duration_fpr_percent'])
        self.assertIsNone(result['eer_percent'])
        self.assertIsNone(result['fully_genuine']['false_alarm_recording_rate_percent'])

    def test_fractional_tail_and_padding_use_same_active_cells(self):
        labels = np.array([0]*10+[1], dtype=np.uint8)
        scores = np.array([-1.]*10+[1.], dtype=np.float32)
        clean = evaluate_report([dict(uid='tail', labels20=labels, scores20=scores, duration=.205)], 0.)
        padded = evaluate_report([dict(uid='tail', labels20=np.r_[labels, np.ones(5, dtype=np.uint8)],
                                       scores20=np.r_[scores, np.full(5, np.nan)], duration=.205)], 0.)
        self.assertEqual(clean, padded)
        self.assertEqual(clean['active_20ms_frames'], 11)
        self.assertEqual(clean['frame_confusion_tp_fp_fn_tn'], [1, 0, 0, 10])
        np.testing.assert_allclose(clean['confusion_seconds_tp_fp_fn_tn'], [.005, 0, 0, .2], atol=1e-12, rtol=0)
        self.assertAlmostEqual(clean['audio_seconds'], .205, places=12)
        self.assertEqual(clean['event_f1_iou05_percent'], 100.)

    def test_zero_negative_nonfinite_duration_rejected(self):
        for duration in (0., -.02, np.nan, np.inf):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                evaluate_report([dict(uid='invalid', labels20=[0], scores20=[0.], duration=duration)], 0.)

    def test_float64_nextafter_threshold_does_not_include_tied_real_frames(self):
        threshold = np.nextafter(np.float64(.25), np.float64(np.inf))
        result = evaluate_report([dict(uid='ties', labels20=[0, 0, 1],
                                       scores20=np.array([.25, .25, 1.], dtype=np.float32), duration=.06)], threshold)
        self.assertEqual(result['frame_confusion_tp_fp_fn_tn'], [1, 0, 0, 2])
        self.assertEqual(result['real_duration_fpr_percent'], 0.)
        self.assertEqual(result['event_f1_iou05_percent'], 100.)
        self.assertGreater(result['threshold'], .25)

    def test_frame_eer_uses_fake_positive_scores(self):
        for scores, expected_eer in [([-2., -1., 1., 2.], 0.),
                                     ([-2., 2., -1., 1.], 50.),
                                     ([2., 1., -1., -2.], 100.)]:
            with self.subTest(scores=scores):
                result = evaluate_report([dict(uid='eer', labels20=[0, 0, 1, 1],
                                               scores20=scores, duration=.08)], 0.)
                self.assertEqual(result['eer_percent'], expected_eer)
                self.assertEqual(result['supplementary_native20']['eer_percent'], expected_eer)


if __name__ == '__main__':
    unittest.main()
