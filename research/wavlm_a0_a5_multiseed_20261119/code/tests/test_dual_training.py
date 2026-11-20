"""Synthetic regression checks for end-to-end boundary training and diagnostics.

No training run is started; backward calls only verify local gradient behavior.
"""
import unittest

import numpy as np
import torch

from cnn_ps.diagnose_events import diagnose_events
from cnn_ps.report_native20 import evaluate_report


class DualTrainingTests(unittest.TestCase):
    @staticmethod
    def loss_inputs():
        final = torch.tensor([[-2., 1., 2., -1., .5, .5, .5, .5,
                               1., -1., 3., 3., 3., 3., 3., 3.],
                              [1., 2., -1., -2., 4., 4., 4., 4.] + [0.]*8], requires_grad=True)
        batch = dict(lengths=torch.tensor([16, 8]), label_lengths=torch.tensor([2, 1]),
                     labels=torch.tensor([[1., 0.], [1., 0.]]),
                     weights=torch.tensor([[.16, .045], [.08, 0.]]),
                     fine_labels=torch.tensor([[0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1.] + [0.]*5,
                                               [1., 1., 0., 0.] + [0.]*12]),
                     fine_weights=torch.tensor([[.02]*10 + [.005] + [0.]*5,
                                                [.02]*4 + [0.]*12]))
        return final, batch

    def test_direct_arm_preserves_original_objective_and_gradient_exactly(self):
        from cnn_ps.finetune import objective
        from cnn_ps.train_dual import training_loss
        final, batch = self.loss_inputs()
        reference = objective(final, batch, 'cpu')
        result, parts = training_loss(final, [], batch, 'cpu', False)
        self.assertTrue(torch.equal(result, reference))
        self.assertEqual(parts['coarse_aux'], 0.)
        self.assertEqual(parts['boundary'], 0.)
        self.assertEqual(parts['consistency'], 0.)
        actual_gradient = torch.autograd.grad(result, final, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(reference, final)[0]
        self.assertTrue(torch.equal(actual_gradient, expected_gradient))
        # Storage tail participates only in the historical coarse mean8.
        self.assertNotEqual(float(actual_gradient[0, 15]), 0.)
        self.assertEqual(float(actual_gradient[1, 8:].abs().sum()), 0.)

    def test_all_terms_use_pooled_batch_duration_including_fractional_tail(self):
        from cnn_ps.train_dual import training_loss
        final, batch = self.loss_inputs()
        outputs = [dict(final_logits=final[0], coarse_logits=torch.tensor([.7, -.2]),
                        boundary_logits=torch.linspace(-1., 1., 16)),
                   dict(final_logits=final[1, :8], coarse_logits=torch.tensor([1.3]),
                        boundary_logits=torch.linspace(.2, 1.6, 8))]
        result, parts = training_loss(final, outputs, batch, 'cpu', True)
        # Independent scalar expressions, with a different duration and loss
        # for each recording, detect an accidental mean of per-record means.
        fine_num = coarse_num = aux_num = boundary_num = inside_num = 0.
        fine_den = coarse_den = boundary_den = inside_den = 0.
        for i, n in enumerate((16, 8)):
            z = final[i, :n].detach().double().numpy()
            labels = batch['fine_labels'][i, :n].double().numpy()
            weights = batch['fine_weights'][i, :n].double().numpy()
            coarse_y = batch['labels'][i, :n//8].double().numpy()
            coarse_w = batch['weights'][i, :n//8].double().numpy()
            probability = 1 / (1 + np.exp(-z))
            for t in range(n):
                fine_num += (np.logaddexp(0., z[t]) - labels[t]*z[t]) * weights[t]
                fine_den += weights[t]
                if t and weights[t] > 0 and weights[t-1] > 0:
                    pair = min(weights[t], weights[t-1])
                    target = float(labels[t] != labels[t-1])
                    boundary_logit = float(outputs[i]['boundary_logits'][t])
                    boundary_num += (np.logaddexp(0., boundary_logit)-target*boundary_logit)*pair
                    boundary_den += pair
                    if labels[t] == labels[t-1]:
                        inside_num += (probability[t]-probability[t-1])**2 * pair
                        inside_den += pair
            for c in range(n//8):
                coarse_z = z[c*8:(c+1)*8].mean()
                auxiliary_z = float(outputs[i]['coarse_logits'][c])
                coarse_num += (np.logaddexp(0., coarse_z)-coarse_y[c]*coarse_z)*coarse_w[c]
                aux_num += (np.logaddexp(0., auxiliary_z)-coarse_y[c]*auxiliary_z)*coarse_w[c]
                coarse_den += coarse_w[c]
        expected_parts = dict(main=fine_num/fine_den + coarse_num/coarse_den,
                              coarse_aux=aux_num/coarse_den,
                              boundary=boundary_num/boundary_den,
                              consistency=inside_num/inside_den)
        for key, value in expected_parts.items():
            self.assertAlmostEqual(parts[key], value, places=6, msg=key)
        expected = (expected_parts['main'] + .5*expected_parts['coarse_aux']
                    + .1*expected_parts['boundary'] + .05*expected_parts['consistency'])
        self.assertAlmostEqual(float(result), expected, places=6)

    def test_boundary_targets_exclude_first_frame_and_invalid_pairs(self):
        from cnn_ps.train_dual import boundary_targets
        labels = torch.tensor([1., 1., 0., 0., 1., 0.])
        weights = torch.tensor([.02, .02, .02, .005, 0., 0.])
        targets, pairweights = boundary_targets(labels, weights)
        torch.testing.assert_close(targets, torch.tensor([0., 0., 1., 0., 0., 0.]), rtol=0, atol=0)
        torch.testing.assert_close(pairweights, torch.tensor([0., .02, .02, .005, 0., 0.]), rtol=0, atol=0)

    def test_boundary_targets_do_not_bridge_invalid_gaps(self):
        from cnn_ps.train_dual import boundary_targets
        targets, pairweights = boundary_targets(torch.tensor([0., 1., 1., 0.]),
                                               torch.tensor([.02, 0., .02, .02]))
        torch.testing.assert_close(targets, torch.tensor([0., 0., 0., 1.]), rtol=0, atol=0)
        torch.testing.assert_close(pairweights, torch.tensor([0., 0., 0., .02]), rtol=0, atol=0)

    def test_consistency_uses_probability_differences_and_fractional_pair_duration(self):
        from cnn_ps.train_dual import within_segment_consistency
        probabilities = torch.tensor([.1, .3, .9, .4, .5, .99], dtype=torch.float64)
        logits = torch.logit(probabilities).requires_grad_()
        labels = torch.tensor([0., 0., 1., 1., 1., 0.], dtype=torch.float64)
        weights = torch.tensor([.02, .02, .02, .02, .005, 0.], dtype=torch.float64)
        loss = within_segment_consistency(logits, labels, weights)
        expected = (.02 * .2**2 + .02 * .5**2 + .005 * .1**2) / .045
        self.assertAlmostEqual(float(loss), expected, places=10)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[-1]), 0.)

    def test_consistency_never_smooths_across_truth_changes(self):
        from cnn_ps.train_dual import within_segment_consistency
        logits = torch.tensor([-10., 10., -10., 10.], requires_grad=True)
        result = within_segment_consistency(logits, torch.tensor([0., 1., 0., 1.]), torch.full((4,), .02))
        self.assertEqual(float(result), 0.)
        result.backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros(4)))

    def test_single_frame_has_no_boundary_or_consistency_term(self):
        from cnn_ps.train_dual import boundary_targets, within_segment_consistency
        logits = torch.tensor([1.], requires_grad=True)
        labels, weights = torch.tensor([1.]), torch.tensor([.005])
        targets, pairweights = boundary_targets(labels, weights)
        self.assertEqual(float(targets.sum()), 0.)
        self.assertEqual(float(pairweights.sum()), 0.)
        result = within_segment_consistency(logits, labels, weights)
        self.assertEqual(float(result), 0.)
        result.backward()
        self.assertEqual(float(logits.grad), 0.)


class EventDiagnosticTests(unittest.TestCase):
    @staticmethod
    def recording(uid, truth_slices, prediction_slices, cells=100):
        truth, scores = np.zeros(cells, dtype=np.uint8), np.full(cells, -1.)
        for start, end in truth_slices:
            truth[start:end] = 1
        for start, end in prediction_slices:
            scores[start:end] = 1.
        return dict(uid=uid, labels20=truth, scores20=scores, duration=cells * .02)

    def test_unmatched_overlap_categories_reconcile_with_real_event_metrics(self):
        # One fragmented truth; the first fragment matches, the small second
        # fragment overlaps but fails IoU. A distant prediction is independent.
        records = [self.recording('fragment', [(10, 30)], [(10, 23), (25, 28), (40, 45)]),
                   self.recording('empty', [], [])]
        original_scores = records[0]['scores20'].copy()
        report = diagnose_events(records, 0.)
        metric = evaluate_report(records, 0.)
        counts = report['counts']
        for key in ('matched', 'predicted', 'truth', 'unmatched_predictions', 'missed_truth'):
            self.assertEqual(counts[key], metric['event_counts'][key])
        self.assertEqual(counts['unmatched_pred_no_truth_overlap'], 1)
        self.assertEqual(counts['unmatched_pred_with_truth_overlap'], 1)
        self.assertEqual(counts['unmatched_pred_overlap_below_iou_threshold'], 1)
        self.assertEqual(counts['truth_with_multiple_overlapping_predictions'], 1)
        np.testing.assert_array_equal(records[0]['scores20'], original_scores)

    def test_touching_intervals_have_zero_overlap_and_joined_truths_are_only_proxy(self):
        records = [self.recording('touch', [(10, 20)], [(20, 25)]),
                   self.recording('merge', [(10, 20), (30, 40)], [(10, 40)])]
        report = diagnose_events(records, 0.)
        self.assertEqual(report['counts']['unmatched_pred_no_truth_overlap'], 1)
        self.assertEqual(report['counts']['prediction_with_multiple_overlapping_truths'], 1)
        self.assertEqual(report['counts']['unmatched_pred_with_truth_overlap'], 1)
        self.assertIn('not causal attribution', report['diagnostic_note'])


if __name__ == '__main__':
    unittest.main()
