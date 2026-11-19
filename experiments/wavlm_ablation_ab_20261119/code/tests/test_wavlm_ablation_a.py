"""Causal switch isolation and exact shared-path equivalence."""
import unittest
import torch
from cnn_ps import wavlm_ablation_a as a
from cnn_ps import train_wavlm_multilayer as old
from cnn_ps.train_dual import forward_batch, training_loss as old_loss
from cnn_ps.finetune import fine_collate
from cnn_ps.dual_resolution import DualResolutionAdapter
from test_four_training import TinyFeatures


class AGroupTests(unittest.TestCase):
    def setUp(self):
        old.setup(666, 2)

    def batch(self, single=False):
        dataset = TinyFeatures('train')
        rows = [dataset[i] for i in range(2)]
        for row in rows:
            row['features'] = torch.randn(1, row['features'].shape[1], 768 if single else 2304)
        return fine_collate(rows)

    def test_all_options_preserve_shared_initialization_and_rng(self):
        expected = None
        for name, options in a.ARMS.items():
            old.setup(666, 2)
            model = a.AblationCNN(old.model_config(), name, 666)
            adapter = a.make_adapter(options, 666, 'cpu')
            current = (old.state_digest(model.cnn), old.state_digest(model.project), old.state_digest(adapter))
            rng = torch.get_rng_state()
            if expected is None:
                expected, expected_rng = current, rng
            self.assertEqual(current, expected)
            self.assertTrue(torch.equal(rng, expected_rng))

    def test_reference_output_loss_gradients_and_update_exact_cpu_cuda(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            for arm, prior in [('A0_full', 'fusion6_9_12'), ('A1_layer12', 'layer12')]:
                batch = self.batch(single=arm == 'A1_layer12')
                states = []
                for original in (True, False):
                    old.setup(666, 2)
                    if original:
                        model = old.ProjectedCNN(old.model_config(), prior, 666).to(device)
                        adapter = old.make_adapter(old.ARMS[prior], 666, device)
                    else:
                        model = a.AblationCNN(old.model_config(), arm, 666).to(device)
                        adapter = a.make_adapter(a.ARMS[arm], 666, device)
                    parameters = list(model.parameters()) + list(adapter.parameters())
                    optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.0001)
                    result, outputs = forward_batch(model, adapter, batch, device)
                    loss, parts = (old_loss(result, outputs, batch, device, True) if original else
                                   a.training_loss(result, outputs, batch, device, a.loss_weights(arm)))
                    loss.backward()
                    gradients = [p.grad.detach().cpu().clone() if p.grad is not None else None for p in parameters]
                    torch.nn.utils.clip_grad_norm_(parameters, 5., error_if_nonfinite=True)
                    optimizer.step()
                    states.append((result.detach().cpu(), loss.detach().cpu(), parts, gradients,
                                   old.state_digest(model), old.state_digest(adapter)))
                for i in (0, 1):
                    self.assertTrue(torch.equal(states[0][i], states[1][i]), (device, arm, i))
                self.assertEqual(states[0][2], states[1][2])
                for g, h in zip(states[0][3], states[1][3]):
                    self.assertTrue((g is None and h is None) or torch.equal(g, h))
                self.assertEqual(states[0][4:], states[1][4:])

    def test_each_loss_switch_changes_only_its_term(self):
        batch = self.batch()
        model = a.AblationCNN(old.model_config(), 'A0_full', 666)
        adapter = a.make_adapter(a.ARMS['A0_full'], 666, 'cpu')
        final, outputs = forward_batch(model, adapter, batch, 'cpu')
        total, parts = a.training_loss(final, outputs, batch, 'cpu', a.loss_weights('A0_full'))
        for arm, term, coefficient in [('A8_no_boundary', 'boundary', .1),
                                        ('A9_no_consistency', 'consistency', .05),
                                        ('A10_no_coarse_aux', 'coarse_aux', .5)]:
            changed, components = a.training_loss(final, outputs, batch, 'cpu', a.loss_weights(arm))
            self.assertAlmostEqual(float(total-changed), coefficient*parts[term], places=6)
            self.assertEqual(components[term], 0.)
            for other in set(parts)-{term}:
                self.assertEqual(parts[other], components[other])

    def test_disabled_modules_cannot_affect_outputs_and_receive_no_gradient(self):
        h, z, weights = torch.randn(24, 256), torch.randn(24), torch.full((24,), .02)
        weights[-3:] = 0
        for arm in ['A5_fixed_pool', 'A6_no_attention', 'A7_fixed_gate', 'A8_no_boundary']:
            adapter = a.make_adapter(a.ARMS[arm], 666, 'cpu')
            before = adapter(h, z, weights)['final_logits']
            with torch.no_grad():
                for name in adapter.disabled_modules:
                    for p in getattr(adapter, name).parameters():
                        p.fill_(123.)
            output = adapter(h, z, weights)
            self.assertTrue(torch.equal(before, output['final_logits']), arm)
            output['final_logits'].sum().backward()
            for name in adapter.disabled_modules:
                for p in getattr(adapter, name).parameters():
                    self.assertFalse(p.requires_grad)
                    self.assertIsNone(p.grad)
            if arm == 'A6_no_attention':
                self.assertEqual(float(output['attention_residual'].abs().sum()), 0.)
            if arm == 'A7_fixed_gate':
                self.assertTrue(torch.equal(output['gate'][weights > 0], torch.full((21,), .5)))
            if arm == 'A8_no_boundary':
                self.assertNotIn('boundary_logits', output)

    def test_single_layer_and_fixed_mean_inputs_are_correct(self):
        states = torch.randn(2, 1, 16, 3, 768)
        lengths = torch.tensor([16, 8])
        for arm, index in [('A1_layer12', 2), ('A2_layer6', 0), ('A3_layer9', 1)]:
            old.setup(666, 2)
            model = a.AblationCNN(old.model_config(), arm, 666).eval()
            old.setup(666, 2)
            reference = old.ProjectedCNN(old.model_config(), 'layer12', 666).eval()
            self.assertTrue(torch.equal(model(states[..., index, :], lengths),
                                        reference(states[..., index, :], lengths)))
            self.assertEqual(model.layer_weights().argmax().item(), index)
        old.setup(666, 2)
        model = a.AblationCNN(old.model_config(), 'A4_mean', 666).eval()
        old.setup(666, 2)
        reference = old.ProjectedCNN(old.model_config(), 'layer12', 666).eval()
        self.assertNotIn('layer_logits', dict(model.named_parameters()))
        self.assertTrue(torch.equal(model(states.flatten(-2), lengths),
                                    reference((states*(1/3)).sum(-2), lengths)))


if __name__ == '__main__':
    unittest.main()
