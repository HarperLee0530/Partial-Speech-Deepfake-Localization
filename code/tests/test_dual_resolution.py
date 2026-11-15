"""Structural adapter checks only; no training loop or optimizer steps."""
import unittest

import torch

from cnn_ps.backbone import CNNBiGRU, ModelConfig, extract_context
from cnn_ps.dual_resolution import DualResolutionAdapter


class DualResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_model(self, variant, seed=666):
        torch.manual_seed(seed)
        return DualResolutionAdapter(torch.zeros(256), torch.ones(256), variant=variant)

    @staticmethod
    def inputs(length=40, active=29):
        generator = torch.Generator().manual_seed(71)
        h = torch.randn(length, 256, generator=generator)
        z = torch.randn(length, generator=generator)
        weights = torch.zeros(length)
        weights[:active] = .02
        if active:
            weights[active - 1] = .007
        return h, z, weights

    @staticmethod
    def activate_outputs(model):
        # Nonzero deterministic test parameters exercise masks and attention;
        # this is not an optimizer update or a fitted experimental checkpoint.
        with torch.no_grad():
            for name in ('fine_output', 'coarse_output', 'attention_output'):
                if hasattr(model, name):
                    layer = getattr(model, name)
                    layer.weight.fill_(.03)
                    layer.bias.fill_(.07)

    def test_zero_initialization_preserves_valid_baseline_for_all_arms(self):
        h, z, weights = self.inputs()
        for variant in ('residual', 'dual', 'fusion'):
            with self.subTest(variant=variant):
                model = self.make_model(variant)
                out = model(h, z, weights)
                torch.testing.assert_close(out['final_logits'], z, rtol=0, atol=0)
                self.assertEqual(out['coarse_logits'].shape, (5,))
                self.assertEqual(out['fine_residual'].shape, z.shape)
                self.assertEqual(out['coarse_residual'].shape, (5,))
                self.assertTrue(torch.isfinite(out['final_logits']).all())

    def test_paired_initialization_and_statistics_are_frozen_buffers(self):
        models = [self.make_model(variant) for variant in ('residual', 'dual', 'fusion')]
        for field in ('fine_projection.weight', 'fine_projection.bias', 'fine_output.weight', 'fine_output.bias'):
            for other in models[1:]:
                self.assertTrue(torch.equal(models[0].state_dict()[field], other.state_dict()[field]), field)
        for field in ('coarse_projection.weight', 'coarse_projection.bias', 'coarse_conv.weight',
                      'coarse_conv.bias', 'coarse_output.weight', 'coarse_output.bias'):
            self.assertTrue(torch.equal(models[1].state_dict()[field], models[2].state_dict()[field]), field)
        self.assertNotIn('feature_mean', dict(models[2].named_parameters()))
        self.assertIn('feature_std', dict(models[2].named_buffers()))

    def test_fractional_pool_ignores_padding_and_uses_duration(self):
        values = torch.tensor([1., 3., float('nan'), 9., 9., 9., 9., 9.] + [99.]*8)
        weights = torch.tensor([.02, .005] + [0.]*14)
        pooled, totals = DualResolutionAdapter.masked_pool(values, weights)
        torch.testing.assert_close(pooled, torch.tensor([1.4, 0.]), rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(totals, torch.tensor([.025, 0.]), rtol=1e-7, atol=2e-9)

    def test_changed_invalid_tails_cannot_change_valid_outputs_or_coarse_auxiliary(self):
        h, z, weights = self.inputs()
        altered_h, altered_z = h.clone(), z.clone()
        altered_h[weights == 0] = float('nan')
        altered_z[weights == 0] = 1e6
        for variant in ('residual', 'dual', 'fusion'):
            model = self.make_model(variant)
            self.activate_outputs(model)
            a, b = model(h, z, weights), model(altered_h, altered_z, weights)
            for key in ('final_logits', 'fine_logits'):
                torch.testing.assert_close(a[key][weights > 0], b[key][weights > 0], rtol=0, atol=0)
                torch.testing.assert_close(b[key][weights == 0], altered_z[weights == 0], rtol=0, atol=0)
            torch.testing.assert_close(a['coarse_logits'], b['coarse_logits'], rtol=0, atol=0)
            # Appending wholly invalid blocks must also preserve all old outputs.
            longer = model(torch.cat((h, torch.full((16, 256), float('nan')))),
                           torch.cat((z, torch.zeros(16))),
                           torch.cat((weights, torch.zeros(16))))
            torch.testing.assert_close(a['final_logits'], longer['final_logits'][:len(z)], rtol=1e-6, atol=1e-7)
            torch.testing.assert_close(a['coarse_logits'], longer['coarse_logits'][:len(z)//8], rtol=1e-6, atol=1e-7)

    def test_attention_is_local_masked_and_safe_without_valid_keys(self):
        model = self.make_model('fusion')
        self.activate_outputs(model)
        h, z, weights = self.inputs(length=80, active=65)
        out = model(h, z, weights)
        attention, indices, mask = out['attention_weights'], out['attention_indices'], out['attention_mask']
        self.assertEqual(attention.shape, (80, 4, 5))
        self.assertEqual(indices.shape, (80, 5))
        center = torch.arange(80)//8
        self.assertTrue(((indices-center[:, None]).abs() <= 2).all())
        self.assertTrue(torch.equal(attention.masked_select(~mask[:, None, :]),
                                    torch.zeros_like(attention.masked_select(~mask[:, None, :]))))
        torch.testing.assert_close(attention.sum(-1)[weights > 0], torch.ones(65, 4))
        empty = model(torch.full_like(h, float('nan')), z, torch.zeros_like(weights))
        for value in empty.values():
            self.assertTrue(torch.isfinite(value).all())
        self.assertEqual(int(torch.count_nonzero(empty['attention_weights'])), 0)

    def test_attention_additional_receptive_field_excludes_distant_blocks(self):
        model = self.make_model('fusion')
        self.activate_outputs(model)
        h, z, weights = self.inputs(length=96, active=96)
        changed = h.clone()
        changed[-8:] += 3.
        before, after = model(h, z, weights), model(changed, z, weights)
        # First query block attends0..2; convolution extends this to input0..3.
        torch.testing.assert_close(before['final_logits'][:8], after['final_logits'][:8], rtol=0, atol=0)

    def test_residual_coarse_supervision_is_pooled_fine_and_dual_averages_residuals(self):
        h, z, weights = self.inputs()
        residual = self.make_model('residual')
        self.activate_outputs(residual)
        out = residual(h, z, weights)
        pooled, _ = residual.masked_pool(out['fine_logits'], weights)
        torch.testing.assert_close(out['coarse_logits'], pooled, rtol=0, atol=0)
        dual = self.make_model('dual')
        self.activate_outputs(dual)
        out = dual(h, z, weights)
        expected = z + .5*out['fine_residual'] + .5*out['coarse_residual'].repeat_interleave(8)
        torch.testing.assert_close(out['final_logits'][weights > 0], expected[weights > 0], rtol=0, atol=0)

    def test_end_to_end_gradients_reach_features_and_original_logits(self):
        model = self.make_model('fusion')
        self.activate_outputs(model)
        h, z, weights = self.inputs()
        h.requires_grad_()
        z.requires_grad_()
        out = model(h, z, weights)
        loss = out['final_logits'].square().sum() + out['coarse_logits'].square().sum()
        loss.backward()
        for name, gradient in (('features', h.grad), ('original_logits', z.grad)):
            self.assertIsNotNone(gradient, name)
            self.assertTrue(torch.isfinite(gradient).all(), name)
            self.assertGreater(int(torch.count_nonzero(gradient)), 0, name)
        for name in ('fine_projection', 'coarse_projection', 'coarse_conv', 'query_projection',
                     'key_projection', 'value_projection', 'attention_output'):
            grad = getattr(model, name).weight.grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(int(torch.count_nonzero(grad)), 0, name)
        self.assertGreater(int(torch.count_nonzero(model.gate_network[0].weight.grad)), 0)

    def test_optional_structure_preserves_initial_baseline_and_common_initialization(self):
        h, z, weights = self.inputs()
        plain = self.make_model('fusion')
        torch.manual_seed(666)
        upgraded = DualResolutionAdapter(torch.zeros(256), torch.ones(256),
            learned_pool=True, boundary_head=True, relative_position=True, reliability_gate=True)
        for name in ('fine_projection', 'coarse_projection', 'coarse_conv',
                     'query_projection', 'key_projection', 'value_projection'):
            for key, value in getattr(plain, name).state_dict().items():
                self.assertTrue(torch.equal(value, getattr(upgraded, name).state_dict()[key]), name)
        out = upgraded(h, z, weights)
        torch.testing.assert_close(out['final_logits'][weights > 0], z[weights > 0], rtol=0, atol=0)
        self.assertEqual(out['boundary_logits'].shape, z.shape)
        self.assertEqual(out['gate_inputs'].shape, (40, 10))
        self.assertEqual(upgraded.relative_bias.shape, (4, 5))
        self.assertEqual(upgraded.coarse_downsample.groups, 256)
        self.assertEqual(upgraded.coarse_downsample.stride, (8,))
        normalized, _, _, _ = upgraded._inputs(h, z, weights)
        expected, _ = upgraded.masked_pool(normalized, weights)
        torch.testing.assert_close(upgraded._coarse_input(normalized, weights), expected, rtol=1e-5, atol=2e-7)

    def test_upgraded_tail_masks_reliability_features_and_parameter_gradients(self):
        torch.manual_seed(19)
        model = DualResolutionAdapter(torch.zeros(256), torch.ones(256),
            learned_pool=True, boundary_head=True, relative_position=True, reliability_gate=True)
        self.activate_outputs(model)
        with torch.no_grad():
            model.boundary_output.weight.fill_(.02)
        h, z, weights = self.inputs()
        h.requires_grad_()
        z.requires_grad_()
        out = model(h, z, weights)
        # Final valid frame has no following valid neighbor, so no padding edge.
        self.assertEqual(float(out['gate_inputs'][28, 8]), 0.)
        self.assertEqual(float(out['gate_inputs'][0, 7]), 0.)
        changed_h, changed_z = h.detach().clone(), z.detach().clone()
        changed_h[weights == 0], changed_z[weights == 0] = float('nan'), 1e6
        changed = model(changed_h, changed_z, weights)
        torch.testing.assert_close(out['final_logits'][weights > 0], changed['final_logits'][weights > 0], rtol=0, atol=0)
        for field in ('coarse_logits', 'boundary_logits', 'gate'):
            torch.testing.assert_close(out[field], changed[field], rtol=0, atol=0)
        (out['final_logits'].square().sum() + out['coarse_logits'].square().sum()
         + out['boundary_logits'].square().sum()).backward()
        for parameter in (model.coarse_downsample.weight, model.boundary_output.weight, model.relative_bias):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(int(torch.count_nonzero(parameter.grad)), 0)
        self.assertTrue(torch.isfinite(h.grad).all())
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_zero_initialized_adapter_preserves_storage_and_end_to_end_cnn_gradients(self):
        torch.manual_seed(12)
        backbone = CNNBiGRU(ModelConfig(dropout=0.))
        adapter = DualResolutionAdapter(torch.zeros(256), torch.ones(256),
            learned_pool=True, boundary_head=True, relative_position=True, reliability_gate=True)
        native, context = extract_context(backbone, torch.randn(1, 1, 16, 80), torch.tensor([16]))
        weights = torch.tensor([.02]*10+[.005]+[0.]*5)
        out = adapter(context[0], native[0], weights)
        torch.testing.assert_close(out['final_logits'], native[0], rtol=0, atol=0)
        torch.testing.assert_close(out['final_logits'].reshape(-1, 8).mean(-1),
                                   native[0].reshape(-1, 8).mean(-1), rtol=0, atol=0)
        # A structural backward pass only; no optimizer, training loop or data.
        out['final_logits'].square().mean().backward()
        for parameter in (backbone.blocks[0][0].weight, backbone.gru.weight_ih_l0,
                          backbone.gru.weight_ih_l0_reverse, backbone.classifier.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(int(torch.count_nonzero(parameter.grad)), 0)

    def test_relative_offset_bias_changes_only_allowed_attention_and_reliability_can_omit_boundary(self):
        model = DualResolutionAdapter(torch.zeros(256), torch.ones(256),
            relative_position=True, reliability_gate=True, boundary_head=False)
        h, z, weights = self.inputs()
        with torch.no_grad():
            model.query_projection.weight.zero_()
            model.query_projection.bias.zero_()
            model.relative_bias.zero_()
            model.relative_bias[:, 2] = 2.
        out = model(h, z, weights)
        self.assertNotIn('boundary_logits', out)
        self.assertEqual(int(torch.count_nonzero(out['gate_inputs'][:, -1])), 0)
        self.assertTrue((out['attention_weights'][8, :, 2] > out['attention_weights'][8, :, 1]).all())
        self.assertEqual(float(out['attention_weights'][0, :, :2].sum()), 0.)

    def test_invalid_statistics_and_grid_are_rejected(self):
        for mean, std in ((torch.full((256,), float('nan')), torch.ones(256)),
                          (torch.zeros(256), -torch.ones(256)),
                          (torch.zeros(255), torch.ones(255))):
            with self.assertRaises(ValueError):
                DualResolutionAdapter(mean, std)
        model = self.make_model('fusion')
        with self.assertRaises(ValueError):
            model(torch.zeros(9, 256), torch.zeros(9), torch.ones(9))
        with self.assertRaises(ValueError):
            model(torch.zeros(8, 256), torch.zeros(8), -torch.ones(8))


if __name__ == '__main__':
    unittest.main()
