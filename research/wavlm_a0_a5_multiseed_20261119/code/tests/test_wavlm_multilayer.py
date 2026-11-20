import unittest
import torch
from cnn_ps.train_wavlm import setup, model_config, state_digest
from cnn_ps.wavlm_backend import ProjectedCNN
from cnn_ps.wavlm_multilayer import LayerFusionCNN


class MultilayerTests(unittest.TestCase):
    def test_baseline_is_exact_original_backend(self):
        setup(666, 2)
        original = ProjectedCNN(model_config(), 'wavlm_frozen', 666).eval()
        setup(666, 2)
        baseline = LayerFusionCNN(model_config(), 'layer12', 666).eval()
        self.assertEqual(state_digest(original), state_digest(baseline))
        x = torch.randn(2, 1, 16, 768)
        lengths = torch.tensor([16, 8])
        with torch.no_grad():
            self.assertTrue(torch.equal(original(x, lengths), baseline(x, lengths)))

    def test_fusion_adds_exactly_three_parameters_without_changing_rng(self):
        setup(666, 2)
        baseline = LayerFusionCNN(model_config(), 'layer12', 666)
        rng = torch.get_rng_state()
        setup(666, 2)
        fusion = LayerFusionCNN(model_config(), 'fusion6_9_12', 666)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(state_digest(baseline.cnn), state_digest(fusion.cnn))
        self.assertEqual(state_digest(baseline.project), state_digest(fusion.project))
        self.assertEqual(sum(p.numel() for p in fusion.parameters()) -
                         sum(p.numel() for p in baseline.parameters()), 3)
        torch.testing.assert_close(fusion.layer_weights(), torch.full((3,), 1/3))

    def test_fusion_matches_explicit_mix_and_receives_gradients(self):
        setup(666, 2)
        fusion = LayerFusionCNN(model_config(), 'fusion6_9_12', 666).eval()
        setup(666, 2)
        baseline = LayerFusionCNN(model_config(), 'layer12', 666).eval()
        states = torch.randn(2, 1, 16, 3, 768)
        lengths = torch.tensor([16, 8])
        actual = fusion(states.flatten(-2), lengths)
        expected = baseline((states * (1/3)).sum(-2), lengths)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        self.assertTrue(torch.isfinite(fusion.layer_logits.grad).all())
        self.assertGreater(float(fusion.layer_logits.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
