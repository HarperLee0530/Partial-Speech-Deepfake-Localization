"""Forward-only B checks: no backward(), optimizer, dataset or training loop."""
import copy
import contextlib
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from transformers import WavLMConfig, WavLMModel
from cnn_ps import wavlm_ablation_b as b
from cnn_ps import wavlm_ablation_a as a
from cnn_ps import train_wavlm_ablation_b as trainer
from cnn_ps.cache_wavlm_ablation_b import build_cache, main as cache_main
from cnn_ps.train_wavlm_ablation_a import model_config, state_digest
import run_wavlm_ablation_b as runner


def tiny_model():
    config = WavLMConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        intermediate_size=32, conv_dim=(8, 8, 8), conv_kernel=(3, 3, 3), conv_stride=(2, 2, 2),
        num_conv_pos_embeddings=8, num_conv_pos_embedding_groups=2,
        num_buckets=8, max_bucket_distance=16, hidden_dropout=0., attention_dropout=0.,
        activation_dropout=0., feat_proj_dropout=0., layerdrop=0., apply_spec_augment=False)
    return WavLMModel(config).eval().requires_grad_(False)


class BGroupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(666)

    @torch.inference_mode()
    def test_control_exact_and_all_switches_preserve_weights_rng(self):
        original = tiny_model()
        wave = torch.randn(1, 320)
        expected = original(wave).last_hidden_state
        digest = state_digest(original)
        for arm in b.ARMS:
            changed = copy.deepcopy(original)
            rng = torch.get_rng_state().clone()
            b.configure_encoder(changed, arm)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(state_digest(changed), digest)
            self.assertFalse(any(p.requires_grad for p in changed.parameters()))
            self.assertFalse(changed.training)
            output = changed(wave).last_hidden_state
            self.assertEqual(output.shape, expected.shape)
            self.assertTrue(torch.isfinite(output).all())
            self.assertEqual(torch.equal(output, expected), arm == 'B0_full')

    @torch.inference_mode()
    def test_disabled_parameters_cannot_affect_output(self):
        wave = torch.randn(1, 320)
        for arm in b.NEW_ARMS:
            model = b.configure_encoder(tiny_model(), arm)
            expected = model(wave).last_hidden_state
            modules = []
            if arm == 'B1_no_conv_position':
                modules.append(model.encoder.pos_conv_embed)
                pos = model.encoder.pos_conv_embed(torch.randn(1, 20, 16))
                self.assertEqual(torch.count_nonzero(pos).item(), 0)
            else:
                for layer in model.encoder.layers:
                    modules.append(layer.attention.gru_rel_pos_linear)
                    layer.attention.gru_rel_pos_const.fill_(100.)
                if arm == 'B3_no_relative_bias':
                    modules.append(model.encoder.layers[0].attention.rel_attn_embed)
            for module in modules:
                for parameter in module.parameters():
                    parameter.fill_(77.)
            self.assertTrue(torch.equal(expected, model(wave).last_hidden_state))

    @torch.inference_mode()
    def test_gate_one_and_zero_bias_match_explicit_attention(self):
        for arm in ('B2_position_gate_one', 'B3_no_relative_bias'):
            model = b.configure_encoder(tiny_model(), arm)
            attention = model.encoder.layers[0].attention
            states = torch.randn(2, 7, 16)
            raw = attention.compute_bias(7, 7).unsqueeze(0).repeat(2, 1, 1, 1).view(4, 7, 7)
            expected_bias = raw if arm == 'B2_position_gate_one' else torch.zeros_like(raw)
            expected = attention.torch_multi_head_self_attention(states, None, expected_bias, False)[0]
            for supplied in (None, raw):
                actual, _, returned_bias = attention(states, position_bias=supplied)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(returned_bias, expected_bias))

    @torch.inference_mode()
    def test_backend_initialization_outputs_losses_match_a0(self):
        reference = a.AblationCNN(model_config(), 'A0_full', 666).eval()
        expected_digest = state_digest(reference)
        features, lengths = torch.randn(1, 1, 16, 2304), torch.tensor([16])
        expected = reference(features, lengths)
        for arm in b.ARMS:
            torch.manual_seed(666)
            model = b.PositionalCNN(model_config(), arm, 666).eval()
            self.assertEqual(state_digest(model), expected_digest)
            self.assertTrue(torch.equal(expected, model(features, lengths)))
            self.assertEqual(b.loss_weights(arm), a.loss_weights('A0_full'))
            self.assertEqual(state_digest(b.make_adapter(b.ARMS[arm], 666, 'cpu')),
                             state_digest(a.make_adapter(a.ARMS['A0_full'], 666, 'cpu')))

    def test_cache_cross_arm_and_stale_source_rejected(self):
        arm = b.NEW_ARMS[0]
        m = dict(schema=b.CACHE_SCHEMA, status='complete', arm=arm,
            encoder_intervention=b.INTERVENTIONS[arm], layers=[6, 9, 12], frozen=True,
            reference_sha256='reference', split='train', recordings=2, channels=3,
            dtype='float32', revision=b.REVISION, model_sha256=b.MODEL_SHA256,
            source_unchanged=True, encoder_sources=b.encoder_sources())
        b.validate_cache_metadata(m, arm, 'reference', 'train', 2, 3)
        for key, value in [('arm', b.NEW_ARMS[1]), ('schema', 'ps_frozen_wavlm_multilayer_cache_v1'),
                           ('encoder_sources', {}), ('model_sha256', 'wrong'), ('split', 'dev')]:
            bad = dict(m, **{key: value})
            with self.assertRaises(ValueError):
                b.validate_cache_metadata(bad, arm, 'reference', 'train', 2, 3)

    def test_eval_guard_precedes_dataset_access(self):
        with patch('cnn_ps.cache_wavlm_ablation_b.FeatureData') as data:
            with self.assertRaises(ValueError):
                build_cache('unopened', 'unopened', 'uncreated', b.NEW_ARMS[0], 'eval')
            data.assert_not_called()

    def test_default_clis_never_train_or_create_outputs(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            output = Path(folder)/'not_created'
            with patch.object(runner, 'run') as run:
                runner.main(['--output', str(output)])
                run.assert_not_called()
            with patch.object(trainer, 'train_arm') as train:
                trainer.main(['--output', str(output)])
                train.assert_not_called()
            with patch('cnn_ps.cache_wavlm_ablation_b.build_cache') as cache:
                cache_main(['--arm', b.NEW_ARMS[0], '--split', 'train', '--reference', 'unused',
                            '--model-folder', 'unused', '--output', str(output)])
                cache.assert_not_called()
            self.assertFalse(output.exists())

    def test_double_intervention_rejected(self):
        model = b.configure_encoder(tiny_model(), b.NEW_ARMS[0])
        with self.assertRaises(ValueError):
            b.configure_encoder(model, b.NEW_ARMS[1])


if __name__ == '__main__':
    unittest.main()
