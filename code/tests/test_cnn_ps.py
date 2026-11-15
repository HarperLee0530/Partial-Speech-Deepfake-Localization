"""Regression checks independent of PS recordings and historical source code.

Run from code/: python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from cnn_ps.augmentation import transform
from cnn_ps.backbone import CNNBiGRU, ModelConfig, load_backbone, masked_bce
from cnn_ps.cached_data import CachedData
from cnn_ps.head import ResidualFrameHead, load_head
from cnn_ps import metrics20, metrics160
from cnn_ps.prepare import allocate_arrays
from cnn_ps.train_head import main as train_head_main, objective


class CNNPSRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_residual_initialization_preserves_native_scores(self):
        torch.manual_seed(7)
        head = ResidualFrameHead()
        native = torch.randn(3, 8)
        context = torch.randn(3, 8, 256).half()
        coarse, weights, fine = head(native, context, return_native=True)
        self.assertTrue(torch.equal(fine, native))
        self.assertTrue(torch.equal(weights, torch.full_like(native, .125)))
        self.assertTrue(torch.equal(coarse, (native*.125).sum(1)))
        self.assertEqual(sum(p.numel() for p in head.parameters()), 8257)
        self.assertFalse(any('gate' in key for key in head.state_dict()))

    def test_checkpoint_roundtrip_and_adaptive_pool_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'head.pt'
            head = ResidualFrameHead()
            payload = dict(state_dict=head.state_dict(), hidden_size=32,
                           fine_aux=True, freeze_gate=True)
            torch.save(payload, path)
            loaded = load_head(path)
            for key, value in head.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[key]))
            payload['freeze_gate'] = False
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                load_head(path)

    def test_backbone_loaded_for_head_is_frozen_and_in_eval_mode(self):
        config = ModelConfig(dropout=.4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cnn.pt'
            torch.save(dict(model_config=config.to_dict(), state_dict=CNNBiGRU(config).state_dict()), path)
            model, frontend = load_backbone(path)
            self.assertFalse(model.training)
            self.assertFalse(frontend.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
            self.assertEqual(model.config.dropout, .4)

    def test_baseline_padding_does_not_enter_loss(self):
        logits = torch.tensor([[1., -1., float('nan')], [2., float('nan'), float('nan')]])
        labels = torch.tensor([[1., 0., float('nan')], [0., float('nan'), float('nan')]])
        result = masked_bce(logits, labels, torch.tensor([2, 1]))
        expected = F.binary_cross_entropy_with_logits(torch.tensor([1., -1., 2.]),
                                                     torch.tensor([1., 0., 0.]))
        self.assertTrue(torch.equal(result, expected))

    def test_actual_duration_weights_and_prediction_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            records = [dict(uid='CON_D_1', length=2, length20=10, duration=.19,
                            labels=np.array([1, 0]), labels20=np.array([1]*3+[0]*7))]
            arrays, _ = allocate_arrays(folder, records, channels=False)
            arrays['native'][:] = 0
            arrays['context'][:] = 0
            for array in arrays.values():
                array.flush()
            meta = dict(split='dev', uids=['CON_D_1'], lengths=[2], lengths20=[10], durations=[.19])
            (folder/'complete.json').write_text(json.dumps(meta), encoding='utf-8')
            cached = CachedData(folder)
            np.testing.assert_allclose(cached.weights.numpy(), [.16, .03], atol=1e-8)
            widths = cached.fine_weights.numpy().reshape(-1)
            np.testing.assert_allclose(widths[:9], .02, atol=1e-8)
            self.assertAlmostEqual(float(widths[9]), .01, places=7)
            self.assertTrue(np.all(widths[10:] == 0))
            scores = np.r_[np.arange(10), np.full(6, 999.)]
            prediction = cached.prediction_records(scores)[0]
            np.testing.assert_array_equal(prediction['scores20'], np.arange(10))
            # Active frames are perfect; inactive trailing values must not alter metrics.
            result = metrics20.evaluate([dict(uid='tail', labels20=[1, 0, 0, 1],
                                              scores20=[1., -1., -1., float('nan')], duration=.05)])
            self.assertAlmostEqual(result['audio_seconds'], .05)
            self.assertAlmostEqual(result['duration_f1_percent'], 100.)
            del cached, arrays, array, widths, prediction

    def test_deterministic_noise_snr_and_channel_length(self):
        rng = np.random.default_rng(9)
        wave = rng.normal(0, .1, 16000).astype(np.float32)
        noise_view = transform(wave, 'one', 1)
        np.testing.assert_array_equal(noise_view, transform(wave, 'one', 1))
        self.assertFalse(np.array_equal(noise_view, transform(wave, 'two', 1)))
        noise = noise_view.astype(np.float64)-wave.astype(np.float64)
        snr = 10*np.log10(np.sum(wave.astype(np.float64)**2)/np.sum(noise**2))
        self.assertAlmostEqual(float(snr), 20., places=5)
        for channel in (0, 1, 2):
            changed = transform(wave, 'one', channel)
            self.assertEqual(changed.shape, wave.shape)
            self.assertEqual(changed.dtype, np.float32)
        self.assertEqual(transform(wave, 'one', 2)[0], wave[0])

    def test_training_loss_is_unbalanced_duration_bce_at_both_grids(self):
        head = ResidualFrameHead()
        native = torch.tensor([[-2., 1., 2., -1., .5, .5, .5, .5],
                               [1., -1., 3., 3., 3., 3., 3., 3.]])
        labels = torch.tensor([1., 0.])
        fine_labels = torch.tensor([[0., 1., 1., 0., 0., 0., 0., 0.], [0.]*8])
        fine_weights = torch.tensor([[.02]*8, [.02, .01, 0., 0., 0., 0., 0., 0.]])
        weights = torch.tensor([.16, .03])
        batch = dict(native=native, context=torch.zeros(2, 8, 256), labels=labels,
                     weights=weights, fine_labels=fine_labels, fine_weights=fine_weights)
        coarse = (native*.125).sum(1)
        # An independent BCE expression makes coefficient/class-weight changes visible.
        expected = (((F.softplus(coarse)-labels*coarse)*weights).sum()/weights.sum()
                    + ((F.softplus(native)-fine_labels*native)*fine_weights).sum()/fine_weights.sum())
        self.assertTrue(torch.allclose(objective(head, batch), expected, atol=1e-7, rtol=1e-7))

    def test_real_score_tie_is_not_split_or_float32_rounded(self):
        records = [dict(uid='tie', labels20=[0, 0, 1],
                        scores20=np.array([.25, .25, 1.], dtype=np.float32), duration=.06)]
        chosen = metrics20.choose_threshold(records, max_real_fpr=.5)
        self.assertGreater(chosen['threshold'], .25)
        np.testing.assert_array_equal(metrics20.decision_mask(records[0]['scores20'], chosen['threshold']),
                                      [False, False, True])
        self.assertEqual(chosen['real_fpr_percent'], 0.)
        self.assertEqual(metrics20.evaluate_dev(records, max_real_fpr=.5)['duration_f1_percent'], 100.)

    def test_short_event_can_be_lost_on_coarse_grid(self):
        labels20 = np.zeros(16, dtype=np.uint8)
        labels20[1] = 1  # a real 20 ms event at [0.02, 0.04)
        fine = metrics20.evaluate([dict(uid='event', labels20=labels20,
                                      scores20=2*labels20.astype(float)-1, duration=.32)])
        coarse = metrics160.evaluate([dict(uid='event', labels=[1, 0], scores=[1., -1.],
                                          labels20=labels20, duration=.32)])
        self.assertEqual(fine['event_counts']['matched'], 1)
        self.assertEqual(coarse['event_counts']['matched'], 0)
        matched, _ = metrics160.match_events([[0, .16]], [[0, .08]])
        self.assertEqual(len(matched), 1)  # IoU exactly 0.5 is included.
        matched, _ = metrics160.match_events([[0, .16]], [[0, .079]])
        self.assertEqual(len(matched), 0)

    def test_tiny_training_pipeline_saves_both_arms_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, letter in [('train', 'T'), ('dev', 'D')]:
                folder = root/split
                folder.mkdir()
                native = np.tile(np.repeat([-.5, .5], 8), 2).reshape(4, 8).astype('f4')
                values = dict(native=native, context=np.zeros((4, 8, 256), 'f2'),
                              labels=np.array([0, 1, 0, 1], 'f4'), weights=np.full(4, .16, 'f4'),
                              fine_labels=(native > 0).astype('f4'), fine_weights=np.full((4, 8), .02, 'f4'))
                for name, array in values.items():
                    np.save(folder/f'{name}.npy', array)
                meta = dict(split=split, uids=[f'CON_{letter}_1', f'CON_{letter}_2'],
                            lengths=[2, 2], lengths20=[16, 16], durations=[.32, .32])
                (folder/'complete.json').write_text(json.dumps(meta), encoding='utf-8')
                if split == 'train':
                    np.save(folder/'native_channels.npy', np.stack([native, native+.1, native-.1]))
                    np.save(folder/'context_channels.npy', np.stack([values['context']]*3))
            argv = ['--train-cache', str(root/'train'), '--dev-cache', str(root/'dev'),
                    '--channel-cache', str(root/'train'), '--output', str(root/'out'),
                    '--arms', 'aug_erm', 'clean_erm', '--seeds', '666', '--threads', '2']
            with contextlib.redirect_stdout(io.StringIO()):
                train_head_main(argv)
            complete = json.loads((root/'out/complete.json').read_text())
            self.assertEqual(complete['count'], 2)
            for run in complete['runs']:
                self.assertTrue(run['reload_exact'])
                self.assertEqual(run['trained_epochs'], 8)
                self.assertFalse(run['eval_read'])
            with self.assertRaises(FileExistsError):
                train_head_main(argv)


if __name__ == '__main__':
    unittest.main()
