"""CNN-BiGRU/channel fine-tuning regressions using synthetic inputs only.

No training loop or optimizer is run. From code/:
  python -m unittest discover -s tests -v
"""
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch
from torch.nn import functional as F

from cnn_ps.augmentation import transform
from cnn_ps.backbone import CNNBiGRU, ModelConfig, load_backbone, masked_bce
from cnn_ps.finetune import FineData, fine_collate, objective
from cnn_ps import metrics20, metrics160
from cnn_ps.predict import predict


class CNNPSRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_original_cnn_checkpoint_format_preserves_weights_and_predictions(self):
        torch.manual_seed(7)
        config = ModelConfig(dropout=.4)
        original = CNNBiGRU(config).eval()
        features, lengths = torch.randn(1, 1, 16, 80), torch.tensor([16])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'cnn.pt'
            torch.save(dict(model_config=config.to_dict(), state_dict=original.state_dict()), path)
            model, frontend = load_backbone(path)
            self.assertFalse(model.training)
            self.assertFalse(frontend.training)
            self.assertTrue(all(not parameter.requires_grad for parameter in model.parameters()))
            self.assertEqual(model.config.dropout, .4)
            self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 911521)
            for key, value in original.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
            with torch.inference_mode():
                self.assertTrue(torch.equal(original(features, lengths), model(features, lengths)))

    def test_retained_entrypoints_import_without_starting_work(self):
        for name in ('baseline', 'finetune', 'evaluate_cnn', 'predict'):
            with self.subTest(module=name):
                self.assertTrue(callable(importlib.import_module(f'cnn_ps.{name}').main))

    def test_baseline_padding_does_not_enter_loss(self):
        logits = torch.tensor([[1., -1., float('nan')], [2., float('nan'), float('nan')]])
        labels = torch.tensor([[1., 0., float('nan')], [0., float('nan'), float('nan')]])
        result = masked_bce(logits, labels, torch.tensor([2, 1]))
        expected = F.binary_cross_entropy_with_logits(torch.tensor([1., -1., 2.]),
                                                     torch.tensor([1., 0., 0.]))
        self.assertTrue(torch.equal(result, expected))

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

    def test_finetune_dataset_preserves_labels_duration_and_rotates_waveform_views(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            uid = 'CON_T_1'
            wave = (.1*np.sin(np.arange(3280)*2*np.pi*440/16000)).astype('f4')
            sf.write(folder/'audio.wav', wave, 16000, subtype='FLOAT')
            np.save(folder/'coarse.npy', {uid: np.array([1, 0])})  # official 1=real
            np.savez_compressed(folder/'fine.npz', **{uid: np.array([0]*8+[1]*3)})
            manifest = dict(split='train', records=[dict(uid=uid, audio='audio.wav', labels_file='coarse.npy')])
            (folder/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            dataset = FineData(folder/'manifest.json', folder/'fine.npz', ModelConfig(), 'train', 666)
            reference = dataset[0]
            self.assertEqual(reference['features'].shape, (1, 16, 80))
            np.testing.assert_array_equal(reference['labels'].numpy(), [0, 1])
            np.testing.assert_allclose(reference['weights'].numpy(), [.16, .045], atol=1e-8)
            np.testing.assert_allclose(reference['fine_weights'][:10].numpy(), .02, atol=1e-8)
            self.assertAlmostEqual(float(reference['fine_weights'][10]), .005, places=8)
            self.assertTrue(torch.equal(reference['fine_weights'][11:], torch.zeros(5)))
            channels = []
            for epoch in (1, 2, 3):
                dataset.epoch = epoch
                changed = dataset[0]
                channels.append(changed['channel'])
                self.assertEqual(changed['duration'], reference['duration'])
                for key in ('labels', 'fine_labels', 'weights', 'fine_weights'):
                    self.assertTrue(torch.equal(changed[key], reference[key]), key)
            self.assertEqual(set(channels), {0, 1, 2})
            batch = fine_collate([reference])
            self.assertEqual(batch['lengths'].tolist(), [16])
            self.assertEqual(batch['label_lengths'].tolist(), [2])
            self.assertEqual(batch['lengths20'], [11])

    @staticmethod
    def loss_batch():
        return dict(lengths=torch.tensor([16]), label_lengths=torch.tensor([2]),
                    labels=torch.tensor([[1., 0.]]), weights=torch.tensor([[.16, .03]]),
                    fine_labels=torch.tensor([[0., 1., 1., 0., 0., 0., 0., 0.]+[0.]*8]),
                    fine_weights=torch.tensor([[.02]*9+[.01]+[0.]*6]))

    def test_direct_finetune_loss_weights_actual_duration_at_both_grids(self):
        native = torch.tensor([[-2., 1., 2., -1., .5, .5, .5, .5,
                                1., -1., 3., 3., 3., 3., 3., 3.]])
        batch = self.loss_batch()
        coarse = native.reshape(1, 2, 8).mean(-1)
        # Independent softplus BCE expression detects wrong coefficients,
        # class balancing and treating the fractional tail as a full cell.
        expected = (((F.softplus(coarse)-batch['labels']*coarse)*batch['weights']).sum()/batch['weights'].sum()
                    + ((F.softplus(native)-batch['fine_labels']*native)*batch['fine_weights']).sum()/batch['fine_weights'].sum())
        self.assertTrue(torch.allclose(objective(native, batch, 'cpu'), expected, atol=1e-7, rtol=1e-7))

    def test_direct_finetune_loss_backpropagates_to_original_cnn_gru_classifier(self):
        torch.manual_seed(11)
        model = CNNBiGRU(ModelConfig(dropout=0.)).train()
        batch = self.loss_batch()
        native = model(torch.randn(1, 1, 16, 80), batch['lengths'])
        loss = objective(native, batch, 'cpu')
        self.assertTrue(torch.isfinite(loss))
        loss.backward()  # No optimizer step or training run.
        for name, parameter in [('cnn', model.blocks[0][0].weight),
                                ('gru_forward', model.gru.weight_ih_l0),
                                ('gru_reverse', model.gru.weight_ih_l0_reverse),
                                ('classifier', model.classifier.weight)]:
            with self.subTest(stage=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(int(torch.count_nonzero(parameter.grad)), 0)

    def test_raw_cnn_prediction_clips_fractional_tail_at_both_grids(self):
        torch.manual_seed(5)
        config = ModelConfig(dropout=.4)
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            path = folder/'cnn.pt'
            torch.save(dict(model_config=config.to_dict(), state_dict=CNNBiGRU(config).state_dict()), path)
            wave = (.1*np.sin(np.arange(3280)*2*np.pi*440/16000)).astype('f4')
            sf.write(folder/'audio.wav', wave, 16000, subtype='FLOAT')
            for resolution in (.02, .16):
                result = predict(folder/'audio.wav', path, folder/f'prediction_{resolution}.json',
                                 threshold=-1e6, resolution=resolution, device='cpu', threads=2)
                self.assertEqual(len(result['scores20']), 11)
                self.assertEqual(len(result['scores160']), 2)
                np.testing.assert_allclose(result['predicted_fake_intervals_seconds'], [[0., .205]], atol=1e-12, rtol=0)

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
        labels20[1] = 1  # A true20 event at [0.02, 0.04).
        fine = metrics20.evaluate([dict(uid='event', labels20=labels20,
                                      scores20=2*labels20.astype(float)-1, duration=.32)])
        coarse = metrics160.evaluate([dict(uid='event', labels=[1, 0], scores=[1., -1.],
                                          labels20=labels20, duration=.32)])
        self.assertEqual(fine['event_counts']['matched'], 1)
        self.assertEqual(coarse['event_counts']['matched'], 0)
        matched, _ = metrics160.match_events([[0, .16]], [[0, .08]])
        self.assertEqual(len(matched), 1)
        matched, _ = metrics160.match_events([[0, .16]], [[0, .079]])
        self.assertEqual(len(matched), 0)


if __name__ == '__main__':
    unittest.main()
