"""Main-checkpoint inference regressions: adapter use, fixed threshold and tails."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from cnn_ps.backbone import CNNBiGRU, LogMel
from cnn_ps.predict_mel import decode_events, load_main_checkpoint, predict, waveform_batch
from cnn_ps.train_four import ARMS, SCHEMA, make_adapter, model_config


class MainPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = model_config()
        cls.frontend = LogMel(cls.config).eval()
        model = CNNBiGRU(cls.config)
        adapter = make_adapter(ARMS['boundary'], 666, 'cpu')
        with torch.no_grad():
            for parameter in list(model.parameters()) + list(adapter.parameters()):
                parameter.zero_()
            model.classifier.bias.fill_(1.)
            adapter.fine_output.bias.fill_(2.)
        # Base logit is1; fine residual2 at gate0.5 makes final logit2.
        # A loader that omits the adapter would produce the opposite decision.
        cls.payload = dict(schema=SCHEMA, protocol=dict(schema=SCHEMA, arm='boundary', seed=666,
                           adapter_options=copy.deepcopy(ARMS['boundary'])),
                           model_config=cls.config.to_dict(), adapter_options=copy.deepcopy(ARMS['boundary']),
                           backbone_state_dict=model.state_dict(), adapter_state_dict=adapter.state_dict(),
                           threshold=1.5, dev_metrics=dict(threshold=1.5, real_fpr_percent=4.9), epoch=3)

    def test_grid_excludes_padding_and_preserves_fractional_duration(self):
        for samples, actual, stored in ((1, 1, 8), (319, 1, 8), (320, 1, 8),
                                        (321, 2, 8), (2559, 8, 8), (2560, 8, 8), (2561, 9, 16)):
            with self.subTest(samples=samples):
                batch, duration, frames = waveform_batch(np.zeros(samples, np.float32), 16000, self.frontend)
                self.assertEqual(frames, actual)
                self.assertEqual(batch['features'].shape, (1, 1, stored, 80))
                self.assertEqual(int(batch['lengths'][0]), stored)
                self.assertEqual(int((batch['fine_weights'] > 0).sum()), actual)
                self.assertTrue(torch.equal(batch['fine_weights'][0, actual:], torch.zeros(stored-actual)))
                self.assertAlmostEqual(float(batch['fine_weights'].sum()), duration, places=7)
                self.assertAlmostEqual(float(batch['weights'].sum()), duration, places=7)

    def test_float64_tie_cut_preserves_gaps_and_clips_last_event(self):
        cut = np.nextafter(np.float64(.25), np.float64(np.inf))
        scores = np.array([.25, .3, .25, .4], dtype=np.float32)
        actual = decode_events(scores, cut, .065)
        np.testing.assert_array_equal(actual, [[.02, .04], [.06, .065]])

    def test_complete_checkpoint_adapter_and_threshold_are_used_without_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint, audio, output = root/'model.pt', root/'audio.wav', root/'result.json'
            torch.save(self.payload, checkpoint)
            sf.write(audio, np.zeros(3252, np.float32), 16000, subtype='FLOAT')
            result = predict(audio, output, checkpoint, threads=1)
            self.assertEqual(result['native20_actual_frames'], 11)
            self.assertEqual(result['stored20_frames'], 16)
            np.testing.assert_array_equal(result['scores20'], np.full(11, 2.))
            self.assertEqual(result['predicted_fake_intervals_seconds'], [[0., 3252/16000]])
            self.assertEqual(result['threshold'], 1.5)
            with self.assertRaises(FileExistsError):
                predict(audio, output, checkpoint, threads=1)
            payload = copy.deepcopy(self.payload)
            payload['threshold'] = float(np.nextafter(np.float64(2), np.float64(np.inf)))
            payload['dev_metrics']['threshold'] = payload['threshold']
            torch.save(payload, checkpoint)
            result = predict(audio, root/'above_tie.json', checkpoint, threads=1)
            self.assertEqual(result['predicted_fake_intervals_seconds'], [])
            self.assertGreater(result['threshold'], 2.)

    def test_legacy_other_arm_and_missing_adapter_state_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.pt'
            torch.save({'schema': 'legacy', 'state_dict': {}}, path)
            with self.assertRaises(ValueError):
                load_main_checkpoint(path)
            wrong = copy.deepcopy(self.payload)
            wrong['adapter_options'] = ARMS['dual']
            torch.save(wrong, path)
            with self.assertRaises(ValueError):
                load_main_checkpoint(path)
            missing = copy.deepcopy(self.payload)
            del missing['adapter_state_dict']['fine_output.bias']
            torch.save(missing, path)
            with self.assertRaises(RuntimeError):
                load_main_checkpoint(path)

    def test_invalid_audio_is_rejected(self):
        for wave, rate in ((np.zeros(0), 16000), (np.zeros((10, 2)), 16000),
                           (np.zeros(10), 8000), (np.array([np.nan]), 16000)):
            with self.assertRaises(ValueError):
                waveform_batch(wave, rate, self.frontend)


if __name__ == '__main__':
    unittest.main()
