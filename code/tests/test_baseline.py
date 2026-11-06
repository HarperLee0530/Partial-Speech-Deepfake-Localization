"""CPU-only synthetic correctness tests; never train on project datasets.

Temporary artifacts stay in a temporary code subdirectory and are removed.
"""
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch


# Allow VS Code's Run Python File from any terminal directory.
import sys
from pathlib import Path
_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

import cnn.baseline as pipeline
from cnn.baseline import collate
from cnn.baseline import (CNNBiGRU, LogMel, ModelConfig, intervals_to_spoof_labels,
                            masked_bce, pool_logits, probabilities_to_intervals, sal_logits)
from tools.reporting import compare_pair
from sal.pipeline import calculate_metrics
import sal.pipeline as sal_evaluation

ROOT = Path(__file__).resolve().parent


def small_config():
    return ModelConfig(n_mels=8, channels=(2,), hidden_size=4, gru_layers=1, dropout=0.)


class SyntheticDataset:
    def __init__(self, split, model_config, crop_seconds=None, data_profile='fixed10'):
        self.resolution = 0.02 if split == 'lps' else 0.16
        self.manifest_sha256 = 'synthetic-only-test-manifest'
        self.split = split
        self.n_mels = model_config.n_mels

    def __len__(self):
        return 4

    def __getitem__(self, index):
        labels = torch.tensor([float(index % 2), float(1-index % 2)])
        frames = len(labels)*round(self.resolution/0.02)
        generator = torch.Generator().manual_seed(100+index)
        return dict(uid=f'{self.split}_{index}', features=torch.randn(1, frames, self.n_mels, generator=generator),
                    labels=labels, duration=len(labels)*self.resolution)


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(666)

    def test_frontend_length_and_silence(self):
        frontend = LogMel(small_config())
        for samples, expected in ((1, 1), (319, 1), (320, 1), (321, 2), (2560, 8)):
            features = frontend(torch.zeros(samples))
            self.assertEqual(tuple(features.shape), (1, expected, 8))
            self.assertTrue(torch.isfinite(features).all())

    def test_sal_combined_entry_routes_without_running_experiments(self):
        with patch.object(sal_evaluation, 'require_no_other_sal_process'), \
             patch.object(sal_evaluation, 'run', return_value=0) as run:
            self.assertEqual(sal_evaluation.click_main(['--mode', 'ps']), 0)
            run.assert_called_once_with(['--split', 'eval'])
        with patch.object(sal_evaluation, 'require_no_other_sal_process'), \
             patch.object(sal_evaluation, 'run', side_effect=[0, 1]) as run, \
             patch('tools.reporting.verify_run') as verify, patch('tools.reporting.summarize') as summarize:
            self.assertEqual(sal_evaluation.click_main(['--mode', 'full']), 1)
            self.assertEqual(run.call_count, 2)
            verify.assert_called_once()
            summarize.assert_not_called()

    def test_variable_lengths_do_not_change_valid_predictions(self):
        model = CNNBiGRU(small_config()).eval()
        short = torch.randn(1, 8, 8)
        long = torch.randn(1, 16, 8)
        items = [dict(features=short, labels=torch.zeros(1), uid='a', duration=.16),
                 dict(features=long, labels=torch.zeros(2), uid='b', duration=.32)]
        batch = collate(items)
        with torch.no_grad():
            alone = model(short.unsqueeze(0), torch.tensor([8]))
            together = model(batch['features'], batch['lengths'])
        torch.testing.assert_close(alone[0], together[0, :8], atol=1e-6, rtol=1e-5)
        self.assertEqual(int(torch.count_nonzero(together[0, 8:])), 0)

    def test_masked_loss_ignores_padding_and_gradients(self):
        logits = torch.tensor([[.3, -.4], [.2, 123.]], requires_grad=True)
        targets = torch.tensor([[1., 0.], [0., float('nan')]])
        loss = masked_bce(logits, targets, torch.tensor([2, 1]))
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits.flatten()[:3], targets.flatten()[:3])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(float(logits.grad[1, 1]), 0.)

    def test_pooling_and_label_orientation(self):
        pooled, lengths = pool_logits(torch.arange(16.).reshape(1, -1), torch.tensor([16]), .16)
        torch.testing.assert_close(pooled, torch.tensor([[3.5, 11.5]]))
        self.assertEqual(lengths.tolist(), [2])
        logits = sal_logits(torch.tensor([-2., 2., 0.]))
        self.assertEqual(logits.argmax(-1).tolist(), [1, 0, 0])
        torch.testing.assert_close(logits[:, 1]-logits[:, 0], torch.tensor([2., -2., 0.]))

    def test_intervals_boundaries_and_empty_prediction(self):
        flags = [0., 0., 1., 1., 1., 0., 0., 1., 1., 0.]
        self.assertEqual(probabilities_to_intervals(flags, .02, .2), [[.04, .10], [.14, .18]])
        self.assertEqual(probabilities_to_intervals([0., 0.], .02, .04), [])
        self.assertEqual(probabilities_to_intervals([1., 1.], .02, .031), [[0., .031]])
        self.assertEqual(intervals_to_spoof_labels([[.03, .07]], 5, .02, .1).tolist(), [0., 1., 1., 0., 0.])
        with self.assertRaises(ValueError):
            probabilities_to_intervals([float('nan')], .02, .02)

    def test_metrics_exactly_match_original_database_evaluator(self):
        scores = torch.tensor([-2.1, .7, -.2, 2.3, 0., -1.])
        spoof = torch.tensor([0., 1., 1., 1., 0., 0.])
        database = sqlite3.connect(':memory:')
        database.execute('CREATE TABLE predictions (ordinal INTEGER, logits BLOB, labels BLOB)')
        database.execute('INSERT INTO predictions VALUES (?,?,?)',
                         (0, sal_logits(scores).numpy().astype(np.float32).tobytes(),
                          (1-spoof).numpy().astype(np.uint8).tobytes()))
        expected = calculate_metrics(database)
        actual = pipeline.official_metrics(scores, spoof)
        database.close()
        for key in ('eer_percent', 'eer_margin_threshold', 'f1_bonafide_percent', 'accuracy_percent', 'frame_count'):
            self.assertEqual(expected[key], actual[key], key)

    def test_train_epoch_updates_cnn_gru_and_head(self):
        model = CNNBiGRU(small_config())
        before = copy.deepcopy(model.state_dict())
        dataset = SyntheticDataset('train', small_config())
        batches = [collate([dataset[0], dataset[1]])]
        loss = pipeline.train_epoch(model, batches, torch.optim.AdamW(model.parameters(), lr=.01), 'cpu')
        self.assertTrue(np.isfinite(loss))
        for name in ('blocks.0.0.weight', 'gru.weight_ih_l0', 'classifier.weight'):
            self.assertFalse(torch.equal(before[name], model.state_dict()[name]), name)

    def test_comparison_rejects_mismatched_manifest_and_incomplete_results(self):
        base = dict(status='complete', split='eval', manifest_sha256='abc', resolution_seconds=.16,
                    utterance_count=10, frame_count=100, f1_rule='same', eer_rule='same', precision='float32',
                    batch_size=1, full_utterance=True, eer_percent=3., f1_bonafide_percent=97.)
        self.assertTrue(compare_pair(base, dict(base))['comparable'])
        self.assertFalse(compare_pair(base, dict(base, manifest_sha256='other'))['comparable'])
        self.assertFalse(compare_pair(base, dict(base, status='running'))['comparable'])

    def test_full_pipeline_order_and_fixed_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='cnn_bigru_test_', dir=ROOT) as directory:
            training_file = Path(directory)/'training.json'
            training_file.write_text(json.dumps(dict(status='complete', checkpoint='../data/models/this_run.pt')))
            calls = []

            def fake_train():
                calls.append('train')
                return training_file

            def fake_evaluate(split, checkpoint_path=None, config=None):
                calls.append((split, checkpoint_path))
                return Path(directory)/(split+'.json')

            with patch.object(pipeline, 'train', side_effect=fake_train), \
                 patch.object(pipeline, 'evaluate', side_effect=fake_evaluate), \
                 patch('tools.reporting.compare', side_effect=lambda **kwargs: calls.append('compare')):
                pipeline.main('full')
            self.assertEqual(calls, ['train', ('eval', '../data/models/this_run.pt'),
                                     ('lps', '../data/models/this_run.pt'), 'compare'])

    def test_full_pipeline_stops_when_training_fails(self):
        with patch.object(pipeline, 'train', side_effect=RuntimeError('training failed')), \
             patch.object(pipeline, 'evaluate') as evaluate, patch('tools.reporting.compare') as compare:
            with self.assertRaises(RuntimeError):
                pipeline.main('full')
            evaluate.assert_not_called()
            compare.assert_not_called()

    def test_named_sal_result_has_priority_over_abandoned_duplicate(self):
        from tools.reporting import find_results
        with tempfile.TemporaryDirectory(prefix='cnn_bigru_test_', dir=ROOT) as directory:
            root = Path(directory)
            selected = dict(run='selected', split='lps', smoke_test=False, started_local='2026-01-01', status='complete')
            (root/'lps_result.json').write_text(json.dumps(selected))
            (root/'sal_wavlm_lps_full_later.json').write_text(json.dumps(dict(selected, run='abandoned', started_local='2026-02-01', status='running')))
            with patch('tools.reporting.RESULT_ROOT', root):
                data, name = find_results('sal', 'lps')
            self.assertEqual(data['run'], 'selected')
            self.assertEqual(name, 'lps_result.json')



    def test_training_checkpoint_evaluation_and_intervals_end_to_end(self):
        # Only four synthetic items, CPU, temporary output: not a real experiment.
        with tempfile.TemporaryDirectory(prefix='cnn_bigru_test_', dir=ROOT) as directory:
            root = Path(directory)
            code, data, results = root/'code', root/'data', root/'result'
            code.mkdir()
            config = dict(seed=666, cpu_threads=2, device='cpu', model=small_config().to_dict(),
                          batch_size=2, crop_seconds=4., num_workers=0, epochs=2, patience=1,
                          learning_rate=.001, weight_decay=.0001, checkpoint=None)
            with patch.multiple(pipeline, CODE_ROOT=code, DATA_ROOT=data, RESULT_ROOT=results), \
                 patch.object(pipeline, 'load_config', return_value=config), \
                 patch.object(pipeline, 'check_training_files'), \
                 patch.object(pipeline, 'audit_splits', return_value={'test_only': True}), \
                 patch.object(pipeline, 'BaselineDataset', SyntheticDataset):
                updates = []
                train_path = pipeline.train(config=config, progress=lambda path, info: updates.append(info['status']))
                self.assertEqual(updates[-1], 'complete')
                training = json.loads(train_path.read_text(encoding='utf-8'))
                self.assertEqual(training['status'], 'complete')
                for split in ('eval', 'lps'):
                    if split == 'lps':
                        config['checkpoint'] = 'unrelated_old_checkpoint.pt'
                        result_path = pipeline.evaluate(split, checkpoint_path=training['checkpoint'])
                        self.assertEqual(config['checkpoint'], 'unrelated_old_checkpoint.pt')
                        config['checkpoint'] = None
                    else:
                        result_path = pipeline.evaluate(split)
                    result = json.loads(result_path.read_text(encoding='utf-8'))
                    self.assertEqual(result['status'], 'complete')
                    self.assertEqual(result['utterance_count'], 4)
                    self.assertEqual(result['frame_count'], 8)
                    self.assertGreater(result['inference_seconds'], 0)
                    saved_metrics = json.loads((results/result['run']/'metrics.json').read_text(encoding='utf-8'))
                    self.assertEqual(saved_metrics['planned_utterances'], saved_metrics['utterance_count'])
                    connection = sqlite3.connect(results/result['predictions_file'])
                    intervals = connection.execute('SELECT spoof_intervals_json FROM intervals').fetchall()
                    self.assertEqual(len(intervals), 4)
                    for row in intervals:
                        for start, end in json.loads(row[0]):
                            self.assertLess(start, end)
                    connection.close()
                # Refuse silently accepting a modified checkpoint.
                checkpoint = (code/training['checkpoint']).resolve()
                with checkpoint.open('ab') as stream:
                    stream.write(b'changed')
                with self.assertRaises(ValueError):
                    pipeline.load_trained_model(config)






if __name__ == '__main__':
    unittest.main(verbosity=2)
