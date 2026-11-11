import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import sqlite3
from contextlib import closing
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from semantic import analysis as semantic
from semantic import full_train_priors as full
from cnn import baseline, grid_search


class FullTrainProtocolTests(unittest.TestCase):
    def make_report(self, root):
        report = root/'report'
        report.mkdir()
        config = semantic.settings()
        config['minimum_count'] = 1
        semantic.write_json(report/'summary.json', dict(status='complete', split='train', data_profile='full',
                            missing_transcript_recordings=0, prediction_recordings=3, config=config, language='en'))
        semantic.write_json(report/'analyzed_uids.json', ['u', 'v', 'w'])
        with (report/'word_diagnostics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=['uid', 'normalized', 'primary_type', 'status'])
            writer.writeheader()
            for uid, word, status in [('u', 'not', 'fn'), ('v', 'not', 'fn'), ('w', 'not', 'tp')]:
                writer.writerow(dict(uid=uid, normalized=word, primary_type='negation', status=status))
        return report

    def test_group_exclusion_matches_refit_and_ignores_own_targets(self):
        with tempfile.TemporaryDirectory(dir=semantic.CODE_ROOT) as directory:
            root = Path(directory)
            report = self.make_report(root)
            mapping = root/'groups.jsonl'
            mapping.write_text('\n'.join(json.dumps(dict(uid=u, source_group=g)) for u,g in
                                        [('u','g1'), ('v','g1'), ('w','g2')]), encoding='utf-8')
            frozen = root/'frozen.json'
            with patch.object(full, 'official_train_ids', return_value={'u','v','w'}):
                semantic.freeze(report, None, frozen, mapping)
            artifact = semantic.read_json(frozen)
            words = [dict(normalized='not', primary_type='negation')]
            result = full.reduced_priors(artifact, words, 'g1')
            expected = semantic.fit_priors([dict(scope='all', dimension=d, value=v, tp=1, fn=0, fp=0, tn=0)
                                           for d,v in [('word','not'),('primary_type','negation')]], artifact['config'])
            self.assertEqual(result, expected)
            changed = copy.deepcopy(artifact)
            for dim, key in [('word','not'), ('primary_type','negation')]:
                changed['counts'][dim][key] = [3,0,0,0]
                changed['group_contributions']['g1'][dim][key] = [2,0,0,0]
            changed['global_counts'] = [3,0,0,0]
            self.assertEqual(full.reduced_priors(changed, words, 'g1'), result)
            transcript = root/'train.jsonl'
            transcript.write_text(json.dumps(dict(uid='u', label='must not propagate',
                 words=[dict(text='not',start=0,end=1,confidence=1.,gain=99)]))+'\n', encoding='utf-8')
            output = root/'features.jsonl'
            semantic.features(transcript, frozen, output, 'train')
            row = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(row['excluded_group'], 'g1')
            self.assertEqual(row['words'][0]['prior_support_counts'], [1,1,0])
            self.assertNotIn('label', row)
            self.assertNotIn('gain', row['words'][0])
            with self.assertRaises(ValueError):
                semantic.features(transcript, frozen, root/'bad.jsonl', 'dev')
            transcript.write_text(json.dumps(dict(uid='new',words=[dict(text='not',start=0,end=1)]))+'\n', encoding='utf-8')
            semantic.features(transcript, frozen, root/'dev.jsonl', 'dev')
            dev = json.loads((root/'dev.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(dev['words'][0]['prior_support_counts'], [3,3,0])
            self.assertIsNone(dev['excluded_group'])

    def test_full_coverage_and_training_split_required(self):
        with tempfile.TemporaryDirectory(dir=semantic.CODE_ROOT) as directory:
            root = Path(directory)
            report = self.make_report(root)
            with patch.object(full, 'official_train_ids', return_value={'u','v','w','missing'}):
                with self.assertRaises(ValueError):
                    semantic.freeze(report, None, root/'frozen.json')
            for split, profile in [('dev','full'),('eval','full'),('lps','full'),('train','fixed10')]:
                meta = semantic.read_json(report/'summary.json')
                meta.update(split=split, data_profile=profile)
                semantic.write_json(report/'summary.json',meta)
                with self.assertRaises(ValueError):
                    semantic.freeze(report, None, root/'frozen.json')

    def test_recording_exclusion_single_record_has_no_available_priors(self):
        config = semantic.settings()
        config['minimum_count'] = 1
        counts = dict(word={'not':[0,1,0,0]}, primary_type={'negation':[0,1,0,0]})
        artifact = dict(config=config, counts=counts, global_counts=[0,1,0,0], group_contributions={'u':counts})
        word = dict(normalized='not',primary_type='negation')
        priors = full.reduced_priors(artifact, [word], 'u')
        self.assertEqual(semantic.lookup(word, priors, .5)['prior_available'], [0,0,0])

    def test_default_check_and_train_export_do_not_call_training(self):
        self.assertEqual(baseline.RUN_MODE, 'check')
        with patch.object(baseline, 'load_config', return_value=dict(training_data_profile='full')), \
             patch.object(baseline, 'evaluate') as evaluate, patch.object(baseline, 'train') as train:
            baseline.main('train-predict')
            evaluate.assert_called_once_with('train', config=dict(training_data_profile='full', evaluation_data_profile='full'))
            train.assert_not_called()

    def test_formal_signature_keeps_profile_while_grid_transfer_allows_full(self):
        config = baseline.load_config(use_grid=False)
        trial = grid_search.candidate_configs(config,dict(learning_rates=[.001], dropouts=[.2]))[0]
        self.assertNotEqual(baseline.configuration_signature(config), baseline.configuration_signature(trial))
        self.assertEqual(baseline.grid_signature(config), baseline.grid_signature(trial))

    def test_formal_model_loader_refuses_grid_checkpoint(self):
        config = baseline.load_config(use_grid=False)
        config['checkpoint'] = 'synthetic_grid.pt'
        with patch.object(baseline.torch, 'load', return_value=dict(config=dict(training_data_profile='fixed10'))):
            with self.assertRaisesRegex(ValueError, 'profile differs'):
                baseline.load_trained_model(config)

    def test_frozen_train_export_has_no_optimizer_and_keeps_weights(self):
        # Forward passes only on four synthetic CPU inputs, no training/real audio.
        from test_baseline import SyntheticDataset, small_config
        with tempfile.TemporaryDirectory(dir=semantic.CODE_ROOT) as directory:
            root = Path(directory)
            weights = root/'fixture.pt'
            weights.write_bytes(b'synthetic fixture; no trained project weights')
            model = baseline.CNNBiGRU(small_config()).eval()
            before = {k:v.clone() for k,v in model.state_dict().items()}
            checkpoint = dict(model_config=small_config().to_dict(), train_run='synthetic',
                              interval_threshold=.7, threshold_source='synthetic dev only')
            config = baseline.load_config(use_grid=False)
            config.update(device='cpu', cpu_threads=1)
            with patch.object(baseline, 'RESULT_ROOT', root/'result'), \
                 patch.object(baseline, 'load_trained_model', return_value=(model, checkpoint, weights)), \
                 patch.object(baseline, 'BaselineDataset', SyntheticDataset), \
                 patch.object(baseline.torch.optim, 'AdamW') as optimizer:
                result_path = baseline.evaluate('train', config=config)
            result = semantic.read_json(result_path)
            self.assertEqual(result['prediction_role'], 'in_sample_training_features')
            self.assertEqual(result['data_profile'], 'full')
            self.assertEqual(result['utterance_count'], 4)
            self.assertEqual(result['interval_threshold'], .7)
            with closing(sqlite3.connect(root/'result'/result['predictions_file'])) as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM predictions').fetchone()[0],4)
            optimizer.assert_not_called()
            self.assertTrue(all(baseline.torch.equal(value, before[key]) for key,value in model.state_dict().items()))


if __name__ == '__main__':
    unittest.main()
