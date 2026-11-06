import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cnn import baseline as p
from cnn import grid_search as grid
from sal.pipeline import training_config


class GridIntegrationTests(unittest.TestCase):
    def test_grid_builds_four_distinct_fixed10_trials(self):
        base = p.load_config(use_grid=False)
        before = copy.deepcopy(base)
        trials = grid.candidate_configs(base, dict(learning_rates=[.001,.0003], dropouts=[.2,.4]))
        self.assertEqual(base, before)
        self.assertEqual(len(trials), 4)
        self.assertTrue(all(not row['use_grid_search_best'] and row['checkpoint'] is None for row in trials))
        with self.assertRaises(ValueError):
            grid.select_winner([dict(status='running')])

    def test_baseline_automatically_uses_dev_winner_not_last_trial(self):
        config = p.load_config(use_grid=False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            module, results = root/'cnn', root/'result'
            module.mkdir(); results.mkdir()
            (module/'config.json').write_text(json.dumps(config))
            checkpoint = root/'winner.pt'
            checkpoint.write_bytes(b'test checkpoint only')
            chosen = copy.deepcopy(config)
            chosen['learning_rate'] = .0003
            chosen['model']['dropout'] = .4
            winner = dict(index=1, status='complete', config=chosen, dev_eer_percent=1.,
                          checkpoint='winner.pt', checkpoint_sha256=p.sha256(checkpoint),
                          train_manifest_sha256='trainhash', dev_manifest_sha256='devhash')
            last = dict(winner, index=2, config=config, dev_eer_percent=2.)
            state = dict(status='complete', base_signature=p.configuration_signature(config, exclude_search=True), trials=[winner,last])
            (results/'search.json').write_text(json.dumps(state))
            pointer = results/'cnn_grid_best.json'
            pointer.write_text(json.dumps(dict(search_file='search.json',winner=winner)))
            with patch.multiple(p, MODULE_ROOT=module, RESULT_ROOT=results, CODE_ROOT=root), \
                 patch('tools.data_tools.read_manifest', side_effect=lambda split: {'manifest_sha256':split+'hash'}):
                selected = p.load_config()
                self.assertEqual(selected['learning_rate'], .0003)
                self.assertEqual(selected['model']['dropout'], .4)
                self.assertEqual(selected['grid_selected_checkpoint'], 'winner.pt')
                self.assertEqual(p.load_config(use_grid=False)['learning_rate'], config['learning_rate'])
                pointer.write_text(json.dumps(dict(search_file='search.json',winner=last)))
                with self.assertRaises(ValueError):
                    p.load_config()

    def test_failed_search_does_not_publish_parameters_or_test(self):
        base = p.load_config(use_grid=False)
        with tempfile.TemporaryDirectory() as tmp, patch.object(p,'RESULT_ROOT',Path(tmp)), \
             patch.object(p,'require_device_idle'), patch.object(p,'train',side_effect=RuntimeError('test failure')), \
             patch.object(p,'evaluate') as evaluate:
            with self.assertRaises(RuntimeError):
                grid.search_grid(base,[base])
            self.assertFalse((Path(tmp)/'cnn_grid_best.json').exists())
            evaluate.assert_not_called()
            state = json.loads(next(Path(tmp).glob('cnn_grid_search_*.json')).read_text())
            self.assertEqual(state['status'],'failed')

    def test_sal_profiles_use_same_official_model_and_separate_data(self):
        # Configuration checks must also work on a code-only clone.
        with patch('tools.data_tools.read_manifest', return_value=dict(seed=20261107, actual_fraction=.1,
                   official_count=10, manifest_sha256='fixture')):
            fixed, fixed_model, fixed_data = training_config('fixed10')
        full, full_model, full_data = training_config('full')
        self.assertEqual(fixed,full)
        self.assertEqual(fixed_model,full_model)
        self.assertNotEqual(fixed_data.root,full_data.root)
        self.assertIn('fixed10',fixed_data.root)
        self.assertNotIn('fixed10',full_data.root)
        self.assertTrue(fixed_model.mixup2)
        self.assertFalse(fixed_model.vis)


if __name__ == '__main__':
    unittest.main()
