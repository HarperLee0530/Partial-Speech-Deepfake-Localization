import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import json
import sqlite3
import tempfile
from contextlib import closing
from unittest.mock import patch
import numpy as np
from tools.data_tools import strata_for, choose
from tools.reporting import compare_pair
from tools import data_tools
from cnn import baseline
from sal import pipeline as sal
from semantic import analysis as semantic
import soundfile as sf


class FixedSubsetTests(unittest.TestCase):
    def test_string_labels_are_stratified_as_numeric_labels(self):
        numeric = {'real': np.ones(10), 'fake': np.zeros(10), 'partial': np.array([1, 0, 1])}
        text = {key: value.astype(int).astype(str) for key, value in numeric.items()}
        protocol = dict.fromkeys(numeric, 'attack')
        self.assertEqual(strata_for(numeric, protocol), strata_for(text, protocol))
        self.assertEqual(len(strata_for(text, protocol)), 3)

    def test_sampling_is_fixed_exact_and_independent_of_input_order(self):
        groups = {'a': [f'a_{i:03}' for i in range(47)], 'b': [f'b_{i:03}' for i in range(56)]}
        first, allocation = choose(groups, 'train')
        second, _ = choose({k: list(reversed(v)) for k, v in groups.items()}, 'train')
        self.assertEqual(first, second)
        self.assertEqual(len(first), 10)
        self.assertEqual(len(first), len(set(first)))
        self.assertEqual(sum(allocation.values()), 10)

    def test_comparison_refuses_full_and_subset_even_if_other_fields_match(self):
        row = dict(status='complete', data_profile='fixed10')
        self.assertFalse(compare_pair(row, dict(row, data_profile='full'))['comparable'])


class PortableSubsetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root/'data'
        self.code = self.root/'code'
        for split in ('train', 'dev', 'eval', 'lps'):
            uid = split+'_cf_sample'
            base = self.data/'fixed10'/('LlamaPartialSpoof' if split == 'lps' else 'PartialSpoof/database')
            audio = base/('cf' if split == 'lps' else split)
            audio.mkdir(parents=True)
            sf.write(audio/(uid+'.wav'), np.zeros(5120), 16000)
            labels = base/'segment_labels'
            labels.mkdir(exist_ok=True)
            name = 'cf_seglab_0.02.npy' if split == 'lps' else f'{split}_seglab_0.16.npy'
            np.save(labels/name, {uid: np.ones(16 if split == 'lps' else 2, dtype=np.int64)})
            manifest = dict(profile='fixed10', split=split, seed=20261107, count=1, official_count=10,
                            actual_fraction=.1, utterance_ids=[uid], manifest_sha256=data_tools.manifest_hash([uid]))
            for folder in (self.data/'fixed10/manifests', self.code/'manifests/fixed10'):
                folder.mkdir(parents=True, exist_ok=True)
                (folder/(split+'.json')).write_text(json.dumps(manifest), encoding='utf-8')
        for target, attrs in ((data_tools, dict(DATA_ROOT=self.data, SUBSET_ROOT=self.data/'fixed10', CODE_ROOT=self.code)),
                              (baseline, dict(DATA_ROOT=self.data)), (sal, dict(STORE=self.data)),
                              (semantic, dict(DATA_ROOT=self.data, CODE_ROOT=self.code))):
            context = patch.multiple(target, **attrs)
            context.start()
            self.addCleanup(context.stop)

    def test_fixed_only_audio_labels_and_features_load_without_originals(self):
        baseline.check_training_files('fixed10')
        audit = baseline.audit_splits('fixed10')
        self.assertEqual(audit['official_count_source'], 'shared_manifest')
        self.assertFalse((self.data/'PartialSpoof').exists())
        for split in ('train', 'dev', 'eval', 'lps'):
            dataset, _ = sal.get_dataset(split, 'fixed10')
            self.assertEqual(len(dataset), 1)
            item = baseline.BaselineDataset(split, baseline.ModelConfig(), data_profile='fixed10')[0]
            self.assertEqual(item['uid'], split+'_cf_sample')
            self.assertEqual(len(item['labels']), 16 if split == 'lps' else 2)
        with self.assertRaises(FileNotFoundError):
            baseline.check_training_files('full')
        with self.assertRaises(AssertionError):
            sal.get_dataset('eval', 'full')

    def test_missing_or_unexpected_audio_is_rejected(self):
        audio = self.data/'fixed10/PartialSpoof/database/train'
        (audio/'train_cf_sample.wav').rename(audio/'wrong.wav')
        with self.assertRaises(ValueError):
            data_tools.validate_fixed10('train')

    def test_semantic_manifest_finds_only_fixed_audio(self):
        run = self.root/'run'
        run.mkdir()
        (run/'metrics.json').write_text(json.dumps(dict(split='eval', data_profile='fixed10')))
        with closing(sqlite3.connect(run/'predictions.sqlite')) as db:
            db.execute('CREATE TABLE predictions (uid TEXT, seconds REAL, ordinal INTEGER)')
            db.execute('INSERT INTO predictions VALUES (?, ?, ?)', ('eval_cf_sample', .32, 0))
            db.commit()
        output = self.root/'audio.jsonl'
        semantic.make_manifest(run, output, 32)
        row = json.loads(output.read_text())
        self.assertEqual(row['uid'], 'eval_cf_sample')
        self.assertIn('fixed10', row['audio'])
        self.assertTrue(semantic.path(row['audio']).is_file())


if __name__ == '__main__':
    unittest.main()
