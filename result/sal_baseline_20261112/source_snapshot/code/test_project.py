"""Reduced-project regression tests. Uses synthetic inputs; never starts real training."""
import copy
import json
import sqlite3
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
import sys
import numpy as np
import torch
import soundfile as sf
import cnn as pipeline
from cnn import collate, CNNBiGRU, LogMel, ModelConfig, intervals_to_spoof_labels, masked_bce, pool_logits, probabilities_to_intervals, sal_logits
from evaluate import compare_pair
from sal import calculate_metrics
import semantic
import sal
from semantic import settings, write_json, read_json, path, annotate, validate_record, align, Accumulator, fit_priors, lookup, analyze, freeze, features
import data
ROOT = Path(__file__).resolve().parent
CODE = ROOT


def small_config():
    return ModelConfig(n_mels=8, channels=(2,), hidden_size=4, gru_layers=1, dropout=0.)

class SyntheticDataset:

    def __init__(self, split, model_config, crop_seconds=None, data_profile='reduced'):
        self.resolution = 0.02 if split == 'lps' else 0.16
        self.manifest_sha256 = 'synthetic-only-test-manifest'
        self.split = split
        self.n_mels = model_config.n_mels

    def __len__(self):
        return 4

    def __getitem__(self, index):
        labels = torch.tensor([float(index % 2), float(1 - index % 2)])
        frames = len(labels) * round(self.resolution / 0.02)
        generator = torch.Generator().manual_seed(100 + index)
        return dict(uid=f'{self.split}_{index}', features=torch.randn(1, frames, self.n_mels, generator=generator), labels=labels, duration=len(labels) * self.resolution)

class BaselineTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(666)

    def test_frontend_length_and_silence(self):
        frontend = LogMel(small_config())
        for (samples, expected) in ((1, 1), (319, 1), (320, 1), (321, 2), (2560, 8)):
            features = frontend(torch.zeros(samples))
            self.assertEqual(tuple(features.shape), (1, expected, 8))
            self.assertTrue(torch.isfinite(features).all())

    def test_variable_lengths_do_not_change_valid_predictions(self):
        model = CNNBiGRU(small_config()).eval()
        short = torch.randn(1, 8, 8)
        long = torch.randn(1, 16, 8)
        items = [dict(features=short, labels=torch.zeros(1), uid='a', duration=0.16), dict(features=long, labels=torch.zeros(2), uid='b', duration=0.32)]
        batch = collate(items)
        with torch.no_grad():
            alone = model(short.unsqueeze(0), torch.tensor([8]))
            together = model(batch['features'], batch['lengths'])
        torch.testing.assert_close(alone[0], together[0, :8], atol=1e-06, rtol=1e-05)
        self.assertEqual(int(torch.count_nonzero(together[0, 8:])), 0)

    def test_masked_loss_ignores_padding_and_gradients(self):
        logits = torch.tensor([[0.3, -0.4], [0.2, 123.0]], requires_grad=True)
        targets = torch.tensor([[1.0, 0.0], [0.0, float('nan')]])
        loss = masked_bce(logits, targets, torch.tensor([2, 1]))
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits.flatten()[:3], targets.flatten()[:3])
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertEqual(float(logits.grad[1, 1]), 0.0)

    def test_pooling_and_label_orientation(self):
        (pooled, lengths) = pool_logits(torch.arange(16.0).reshape(1, -1), torch.tensor([16]), 0.16)
        torch.testing.assert_close(pooled, torch.tensor([[3.5, 11.5]]))
        self.assertEqual(lengths.tolist(), [2])
        logits = sal_logits(torch.tensor([-2.0, 2.0, 0.0]))
        self.assertEqual(logits.argmax(-1).tolist(), [1, 0, 0])
        torch.testing.assert_close(logits[:, 1] - logits[:, 0], torch.tensor([2.0, -2.0, 0.0]))

    def test_intervals_boundaries_and_empty_prediction(self):
        flags = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
        self.assertEqual(probabilities_to_intervals(flags, 0.02, 0.2), [[0.04, 0.1], [0.14, 0.18]])
        self.assertEqual(probabilities_to_intervals([0.0, 0.0], 0.02, 0.04), [])
        self.assertEqual(probabilities_to_intervals([1.0, 1.0], 0.02, 0.031), [[0.0, 0.031]])
        self.assertEqual(intervals_to_spoof_labels([[0.03, 0.07]], 5, 0.02, 0.1).tolist(), [0.0, 1.0, 1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            probabilities_to_intervals([float('nan')], 0.02, 0.02)

    def test_metrics_exactly_match_original_database_evaluator(self):
        scores = torch.tensor([-2.1, 0.7, -0.2, 2.3, 0.0, -1.0])
        spoof = torch.tensor([0.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        database = sqlite3.connect(':memory:')
        database.execute('CREATE TABLE predictions (ordinal INTEGER, logits BLOB, labels BLOB)')
        database.execute('INSERT INTO predictions VALUES (?,?,?)', (0, sal_logits(scores).numpy().astype(np.float32).tobytes(), (1 - spoof).numpy().astype(np.uint8).tobytes()))
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
        loss = pipeline.train_epoch(model, batches, torch.optim.AdamW(model.parameters(), lr=0.01), 'cpu')
        self.assertTrue(np.isfinite(loss))
        for name in ('blocks.0.0.weight', 'gru.weight_ih_l0', 'classifier.weight'):
            self.assertFalse(torch.equal(before[name], model.state_dict()[name]), name)

    def test_comparison_rejects_mismatched_manifest_and_incomplete_results(self):
        base = dict(status='complete', split='eval', manifest_sha256='abc', resolution_seconds=0.16, utterance_count=10, frame_count=100, f1_rule='same', eer_rule='same', precision='float32', batch_size=1, full_utterance=True, eer_percent=3.0, f1_bonafide_percent=97.0)
        self.assertTrue(compare_pair(base, dict(base))['comparable'])
        self.assertFalse(compare_pair(base, dict(base, manifest_sha256='other'))['comparable'])
        self.assertFalse(compare_pair(base, dict(base, status='running'))['comparable'])

class SemanticsTest(unittest.TestCase):

    def setUp(self):
        self.config = settings()

    def words(self, *items):
        return annotate(dict(uid='u', language='en', words=[dict(text=t, start=s, end=e, confidence=0.99) for (t, s, e) in items]), self.config)['words']

    def alignment(self, words, labels, pred, seconds=None, dt=1):
        logits = np.zeros((len(pred), 2), np.float32)
        logits[np.arange(len(pred)), pred] = 2
        return align(words, logits, np.array(labels), len(pred) * dt if seconds is None else seconds, dt, self.config)

    def test_all_four_outcomes_and_denominators(self):
        words = self.words(('not', 0, 1), ('three', 1, 2), ('hello', 2, 3), ('today', 3, 4))
        (rows, audit) = self.alignment(words, [0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual([r['status'] for r in rows], ['tp', 'fn', 'fp', 'tn'])
        self.assertEqual(audit['scope'], 'partial')
        a = Accumulator()
        for row in rows:
            a.add(row, 'partial')
        counts = {r['value']: r for r in a.rows() if r['scope'] == 'all' and r['dimension'] == 'primary_type'}
        self.assertEqual(counts['quantity']['miss_rate'], 1)
        self.assertIsNone(counts['quantity']['false_alarm_rate'])
        self.assertEqual(counts['other']['false_alarm_rate'], 1)

    def test_boundary_partial_is_not_forced_into_clean_denominator(self):
        (rows, _) = self.alignment(self.words(('not', 0.5, 1.5)), [0, 1], [1, 1])
        self.assertEqual(rows[0]['status'], 'partial_truth')
        self.assertEqual(rows[0]['threshold_status'], 'fn')
        self.assertAlmostEqual(rows[0]['false_negative_fraction'], 0.5)

    def test_unscored_audio_tail_is_not_assumed_real(self):
        (rows, _) = self.alignment(self.words(('tail', 0.8, 1.2)), [0], [0], seconds=1.2)
        self.assertEqual(rows[0]['status'], 'unscored')
        self.assertAlmostEqual(rows[0]['scored_fraction'], 0.5)

    def test_last_grid_cell_clips_to_audio(self):
        (rows, audit) = self.alignment(self.words(('last', 1, 1.2)), [1, 0], [1, 1], seconds=1.2)
        self.assertEqual(rows[0]['status'], 'fn')
        self.assertAlmostEqual(audit['false_negative_seconds'], 0.2)

    def test_complete_miss_is_not_any_partial_miss(self):
        (rows, _) = self.alignment(self.words(('one', 0, 1), ('two', 1, 2)), [0, 0], [0, 1])
        self.assertEqual(rows[1]['false_negative_fraction'], 1)
        self.assertEqual(rows[1]['complete_miss_event_fraction'], 0)

    def test_false_alarm_event_requires_no_true_overlap(self):
        (rows, _) = self.alignment(self.words(('word', 1, 2)), [0, 1], [0, 0])
        self.assertEqual(rows[0]['false_positive_fraction'], 1)
        self.assertEqual(rows[0]['no_overlap_false_alarm_event_fraction'], 0)

    def test_ties_match_sal_argmax(self):
        (rows, _) = align(self.words(('tie', 0, 1)), np.zeros((1, 2)), np.array([0]), 1, 1, self.config)
        self.assertEqual(rows[0]['status'], 'tp')

    def test_low_confidence_and_no_word_coverage_are_visible(self):
        words = self.words(('bad', 0, 1))
        words[0]['confidence'] = 0.1
        (rows, audit) = self.alignment(words, [0, 0], [1, 1])
        self.assertEqual(rows[0]['status'], 'low_confidence')
        self.assertEqual(audit['false_negative_without_words_seconds'], 1)

    def test_invalid_word_intervals_rejected(self):
        for words in [[dict(text='x', start=0, end=0)], [dict(text='x', start=0, end=2), dict(text='y', start=1, end=3)]]:
            with self.assertRaises(ValueError):
                validate_record(dict(uid='u', words=words))

    def test_multi_tags_keep_one_primary_bucket(self):
        result = annotate(dict(uid='u', words=[dict(text='May', start=0, end=1, types=['person', 'time'])]), self.config)
        self.assertEqual(result['words'][0]['primary_type'], 'time')
        self.assertIn('person', result['words'][0]['types'])

    def test_ner_char_alignment_with_stub(self):

        class Entity:
            (start_char, end_char, label_) = (0, 5, 'PERSON')

        class Token:
            (idx, text, pos_, lemma_) = (0, 'Alice', 'PROPN', 'Alice')

        class Doc(list):
            ents = [Entity()]
        result = annotate(dict(uid='u', words=[dict(text='Alice', start=0, end=1)]), self.config, lambda text: Doc([Token()]))
        self.assertEqual(result['words'][0]['primary_type'], 'person')
        self.assertEqual(result['words'][0]['pos'], 'PROPN')

    def test_paths_do_not_depend_on_working_directory(self):
        self.assertEqual(path('config.json'), Path(__file__).resolve().parent / 'config.json')

    def test_entity_type_precedes_numeric_surface(self):
        result = annotate(dict(uid='u', words=[dict(text='2026', start=0, end=1, types=['time'])]), self.config)
        self.assertEqual(result['words'][0]['primary_type'], 'time')
        self.assertEqual(set(result['words'][0]['types']), {'time', 'quantity'})

    def test_backoff_and_missing_mask(self):
        rows = [dict(scope='all', dimension='primary_type', value='negation', tp=10, fn=10, fp=2, tn=18)]
        priors = fit_priors(rows, self.config)
        word = self.words(('never', 0, 1))[0]
        result = lookup(word, priors, 0.5)
        self.assertEqual(result['prior_sources'], ['type'] * 3)
        word['confidence'] = 0.1
        self.assertEqual(lookup(word, priors, 0.5)['prior_available'], [0, 0, 0])

class ReducedSemanticTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = semantic.settings()
        self.manifests = {role: dict(utterance_ids=[role + '_id'], manifest_sha256=chr(97 + i) * 64, official_split='train' if role.startswith('train_') else role) for (i, role) in enumerate(semantic.ROLES)}
        self.patcher = patch.object(semantic, 'role_manifest', side_effect=lambda role: self.manifests[semantic.role_name(role)])
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def prediction(self, role, checkpoint='0' * 64):
        run = self.root / role
        run.mkdir()
        semantic.write_json(run / 'metrics.json', dict(split=role, data_profile='reduced', manifest_sha256=self.manifests[role]['manifest_sha256'], checkpoint_sha256=checkpoint, training_data_profile='reduced', training_split='train_detector', training_manifest_sha256=self.manifests['train_detector']['manifest_sha256'], resolution_seconds=0.16, checkpoint_kind='trained_reduced'))
        semantic.write_json(run / 'status.json', dict(status='complete'))
        db = sqlite3.connect(run / 'predictions.sqlite')
        db.execute('CREATE TABLE predictions (uid TEXT PRIMARY KEY, ordinal INTEGER, logits BLOB, labels BLOB, seconds REAL)')
        db.execute('INSERT INTO predictions VALUES (?,?,?,?,?)', (role + '_id', 0, np.array([[0, 2]], np.float32).tobytes(), np.array([0], np.uint8).tobytes(), 0.16))
        db.commit()
        db.close()
        words = self.root / (role + '_words.jsonl')
        words.write_text(json.dumps(dict(uid=role + '_id', label='must-not-export', words=[dict(text='not', start=0, end=0.16, confidence=0.99, label=0, status='fn')])) + '\n', encoding='utf-8')
        return (run, words)

    def fit(self):
        (run, words) = self.prediction('train_edit')
        report = self.root / 'report'
        semantic.analyze(run, words, report, self.config)
        frozen = self.root / 'frozen.json'
        semantic.freeze(report, None, frozen)
        return (run, words, report, frozen)

    def test_reduced_fit_apply_and_label_free_whitelist(self):
        (_, edit_words, _, frozen) = self.fit()
        (run, words) = self.prediction('train_review')
        output = self.root / 'features.jsonl'
        semantic.features(words, frozen, output, 'train_review', run)
        result = json.loads(output.read_text())
        self.assertEqual(result['feature_mode'], 'frozen_risk')
        self.assertIn('prior_available', result['words'][0])
        self.assertNotIn('label', result)
        self.assertNotIn('label', result['words'][0])
        self.assertNotIn('status', result['words'][0])
        raw = self.root / 'raw.jsonl'
        semantic.features(edit_words, None, raw, 'train_edit')
        self.assertEqual(json.loads(raw.read_text())['feature_mode'], 'label_free')
        self.assertNotIn('prior_available', json.loads(raw.read_text())['words'][0])
        with self.assertRaisesRegex(ValueError, 'fitting role'):
            semantic.features(edit_words, frozen, self.root / 'leak.jsonl', 'train_edit')
        self.assertFalse((self.root / 'leak.jsonl').exists())

    def test_legacy_artifacts_and_checkpoint_mismatch_rejected(self):
        (_, _, _, frozen) = self.fit()
        (run, words) = self.prediction('dev', '1' * 64)
        with self.assertRaisesRegex(ValueError, 'different detector'):
            semantic.features(words, frozen, self.root / 'wrong.jsonl', 'dev', run)
        legacy = self.root / 'legacy.json'
        semantic.write_json(legacy, dict(schema_version=2, source_split='train'))
        with self.assertRaisesRegex(ValueError, 'Legacy risk'):
            semantic.features(words, legacy, self.root / 'old.jsonl', 'dev', run)
        self.assertFalse((self.root / 'old.jsonl').exists())

    def test_manifest_mismatch_rejected_before_output(self):
        (run, _) = self.prediction('train_edit')
        metrics = semantic.read_json(run / 'metrics.json')
        metrics['manifest_sha256'] = 'wrong'
        semantic.write_json(run / 'metrics.json', metrics)
        with self.assertRaisesRegex(ValueError, 'manifest differs'):
            semantic.make_manifest(run, self.root / 'audio.jsonl')
        self.assertFalse((self.root / 'audio.jsonl').exists())

    def test_only_train_edit_and_reduced_detector_can_fit(self):
        (_, _, report, _) = self.fit()
        summary = semantic.read_json(report / 'summary.json')
        summary['role'] = 'dev'
        semantic.write_json(report / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'train_edit report'):
            semantic.freeze(report, None, self.root / 'dev_prior.json')
        summary['role'] = 'train_edit'
        summary['checkpoint_kind'] = 'released_reference'
        semantic.write_json(report / 'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'released/legacy'):
            semantic.freeze(report, None, self.root / 'released_prior.json')

    def test_partial_coverage_and_source_group_overlap_cannot_fit(self):
        (run, words) = self.prediction('train_edit')
        self.manifests['train_edit']['utterance_ids'].append('absent_transcript')
        report = self.root / 'partial'
        semantic.analyze(run, words, report, self.config)
        with self.assertRaisesRegex(ValueError, 'complete train_edit'):
            semantic.freeze(report, None, self.root / 'partial_prior.json')
        self.manifests['train_edit']['utterance_ids'].pop()
        groups = self.root / 'groups.jsonl'
        groups.write_text('\n'.join((json.dumps(dict(uid=role + '_id', role=role, source_group='one')) for role in semantic.ROLES)), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Source-group leakage'):
            semantic.freeze(report, groups, self.root / 'group_prior.json')

    def test_reduced_audio_manifest_does_not_need_full_roots(self):
        (run, _) = self.prediction('train_edit')
        audio = self.root / 'audio'
        audio.mkdir()
        (audio / 'train_edit_id.wav').write_bytes(b'fake audio path fixture')
        data_stub = types.SimpleNamespace(dataset_paths=lambda role, profile: (audio, self.root / 'labels'))
        output = self.root / 'audio.jsonl'
        with patch.dict(sys.modules, {'data': data_stub}):
            semantic.make_manifest(run, output)
        result = json.loads(output.read_text())
        self.assertEqual(result['role'], 'train_edit')
        self.assertEqual(Path(semantic.path(result['audio'])), audio / 'train_edit_id.wav')

    def test_full_predictions_refused(self):
        with self.assertRaisesRegex(ValueError, 'old full/fixed10'):
            semantic.prediction_profile(dict(data_profile='fixed10'))


class ReducedDataContractTests(unittest.TestCase):
    def test_train_alias_never_includes_gain_roles(self):
        self.assertEqual(data.canonical_split('train'), 'train_detector')
        self.assertNotEqual(data.dataset_paths('train'), data.dataset_paths('train_edit'))
        self.assertEqual(data.label_split('train_review'), 'train')
        with self.assertRaises(ValueError):
            data.dataset_paths('train', 'full')
        with self.assertRaises(ValueError):
            data.dataset_paths('dev', 'fixed10')

    def test_string_labels_preserve_authenticity(self):
        values = np.array(['1', '0', '0', '1'], dtype='<U21')
        features = data._features(values, .16)
        self.assertEqual(features['kind'], 'partial')
        self.assertAlmostEqual(features['fake_seconds'], .32)

    def test_train_partitions_are_exact_disjoint_and_deterministic(self):
        metadata = {}
        for i in range(200):
            row = data._features(np.array([1, 0] if i % 3 else [1, 1]), .16)
            row.update(speaker=f'speaker{i % 4}', attack='CON', source_group=None)
            metadata[f'uid{i:04}'] = row
        available = set(metadata)
        selected = []
        for role, size in zip(data.TRAIN_ROLES, (40, 10, 10)):
            result = data._choose_records(metadata, available, size, role)
            self.assertEqual(len(result), size)
            self.assertEqual(result, data._choose_records(dict(reversed(list(metadata.items()))), available, size, role))
            self.assertFalse(set(result) & set().union(*(set(ids) for ids in selected)))
            selected.append(result)
            available.difference_update(result)

    def test_lps_inferred_source_variants_are_kept_together(self):
        metadata = {}
        groups = {}
        for i in range(40):
            group = f'{i}_1_1'
            groups[group] = []
            for prefix, values in [('clean', [1, 1]), ('tts', [0, 0])]:
                uid = f'{prefix}_{i}_1_1_0'
                row = data._features(np.asarray(values), .02)
                row.update(speaker=str(i), attack=prefix, source_group=group)
                metadata[uid] = row
                groups[group].append(uid)
        selected, total_groups, selected_groups = data._choose_lps(metadata)
        self.assertEqual(total_groups, 40)
        self.assertGreater(selected_groups, 0)
        for ids in groups.values():
            overlap = set(ids) & set(selected)
            self.assertIn(len(overlap), (0, len(ids)))

    def test_frozen_validation_detects_label_and_audio_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            role = 'train_detector'
            audio = root / role / 'audio'
            labels = root / role / 'segment_labels'
            audio.mkdir(parents=True)
            labels.mkdir(parents=True)
            wave = audio / 'sample.wav'
            wave.write_bytes(b'unchanged audio fixture')
            label_file = labels / 'train_seglab_0.16.npy'
            np.save(label_file, {'sample': np.array([1, 0], dtype=np.int8)})
            manifest = dict(profile='reduced', split=role, count=1, utterance_ids=['sample'],
                            manifest_sha256=data.manifest_hash(['sample']),
                            label_files={label_file.name: {'sha256': data.file_hash(label_file)}},
                            audio_files={'sample': {'bytes': wave.stat().st_size, 'sha256': data.file_hash(wave)}},
                            audio_bytes=wave.stat().st_size)
            (root / 'manifests').mkdir()
            (root / 'manifests' / (role + '.json')).write_text(json.dumps(manifest), encoding='utf-8')
            with patch.object(data, 'SUBSET_ROOT', root), patch.object(data, 'CODE_ROOT', root / 'isolated_code'):
                self.assertTrue(data.validate_reduced('train', deep=True)['audio_hashes_verified'])
                # Same-length corruption must still fail the deep audit.
                wave.write_bytes(b'X' * wave.stat().st_size)
                with self.assertRaisesRegex(ValueError, 'Audio hash changed'):
                    data.validate_reduced('train', deep=True)
                wave.write_bytes(b'unchanged audio fixture')
                np.save(label_file, {'sample': np.array([0, 0], dtype=np.int8)})
                with self.assertRaisesRegex(ValueError, 'Labels changed'):
                    data.validate_reduced('train')

class SalDeviceAndMixingTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for deterministic EER regression')
    def test_validation_eer_with_cuda_inputs_and_strict_determinism(self):
        from src.evaluator.evaluator import EERMetric
        enabled = torch.are_deterministic_algorithms_enabled()
        warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        try:
            torch.use_deterministic_algorithms(True)
            scores = torch.tensor([.9, .8, .7, .6, .4, .3, .2, .1])
            labels = torch.tensor([1., 0., 1., 0., 0., 1., 0., 1.])
            reference = EERMetric()
            reference.update(scores, labels)
            model = types.SimpleNamespace(val_eer=EERMetric().cuda())
            sal.enable_cpu_validation_eer(model)
            model.val_eer.update(scores[:4].cuda(), labels[:4].cuda())
            model.val_eer.update(scores[4:].cuda(), labels[4:].cuda())
            self.assertTrue(all(x.device.type == 'cpu' for x in model.val_eer.y_pred))
            self.assertEqual(model.val_eer.compute(), reference.compute())
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertFalse(torch.is_deterministic_algorithms_warn_only_enabled())
        finally:
            torch.use_deterministic_algorithms(enabled, warn_only=warn_only)

    def test_configured_cpu_never_requires_cuda(self):
        with patch.dict(sal.CONFIG,{'sal':{'device':'cpu'}}),patch.object(torch.cuda,'is_available',side_effect=AssertionError('CUDA queried')):
            self.assertEqual(sal.configured_device(),'cpu')

    def test_cpu_inference_float32_without_gpu_calls(self):
        model=Mock(return_value=(None,torch.zeros(1,2,2,dtype=torch.float64)))
        with patch.object(torch.cuda,'synchronize',side_effect=AssertionError('CUDA sync')),patch.object(torch.cuda,'empty_cache',side_effect=AssertionError('CUDA cache')):
            logits,device=sal.infer_waveform(model,torch.zeros(5120,dtype=torch.float64),'cpu')
        self.assertEqual(device,'cpu');self.assertEqual(logits.dtype,torch.float32)
        value=model.call_args.args[0];self.assertEqual(value.device.type,'cpu');self.assertEqual(value.dtype,torch.float32)

    def test_cpu_oom_is_not_retried_as_cuda_oom(self):
        model=Mock(side_effect=torch.cuda.OutOfMemoryError('synthetic'))
        with self.assertRaises(torch.cuda.OutOfMemoryError):sal.infer_waveform(model,torch.zeros(10),'cpu')
        self.assertEqual(model.call_count,1)

    def test_cuda_oom_retries_on_cpu_and_restores_device(self):
        wave=Mock();wave.unsqueeze.return_value=wave;wave.to.return_value=torch.zeros(1,5120)
        model=Mock(side_effect=[torch.cuda.OutOfMemoryError('synthetic'),(None,torch.zeros(1,2,2))])
        with patch.object(torch.cuda,'synchronize'),patch.object(torch.cuda,'empty_cache'):
            logits,device=sal.infer_waveform(model,wave,'cuda')
        self.assertEqual(device,'cpu_oom_fallback');self.assertEqual(model.call_count,2)
        self.assertEqual([call.kwargs['device'] for call in wave.to.call_args_list],['cuda','cpu'])
        model.cpu.assert_called_once();model.to.assert_called_once_with('cuda')

    def test_cpu_prediction_export_counts_no_oom_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result=root/'result';result.mkdir();weights=root/'fake.ckpt';weights.write_bytes(b'synthetic')
            paths=[]
            for i in range(2):
                p=root/f'{i}.wav';sf.write(p,np.zeros(5120,dtype=np.float32),16000);paths.append(str(p))
            class Dataset:
                utt_ids=['a','b'];sample_list=paths;labels={'a':np.array([0,1]),'b':np.array([0,1])}
                def __len__(self):return 2
                def __getitem__(self,i):return self.utt_ids[i],torch.zeros(5120),torch.tensor([0.,1.]),2
            class Model(torch.nn.Module):
                def forward(self,x):
                    assert x.device.type=='cpu' and x.dtype==torch.float32
                    return None,torch.tensor([[[2.,-2.],[-2.,2.]]])
            def load(*args,**kwargs):
                self.assertEqual(kwargs['device'],'cpu');return Model(),{'strict_load':'synthetic CPU test'}
            with patch.dict(sal.CONFIG,{'sal':{'device':'cpu'}}),patch.object(sal,'RESULT_ROOT',result),patch.object(sal,'resolve_checkpoint',return_value=weights),patch.object(sal,'get_dataset',return_value=(Dataset(),.16)),patch.object(sal,'load_model',side_effect=load),patch.object(sal,'profile_info',return_value={'data_profile':'reduced'}),patch.object(sal,'checkpoint_digest',return_value='synthetic-only'),patch.object(torch.cuda,'synchronize',side_effect=AssertionError('CUDA sync')),patch.object(torch.cuda,'get_device_name',side_effect=AssertionError('CUDA info')),patch.object(torch.cuda,'empty_cache',side_effect=AssertionError('CUDA cache')):
                self.assertEqual(sal.run(['--split','dev','--checkpoint','public','--run-name','cpu_test']),0)
            metrics=json.loads((result/'cpu_test/metrics.json').read_text())
            self.assertEqual(metrics['inference_device'],'cpu');self.assertEqual(metrics['cpu_fallback_utterances'],0)
            conn=sqlite3.connect(result/'cpu_test/predictions.sqlite')
            try:self.assertEqual(conn.execute('SELECT DISTINCT device FROM predictions').fetchall(),[('cpu',)])
            finally:conn.close()

    def test_csm_cross_recording_and_cut_alignment_all_ratios(self):
        batch_size,frames,scale=8,12,4
        labels=torch.stack([torch.tensor([(i>>(j%3))&1 for j in range(frames)]) for i in range(batch_size)])
        inputs=torch.arange(batch_size,dtype=torch.float32)[:,None].expand(batch_size,frames*scale).clone()
        lengths=torch.full((batch_size,),frames);ids=list(map(str,range(batch_size)))
        for ratio in (.2,.5,1.):
            torch.manual_seed(888)
            _,audio,y,n=sal.cross_recording_mixup(types.SimpleNamespace(mixup_ratio=ratio),(ids,inputs,labels,lengths))
            changed=[i for i in range(batch_size) if not torch.equal(audio[i],inputs[i])]
            self.assertEqual(len(changed),int(batch_size*ratio))
            for i in changed:
                sample_cut=int(torch.nonzero(audio[i]!=inputs[i])[0]);self.assertEqual(sample_cut%scale,0)
                cut=sample_cut//scale;donor=int(audio[i,-1]);self.assertNotEqual(donor,i)
                torch.testing.assert_close(y[i,:cut],labels[i,:cut]);torch.testing.assert_close(y[i,cut:],labels[donor,cut:])
                self.assertEqual(int(n[i]),frames)
            torch.testing.assert_close(inputs,torch.arange(batch_size,dtype=torch.float32)[:,None].expand_as(inputs))

    def test_csm_short_valid_lengths_and_singleton(self):
        inputs=torch.tensor([[0.]*24,[1.]*24,[2.]*24]);labels=torch.tensor([[0]*6,[1]*6,[0,1,0,1,0,1]])
        lengths=torch.tensor([10,3,1]);batch=(['a','b','c'],inputs,labels,lengths)
        _,audio,y,n=sal.cross_recording_mixup(types.SimpleNamespace(mixup_ratio=1.),batch)
        self.assertTrue(bool(torch.all((n>=0)&(n<=6))))
        for i in range(3):
            changes=torch.nonzero(audio[i]!=inputs[i])
            if len(changes):
                cut=int(changes[0])//4;donor=int(audio[i,-1]);self.assertLess(cut,min(int(lengths[i].clamp(max=6)),int(lengths[donor].clamp(max=6))))
                self.assertEqual(int(n[i]),int(lengths[donor].clamp(max=6)))
        single=(['a'],inputs[:1],labels[:1],lengths[:1])
        self.assertIs(sal.cross_recording_mixup(types.SimpleNamespace(mixup_ratio=1.),single),single)

if __name__ == '__main__':
    unittest.main(verbosity=2)
