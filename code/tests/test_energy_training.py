"""Paired energy conditioning, gradient equivalence and immutable evaluation."""
from argparse import Namespace
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from cnn_ps.backbone import CNNBiGRU
from cnn_ps.energy_adapter import make_energy_adapter
from cnn_ps.energy_cache import energy_collate, NORMALIZATION_SCHEMA
from cnn_ps.energy_features import FEATURE_NAMES
from cnn_ps.io import read_json, save_json, sha256
from cnn_ps import train_four as old
from cnn_ps import train_energy as training
from cnn_ps.train_dual import forward_batch as old_forward, training_loss
from test_four_training import TinyFeatures


STATS = dict(mean=[.1, .2, .3, .4, .5], std=[1., 2., 3., 4., 5.])


class TinyEnergy(TinyFeatures):
    def __init__(self, split, root):
        super().__init__(split)
        self.folder, self.energy_folder = root/(split+'_acoustic'), root/(split+'_energy')
        self.folder.mkdir()
        self.energy_folder.mkdir()
        save_json(self.folder/'complete.json', dict(split=split, synthetic=True))
        self.energy_metadata = dict(array=dict(sha256='synthetic-energy-array'), normalization=None)
        if split == 'train':
            stats = dict(STATS, schema=NORMALIZATION_SCHEMA, fit_split='train', fit_channel=0,
                         feature_names=list(FEATURE_NAMES), valid_train_frames=15,
                         frontend_complete_sha256=sha256(self.folder/'complete.json'),
                         energy_sha256='synthetic-energy-array', padded_frames_in_statistics=False)
            save_json(self.energy_folder/'normalization.json', stats)
            self.energy_metadata['normalization'] = dict(sha256=sha256(self.energy_folder/'normalization.json'))
        save_json(self.energy_folder/'complete.json', self.energy_metadata)

    def get_record(self, index, channel=0):
        row = super().get_record(index, channel)
        generator = torch.Generator().manual_seed(765+index+channel)
        row['energy_features'] = torch.randn(len(row['fine_weights']), 5, generator=generator)
        row['energy_features'][row['fine_weights'] == 0] = 0.
        return row


class EnergyTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_original_initialization_and_rng_preservation(self):
        for device in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
            original = old.make_adapter(old.ARMS['boundary'], 666, device)
            common = old.state_digest(original)
            initial = None
            for enabled in (False, True):
                cpu = torch.get_rng_state().clone()
                cuda = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
                adapter = make_energy_adapter(enabled, STATS, 666, device)
                self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
                if cuda is not None:
                    self.assertTrue(torch.equal(cuda, torch.cuda.get_rng_state()))
                self.assertEqual(training.dictionary_digest(adapter.common_state_dict()), common)
                if initial is None:
                    initial = old.state_digest(adapter)
                self.assertEqual(old.state_digest(adapter), initial)
                self.assertEqual(adapter.energy_projection.weight.numel(), 160)
                self.assertEqual(bool(adapter.energy_projection.weight.requires_grad), enabled)
                self.assertEqual(int(torch.count_nonzero(adapter.energy_projection.weight)), 0)
                if not enabled:
                    self.assertEqual([n for n,p in adapter.named_parameters() if p.requires_grad],
                                     [n for n,p in original.named_parameters()])

    def test_control_forward_gradients_and_adam_update_match_original_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = TinyEnergy('train', Path(temporary))
            batch = energy_collate([data[i] for i in range(2)])
            old.setup(666, 2)
            model = CNNBiGRU(old.model_config())
            reference = old.make_adapter(old.ARMS['boundary'], 666, 'cpu')
            # Nonzero trained-like residuals exercise every shared path.
            with torch.no_grad():
                for name, value in reference.named_parameters():
                    if 'output' in name:
                        value.uniform_(-.2, .2)
            control = make_energy_adapter(False, STATS, 666, 'cpu')
            control.load_state_dict({**control.state_dict(), **reference.state_dict()}, strict=True)
            candidate = make_energy_adapter(True, STATS, 666, 'cpu')
            candidate.load_state_dict({**candidate.state_dict(), **reference.state_dict()}, strict=True)
            models = [copy.deepcopy(model) for _ in range(3)]
            outputs, losses, optimizers = [], [], []
            for current, adapter in zip(models, (reference, control, candidate)):
                parameters = [p for p in list(current.parameters())+list(adapter.parameters()) if p.requires_grad]
                optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=.0001)
                optimizers.append(optimizer)
                torch.manual_seed(814)
                scores, info = (old_forward(current, adapter, batch, 'cpu') if adapter is reference
                                else training.forward_batch(current, adapter, batch, 'cpu'))
                loss, _ = training_loss(scores, info, batch, 'cpu', True)
                outputs.append(scores)
                losses.append(loss)
                loss.backward()
            for scores, loss in zip(outputs[1:], losses[1:]):
                torch.testing.assert_close(scores, outputs[0], rtol=0, atol=0)
                torch.testing.assert_close(loss, losses[0], rtol=0, atol=0)
            for pair in zip(models[0].parameters(), models[1].parameters()):
                torch.testing.assert_close(pair[0].grad, pair[1].grad, rtol=0, atol=0)
            for name, parameter in reference.named_parameters():
                torch.testing.assert_close(parameter.grad, dict(control.named_parameters())[name].grad, rtol=0, atol=0)
            self.assertIsNone(control.energy_projection.weight.grad)
            self.assertGreater(int(torch.count_nonzero(candidate.energy_projection.weight.grad)), 0)
            for optimizer in optimizers[:2]:
                optimizer.step()
            self.assertEqual(old.state_digest(models[0]), old.state_digest(models[1]))
            self.assertEqual(old.state_digest(reference), training.dictionary_digest(control.common_state_dict()))

    def test_energy_uses_valid_zero_vectors_and_masks_invalid_tail(self):
        adapter = make_energy_adapter(True, STATS, 666, 'cpu')
        h, z = torch.randn(16,256), torch.randn(16)
        weights = torch.tensor([.02]*9+[.005]+[0.]*6)
        energy = torch.zeros(16,5)
        normalized = adapter.normalize_energy(energy, weights)
        self.assertGreater(int(torch.count_nonzero(normalized[:10])), 0)
        self.assertEqual(int(torch.count_nonzero(normalized[10:])), 0)
        with torch.no_grad():
            adapter.energy_projection.weight.fill_(.1)
            adapter.fine_output.weight.fill_(.2)
        baseline = adapter(h,z,weights,energy)['final_logits']
        padded = energy.clone()
        padded[10:] = float('nan')
        torch.testing.assert_close(adapter(h,z,weights,padded)['final_logits'], baseline, rtol=0, atol=0)
        torch.testing.assert_close(baseline[10:], z[10:], rtol=0, atol=0)
        changed = energy.clone()
        changed[:10,0] = 2.
        self.assertFalse(torch.equal(adapter(h,z,weights,changed)['final_logits'][:10], baseline[:10]))
        invalid = energy.clone()
        invalid[0,0] = float('nan')
        with self.assertRaises(ValueError):
            adapter(h,z,weights,invalid)
        with self.assertRaises(ValueError):
            adapter(h,z,weights,energy[:10])
        plain = old.make_adapter(old.ARMS['boundary'], 666, 'cpu')
        with self.assertRaises(ValueError):
            plain(h,z,weights,fine_condition=torch.zeros(16,31))
        bad = torch.zeros(16,32)
        bad[0,0] = float('nan')
        with self.assertRaises(ValueError):
            plain(h,z,weights,fine_condition=bad)

    def test_archives_paired_freeze_and_no_test_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, dev, test = (TinyEnergy(split, root) for split in ('train','dev','eval'))
            args = Namespace(output=root/'run', seed=666, epochs=2, batch_size=2, threads=2, device='cpu',
                             train_cache=train.folder, dev_cache=dev.folder,
                             train_energy=train.energy_folder, dev_energy=dev.energy_folder)
            completed = {}
            for arm in training.ARMS:
                args.arm = arm
                with contextlib.redirect_stdout(io.StringIO()):
                    completed[arm] = training.train_arm(args, train, dev)
                self.assertEqual(completed[arm]['updates'], 2)
                self.assertEqual(completed[arm]['archived_epochs'], 3)
                for epoch in range(3):
                    prefix = args.output/arm/'epochs'/f'epoch_{epoch:02d}'
                    self.assertTrue(prefix.with_suffix('.pt').exists())
                    self.assertTrue(prefix.with_name(prefix.name+'_scores20.npz').exists())
                    metrics = read_json(prefix.with_name(prefix.name+'_metrics.json'))
                    self.assertNotIn('selection_score_j_percent', metrics)
                    self.assertLessEqual(metrics['real_fpr_percent'], 5+1e-10)
            frozen = training.freeze_eval_plan(args.output)
            digest = sha256(args.output/'frozen_eval_plan.json')
            training.freeze_eval_plan(args.output)
            self.assertEqual(sha256(args.output/'frozen_eval_plan.json'), digest)
            with patch.object(training, 'evaluate_dev', side_effect=AssertionError('Test calibration is forbidden')):
                with contextlib.redirect_stdout(io.StringIO()):
                    reports = training.evaluate_arms(args, test, frozen)
            for arm in training.ARMS:
                self.assertEqual(reports[arm]['threshold'], completed[arm]['threshold'])
            self.assertEqual(read_json(args.output/'comparison.json')['status'], 'complete')
            with self.assertRaises(FileExistsError):
                training.evaluate_arms(args, test, frozen)

    def test_training_cli_rejects_eval_and_pretrained_options(self):
        prefix = ['--output','unused','--device','cpu']
        for options in (['--arm','boundary_energy5','--eval-energy','test'],
                        ['--freeze-only','--eval-cache','test'],
                        ['--evaluate','--train-energy','train','--eval-cache','test','--eval-energy','energy'],
                        ['--checkpoint','old.pt']):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    training.main(prefix+options)


if __name__ == '__main__':
    unittest.main()
