"""Paired frontend training, meaningful gradient checks and immutable selection."""
from argparse import Namespace
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cnn_ps.backbone import CNNBiGRU
from cnn_ps.io import read_json, save_json, sha256
from cnn_ps.learnable_filterbank import LearnableFilterbank
from cnn_ps.power_cache import power_collate
from cnn_ps import train_four as original
from cnn_ps import train_filterbank as training
from cnn_ps.train_dual import forward_batch as original_forward, training_loss
from test_four_training import TinyFeatures


class TinyPower(TinyFeatures):
    def __init__(self, split, root):
        super().__init__(split)
        self.folder, self.power_folder = root/(split+'_acoustic'), root/(split+'_power')
        self.folder.mkdir()
        self.power_folder.mkdir()
        save_json(self.folder/'complete.json', dict(split=split, synthetic=True))
        self.power_metadata = dict(split=split, synthetic=True)
        save_json(self.power_folder/'complete.json', self.power_metadata)

    def get_record(self, index, channel=0):
        row = super().get_record(index, channel)
        generator = torch.Generator().manual_seed(765+index+channel)
        row['power_features'] = torch.rand(1, len(row['fine_weights']), 257, generator=generator) + .05
        row['features'] = LearnableFilterbank(learnable=False)(row['power_features'])
        return row


class FilterbankTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_initial_forward_and_backend_gradients_match_unchanged_boundary(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = TinyPower('train', Path(temporary))
            batch = power_collate([data[i] for i in range(2)])
            # Match this experiment's batched projection arithmetic. Historical
            # per-record CPU matrix products may differ by float32 roundoff.
            reference_batch = dict(batch, features=LearnableFilterbank(learnable=False)(batch['power_features']))
            original.setup(666, 2)
            initial_model = CNNBiGRU(original.model_config())
            initial_adapter = original.make_adapter(original.ARMS['boundary'], 666, 'cpu')
            # Exercise trained-like nonzero auxiliary/residual paths too.
            with torch.no_grad():
                for name, value in initial_adapter.named_parameters():
                    if 'output' in name:
                        value.uniform_(-.2, .2)
            scores, losses, model_gradients, adapter_gradients = [], [], [], []
            for enabled in (None, False, True):
                model, adapter = copy.deepcopy(initial_model), copy.deepcopy(initial_adapter)
                frontend = None if enabled is None else LearnableFilterbank(original.model_config(), learnable=enabled)
                torch.manual_seed(814)
                final, info = (original_forward(model, adapter, reference_batch, 'cpu') if frontend is None else
                               training.forward_batch(model, adapter, frontend, batch, 'cpu'))
                loss, _ = training_loss(final, info, batch, 'cpu', True)
                scores.append(final.detach())
                losses.append(loss.detach())
                loss.backward()
                model_gradients.append({name: p.grad.clone() for name, p in model.named_parameters()})
                adapter_gradients.append({name: None if p.grad is None else p.grad.clone() for name, p in adapter.named_parameters()})
                if enabled is True:
                    self.assertTrue(torch.isfinite(frontend.raw_shift.grad).all())
                    self.assertGreater(int(torch.count_nonzero(frontend.raw_shift.grad)), 0)
                    optimizer = torch.optim.AdamW(frontend.parameters(), lr=.001, weight_decay=.0001)
                    optimizer.step()
                    self.assertGreater(int(torch.count_nonzero(frontend.raw_shift)), 0)
                elif enabled is False:
                    self.assertIsNone(frontend.raw_shift.grad)
            for current in (1, 2):
                torch.testing.assert_close(scores[0], scores[current], rtol=0, atol=0)
                torch.testing.assert_close(losses[0], losses[current], rtol=0, atol=0)
                for key, value in model_gradients[0].items():
                    torch.testing.assert_close(value, model_gradients[current][key], rtol=0, atol=0)
                for key, value in adapter_gradients[0].items():
                    if value is None:
                        self.assertIsNone(adapter_gradients[current][key])
                    else:
                        torch.testing.assert_close(value, adapter_gradients[current][key], rtol=0, atol=0)

    def test_shared_initialization_rng_and_canonical_nonzero_diagnostics(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        with tempfile.TemporaryDirectory() as temporary:
            data = TinyPower('dev', Path(temporary))
            batch = power_collate([data[i] for i in range(2)])
            for device in devices:
                outputs, digests = [], []
                for enabled in (False, True):
                    original.setup(666, 2)
                    model = CNNBiGRU(original.model_config()).to(device).eval()
                    adapter = original.make_adapter(original.ARMS['boundary'], 666, device).eval()
                    cpu = torch.get_rng_state().clone()
                    cuda = torch.cuda.get_rng_state().clone() if torch.cuda.is_available() else None
                    frontend = LearnableFilterbank(original.model_config(), learnable=enabled).to(device)
                    self.assertTrue(torch.equal(cpu, torch.get_rng_state()))
                    if cuda is not None:
                        self.assertTrue(torch.equal(cuda, torch.cuda.get_rng_state()))
                    digests.append(tuple(original.state_digest(m) for m in (model, adapter, frontend)))
                    with torch.no_grad():
                        outputs.append(training.forward_batch(model, adapter, frontend, batch, device)[0])
                    initial = frontend.frequencies_hz().detach().cpu().tolist()
                    with torch.no_grad():
                        frontend.raw_shift.copy_(torch.linspace(-2., 2., 80, device=device))
                    diagnostics = training.frontend_diagnostics(frontend, initial)
                    reloaded = LearnableFilterbank(original.model_config(), learnable=enabled)
                    reloaded.load_state_dict(training._cpu_state(frontend), strict=True)
                    self.assertEqual(diagnostics, training.frontend_diagnostics(reloaded, initial))
                self.assertEqual(digests[0], digests[1])
                torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

    def test_batch_padding_is_excluded_but_waveform_storage_tail_is_retained(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = TinyPower('dev', Path(temporary))
            batch = power_collate([data[i] for i in range(2)])
            original.setup(666, 2)
            model = CNNBiGRU(original.model_config()).eval()
            adapter = original.make_adapter(original.ARMS['boundary'], 666, 'cpu').eval()
            frontend = LearnableFilterbank(original.model_config(), learnable=True)
            reference = training.forward_batch(model, adapter, frontend, batch, 'cpu')[0]
            altered = dict(batch, power_features=batch['power_features'].clone())
            altered['power_features'][1, :, 8:] = float('nan')
            torch.testing.assert_close(training.forward_batch(model, adapter, frontend, altered, 'cpu')[0],
                                       reference, rtol=0, atol=0)
            # Record0 valid storage includes frames11..15 even though actual
            # duration has ended; corrupting these is not ignored as batch pad.
            altered['power_features'][0, 0, 15, 0] = float('nan')
            with self.assertRaises(ValueError):
                training.forward_batch(model, adapter, frontend, altered, 'cpu')

    def test_archives_freeze_all_epochs_reject_tamper_and_never_calibrate_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            datasets = {split: TinyPower(split, root) for split in ('train', 'dev', 'eval')}
            train, dev, test = (datasets[split] for split in ('train', 'dev', 'eval'))
            args = Namespace(output=root/'run', seed=666, epochs=2, batch_size=2, threads=2,
                             device='cpu', train_cache=train.folder, dev_cache=dev.folder,
                             train_power=train.power_folder, dev_power=dev.power_folder)
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
                    metrics = read_json(prefix.with_name(prefix.name+'_metrics.json'))
                    self.assertNotIn('selection_score_j_percent', metrics)
                    self.assertLessEqual(metrics['real_fpr_percent'], 5+1e-10)
            self.assertEqual(completed['fixed_mel']['final_frontend_diagnostics']['nonzero_parameters'], 0)
            self.assertGreater(completed['learnable_mel']['final_frontend_diagnostics']['nonzero_parameters'], 0)
            def fixture_cache(acoustic, power, **options):
                self.assertTrue(options['verify_hashes'])
                self.assertNotEqual(options['split'], 'eval')
                return datasets[options['split']]
            with patch.object(training, 'PowerFeatureData', side_effect=fixture_cache) as cache_reader:
                frozen = training.freeze_eval_plan(args.output)
                self.assertEqual(cache_reader.call_count, 2)
                digest = sha256(args.output/'frozen_eval_plan.json')
                training.freeze_eval_plan(args.output)
                self.assertEqual(sha256(args.output/'frozen_eval_plan.json'), digest)
                # A non-selected archived score is also protected.
                path = args.output/'fixed_mel'/'epochs'/'epoch_01_scores20.npz'
                original_bytes = path.read_bytes()
                path.write_bytes(original_bytes+b'tampered')
                with self.assertRaisesRegex(ValueError, 'Epoch archive changed'):
                    training.freeze_eval_plan(args.output)
                path.write_bytes(original_bytes)
            with patch.object(training, 'evaluate_dev', side_effect=AssertionError('Test calibration forbidden')):
                with contextlib.redirect_stdout(io.StringIO()):
                    reports = training.evaluate_arms(args, test, frozen)
            for arm in training.ARMS:
                self.assertEqual(reports[arm]['threshold'], completed[arm]['threshold'])
            self.assertEqual(read_json(args.output/'comparison.json')['status'], 'complete')
            with self.assertRaises(FileExistsError):
                training.evaluate_arms(args, test, frozen)

    def test_training_cli_rejects_eval_pretrained_and_mixed_inputs(self):
        prefix = ['--output', 'unused', '--device', 'cpu']
        for options in (['--arm', 'learnable_mel', '--eval-power', 'test'],
                        ['--freeze-only', '--eval-cache', 'test'],
                        ['--evaluate', '--train-power', 'train', '--eval-cache', 'test', '--eval-power', 'power'],
                        ['--checkpoint', 'old.pt']):
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    training.main(prefix+options)


if __name__ == '__main__':
    unittest.main()
